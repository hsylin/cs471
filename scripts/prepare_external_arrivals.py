#!/usr/bin/env python3
"""Prepare pinned Azure2024 service-specific day blocks without loading CSVs."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generalization.external_arrivals import scan_trace, select_service_days


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloads',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh output directory required')
    inventory_path=args.downloads/'azure-week-download-manifest.json'
    inventory=json.loads(inventory_path.read_text())
    by_name={x['name']:x for x in inventory}
    sources=[('code','AzureLLMInferenceTrace_code_1week.csv'),
             ('conversation','AzureLLMInferenceTrace_conv_1week.csv')]
    for _,name in sources:
        if name not in by_name:raise ValueError(f'Missing pinned asset {name}')
        if (args.downloads/name).stat().st_size != by_name[name]['bytes']:
            raise ValueError(f'Wrong source size: {name}')
    args.output.mkdir(parents=True)
    scans=[];candidate_files=[]
    try:
        for trace,name in sources:
            target=args.output/f'{trace}-all-candidates.jsonl.gz'
            print(json.dumps({'event':'scan_start','trace':trace,'source':name}),flush=True)
            with open(target,'wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as compressed:
                    with io.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as stream:
                        scan=scan_trace(args.downloads/name,trace=trace,expected_sha256=by_name[name]['sha256'],
                                        seed=471502,jobs=256,keep_per_day=8,inventory_stream=stream)
            scan['release_asset']=by_name[name]
            (args.output/f'{trace}-scan.json').write_text(json.dumps(scan,indent=2)+'\n')
            scans.append(scan)
            candidate_files.append({'file':target.name,'sha256':sha256(target),'bytes':target.stat().st_size,
                                    'candidate_rows':sum(d['candidate_bins'] for d in scan['days'])})
            print(json.dumps({'event':'scan_complete','trace':trace,'rows':scan['rows'],
                              'dates':[d['date_utc'] for d in scan['days']],
                              'precision_counts':scan['fractional_precision_counts'],
                              'inversions':scan['timestamp_inversions']}),flush=True)
        selected=select_service_days(scans)
        selected['source_inventory_sha256']=sha256(inventory_path)
        selected['candidate_inventories']=candidate_files
        selected['source_documentation']={name:{'sha256':sha256(args.downloads/name),'bytes':(args.downloads/name).stat().st_size}
            for name in ('azure-2024-release.json','azure-2024-readme.md','azure-license.txt')}
        selected['documentation_date_note']='README describes May10–19. Actual assets cover code May10–16 and conversation May12–18; scanned dates are authoritative for this study.'
        selected['selector_source_sha256']={'generalization/external_arrivals.py':sha256(Path(__file__).resolve().parents[1]/'generalization/external_arrivals.py'),
                                           'scripts/prepare_external_arrivals.py':sha256(__file__)}
        from generalization.data import digest
        selected['content_sha256']=digest({k:v for k,v in selected.items() if k!='content_sha256'})
        (args.output/'arrival-manifest.json').write_text(json.dumps(selected,indent=2)+'\n')
        report={'status':'complete','blocks':len(selected['blocks']),
                'pilot_blocks':sum(b['split']=='pilot' for b in selected['blocks']),
                'final_blocks':sum(b['split']=='final' for b in selected['blocks']),
                'final_calendar_date_clusters':selected['final_calendar_date_clusters'],
                'cross_service_pilot_final_calendar_overlap':selected['cross_service_pilot_final_calendar_overlap'],
                'global_calendar_future_holdout':selected['global_calendar_future_holdout'],
                'within_service_temporal_holdout':True,'manifest_sha256':sha256(args.output/'arrival-manifest.json')}
        (args.output/'preparation-status.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    except Exception as error:
        (args.output/'preparation-status.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n')
        raise

if __name__=='__main__':main()
