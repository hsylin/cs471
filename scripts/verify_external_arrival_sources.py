#!/usr/bin/env python3
"""Independent selected Azure row, candidate-ranking and source identity audit.

No imports from the production selection code. Streams the original CSVs and
checks the full candidate inventories plus every selected job/boundary row.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path


def canonical_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ns(value):
    assert value.endswith('+00:00'),'Source must explicitly use UTC'
    whole,_,fraction=value[:-6].partition('.')
    delta=datetime.fromisoformat(whole).replace(tzinfo=timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    assert len(fraction)<=9
    return delta.days*86400*10**9+delta.seconds*10**9+int(fraction.ljust(9,'0'))


def verify_raw_block(block,rows):
    jobs=block['jobs']
    assert len(rows)==jobs+1,'Wrong source row count'
    assert canonical_sha(rows)==block['source_rows_sha256'],'Original source-row hash differs'
    assert [r[0] for r in rows[:-1]]==block['raw_timestamps'],'Raw timestamps differ'
    assert [r[1] for r in rows[:-1]]==block['context_tokens'],'Context metadata differs'
    assert [r[2] for r in rows[:-1]]==block['generated_tokens'],'Generated metadata differs'
    assert rows[-1]==[block['boundary_timestamp'],block['boundary_context_tokens'],block['boundary_generated_tokens']],'Boundary row differs'
    assert ns(rows[0][0])==block['source_origin_ns'],'Origin differs'
    assert ns(rows[-1][0])==block['boundary_timestamp_ns'],'Boundary timestamp ns differs'
    assert [ns(r[0])-ns(rows[0][0]) for r in rows[:-1]]==block['original_offsets_ns'],'Offsets differ'
    assert ns(rows[-1][0])-ns(rows[0][0])==block['original_horizon_ns'],'Horizon differs'
    assert ns(rows[-1][0])-ns(rows[-2][0])==block['terminal_gap_ns'],'Terminal gap differs'
    assert rows[0][0][:10]==rows[-1][0][:10]==block['source_date_utc'],'Candidate crosses source date'


def verify_rank_selection(blocks,candidates):
    expected=sorted(candidates,key=lambda c:(c['hash_rank_score'],c['row_start']))[:len(blocks)]
    actual=sorted(blocks,key=lambda b:b['day_hash_rank'])
    assert [b['day_hash_rank'] for b in actual]==list(range(1,len(actual)+1)),'Noncontiguous rank labels'
    assert [(b['row_start'],b['hash_rank_score']) for b in actual]==[(c['row_start'],c['hash_rank_score']) for c in expected],'Selected blocks are not hash winners'


def audit(manifest_path,downloads,candidate_root):
    manifest=json.loads(manifest_path.read_text())
    assert manifest['content_sha256']==canonical_sha({k:v for k,v in manifest.items() if k!='content_sha256'}),'Manifest canonical hash differs'
    assert file_sha(downloads/'azure-week-download-manifest.json')==manifest['source_inventory_sha256'],'Download inventory differs'
    for name,pin in manifest['source_documentation'].items():
        assert file_sha(downloads/name)==pin['sha256'],'Documentation artifact differs'
    report={'status':'pass','manifest_sha256':file_sha(manifest_path),'verifier_sha256':file_sha(__file__),
            'selected_blocks':len(manifest['blocks']),'selected_source_rows':0,'sources':{}}
    inventory_by_name={r['file']:r for r in manifest['candidate_inventories']}
    for trace in ('code','conversation'):
        blocks=[b for b in manifest['blocks'] if b['trace']==trace];assert blocks
        jobs={b['jobs'] for b in blocks};assert len(jobs)==1;jobs=jobs.pop()
        bydate=defaultdict(list)
        inventory=candidate_root/(trace+'-all-candidates.jsonl.gz')
        pin=inventory_by_name[inventory.name]
        assert file_sha(inventory)==pin['sha256'] and inventory.stat().st_size==pin['bytes'],'Candidate inventory identity differs'
        with gzip.open(inventory,'rt') as stream:
            for line in stream:
                candidate=json.loads(line)
                assert candidate['trace']==trace
                assert candidate['hash_rank_score']==canonical_sha([manifest['selection']['seed'],trace,candidate['source_date_utc'],candidate['row_start']]),'Candidate score differs'
                assert candidate['row_stop_exclusive']==candidate['boundary_row']==candidate['row_start']+jobs
                bydate[candidate['source_date_utc']].append(candidate)
        assert sum(map(len,bydate.values()))==pin['candidate_rows']
        for date,entries in bydate.items():
            selected=[b for b in blocks if b['source_date_utc']==date]
            if selected:verify_rank_selection(selected,entries)
        wanted={};found={b['id']:[] for b in blocks}
        for block in blocks:
            for index in range(block['row_start'],block['boundary_row']+1):
                assert index not in wanted,'Selected source row reused';wanted[index]=block['id']
        source_name='AzureLLMInferenceTrace_'+('code' if trace=='code' else 'conv')+'_1week.csv'
        h=hashlib.sha256();dates=Counter();first={};precisions=Counter();previous=None;ties=0;inversions=0;rows=0
        with (downloads/source_name).open('rb') as stream:
            header=next(stream);h.update(header);assert header.strip()==b'TIMESTAMP,ContextTokens,GeneratedTokens'
            for i,line in enumerate(stream):
                h.update(line);stamp=line.split(b',',1)[0]
                # Both pinned assets have only integer-second or6-digit UTC
                # timestamps. Normalization preserves equality at .000000.
                assert len(stamp) in (25,32) and stamp.endswith(b'+00:00'),'Unexpected timestamp precision/zone'
                frac=stamp[20:26] if len(stamp)==32 else b'000000'
                if len(stamp)==32:assert stamp[19:20]==b'.' and frac.isdigit()
                key=(stamp[:19],frac);precision=6 if len(stamp)==32 else 0;precisions[precision]+=1
                if previous is not None:
                    ties+=key==previous;inversions+=key<previous
                previous=key;day=stamp[:10].decode();dates[day]+=1;first.setdefault(day,i);rows+=1
                if i in wanted:
                    timestamp,context,generated=next(csv.reader([line.decode()]));found[wanted[i]].append([timestamp,int(context),int(generated)])
        assert inversions==0,'Original source contains inversions'
        assert {b['source_sha256'] for b in blocks}=={h.hexdigest()},'Original source byte hash differs'
        for date,count in dates.items():
            entries=bydate[date]
            assert len(entries)==count//(jobs+1),'Candidate count differs'
            assert sorted(e['row_start'] for e in entries)==list(range(first[date],first[date]+(jobs+1)*(count//(jobs+1)),jobs+1)),'Candidate day-bin positions differ'
        assert sorted(dates)==manifest['services'][trace]['all_dates'],'Source dates differ'
        for block in blocks:verify_raw_block(block,found[block['id']])
        report['selected_source_rows']+=len(wanted)
        report['sources'][trace]={'source_sha256':h.hexdigest(),'rows':rows,'dates':dict(dates),
            'candidate_bins':sum(map(len,bydate.values())),'selected_blocks':len(blocks),'source_rows_verified':len(wanted),
            'timestamp_ties':ties,'timestamp_inversions':inversions,'fractional_precision_counts':{str(k):v for k,v in sorted(precisions.items())}}
        print(json.dumps({'trace':trace,'verified_source_rows':rows,'selected_blocks':len(blocks)}),flush=True)
    report['checks']=['Manifest/documentation/download inventory identity','All candidate inventory bytes, scores and disjoint day-bin positions',
        'Hash-ranked selection','Every selected original timestamp/token-length row and boundary','Integer time offsets/horizon and raw-row SHA',
        'Full source byte SHA','No selected source-row reuse','Source order, ties, fractional precision and date coverage']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--downloads',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh audit output required; preserve previous audit')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:report=audit(args.manifest,args.downloads,args.candidate_root or args.manifest.parent)
    except Exception as error:
        report={'status':'failed','verifier_sha256':file_sha(__file__),'error_type':type(error).__name__,'error':str(error)}
        args.output.write_text(json.dumps(report,indent=2)+'\n');raise
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'selected_blocks':report['selected_blocks']}))

if __name__=='__main__':main()
