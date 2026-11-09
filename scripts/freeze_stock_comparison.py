#!/usr/bin/env python3
"""Freeze source, input identities, order and absolute calendars before timing."""
import argparse,itertools,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stock_comparison.common import ARMS,HOST,GPU_UUIDS,MODEL_REVISION,VLLM_COMMIT,source_identity,seal,scale_calendar
from followup.router_tapes import file_sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['calibration','pilot','final'],required=True)
    p.add_argument('--split',choices=['pilot','final'],required=True)
    for name in ('capture-index','arrivals','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    capture=json.loads(a.capture_index.read_text());arrivals=json.loads(a.arrivals.read_text())
    banks=[dict(block_id=b['block_id'],dataset=b['dataset'],split=b['split'],capture_path=Path(b['bank_file']).name,capture_sha256=b['capture_sha256'],selection_content_sha256=arrivals['source_selection_sha256']) for b in capture if b['split']==a.split]
    expected=6 if a.split=='pilot' else 18
    if len(banks)!=expected or any(not b['correctness_pass'] or b['documents']!=32 for b in capture):raise ValueError('Invalid fresh capture count')
    random.Random(471604).shuffle(banks)
    sha,files=source_identity()
    plan=dict(schema='stock_comparison_plan_v1',phase=a.phase,split=a.split,
        arms=list(ARMS),jobs=4096,batch_tokens=128,source_sha256=sha,source_files=files,
        capture_index_sha256=file_sha256(a.capture_index),arrivals_sha256=file_sha256(a.arrivals),
        hardware=dict(hostname=HOST,gpu_uuids=GPU_UUIDS,allocation_id='4872248'),
        checkpoint_revision=MODEL_REVISION,vllm_commit=VLLM_COMMIT,
        banks=banks,cases=[],order_seed=471604,protocol_sha256=file_sha256(Path(__file__).resolve().parents[1]/'docs/stock-comparison/protocol-v1.md'),
        validity='layer-job only; native module/router included; same GPUs and max32extra replica allowance; stockoff has128slots others160')
    if a.phase!='calibration':
        if a.calibration is None:raise ValueError('Native-only calibration required')
        rows=[]
        for f in sorted((a.calibration/'trials').glob('*.jsonl')):
            rows.extend(json.loads(s) for s in f.read_text().splitlines() if s)
        from stock_comparison.analysis import analyze
        calibration_plan_path=a.calibration/'provenance/plan.json'
        calibration_plan=json.loads(calibration_plan_path.read_text())
        checked=analyze(calibration_plan,rows)
        if not checked['valid']:
            raise ValueError('Calibration audit failed before deriving rates: '+str(checked['errors']))
        if calibration_plan['source_sha256']!=sha:
            raise ValueError('Execution source changed since calibration; recalibrate before treatment')
        by={r['content_group_id']:r for r in rows}
        if len(by)!=expected or len(rows)!=expected or any(r['arm']!='stockoff' or r['phase']!='calibration' for r in rows):raise ValueError('Wrong calibration records')
        sources={b['id']:b for b in arrivals['blocks']}
        # Arrival selector stores explicit content-group assignments. No outcome
        # participates in which source block is attached to which group.
        assigned=arrivals['assignments']
        permutations=list(itertools.permutations(ARMS))
        for dataset_index,dataset in enumerate(('gsm8k','mbpp','wildchat')):
            group_ids=sorted(b['block_id'] for b in banks if b['dataset']==dataset)
            for loadidx,rho in enumerate((.5,.85)):
                j=dataset_index*4 if a.split=='pilot' else 0
                for gid in group_ids:
                    for service in ('code','conversation'):
                        assignment=[x for x in assigned if x['block_id']==gid and x['trace']==service]
                        if len(assignment)!=1:raise ValueError('Missing unique content/calendar assignment')
                        source=sources[assignment[0]['arrival_block_id']]
                        row=by[gid];elapsed=row['calibration_capacity']['calibration_elapsed_ns']
                        calendar=scale_calendar(source,elapsed,rho=rho)
                        order=list(permutations[(j+loadidx*3)%6]);j+=1
                        plan['cases'].append(dict(block_id=gid,dataset=dataset,source_block=source,calendar=calendar,arm_order=order))
        random.Random(471605).shuffle(plan['cases'])
        plan['calibration_record_ids']={gid:r['record_id'] for gid,r in by.items()}
        plan['calibration_provenance']={gid:dict(record_id=r['record_id'],
            calibration_elapsed_ns=r['calibration_capacity']['calibration_elapsed_ns'],
            calibration_plan_sha256=r['plan_sha256'],allocation_id=r['allocation_id'],
            source_sha256=r['source_sha256'],raw_record_sha256=__import__('stock_comparison.data',fromlist=['digest']).digest(r))
            for gid,r in by.items()}
        plan['calibration_files']={f.name:file_sha256(f) for f in sorted((a.calibration/'trials').glob('*.jsonl'))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(seal(plan),indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output),phase=a.phase,banks=len(banks),cases=len(plan['cases']),sha=seal(plan)['content_sha256'])))
if __name__=='__main__':main()
