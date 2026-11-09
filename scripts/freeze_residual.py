#!/usr/bin/env python3
"""Freeze source, fresh input identities, order and common absolute calendars."""
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resident_residual.common import (ARMS,HOST,GPU_UUIDS,MODEL_REVISION,VLLM_COMMIT,
    PREVIOUS_CLOSURE,source_identity,seal,scale_calendar)
from resident_residual.inference import build_arm_orders,validate_arm_orders
from stock_comparison.data import digest
from followup.router_tapes import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['calibration','pilot','final'],required=True)
    p.add_argument('--split',choices=['pilot','final'],required=True)
    for n in ('capture-index','arrivals','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.phase!='calibration' and a.phase!=a.split:raise ValueError('Treatment phase/split mismatch')
    root=Path(__file__).resolve().parents[1]
    chosen_path=root/'configs/resident-residual-selected-v1.json'
    chosen=json.loads(chosen_path.read_text())
    capture=json.loads(a.capture_index.read_text());arrivals=json.loads(a.arrivals.read_text())
    banks=[dict(block_id=b['block_id'],dataset=b['dataset'],split=b['split'],
        capture_path=Path(b['bank_file']).name,capture_sha256=b['capture_sha256'],
        selection_content_sha256=arrivals['source_selection_sha256']) for b in capture if b['split']==a.split]
    expected=4 if a.split=='pilot' else 16
    if len(banks)!=expected or any(not b['correctness_pass'] or b['documents']!=32 for b in capture):
        raise ValueError('Invalid fresh capture matrix')
    from collections import Counter
    n=2 if a.split=='pilot' else 8
    if len({b['block_id'] for b in banks})!=expected or Counter(b['dataset'] for b in banks)!=Counter({'gsm8k_train':n,'wildchat':n}):
        raise ValueError('Wrong/duplicate two-dataset capture allocation')
    random.Random(471804).shuffle(banks)
    sha,files=source_identity()
    plan=dict(schema='resident_residual_plan_v1',phase=a.phase,split=a.split,arms=list(ARMS),jobs=4096,
        batch_tokens=128,source_sha256=sha,source_files=files,previous_closure_sha256=PREVIOUS_CLOSURE,
        capture_index_sha256=file_sha256(a.capture_index),arrivals_sha256=file_sha256(a.arrivals),
        hardware=dict(hostname=HOST,gpu_uuids=GPU_UUIDS,allocation_id='4872248'),
        checkpoint_revision=MODEL_REVISION,vllm_commit=VLLM_COMMIT,banks=banks,cases=[],order_seed=471804,
        protocol_sha256=file_sha256(root/'docs/resident-residual/protocol-v1.md'),
        selected_fix=chosen,selected_fix_sha256=file_sha256(chosen_path),selected_fix_content_sha256=digest(chosen),
        validity='One actual model-derived layer; same GPUs/max32replica allowance; GSM8K-train/WildChat content bundles; no MBPP')
    if a.phase!='calibration':
        if a.calibration is None:raise ValueError('Native-only calibration required')
        from stock_comparison.analysis import read_records
        rows,calibration_files=read_records(a.calibration)
        from resident_residual.analysis import analyze
        calibration_plan=json.loads((a.calibration/'provenance/plan.json').read_text())
        checked=analyze(calibration_plan,rows)
        if not checked['valid']:raise ValueError('Calibration validation failed: '+str(checked['errors']))
        if calibration_plan['source_sha256']!=sha:raise ValueError('Source changed since calibration')
        if calibration_plan['phase']!='calibration' or calibration_plan['split']!=a.split:
            raise ValueError('Calibration phase/split differs from treatment inputs')
        calibrated_banks={b['block_id']:b for b in calibration_plan['banks']}
        if calibrated_banks!={b['block_id']:b for b in banks}:
            raise ValueError('Capture/content identities changed since baseline calibration')
        by={r['content_group_id']:r for r in rows}
        if len(by)!=expected or len(rows)!=expected or any(r['arm']!='stockoff' or r['phase']!='calibration' for r in rows):
            raise ValueError('Wrong calibration rows')
        groups=[dict(id=b['block_id'],dataset=b['dataset']) for b in banks]
        orders=build_arm_orders(groups,seed=471804,phase=a.split)
        order_index={(c['content_group_id'],c['arrival_type'],c['rho']):c['arm_order'] for c in orders}
        plan['order_audit']=validate_arm_orders(orders,groups,phase=a.split)
        sources={b['id']:b for b in arrivals['blocks']}
        for b in banks:
            gid=b['block_id'];elapsed=by[gid]['calibration_capacity']['calibration_elapsed_ns']
            for service in ('code','conversation'):
                assigned=[x for x in arrivals['assignments'] if x['block_id']==gid and x['trace']==service]
                if len(assigned)!=1:raise ValueError('Missing unique source calendar assignment')
                source=sources[assigned[0]['arrival_block_id']]
                for rho in (.5,.85):
                    plan['cases'].append(dict(block_id=gid,content_group_id=gid,dataset=b['dataset'],
                        arrival_type=service,rho=rho,source_block=source,
                        calendar=scale_calendar(source,elapsed,rho=rho),arm_order=order_index[gid,service,rho]))
        random.Random(471805).shuffle(plan['cases'])
        plan['calibration_record_ids']={gid:r['record_id'] for gid,r in by.items()}
        plan['calibration_provenance']={gid:dict(record_id=r['record_id'],
            calibration_elapsed_ns=r['calibration_capacity']['calibration_elapsed_ns'],
            calibration_plan_sha256=r['plan_sha256'],allocation_id=r['allocation_id'],
            source_sha256=r['source_sha256'],raw_record_sha256=digest(r)) for gid,r in by.items()}
        plan['calibration_files']={Path(f['path']).name.removesuffix('.gz'):f['uncompressed_bytes_sha256'] for f in calibration_files}
        if len(plan['calibration_files'])!=len(calibration_files):raise ValueError('Duplicate compressed/uncompressed calibration files')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    final=seal(plan)
    from resident_residual.analysis import validate_plan
    validate_plan(final)
    a.output.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(dict(plan=str(a.output),phase=a.phase,banks=len(banks),cases=len(plan['cases']),sha=final['content_sha256'])))


if __name__=='__main__':main()
