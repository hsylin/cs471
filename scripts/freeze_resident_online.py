#!/usr/bin/env python3
"""Freeze the single online algorithm and fixed development/fresh inventories."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from optimization.common import manifest_hash
from resident_dispatch.common import execution_fingerprint
from resident_dispatch.online import POLICY_ID

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validation','final'),required=True)
    p.add_argument('--capture-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads((ROOT/'configs/resident-headroom-frozen-v2.json').read_text())
    m.pop('freeze');m.pop('calibration_revision');m.pop('pilot')
    m.update(name='resident-online-'+a.phase+'-v1',phase=a.phase,online_algorithm=POLICY_ID)
    m['environment']['resident_dispatch_sha256'],m['execution_source_sha256']=execution_fingerprint()
    m['analysis_source_sha256']=hashlib.sha256((ROOT/'resident_dispatch/analysis.py').read_bytes()).hexdigest()
    m['online_protocol_sha256']=hashlib.sha256((ROOT/'docs/resident-online-validation-protocol.md').read_bytes()).hexdigest()
    m['matrix'].pop('domains')
    m['matrix'].update(arms=['native','online'],regimes=['domain0','domain1'])
    m['execution']['timing_scope']='Completed max-rank elapsed forward+native SKIP step+normal synchronous completion, including current-batch selector, recording, expert kernels, collective and planning; static materialization/JIT/reference checks/initialized timing events and prefix+64 full-policy conditioning excluded; GC enabled'
    m['execution'].update(measurement_setup_revision=2,full_policy_conditioning_steps=64,
        measured_steps=256 if a.phase=='validation' else 512,
        final_repetitions_in_distinct_allocations=True,performance_retuning_allowed=False)
    if a.phase=='validation':
        plan=json.loads((ROOT/'configs/resident-online-validation-analysis-v1.json').read_text())
    else:
        if a.capture_dir is None:raise ValueError('Final requires immutable capture inventory')
        m.update(split='final');m['trace']['seed']=20471
        source=ROOT/'resident_dispatch/corpus/sources.json'
        corpus=json.loads(source.read_text())
        m['source_manifest']={'path':'resident_dispatch/corpus/sources.json','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        m['excluded_source_document_ids']=[]
        for path in ('followup/corpus/sources.json','optimization/corpus/sources.json'):
            for block in json.loads((ROOT/path).read_text())['blocks']:
                m['excluded_source_document_ids'] += [s['source_id'] for s in block['sources']]
        m['blocks']=[dict(block_id=b['block_id'],source_document_ids=[s['source_id'] for s in b['sources']]) for b in corpus['blocks']]
        m['matrix']['block_ids']=[b['block_id'] for b in m['blocks']]
        m['matrix']['regimes']=['stable','transient','persistent','switching']
        index=json.loads((a.capture_dir/'capture-index.json').read_text())
        m['banks']=[];m['tapes']=[]
        for bank in index:
            block=bank['block_id'];cap=bank['capture_sha256']
            m['banks'].append(dict(block_id=block,capture_path='banks/'+block+'.npz',capture_sha256=cap))
            for replay in bank['replays']:
                filename=Path(replay['file']).name
                suffix=filename.removeprefix(block+'-').removesuffix('.npz')
                regime,batch=suffix.rsplit('-b',1)
                m['tapes'].append(dict(block_id=block,regime=regime,batch_tokens=int(batch),
                    capture_path='banks/'+block+'.npz',capture_sha256=cap,
                    replay_path='tapes/'+filename,replay_sha256=replay['sha256']))
        if len(m['banks'])!=8 or len(m['tapes'])!=64:raise ValueError('Incomplete fresh inventory')
        plan=dict(phase='final',expected_blocks=m['matrix']['block_ids'],expected_regimes=m['matrix']['regimes'],
            expected_batches=[128,1024],expected_repetitions=[0,1],native_arm='native',treatment_arm='online',
            bootstrap_samples=20000,bootstrap_seed=10471,useful_gain_ratio=.95,stable_ratio_limit=1.02)
    m['analysis']=plan
    m['freeze']={'algorithm':'sha256','sha256':manifest_hash(m)}
    a.output.write_text(json.dumps(m,indent=2)+'\n')
    print(m['freeze']['sha256'])

if __name__=='__main__':main()
