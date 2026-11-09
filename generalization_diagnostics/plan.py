"""Outcome-independent selection and ordering of exactly 24 diagnostic trials."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from generalization.common import load_plan as load_main_plan,source_identity as main_identity
from optimization.common import canonical_hash,manifest_hash
from resident_dispatch.common import execution_fingerprint

ROOT=Path(__file__).resolve().parents[1]
SEED=471520
DATASETS=('gsm8k','mbpp','wildchat')
PAIRS=('main','cycle0','cycle16','offline')
CONDITIONS={
 'main_shortcut':('main','native'), 'main_combined':('main','online'),
 'cycle0_shortcut':('cycle0','native'), 'cycle0_combined':('cycle0','online'),
 'cycle16_shortcut':('cycle16','native'), 'cycle16_combined':('cycle16','online'),
 'offline_native':('main','pre_native'), 'offline_online':('main','pre_online')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    _,files=main_identity()
    files.update(execution_fingerprint()[1])
    names=['generalization_diagnostics/'+x for x in ('__init__.py','plan.py','runner.py','analysis.py')]
    names+=['scripts/freeze_generalization_diagnostics.py','scripts/run-generalization-diagnostics-kuma.sh',
        'docs/generalization/diagnostic-protocol-v1.md','generalization/analysis.py','generalization/inference.py']
    files.update({name:sha(ROOT/name) for name in names})
    return canonical_hash(files),files


def sequence_for(name):
    if name=='main':return list(range(16,272))
    if name not in ('cycle0','cycle16'):raise ValueError('Unknown sequence')
    phase=0 if name=='cycle0' else 16
    return [16+(i+phase)%32 for i in range(256)]


def build(pilot_path):
    pilot_path=Path(pilot_path).resolve();pilot,_=load_main_plan(pilot_path)
    if pilot['phase']!='pilot' or pilot['split']!='pilot':raise ValueError('Frozen development pilot plan required')
    groups=[]
    for dataset in DATASETS:
        candidates=[g for g in pilot['groups'] if g['dataset']==dataset]
        if len(candidates)!=2:raise ValueError('Two pilot groups per dataset required')
        groups.append(min(candidates,key=lambda g:canonical_hash([SEED,'select',dataset,g['id']])))
    order=list(PAIRS);random.Random(SEED).shuffle(order)
    states=[]
    for di,g in enumerate(groups):
        source=[c for c in pilot['cases'] if c['block_id']==g['id'] and c['source_block']['trace']=='code' and c['calendar']['rate_kind']=='calibrated']
        if len(source)!=1:raise ValueError('Unique frozen calibrated code case required')
        rotated=order[di:]+order[:di];conditions=[]
        for pair in rotated:
            names=['offline_native','offline_online'] if pair=='offline' else [pair+'_shortcut',pair+'_combined']
            if (di+PAIRS.index(pair))%2:names.reverse()
            conditions.extend(names)
        states.append(dict(content_group_id=g['id'],dataset=g['dataset'],source_case=source[0],condition_order=conditions))
    random.Random(SEED+1).shuffle(states)
    execution,files=source_identity()
    result=dict(generalization_diagnostic_schema_version=1,phase='diagnostic',seed=SEED,
        pilot_plan={'path':str(pilot_path.relative_to(ROOT)),'sha256':pilot['freeze']['sha256'],'file_sha256':sha(pilot_path)},
        main_execution_source_sha256=pilot['execution_source_sha256'],diagnostic_source_sha256=execution,
        diagnostic_source_files=files,hardware=pilot['hardware'],batch_tokens=128,physical_to_logical_map=pilot['physical_to_logical_map'],
        states=states,expected_trials=24,measured_jobs=256,evidence_eligible=False,
        sequence_definitions={k:sequence_for(k) for k in ('main','cycle0','cycle16')},
        comparison_pairs={k:(['offline_native','offline_online'] if k=='offline' else [k+'_shortcut',k+'_combined']) for k in PAIRS},
        independent_unit='three selected development content states; no inferential interval or acceptance gate',
        selection_uses_treatment_outcomes=False,selection_uses_final_inputs=False)
    result['freeze']={'sha256':manifest_hash(result),'stage':'before diagnostic timing'}
    return result


def validate(plan):
    if plan.get('generalization_diagnostic_schema_version')!=1 or plan.get('freeze',{}).get('sha256')!=manifest_hash(plan):raise ValueError('Changed diagnostic plan')
    original=ROOT/plan['pilot_plan']['path']
    if sha(original)!=plan['pilot_plan']['file_sha256']:raise ValueError('Changed referenced pilot plan')
    pilot,base=load_main_plan(original)
    if pilot['freeze']['sha256']!=plan['pilot_plan']['sha256']:raise ValueError('Wrong pilot manifest')
    if plan!=build(original):raise ValueError('Diagnostic selection/source/order differs from deterministic freeze')
    return pilot,base
