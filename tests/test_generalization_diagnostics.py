"""CPU-only diagnostic contracts; synthetic clocks are not benchmark evidence."""
import copy
from collections import Counter
import itertools
from pathlib import Path
import sys
import types
import numpy as np
import pytest
from generalization_diagnostics.plan import build,validate,sequence_for,CONDITIONS,PAIRS
from optimization.common import canonical_hash,manifest_hash
from tests.test_generalization_analysis import fixture as main_fixture


def test_frozen_pilot_selection_and_24_conditions_do_not_change_main_source():
    from generalization.common import source_identity
    before=source_identity()[0];plan=build('configs/generalization-pilot-v2.json')
    assert source_identity()[0]==before==plan['main_execution_source_sha256']
    assert plan==build('configs/generalization-pilot-v2.json')
    assert len(plan['states'])==3 and sum(len(s['condition_order']) for s in plan['states'])==24
    assert {s['dataset'] for s in plan['states']}=={'gsm8k','mbpp','wildchat'}
    for state in plan['states']:
        assert set(state['condition_order'])==set(CONDITIONS)
        assert state['source_case']['source_block']['trace']=='code'
        assert state['source_case']['calendar']['rate_kind']=='calibrated'
    for pair,names in plan['comparison_pairs'].items():
        control_first=sum(s['condition_order'].index(names[0])<s['condition_order'].index(names[1]) for s in plan['states'])
        assert control_first in (1,2)
    changed=copy.deepcopy(plan);changed['states'][0]['condition_order'].reverse()
    changed['freeze']['sha256']=manifest_hash(changed)
    with pytest.raises(ValueError):validate(changed)


def test_phase_sensitivity_preserves_multiset_but_changes_observation_window():
    a,b=sequence_for('cycle0'),sequence_for('cycle16')
    assert Counter(a)==Counter(b)==Counter({i:8 for i in range(16,48)})
    assert len(set(sequence_for('main')))==256
    for end in (63,127,191,255):
        assert a[end-15:end+1]==list(range(32,48))
        assert b[end-15:end+1]==list(range(16,32))
    with pytest.raises(ValueError):sequence_for('unknown')


def test_precomputed_adapter_selects_exact_pointer_and_charges_each_call(monkeypatch):
    class Backend:
        def __init__(self,torch,physical):self.raw_index=0;self.physical=physical
        def __call__(self,ids,*args):return self.physical[self.raw_index]
    monkeypatch.setitem(sys.modules,'resident_dispatch.gpu',types.SimpleNamespace(PrecomputedMapper=Backend))
    from generalization_diagnostics.runner import IndexedPrecomputedMapper
    class Tensor:
        def __init__(self,ptr):self.ptr=ptr
        def data_ptr(self):return self.ptr
    inputs=[Tensor(100),Tensor(200)];routes=['route0','route1']
    mapper=IndexedPrecomputedMapper(None,routes,inputs)
    assert mapper(Tensor(200))=='route1' and mapper(Tensor(100))=='route0'
    assert mapper.calls==2
    with pytest.raises(KeyError):mapper(Tensor(300))
    with pytest.raises(ValueError):IndexedPrecomputedMapper(None,routes,[Tensor(100),Tensor(100)])


def diagnostic_fixture():
    pilot,source_rows=main_fixture();chosen=[g for g in pilot['groups'] if g['id'].endswith('-0')]
    states=[]
    for group in chosen:
        case=next(c for c in pilot['cases'] if c['block_id']==group['id'] and c['source_block']['trace']=='code' and c['calendar']['rate_kind']=='calibrated')
        states.append(dict(content_group_id=group['id'],dataset=group['dataset'],source_case=case,condition_order=list(CONDITIONS)))
    plan=dict(states=states,diagnostic_source_sha256='synthetic-diagnostic-source',hardware=pilot['hardware'],
        physical_to_logical_map=pilot['physical_to_logical_map'],comparison_pairs={p:(['offline_native','offline_online'] if p=='offline' else [p+'_shortcut',p+'_combined']) for p in PAIRS})
    rows=[]
    for state in states:
        for condition in state['condition_order']:
            seq_name,mode=CONDITIONS[condition];arm='combined' if mode in ('online','pre_online') else 'shortcut'
            row=copy.deepcopy(next(r for r in source_rows if r['content_group_id']==state['content_group_id'] and r['arrival_type']=='code' and r['rate_kind']=='calibrated' and r['arm']==arm))
            seq=sequence_for(seq_name)
            row.update(generalization_diagnostic_schema_version=1,phase='diagnostic',diagnostic_only=True,
                diagnostic_config_hash=manifest_hash(plan),diagnostic_source_sha256=plan['diagnostic_source_sha256'],
                main_execution_source_sha256=pilot['execution_source_sha256'],pilot_plan_sha256=pilot['freeze']['sha256'],
                condition=condition,sequence_name=seq_name,dispatch_mode=mode,condition_order=state['condition_order'],
                measured_raw_indices=seq,unique_measured_job_combinations=len(set(seq)),raw_job_frequency={str(i):seq.count(i) for i in sorted(set(seq))},
                online_selection_cost_included=mode=='online',precomputed_selection_cost_excluded=mode.startswith('pre_'),precomputed_adapter_cost_included=mode.startswith('pre_'),
                mapper_calls_measured=0 if mode=='native' else 256,precomputed_gpu_adapter_checks=544,postflight_reference_raw_indices=[16,271],
                executed_logical_routing_sha256=state['content_group_id']+seq_name,executed_combining_weights_sha256=canonical_hash('weights-'+seq_name),
                executed_combining_weights_encoding='little-endian float32; C-order; no header',instrumentation_revision=2)
            row['assignment_metrics']=[dict(per_rank_tokens=[512,512],per_rank_active_copies=[64,64],per_rank_m_tiles=[64,64],per_rank_padded_rows=[1024,1024]) for _ in range(256)]
            row['record_id']=canonical_hash({k:row[k] for k in ('diagnostic_config_hash','phase','content_group_id','condition')})
            rows.append(row)
    return pilot,plan,rows


def test_descriptive_analysis_has_no_ci_or_acceptance_and_preserves_all_pairs(monkeypatch):
    from generalization_diagnostics import analysis
    pilot,plan,rows=diagnostic_fixture();monkeypatch.setattr(analysis,'validate',lambda _: (pilot,{}))
    out=analysis.analyze(plan,rows)
    assert out['valid'],out['errors']
    assert out['valid_records']==24 and len(out['paired_cases'])==30
    assert out['confidence_intervals'] is None and out['acceptance'] is None
    assert all(p['ratios']['queue_wait_mean_ns'] is None for p in out['paired_cases'])


@pytest.mark.parametrize('mutation',['missing','sequence','calendar','offline_route','omit_adapter','order'])
def test_descriptive_audit_rejects_changed_execution(monkeypatch,mutation):
    from generalization_diagnostics import analysis
    pilot,plan,rows=diagnostic_fixture();monkeypatch.setattr(analysis,'validate',lambda _: (pilot,{}))
    if mutation=='missing':rows.pop()
    elif mutation=='sequence':rows[2]['measured_raw_indices'][0]=99
    elif mutation=='calendar':rows[0]['arrival_offsets_ns'][4]+=1
    elif mutation=='offline_route':rows[6]['assignment_sha256']='wrong'
    elif mutation=='omit_adapter':rows[6]['precomputed_adapter_cost_included']=False
    elif mutation=='order':rows[0],rows[1]=rows[1],rows[0]
    assert not analysis.analyze(plan,rows)['valid']


def test_float_weight_identity_preserves_subunit_values_and_normalizes_encoding():
    import hashlib,struct
    from generalization_diagnostics.runner import floating_weight_hash
    a=np.array([[.2,.8],[.3,.7]],dtype=np.float32)
    b=np.array([[.1,.9],[.4,.6]],dtype=np.float32)
    # The superseded integer encoding loses every value in both valid examples.
    assert np.array_equal(a.astype(np.int64),b.astype(np.int64))
    assert floating_weight_hash(a)!=floating_weight_hash(b)
    assert floating_weight_hash(a)==hashlib.sha256(struct.pack('<4f',.2,.8,.3,.7)).hexdigest()
    assert floating_weight_hash(a)==floating_weight_hash(np.asfortranarray(a))
    assert floating_weight_hash(a)==floating_weight_hash(a.astype('>f4'))
    with pytest.raises(ValueError):floating_weight_hash(np.array([float('nan')]))


def test_old_diagnostic_weight_encoding_is_ineligible(monkeypatch):
    from generalization_diagnostics import analysis
    pilot,plan,rows=diagnostic_fixture();monkeypatch.setattr(analysis,'validate',lambda _: (pilot,{}))
    for row in rows:
        row.pop('executed_combining_weights_encoding');row.pop('instrumentation_revision')
    out=analysis.analyze(plan,rows)
    assert not out['valid'] and out['valid_records']==0
