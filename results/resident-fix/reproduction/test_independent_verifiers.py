"""CPU-only QA; synthetic fixtures are never experimental observations.

Production imports occur only here as comparison targets, never in either
independent auditor. These tests exercise corruption detection and semantics.
"""
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
stats=load('independent_stats',HERE/'verify_resident_fix_statistics.py')
raw=load('independent_raw',HERE/'audit_resident_fix_records.py')
fixtures=load('synthetic_inference',REPO/'tests/test_resident_fix_inference.py')
state_fixtures=load('synthetic_static',REPO/'tests/test_resident_fix_analysis.py')
clock_fixtures=load('synthetic_clocks',REPO/'tests/test_stock_comparison_analysis.py')
from resident_fastpath.inference import infer_endpoints
from stock_comparison.analysis import clock_metrics


def independently_infer(pairs,groups,n=300):
    by={d:sorted(g['id'] for g in groups if g['dataset']==d) for d in ('gsm8k','wildchat','mbpp')}
    index={tuple(p[k] for k in ('content_group_id','arrival_type','rho','contrast')):p['log_ratios'] for p in pairs}
    cases={tuple(p[k] for k in ('content_group_id','arrival_type','rho')):{'calendar':{'arrival_date':p['arrival_date']}} for p in pairs}
    return stats.infer_pairs(index,by,cases,draw_count=n)


def compare_subset(actual,reference):
    if isinstance(actual,dict):
        for key,value in actual.items():compare_subset(value,reference[key])
    elif isinstance(actual,list):
        assert len(actual)==len(reference)
        for a,b in zip(actual,reference):compare_subset(a,b)
    elif isinstance(actual,float):assert actual==pytest.approx(reference,rel=2e-12,abs=2e-10)
    else:assert actual==reference


@pytest.mark.parametrize('kwargs',({}, {'main_fix':1.1}, {'sentinel_fix':10.}, {'low_fix':1.02}))
def test_independent_hierarchy_and_sentinel_match_registered_statistics(kwargs):
    pairs,groups=fixtures.evidence(**kwargs)
    actual=independently_infer(pairs,groups)
    compare_subset(actual,infer_endpoints(pairs,groups,bootstrap_samples=300))
    if kwargs.get('sentinel_fix')==10.:
        assert actual['main_acceptance']=='pass' and actual['overall_advance']=='do_not_advance'
    if kwargs.get('main_fix')==1.1:assert actual['secondary_current']['ci95'] is None


def test_varying_bundles_full_100k_draws_and_date_deletion():
    pairs,groups=fixtures.evidence()
    for p in pairs:
        i=int(p['content_group_id'].rsplit('-',1)[-1]);factor=.7+.03*i+(.07 if p['arrival_type']=='conversation' else 0)
        value=math.log(1.2) if p['contrast']=='current_over_stockoff' else math.log(factor/(1.2 if p['contrast']=='fix_over_current' else 1))
        p['log_ratios']={'response_mean_ns':value,'response_p95_ns':value+.005 if p['contrast']!='current_over_stockoff' else value}
        if p['dataset']=='gsm8k':p['arrival_date']='2024-05-14'
    actual=independently_infer(pairs,groups,100000)
    compare_subset(actual,infer_endpoints(pairs,groups))
    assert actual['calendar_date_robustness']['passes'] is False
    assert actual['bootstrap']['samples']==100000


def test_unregistered_secondary_ci_is_detected_by_summary_comparison():
    pairs,groups=fixtures.evidence(main_fix=1.1)
    independent=independently_infer(pairs,groups)
    production=infer_endpoints(pairs,groups,bootstrap_samples=300)
    actual=dict(phase='final',trial_count=0,matched_triplets=0,paired_comparisons=0,pairs_per_contrast={},trial_metrics={},paired_cases=[],confirmatory=independent)
    summary=dict(valid=True,phase='final',observed_records=0,expected_records=0,valid_records=0,matched_triplets=0,paired_comparisons=0,pairs_per_contrast={},missing_keys=[],errors=[],trial_summaries=[],paired_cases=[],confirmatory=production)
    assert stats.compare_summary(actual,summary)['passed']
    summary['confirmatory']['secondary_current']['ci95']=[.1,.9]
    with pytest.raises(ValueError,match='ci95'):stats.compare_summary(actual,summary)


def test_raw_clock_reconstruction_and_tampered_response_detection():
    row=clock_fixtures.clocks();actual=stats.clock_endpoints(row);target=clock_metrics(row)
    compare_subset(actual,target)
    assert raw.stock.raw_clock_check(row)['response_mean_ns']==12
    row['coordinator_finish_ns'][100]+=1
    with pytest.raises(ValueError,match='Stored response'):stats.clock_endpoints(row)
    with pytest.raises(ValueError,match='Stored vector'):raw.stock.raw_clock_check(row)


@pytest.mark.parametrize('field',('completed_elapsed_ns','job_completed_elapsed_ns','controller_tail_elapsed_ns'))
def test_raw_timing_cannot_hide_tail_or_change_extent(field):
    row=clock_fixtures.clocks();row[field]+=1
    with pytest.raises(ValueError,match='accounting'):stats.clock_endpoints(row)


def test_known_greedy_assignment_ties_and_all_single_copy_coverage():
    ids=np.tile(np.arange(8,dtype=np.int32),(128,1))
    got=raw.greedy_oracle(ids)
    expected=np.tile(np.array([0,129,2,131,4,133,6,135]),(128,1))
    assert np.array_equal(got['physical_ids'],expected)
    assert got['per_rank_tokens'].tolist()==[512,512]
    assert got['per_rank_active_copies'].tolist()==[4,4]
    ids=np.tile(np.array([32,33,34,35,80,81,82,83]),(128,1))
    assert np.array_equal(raw.greedy_oracle(ids)['physical_ids'],ids)


def test_independent_oracle_matches_separate_reference_on_adversarial_and_seeded_inputs():
    from resident_dispatch.online import cpu_greedy_assignment,adversarial_inputs
    rng=np.random.default_rng(17);mapping=np.r_[np.arange(128),np.arange(32)]
    cases=[c['logical_ids'] for c in adversarial_inputs(128) if c['real_tokens']==128]
    cases += [np.stack([rng.choice(128,8,replace=False) for _ in range(128)]) for _ in range(10)]
    for ids in cases:
        got=raw.greedy_oracle(ids);ref=cpu_greedy_assignment(ids,mapping)
        assert np.array_equal(got['physical_ids'],ref.physical_ids)
        assert got['physical_counts'].sum()==1024


def test_static_recording_mutation_and_fused_context_are_rejected():
    row=state_fixtures.fused_row()
    assert raw.static_state(row,'fused')['static_trials']==1
    broken=copy.deepcopy(row);broken['per_rank_state'][0]['audit']['fused_stream_context']['device']='cuda:1'
    with pytest.raises(ValueError,match='stream/storage'):raw.static_state(broken,'fused')
    broken=copy.deepcopy(row);broken['per_rank_state'][1]['end']['controller_step_calls']=1
    with pytest.raises(ValueError,match='Static contract'):raw.static_state(broken,'fused')


def matrix_fixture(monkeypatch,phase='final'):
    """Only matrix-test clock work is stubbed; raw-clock tests above are separate."""
    p=state_fixtures.plan(phase)
    p['protocol_sha256']=stats.digest(b'CPU QA protocol')
    p['hardware']=dict(allocation_id='1',hostname='qa',gpu_uuids=['GPU-A','GPU-B'])
    p={k:v for k,v in p.items() if k!='content_sha256'};p['content_sha256']=stats.digest(p)
    rows=[]
    for ci,c in enumerate(p['cases']):
        cal=c['calendar']
        for pos,arm in enumerate(c['arm_order']):
            row=dict(schema='resident_fix_trial_v1',phase=phase,status='complete',evidence_eligible=phase=='final',
                content_group_id=c['block_id'],block_id=c['block_id'],dataset=c['dataset'],arrival_type=c['arrival_type'],rho=c['rho'],
                arrival_date=cal['arrival_date'],inference_role='sentinel_descriptive' if c['dataset']=='mbpp' else 'main',
                plan_sha256=p['content_sha256'],source_sha256=p['source_sha256'],old_closure_sha256=p['old_closure_sha256'],
                allocation_id='1',hostname='qa',gpu_uuids=['GPU-A','GPU-B'],arm=arm,calendar=cal,arm_order=c['arm_order'],
                arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],epoch_monotonic_ns=1000000*(1+ci*3+pos),
                input_meta=dict(capture_sha256='a'*64,indices_sha256='b'*64,gate_weight_sha256='c'*64),
                preflight=dict(correctness_pass=True,logical_routing_sha256='d'*64,float32_combining_weights_sha256='e'*64,physical_assignment_digest='f'*64),
                postflight=dict(correctness_pass=True))
            row['record_id']=stats.digest(row);rows.append(row)
    def fake_clock(row):
        value={'stockoff':100.,'current':120.,'fix':90.}[row['arm']]
        return dict(response_mean_ns=value,response_p95_ns=value*2,service_mean_ns=value/2,queue_mean_ns=value/2)
    monkeypatch.setattr(stats,'clock_endpoints',fake_clock)
    return p,rows


@pytest.mark.parametrize('phase',('pilot','final'))
def test_complete_matrix_reconstruction_retains_correct_phase(monkeypatch,phase):
    p,rows=matrix_fixture(monkeypatch,phase)
    result=stats.reconstruct(p,rows,b'CPU QA protocol',draw_count=300)
    assert result['trial_count']==(60 if phase=='pilot' else 204)
    assert result['confirmatory'] is None if phase=='pilot' else result['confirmatory']['primary']['ratio']==pytest.approx(.9)


@pytest.mark.parametrize('mutation',('missing','duplicate','source','calendar','assignment','order','input'))
def test_matrix_or_pair_corruption_is_rejected(monkeypatch,mutation):
    p,rows=matrix_fixture(monkeypatch,'pilot')
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    else:
        r=rows[0]
        if mutation=='source':r['source_sha256']='0'*64
        elif mutation=='calendar':r['horizon_ns']+=1
        elif mutation=='order':r['epoch_monotonic_ns']=rows[1]['epoch_monotonic_ns']+1
        elif mutation=='input':r['input_meta']['indices_sha256']='0'*64
        elif mutation=='assignment':
            r=next(x for x in rows if x['arm']=='fix');r['preflight']['physical_assignment_digest']='0'*64
        r['record_id']=stats.digest({k:v for k,v in r.items() if k!='record_id'})
    with pytest.raises(ValueError):stats.reconstruct(p,rows,b'CPU QA protocol',draw_count=10)
