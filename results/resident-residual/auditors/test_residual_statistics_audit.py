"""Synthetic CPU QA for the external independent residual statistics auditor.

Production helpers are used only as comparison targets in this test module.
The auditor itself imports no production analysis or inference code.
"""
import copy
import importlib.util
import hashlib
import itertools
import math
from pathlib import Path
import sys
import numpy as np
import pytest

WORK=Path(__file__).resolve().parent
REPO=next((p for p in WORK.parents if (p/'resident_residual').is_dir()),WORK/'cs471')
sys.path.insert(0,str(REPO))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
stats=load('residual_statistics',WORK/'audit_residual_statistics.py')
fixtures=load('new_inference_fixture',REPO/'tests/test_resident_residual_inference.py')
plan_fixture=load('new_analysis_fixture',REPO/'tests/test_resident_residual_analysis.py')
clocks_fixture=load('frozen_clock_fixture',REPO/'tests/test_stock_comparison_analysis.py')
from resident_residual.inference import infer_endpoints
from stock_comparison.analysis import clock_metrics


def subset_equal(actual,reference):
    if isinstance(actual,dict):
        for key,value in actual.items():subset_equal(value,reference[key])
    elif isinstance(actual,list):
        assert len(actual)==len(reference)
        for a,b in zip(actual,reference):subset_equal(a,b)
    elif isinstance(actual,float):assert actual==pytest.approx(reference,rel=2e-12,abs=2e-10)
    else:assert actual==reference


def independent(pairs,groups,n=300):
    by={d:sorted(g['id'] for g in groups if g['dataset']==d) for d in ('gsm8k_train','wildchat')}
    index={tuple(p[k] for k in ('content_group_id','arrival_type','rho','contrast')):p['log_ratios'] for p in pairs}
    cases={tuple(p[k] for k in ('content_group_id','arrival_type','rho')):{'calendar':{'arrival_date':p['arrival_date']}} for p in pairs}
    return stats.infer_pairs(index,by,cases,draw_count=n)


@pytest.mark.parametrize('settings',({}, {'main_fix':1.1}, {'main_fix':.95}, {'low_fix':1.02}, {'low_fix':1.03}))
def test_exact_thresholds_guards_and_closed_hierarchy(settings):
    pairs,groups=fixtures.evidence(**settings)
    actual=independent(pairs,groups)
    subset_equal(actual,infer_endpoints(pairs,groups,bootstrap_samples=300))
    assert actual['total_treatment_trials']==192 and actual['matched_triplets']==64
    assert len(actual['guards'])==10 and len(actual['descriptive_cells'])==24
    if settings.get('main_fix')==.95:assert not actual['primary']['passes']
    if settings.get('low_fix')==1.02:assert all(x['passes'] for x in actual['guards'])
    if settings.get('main_fix')==1.1 or settings.get('low_fix')==1.03:
        assert actual['secondary_previous']['status']=='not_tested_gate_closed' and actual['secondary_previous']['ci95'] is None


def test_nonconstant_content_full_100k_draws_and_unestimable_date_deletion():
    pairs,groups=fixtures.evidence()
    for p in pairs:
        idx=int(p['content_group_id'].rsplit('-',1)[1]);factor=.72+.03*idx+(.04 if p['arrival_type']=='conversation' else 0.)
        ratio=1.2 if p['contrast']=='previous_over_stockoff' else factor/(1.2 if p['contrast']=='fix_over_previous' else 1.)
        p['log_ratios']={k:math.log(ratio) for k in p['log_ratios']}
        if p['dataset']=='gsm8k_train':p['arrival_date']='2024-05-14'
    actual=independent(pairs,groups,100000);subset_equal(actual,infer_endpoints(pairs,groups))
    rng=np.random.default_rng(471812)
    payload=b''.join(rng.integers(0,8,(100000,8)).astype('<i8').tobytes() for _ in range(2))
    assert actual['bootstrap']['draw_indices_sha256']==hashlib.sha256(payload).hexdigest()
    assert not actual['calendar_date_robustness']['passes']
    assert actual['secondary_previous']['ci95'] is None


def test_raw_clocks_reproduce_all_service_queue_elapsed_and_throughput_metrics():
    row=clocks_fixture.clocks();actual=stats.clock_endpoints(row)
    subset_equal(actual,clock_metrics(row))
    assert actual['service_mean_ns']==10 and actual['queue_mean_ns']==2 and actual['response_mean_ns']==12
    assert actual['throughput_jobs_per_second']==4096*1e9/actual['completed_elapsed_ns']
    row['coordinator_finish_ns'][100]+=1
    with pytest.raises(ValueError,match='Stored response'):stats.clock_endpoints(row)


@pytest.mark.parametrize('field',('completed_elapsed_ns','controller_tail_elapsed_ns','quiescence_call_elapsed_ns'))
def test_timing_extent_mutations_rejected(field):
    row=clocks_fixture.clocks();row[field]+=1
    with pytest.raises(ValueError,match='accounting'):stats.clock_endpoints(row)


def matrix(monkeypatch,phase='final'):
    """Clock work stubbed only for matrix tests; genuine raw-clock tests above."""
    p=plan_fixture.plan(phase);p['protocol_sha256']=stats.digest(b'SYNTHETIC QA protocol')
    p['hardware']=dict(allocation_id='1',hostname='qa',gpu_uuids=['GPU-A','GPU-B'])
    p={k:v for k,v in p.items() if k!='content_sha256'};p['content_sha256']=stats.digest(p)
    rows=[]
    for ci,c in enumerate(p['cases']):
        cal=c['calendar']
        for pos,arm in enumerate(c['arm_order']):
            row=dict(schema='resident_residual_trial_v1',phase=phase,status='complete',evidence_eligible=phase=='final',content_group_id=c['block_id'],block_id=c['block_id'],dataset=c['dataset'],arrival_type=c['arrival_type'],rho=c['rho'],arrival_date=cal['arrival_date'],inference_role='main',plan_sha256=p['content_sha256'],source_sha256=p['source_sha256'],previous_closure_sha256=p['previous_closure_sha256'],allocation_id='1',hostname='qa',gpu_uuids=['GPU-A','GPU-B'],arm=arm,calendar=cal,arm_order=c['arm_order'],arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],epoch_monotonic_ns=1000000*(1+ci*3+pos),input_meta=dict(capture_sha256='a'*64,indices_sha256='b'*64,gate_weight_sha256='c'*64),preflight=dict(correctness_pass=True,logical_routing_sha256='d'*64,float32_combining_weights_sha256='e'*64,physical_assignment_digest='f'*64),postflight=dict(correctness_pass=True))
            row['record_id']=stats.digest(row);rows.append(row)
    def clocks(row):
        value={'stockoff':100.,'previous':120.,'fix':90.}[row['arm']]
        return dict(response_mean_ns=value,response_p95_ns=value*2,service_mean_ns=value/2,queue_mean_ns=value/2,completed_elapsed_ns=value*4096,throughput_jobs_per_second=1e9/value)
    monkeypatch.setattr(stats,'clock_endpoints',clocks)
    return p,rows


@pytest.mark.parametrize('phase,count',(('pilot',48),('final',192)))
def test_complete_phase_matrix(monkeypatch,phase,count):
    p,rows=matrix(monkeypatch,phase);r=stats.reconstruct(p,rows,b'SYNTHETIC QA protocol',draw_count=300)
    assert r['trial_count']==count and r['paired_comparisons']==count
    if phase=='pilot':assert r['confirmatory'] is None
    else:assert r['confirmatory']['primary']['ratio']==pytest.approx(.9)


@pytest.mark.parametrize('mutation',('missing','duplicate','source','calendar','assignment','order','input','dataset'))
def test_complete_pairing_and_identity_mutations_rejected(monkeypatch,mutation):
    p,rows=matrix(monkeypatch,'pilot')
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    else:
        r=rows[0]
        if mutation=='source':r['source_sha256']='0'*64
        elif mutation=='calendar':r['horizon_ns']+=1
        elif mutation=='order':r['epoch_monotonic_ns']=rows[1]['epoch_monotonic_ns']+1
        elif mutation=='input':r['input_meta']['indices_sha256']='0'*64
        elif mutation=='dataset':r['dataset']='gsm8k'
        else:r=next(x for x in rows if x['arm']=='fix');r['preflight']['physical_assignment_digest']='0'*64
        r['record_id']=stats.digest({k:v for k,v in r.items() if k!='record_id'})
    with pytest.raises(ValueError):stats.reconstruct(p,rows,b'SYNTHETIC QA protocol',draw_count=10)


def test_comparison_rejects_an_unregistered_secondary_interval():
    pairs,groups=fixtures.evidence(main_fix=1.1);actual_inference=independent(pairs,groups)
    reference=infer_endpoints(pairs,groups,bootstrap_samples=300)
    actual=dict(phase='final',trial_count=0,matched_triplets=0,paired_comparisons=0,pairs_per_contrast={},trial_metrics={},paired_cases=[],confirmatory=actual_inference)
    summary=dict(valid=True,phase='final',observed_records=0,expected_records=0,valid_records=0,matched_triplets=0,paired_comparisons=0,pairs_per_contrast={},missing_keys=[],errors=[],trial_summaries=[],paired_cases=[],confirmatory=reference)
    assert stats.compare_summary(actual,summary)['passed']
    summary['confirmatory']['secondary_previous']['ci95']=[.1,.9]
    with pytest.raises(ValueError,match='ci95'):stats.compare_summary(actual,summary)
