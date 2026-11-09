"""Synthetic paired-effect contracts, not stock-comparator measurements."""
import copy
import hashlib
import itertools
import math
import numpy as np
import pytest
from stock_comparison.inference import DATASETS,ARRIVAL_TYPES,LOADS,CONTRASTS,infer_endpoints


def fixture(off=.85,on=.9):
    groups=[dict(id=f'{d}-{i:02d}',dataset=d) for d in DATASETS for i in range(6)]
    pairs=[]
    for g in groups:
        i=int(g['id'][-2:])
        for t,rho,c in itertools.product(ARRIVAL_TYPES,LOADS,CONTRASTS):
            ratio={'ours_over_stockoff':off,'ours_over_stockon32':on,'stockon32_over_stockoff':off/on}[c]
            day=12+i%5 if t=='code' else 14+(i+2)%5
            pairs.append(dict(content_group_id=g['id'],dataset=g['dataset'],arrival_type=t,rho=rho,contrast=c,
                arrival_date=f'2024-05-{day:02d}',log_ratios=dict(response_mean_ns=math.log(ratio),response_p95_ns=math.log(ratio))))
    return groups,pairs


def set_ours_ratios(pairs,predicate,off,on,metric='response_mean_ns'):
    for p in pairs:
        if predicate(p):p['log_ratios'][metric]=math.log({'ours_over_stockoff':off,'ours_over_stockon32':on,'stockon32_over_stockoff':off/on}[p['contrast']])


def test_both_stock_controls_required_and_hierarchy_not_weaker_posthoc_choice():
    groups,pairs=fixture(off=.96,on=.8)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    assert not out['primary']['passes'] and out['secondary_stockon32']['status']=='not_tested_gate_closed'
    assert out['acceptance']=='not_established'
    groups,pairs=fixture(off=.85,on=.96)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    assert out['primary_acceptance']=='pass' and out['secondary_stockon32']['confirmatory']
    assert not out['secondary_stockon32']['passes'] and out['acceptance']=='not_established'
    assert out['worst_stock_relative_point_ratio']==pytest.approx(.96)


def test_symmetrical14_guards_and_constant_effects():
    groups,pairs=fixture();out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    assert out['acceptance']=='pass'
    assert out['primary']['ci95']==pytest.approx([.85,.85])
    assert out['secondary_stockon32']['ci95']==pytest.approx([.9,.9])
    assert len(out['guards'])==14
    assert all(g['one_sided_confidence_level']==pytest.approx(1-.05/14) for g in out['guards'])
    assert out['treatment_trials']==216 and out['matched_triplets']==72
    assert out['calendar_date_robustness']['date_count']==7


def test_dataset_load_regression_against_either_stock_closes_gate():
    groups,pairs=fixture()
    set_ours_ratios(pairs,lambda p:p['dataset']=='mbpp' and p['rho']==.5,.98,1.04)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    assert out['primary']['passes']
    guard=next(g for g in out['guards'] if g['name']=='stockon32_mbpp_rho0.5_mean_response')
    assert not guard['passes'] and out['secondary_stockon32']['ci95'] is None


def test_bootstrap_independent_oracle_same_tables_all_endpoints_and_hash():
    groups,pairs=fixture()
    for p in pairs:
        i=int(p['content_group_id'][-2:]);di=DATASETS.index(p['dataset'])
        # Changing ours against both controls by the same log amount preserves
        # the ratio identity; the stock-on/off contrast remains fixed.
        if p['contrast']!='stockon32_over_stockoff':p['log_ratios']['response_mean_ns']+=(i-2.5)*.01+di*.005
    out=infer_endpoints(pairs,groups,bootstrap_samples=700,bootstrap_seed=33)
    rng=np.random.default_rng(33);samples=[];encoded=[]
    for di,d in enumerate(DATASETS):
        idx=rng.integers(0,6,size=(700,6));encoded.append(np.asarray(idx,dtype='<i8').tobytes(order='C'))
        values=np.log(.85)+(np.arange(6)-2.5)*.01+di*.005;samples.append(values[idx].mean(axis=1))
    expected=np.exp(np.mean(samples,axis=0))
    assert out['primary']['ci95']==pytest.approx(np.quantile(expected,[.025,.975],method='linear'))
    assert out['bootstrap']['draw_indices_sha256']==hashlib.sha256(b''.join(encoded)).hexdigest()
    shuffled=infer_endpoints(pairs[::-1],groups[::-1],bootstrap_samples=700,bootstrap_seed=33)
    assert out==shuffled


@pytest.mark.parametrize('mutation',['missing','duplicate','date','dataset','ratio_identity','nan','pilot_groups'])
def test_invalid_paired_units_are_not_resampled(mutation):
    groups,pairs=fixture()
    if mutation=='missing':pairs.pop()
    elif mutation=='duplicate':pairs.append(copy.deepcopy(pairs[0]))
    elif mutation=='date':pairs[0]['arrival_date']='2024-05-21'
    elif mutation=='dataset':pairs[0]['dataset']='mbpp'
    elif mutation=='ratio_identity':pairs[0]['log_ratios']['response_mean_ns']+=.01
    elif mutation=='nan':pairs[0]['log_ratios']['response_p95_ns']=float('nan')
    elif mutation=='pilot_groups':groups=[g for g in groups if int(g['id'][-2:])<2]
    with pytest.raises(ValueError):infer_endpoints(pairs,groups,bootstrap_samples=100)


def test_date_deletion_links_services_and_checks_both_stock_controls():
    groups,pairs=fixture()
    for day in {p['arrival_date'] for p in pairs}:
        set_ours_ratios(pairs,lambda p,d=day:p['arrival_date']==d,.7 if day=='2024-05-14' else 1.03,.65 if day=='2024-05-14' else 1.01)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    removed=[r for r in out['calendar_date_robustness']['results'] if r['excluded_date']=='2024-05-14']
    assert len(removed)==2 and all(r['removed_arrival_bundles']==6 for r in removed)
    off=next(r for r in removed if r['control']=='stockoff');on=next(r for r in removed if r['control']=='stockon32')
    assert off['ratio']==pytest.approx(1.03) and not off['passes']
    assert on['ratio']==pytest.approx(1.01) and on['passes']
    assert not out['calendar_date_robustness']['passes']
