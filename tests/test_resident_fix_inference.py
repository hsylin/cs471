import copy
import hashlib
import itertools
import math
import numpy as np
import pytest
from resident_fastpath.inference import DATASETS,CONTRASTS,infer_endpoints


def evidence(*,main_fix=.9,sentinel_fix=.9,low_fix=None):
    groups=[dict(id=f'{d}-{i}',dataset=d) for d,n in (('gsm8k',8),('wildchat',8),('mbpp',1)) for i in range(n)]
    pairs=[]
    for g,typ,rho in itertools.product(groups,('code','conversation'),(.5,.85)):
        idx=int(g['id'].rsplit('-',1)[1]);day=12+idx%5+(2 if typ=='conversation' else 0)
        fix=sentinel_fix if g['dataset']=='mbpp' else low_fix if rho==.5 and low_fix is not None else main_fix
        values={'stockoff':1.,'current':1.2,'fix':fix}
        for contrast in CONTRASTS:
            treatment,control=contrast.split('_over_');log=math.log(values[treatment])-math.log(values[control])
            pairs.append(dict(content_group_id=g['id'],dataset=g['dataset'],arrival_type=typ,arrival_date=f'2024-05-{day:02d}',rho=rho,
                contrast=contrast,log_ratios={'response_mean_ns':log,'response_p95_ns':log}))
    return pairs,groups


def infer(pairs,groups,**kwargs):return infer_endpoints(pairs,groups,bootstrap_samples=200,**kwargs)


def test_complete_main_and_sentinel_contract_passes():
    result=infer(*evidence())
    assert result['main_acceptance']=='pass' and result['overall_advance']=='pass'
    assert result['total_treatment_trials']==204 and result['main_treatment_trials']==192
    assert result['main_content_groups']==16 and result['sentinel_content_groups']==1
    assert result['matched_triplets']==68 and len(result['guards'])==10
    assert all(g['one_sided_confidence_level']==.995 for g in result['guards'])
    assert len(result['mbpp_sentinel']['checks'])==6
    assert result['secondary_current']['confirmatory'] is True
    assert result['primary']['ratio']==pytest.approx(.9)
    assert result['secondary_current']['ratio']==pytest.approx(.75)


def test_sentinel_failure_does_not_change_or_rescue_main_inference():
    good=infer(*evidence());bad=infer(*evidence(sentinel_fix=10.))
    for key in ('primary','guards','secondary_current','calendar_date_robustness','bootstrap'):
        assert good[key]==bad[key]
    assert bad['main_acceptance']=='pass' and bad['overall_advance']=='do_not_advance'
    assert bad['mbpp_sentinel']['passes'] is False
    assert all(x['ci95'] is None and x['inferential'] is False for x in bad['mbpp_sentinel']['checks'])


def test_primary_target_is_strict_but_regression_guard_is_inclusive():
    boundary=infer(*evidence(main_fix=.95))
    assert boundary['primary']['upper_confidence_bound']==pytest.approx(.95)
    assert boundary['primary']['passes'] is False
    assert boundary['secondary_current']['ci95'] is None
    inclusive=infer(*evidence(main_fix=.9,low_fix=1.02))
    guards=[g for g in inclusive['guards'] if g['rho']==.5 and g['contrast']=='fix_over_stockoff']
    assert all(g['upper_confidence_bound']==pytest.approx(1.02) and g['passes'] for g in guards)


def test_main_failure_closes_secondary_even_when_current_is_slow():
    result=infer(*evidence(main_fix=1.1))
    assert result['primary_acceptance']=='not_established'
    assert result['secondary_current']['status']=='not_tested_gate_closed'
    assert result['secondary_current']['ratio']<1
    assert result['secondary_current']['ci95'] is None and result['secondary_current']['passes'] is None
    assert result['overall_advance']=='do_not_advance'


def test_bootstrap_tables_are_exactly_two_main_strata_in_declared_order():
    seed=471712;r=infer(*evidence(),bootstrap_seed=seed);rng=np.random.default_rng(seed)
    payload=b''.join(rng.integers(0,8,size=(200,8)).astype('<i8').tobytes(order='C') for _ in DATASETS)
    assert r['bootstrap']['draw_indices_sha256']==hashlib.sha256(payload).hexdigest()
    assert r['bootstrap']['strata']==['gsm8k','wildchat']


def test_within_bundle_types_are_averaged_before_equal_main_dataset_weight():
    pairs,groups=evidence()
    for p in pairs:
        gid=p['content_group_id'];factor=.5 if p['dataset']=='gsm8k' else .8 if p['dataset']=='wildchat' else .1
        if p['arrival_type']=='conversation':factor*=.8
        if p['contrast']=='fix_over_stockoff':v=math.log(factor)
        elif p['contrast']=='fix_over_current':v=math.log(factor/1.2)
        else:v=math.log(1.2)
        p['log_ratios']={k:v for k in p['log_ratios']}
    r=infer(pairs,groups)
    assert r['primary']['ratio']==pytest.approx(math.sqrt(math.sqrt(.5*.4)*math.sqrt(.8*.64)))


@pytest.mark.parametrize('mutation',('missing','duplicate','wrong_dataset','date','contrast_identity','nonfinite'))
def test_invalid_paired_matrix_is_rejected(mutation):
    pairs,groups=evidence()
    if mutation=='missing':pairs.pop()
    elif mutation=='duplicate':pairs.append(copy.deepcopy(pairs[0]))
    elif mutation=='wrong_dataset':pairs[0]['dataset']='mbpp'
    elif mutation=='date':pairs[0]['arrival_date']='2024-05-19'
    elif mutation=='contrast_identity':pairs[0]['log_ratios']['response_mean_ns']+=.01
    else:pairs[0]['log_ratios']['response_mean_ns']=float('nan')
    with pytest.raises(ValueError):infer(pairs,groups)


def test_wrong_main_group_count_not_silently_reweighted():
    pairs,groups=evidence();groups=[g for g in groups if g['id']!='gsm8k-7']
    pairs=[p for p in pairs if p['content_group_id']!='gsm8k-7']
    with pytest.raises(ValueError,match='Exactly'):infer(pairs,groups)


def test_unestimable_date_deletion_cannot_pass():
    pairs,groups=evidence()
    for p in pairs:
        if p['dataset']=='gsm8k':p['arrival_date']='2024-05-14'
    result=infer(pairs,groups)
    failed=[x for x in result['calendar_date_robustness']['results'] if x['excluded_date']=='2024-05-14']
    assert all(x['ratio'] is None and x['missing_datasets']==['gsm8k'] and not x['passes'] for x in failed)
    assert result['secondary_current']['confirmatory'] is False


@pytest.mark.parametrize('phase,n',(('pilot',2),('final',8)))
def test_near_balanced_orders_and_exact_arrival_reversal(phase,n):
    from resident_fastpath.inference import build_arm_orders,validate_arm_orders
    groups=[dict(id=f'{d}-{i}',dataset=d) for d,k in [('gsm8k',n),('wildchat',n),('mbpp',1)] for i in range(k)]
    cases=build_arm_orders(groups,seed=471711,phase=phase)
    assert len(cases)==4*(2*n+1)
    diagnostics=validate_arm_orders(cases,groups,phase=phase)
    assert all(x['pairwise_precedence_balanced'] for x in diagnostics)
    if phase=='final':
        pooled=[x for x in diagnostics if len(x['datasets'])==2]
        assert all(set(x['permutation_counts'].values())=={5,6} for x in pooled)
        individual=[x for x in diagnostics if len(x['datasets'])==1]
        assert all(set(x['permutation_counts'].values())=={2,3} for x in individual)
    assert cases==build_arm_orders(groups,seed=471711,phase=phase)


def test_permutation_counts_alone_cannot_hide_wrong_within_group_order():
    from resident_fastpath.inference import build_arm_orders,validate_arm_orders
    _,groups=evidence();cases=build_arm_orders(groups,seed=471711,phase='final')
    selected=[c for c in cases if c['rho']==.5 and c['arrival_type']=='conversation' and c['content_group_id'].startswith('gsm8k')]
    first=selected[0];other=next(c for c in selected if c['arm_order']!=first['arm_order'])
    first['arm_order'],other['arm_order']=other['arm_order'],first['arm_order']
    with pytest.raises(ValueError,match='reversals'):validate_arm_orders(cases,groups,phase='final')
