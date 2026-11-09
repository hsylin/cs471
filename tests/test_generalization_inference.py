import copy
import itertools
import math
from collections import Counter
import numpy as np
import pytest
from generalization.inference import (DATASETS,ARRIVAL_TYPES,RATE_KINDS,CONTRASTS,
    assign_calendars,select_content_groups,choose_final_sample_size,infer_endpoints)


def fixture(n=6,ratio=.9):
    groups=[dict(id=f'{d}-{i:02d}',dataset=d) for d in DATASETS for i in range(n)]
    pairs=[]
    for group in groups:
        i=int(group['id'][-2:])
        for t,r,c in itertools.product(ARRIVAL_TYPES,RATE_KINDS,CONTRASTS):
            day=12+i%5 if t=='code' else 14+(i+2)%5
            value={'combined_over_shortcut':ratio,'combined_over_native':ratio*.6,'shortcut_over_native':.6}[c]
            pairs.append(dict(content_group_id=group['id'],dataset=group['dataset'],arrival_type=t,
                arrival_date=f'2024-05-{day:02d}',rate_kind=r,contrast=c,
                log_ratios={'response_mean_ns':math.log(value),'response_p95_ns':math.log(value)}))
    return groups,pairs


def test_complete_units_required_not_jobs_or_duplicate_contrasts():
    groups,pairs=fixture()
    for bad in (pairs[:-1],pairs+[pairs[0]]):
        with pytest.raises(ValueError):infer_endpoints(bad,groups,bootstrap_samples=100)
    bad=copy.deepcopy(pairs);bad[0]['arrival_date']='2024-05-19'
    with pytest.raises(ValueError):infer_endpoints(bad,groups,bootstrap_samples=100)
    bad=copy.deepcopy(pairs);bad[0]['log_ratios']['response_p95_ns']=float('nan')
    with pytest.raises(ValueError):infer_endpoints(bad,groups,bootstrap_samples=100)


def test_identical_effect_exact_bounds_and_hierarchy():
    groups,pairs=fixture(ratio=.9)
    out=infer_endpoints(pairs,groups,bootstrap_samples=1000)
    assert out['primary']['ci95']==pytest.approx([.9,.9])
    assert out['acceptance']=='pass' and out['calendar_date_robustness']['date_count']==7
    assert out['calendar_date_robustness']['ratio_range']==pytest.approx([.9,.9])
    assert out['paired_comparisons']==216 and out['treatment_trials']==216
    # Clear primary failure closes secondary despite a strong baseline contrast.
    groups,pairs=fixture(ratio=.96)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    assert out['secondary_baseline']['ci95'] is None
    assert out['secondary_baseline']['status']=='not_tested_gate_closed'


def test_guard_family_and_whole_bundle_draws_match_independent_oracle():
    groups,pairs=fixture()
    for p in pairs:
        i=int(p['content_group_id'][-2:]);di=DATASETS.index(p['dataset'])
        p['log_ratios']['response_mean_ns'] += (i-2.5)*.02+di*.01
    out=infer_endpoints(pairs,groups,bootstrap_samples=800,bootstrap_seed=19)
    rng=np.random.default_rng(19)
    blocks=[];draw_bytes=[]
    for di,d in enumerate(DATASETS):
        values=np.log(.9)+(np.arange(6)-2.5)*.02+di*.01
        idx=rng.integers(0,6,size=(800,6))
        draw_bytes.append(np.asarray(idx,dtype='<i8').tobytes(order='C'))
        blocks.append(values[idx].mean(axis=1))
    expected=np.exp(np.mean(blocks,axis=0))
    import hashlib
    assert out['bootstrap']['draw_indices_sha256']==hashlib.sha256(b''.join(draw_bytes)).hexdigest()
    assert out['primary']['ci95']==pytest.approx(np.quantile(expected,[.025,.975],method='linear'))
    assert len(out['guards'])==7
    assert all(g['one_sided_confidence_level']==pytest.approx(1-.05/7) for g in out['guards'])


def test_variance_planning_is_translation_invariant_and_bounded():
    groups,pairs=fixture(n=2)
    assert choose_final_sample_size(pairs,groups)['groups_per_dataset']==6
    for p in pairs:
        p['log_ratios']['response_mean_ns'] += (-1 if p['content_group_id'].endswith('00') else 1)*.2
    a=choose_final_sample_size(pairs,groups)
    for p in pairs:p['log_ratios']['response_mean_ns']+=10
    b=choose_final_sample_size(pairs,groups)
    assert a['groups_per_dataset']==b['groups_per_dataset']==8
    assert a['pooled_sample_sd']==pytest.approx(b['pooled_sample_sd'])
    assert a['final_treatment_trials']==288


def block_pool(phase):
    blocks=[]
    for typ in ARRIVAL_TYPES:
        start=(10 if typ=='code' else 12)+(0 if phase=='pilot' else 2)
        for date_i in range(2 if phase=='pilot' else 5):
            for k in range(3 if phase=='pilot' else 8):
                day=f'2024-05-{start+date_i:02d}'
                blocks.append(dict(id=f'{typ}-{day}-{k}',trace=typ,source_date_utc=day,split=phase,
                    hash_rank_score=f'{k:064d}',source_rows_sha256=f'{k:064d}',
                    pilot_dataset_assignment=DATASETS[k] if phase=='pilot' else None))
    return blocks


@pytest.mark.parametrize('n',[6,8])
def test_assignment_balances_each_service_and_dataset_without_reuse(n):
    groups,_=fixture(n=n);blocks=block_pool('final')
    assigned=assign_calendars(groups,blocks,phase='final')
    assert assigned==assign_calendars(list(reversed(groups)),list(reversed(blocks)),phase='final')
    assert len(assigned)==6*n and len({x['arrival_block_id'] for x in assigned})==6*n
    for typ in ARRIVAL_TYPES:
        counts=Counter(x['arrival_date'] for x in assigned if x['arrival_type']==typ)
        assert len(counts)==5 and max(counts.values())-min(counts.values())<=1
        for d in DATASETS:
            dc=Counter(x['arrival_date'] for x in assigned if x['arrival_type']==typ and x['dataset']==d)
            assert len(dc)==5 and max(dc.values())-min(dc.values())<=1
    with pytest.raises(ValueError):assign_calendars(groups,blocks[:2],phase='final')


def test_pilot_preassigned_blocks_and_final_subset_hash_selection():
    groups,_=fixture(n=2)
    assigned=assign_calendars(groups,block_pool('pilot'),phase='pilot')
    assert len(assigned)==12
    assert all(x['arrival_block_id'].endswith(str(DATASETS.index(x['dataset']))) for x in assigned)
    groups,_=fixture(n=8)
    a=select_content_groups(groups,6)
    assert a==select_content_groups(groups[::-1],6) and len(a)==18
    assert set(x['id'] for x in a)<=set(x['id'] for x in groups)


def test_date_deletion_links_services_and_renormalizes_surviving_bundle():
    groups,pairs=fixture()
    for p in pairs:
        if p['contrast']=='combined_over_shortcut':
            # Deliberately heterogeneous dates; verify the point calculation,
            # without interpreting this synthetic fixture as benchmark data.
            p['log_ratios']['response_mean_ns']=math.log(.7 if p['arrival_date']=='2024-05-14' else 1.03)
    out=infer_endpoints(pairs,groups,bootstrap_samples=100)
    deletion=next(x for x in out['calendar_date_robustness']['results'] if x['excluded_date']=='2024-05-14')
    assert deletion['removed_arrival_bundles']==6 # code and conversation together, across three datasets
    assert deletion['retained_content_groups']==dict.fromkeys(DATASETS,6)
    assert deletion['ratio']==pytest.approx(1.03)
    assert not deletion['passes'] and not out['calendar_date_robustness']['passes']
    assert out['secondary_baseline']['status']=='not_tested_gate_closed'
