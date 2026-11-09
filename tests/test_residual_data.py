"""Synthetic source-split/overlap checks; no timing or captured-data claims."""
from collections import Counter
import copy
import pytest
from resident_residual.data import DATASETS,ALLOCATION,digest,split_sources,combine_prior_arrivals,assign_arrivals


def records():
    return [dict(dataset=d,source_id=str(i),text_sha256=digest([d,i]),normalized_text_sha256=digest(['text',d,i])) for d in DATASETS for i in range(350)]


def priors():
    out=[{'blocks':[]} for _ in range(3)]
    for trace in ('code','conversation'):
        for j,n in enumerate((46,24,22)):
            for i in range(n):
                lo=j*1000000+i*5000
                out[j]['blocks'].append(dict(trace=trace,row_start=lo,boundary_row=lo+(256 if j==0 else 4096),source_sha256=digest(trace)))
    return out


def scans():
    return [dict(trace=t,source_sha256=digest(t),retained_candidates={f'2024-05-{first+i:02d}':[dict(trace=t,source_date_utc=f'2024-05-{first+i:02d}',row_start=(i*4+j)*4097,boundary_row=(i*4+j)*4097+4096,jobs=4096,original_horizon_ns=4097,terminal_gap_ns=1) for j in range(4)] for i in range(7)}) for t,first in [('code',10),('conversation',12)]]


def test_exact_twenty_groups_source_disjoint_and_new_frame_named():
    groups=split_sources(records());assert groups==split_sources(records()[::-1])
    assert len(groups)==20 and sum(len(g['sources']) for g in groups)==640
    assert Counter((g['dataset'],g['split']) for g in groups)=={(d,s):n for d in DATASETS for s,n in ALLOCATION[d].items()}
    assert all(g['inference_role']=='main' and g['block_id'].startswith('residual-') for g in groups)
    assert {g['dataset'] for g in groups}=={'gsm8k_train','wildchat'}


def test_cannot_fill_train_frame_from_old_test_or_shortfall():
    x=records()
    for r in x:
        if r['dataset']=='gsm8k_train':r['dataset']='gsm8k'
    with pytest.raises(ValueError,match='need320'):split_sources(x)
    x=[r for r in records() if r['dataset']!='gsm8k_train' or int(r['source_id'])<319]
    with pytest.raises(ValueError,match='need320'):split_sources(x)


def test_requires_all184_prior_arrival_blocks_and_boundaries():
    x=priors();assert len(combine_prior_arrivals(x))==184
    with pytest.raises(ValueError,match='44'):combine_prior_arrivals(x[:2])
    x=priors();x[2]['blocks'][0].update(row_start=256,boundary_row=4352)
    with pytest.raises(ValueError,match='overlap'):combine_prior_arrivals(x)


def test_forty_new_intervals_have_exact_split_and_dates():
    groups=split_sources(records());rows=assign_arrivals(scans(),groups)
    assert len(rows)==40 and Counter(r['split'] for r in rows)=={'pilot':8,'final':32}
    assert len({(r['trace'],r['row_start']) for r in rows})==40
    assert all(r['id'].startswith('residual-azure-') for r in rows)
    for d in DATASETS:
        for t in ['code','conversation']:
            assert len({r['source_date_utc'] for r in rows if r['dataset']==d and r['trace']==t and r['split']=='final'})==5


def test_invalid_selected_interval_is_not_replaced():
    x=scans();x[0]['retained_candidates']['2024-05-10'][0]['terminal_gap_ns']=0
    with pytest.raises(ValueError,match='no silent replacement'):assign_arrivals(x,split_sources(records()))


def test_capture_rejects_preceding_study_schema(tmp_path):
    import json
    from scripts.capture_residual_inputs import validate_sources
    x=dict(schema='resident_fix_sources_v1',blocks=[]);x['content_sha256']=digest(x)
    path=tmp_path/'sources.json';path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='Changed/unfrozen'):validate_sources(path)
