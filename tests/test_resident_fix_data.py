"""Synthetic data-contract checks; not measured workloads or timing evidence."""
import copy,itertools
from collections import Counter
import pytest
from resident_fastpath.data import DATASETS,ALLOCATION,digest,split_sources,combine_prior_arrivals,assign_arrivals,exclude_and_deduplicate


def records():
    return [dict(dataset=d,source_id=str(i),text_sha256=digest([d,i]),normalized_text_sha256=digest(['normalized',d,i]))
            for d in DATASETS for i in range(73 if d=='mbpp' else 423)]


def scans():
    out=[]
    for trace,base in [('code',10),('conversation',12)]:
        days={}
        for i in range(7):
            day=f'2024-05-{base+i:02d}';days[day]=[
                dict(trace=trace,source_date_utc=day,row_start=(i*4+j)*4097,boundary_row=(i*4+j)*4097+4096,jobs=4096,original_horizon_ns=4097,terminal_gap_ns=1)
                for j in range(4)]
        out.append(dict(trace=trace,source_sha256=digest(trace),retained_candidates=days))
    return out


def prior_arrivals():
    old={'blocks':[]};stock={'blocks':[]}
    for trace in ('code','conversation'):
        for i in range(46):old['blocks'].append(dict(trace=trace,row_start=i*1000,boundary_row=i*1000+256,source_sha256=digest(trace)))
        for i in range(24):stock['blocks'].append(dict(trace=trace,row_start=100000+i*5000,boundary_row=100000+i*5000+4096,source_sha256=digest(trace)))
    return [old,stock]


def test_declared_allocation_is_exact_disjoint_and_order_invariant():
    a=split_sources(records());b=split_sources(records()[::-1]);assert a==b and len(a)==22
    assert Counter((r['dataset'],r['split']) for r in a)==Counter({(d,s):n for d in DATASETS for s,n in ALLOCATION[d].items()})
    assert sum(len(b['sources']) for b in a)==704
    assert sum(b['split']=='pilot' for b in a)==5
    assert sum(b['split']=='final' and b['inference_role']=='main' for b in a)==16
    assert all(b['inference_role']=='descriptive_sentinel' for b in a if b['dataset']=='mbpp')
    keys=[(r['dataset'],r['source_id']) for b in a for r in b['sources']];assert len(keys)==len(set(keys))


def test_mbpp_shortfall_is_not_silently_reused_or_reallocated():
    r=[x for x in records() if x['dataset']!='mbpp' or int(x['source_id'])<63]
    with pytest.raises(ValueError,match='need64'):split_sources(r)


def test_stock_sources_and_visible_prefixes_remain_excluded():
    refs=[dict(dataset='mbpp',source_id='stock',text='old full source',prefix_views=['old captured prefix'])]
    candidates=[dict(dataset='mbpp',source_id='stock',text='changed body',captured_prefix_text='changed'),
                dict(dataset='mbpp',source_id='new-id',text='new longer body',captured_prefix_text='OLD CAPTURED PREFIX')]
    kept,rejected=exclude_and_deduplicate(candidates,refs)
    assert not kept and len(rejected)==2 and rejected[1]['view']=='captured_prefix'


def test_all140_prior_arrival_intervals_required():
    x=prior_arrivals();assert len(combine_prior_arrivals(x))==140
    x[1]['blocks'].pop()
    with pytest.raises(ValueError,match='48 stock'):combine_prior_arrivals(x)


def test_prior_boundary_overlap_and_source_pin_change_rejected():
    x=prior_arrivals();x[1]['blocks'][0].update(row_start=256,boundary_row=4352)
    with pytest.raises(ValueError,match='overlap'):combine_prior_arrivals(x)
    x=prior_arrivals();x[1]['blocks'][0]['source_sha256']='changed'
    with pytest.raises(ValueError,match='pins'):combine_prior_arrivals(x)


def test_arrival_assignment_has44_unique_intervals_with_split_roles():
    groups=split_sources(records());assigned=assign_arrivals(scans(),groups)
    assert len(assigned)==44 and Counter(x['split'] for x in assigned)=={'pilot':10,'final':34}
    assert len({(x['trace'],x['row_start']) for x in assigned})==44
    for g in groups:
        rows=[x for x in assigned if x['content_group_id']==g['block_id']]
        assert {x['trace'] for x in rows}=={'code','conversation'}
        assert all(x['inference_role']==g['inference_role'] for x in rows)
    for dataset,trace in itertools.product(('gsm8k','wildchat'),('code','conversation')):
        assert len({x['source_date_utc'] for x in assigned if x['split']=='final' and x['dataset']==dataset and x['trace']==trace})==5


def test_selected_invalid_calendar_not_replaced():
    x=scans();x[0]['retained_candidates']['2024-05-10'][0]['terminal_gap_ns']=0
    with pytest.raises(ValueError,match='no silent replacement'):assign_arrivals(x,split_sources(records()))


def test_capture_schema_cannot_accept_old_study(tmp_path):
    from scripts.capture_resident_fix_inputs import validate_sources
    x={'schema':'stock_comparison_sources_v1','blocks':[]};x['content_sha256']=digest(x)
    import json
    p=tmp_path/'sources.json';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='Changed/unfrozen'):validate_sources(p)
