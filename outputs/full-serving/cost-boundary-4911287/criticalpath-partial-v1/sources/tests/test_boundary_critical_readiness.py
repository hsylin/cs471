import copy
from full_serving.boundary_critical_readiness import clock_box,latest_resolved,range_bounds,pair_calls,summarize_pairing,commit_summary

def fixture():
    clocks={r:dict(low_us=-1,high_us=1) for r in range(8)}
    calls={r:{10:[dict(start_wall_ns=(100+r)*1000,end_wall_ns=120000,duration_ns=(20-r)*1000,local_pending_at_entry=True,pending_layer_at_entry=0)]} for r in range(8)}
    results={r:{10:[dict(all_ready=True)]} for r in range(8)}
    return calls,results,clocks

def test_clock_interval_sign_and_order():
    assert clock_box(100000,dict(low_us=3,high_us=7))==(93,97)
    assert range_bounds({0:(0,2),1:(10,12)})==(8,12)
    assert latest_resolved({0:(0,2),1:(10,12)})==1
    assert latest_resolved({0:(0,2),1:(1,3)}) is None

def test_complete_pair_and_uncertain_last_arrival():
    c,r,k=fixture();p,u=pair_calls(c,r,k)
    assert not u and len(p)==1 and p[0]['all_ready']
    assert p[0]['latest_entry_rank_resolved'] is None
    assert p[0]['arrival_spread_bounds_us']==(5,9)
    assert p[0]['tail_after_latest_entry_bounds_us']==(11,15)

def test_publication_race_is_not_result_corruption():
    c,r,k=fixture();c[3][10][0]['local_pending_at_entry']=False
    p,u=pair_calls(c,r,k);s=summarize_pairing(p,u)
    assert s['publication_race_compatible_calls']==1 and s['ready_calls']==1
    assert not s['entry_missing_in_deferred_calls']

def test_false_readiness_is_deferral_not_serving_block():
    c,r,k=fixture()
    for rank in r:r[rank][10][0]['all_ready']=False
    c[6][10][0]['local_pending_at_entry']=False
    p,u=pair_calls(c,r,k);s=summarize_pairing(p,u)
    assert s['deferred_calls']==1 and s['entry_missing_in_deferred_calls']=={6:1}

def test_missing_result_or_span_not_zero_filled():
    c,r,k=fixture();del c[2][10];p,u=pair_calls(c,r,k)
    assert not p and u[0]['reason']=='missing_or_multiple_calls'
    p,u=pair_calls({},r,k);assert not p and len(u)==1

def test_duplicate_calls_and_disagreement_unknown():
    c,r,k=fixture();c[0][10]*=2
    assert pair_calls(c,r,k)[1][0]['reason']=='missing_or_multiple_calls'
    c,r,k=fixture();r[0][10][0]['all_ready']=False
    assert pair_calls(c,r,k)[1][0]['reason']=='readiness_result_disagreement'

def test_impossible_collective_order_does_not_imply_laggard():
    c,r,k=fixture();c[0][10][0]['end_wall_ns']=90000
    p,u=pair_calls(c,r,k)
    assert not p and u[0]['reason']=='return_before_latest_entry_beyond_empirical_clock_bounds'

def test_commit_full_48_and_hash_disagreement():
    events={r:[dict(kind='commit_completed',cycle_step=750,layer=l,map_sha256=str(l)) for l in range(48)] for r in range(8)}
    a=commit_summary(events)[0];assert a['full_48layer_all_rank_completion'] and a['all_observed_layer_hashes_agree']
    events[7][0]['map_sha256']='different'
    assert not commit_summary(events)[0]['all_observed_layer_hashes_agree']

def test_censored_and_duplicate_completion_preserved():
    events={0:[dict(kind='commit_enqueued',cycle_step=750,layer=1),dict(kind='commit_completed',cycle_step=750,layer=0,map_sha256='a')]*2}
    a=commit_summary(events)[0]
    assert not a['full_48layer_all_rank_completion'] and a['per_rank_queued_without_observed_completion'][0]==[1]
    assert a['duplicate_completions'][0]==1 and a['all_rank_observed_layers']==[]
