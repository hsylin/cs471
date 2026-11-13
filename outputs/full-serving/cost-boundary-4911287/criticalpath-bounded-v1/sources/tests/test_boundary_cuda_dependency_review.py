from full_serving.boundary_cuda_dependency_review import inspect_edge

def edge():return dict(source_activity_interval_us=[10,30],source_stream=[1,2,3],destination_stream=[1,2,4],wait_CPU_api_interval_us=[20,21])
def work(corr,start,end,submit):
 return [dict(cat='cuda_runtime',ph='X',ts=submit,dur=1,args={'correlation':corr}),dict(cat='gpu_memcpy',ph='X',name='copy',ts=start,dur=end-start,args=dict(correlation=corr,device=1,context=2,stream=4))]
def test_source_and_successor_order_does_not_claim_exclusive_delay():
 e=edge();v=inspect_edge(e,work(1,5,15,1)+work(2,31,35,22),[e]);assert v['status']=='observed_dependency_order' and v['source_unfinished_at_wait_submission'];assert not v['exclusive_dependency_established']
def test_missing_prior_activity_is_unknown_not_idle():
 e=edge();v=inspect_edge(e,work(2,31,35,22),[e]);assert v['destination_previous_observed_activity_end_us'] is None and v['source_finishes_after_previous_observed_destination_work'] is None
def test_missing_source_or_successor_is_unknown():
 e=edge();assert inspect_edge(dict(e,source_activity_interval_us=None),[],[])['status']=='unknown_source_activity';assert inspect_edge(e,[],[])['status']=='next_destination_activity_unobserved'
def test_inconsistent_order_is_not_forced_valid():
 e=edge();assert inspect_edge(e,work(2,25,35,22),[e])['status']=='timing_order_inconsistent'
def test_intervening_wait_is_retained():
 e=edge();other=dict(e,wait_CPU_api_interval_us=[21,22]);v=inspect_edge(e,work(2,31,35,23),[e,other]);assert v['intervening_linked_event_waits']==1

def test_saved_link_receipt_rejects_changed_input(tmp_path):
 import json,pytest
 from full_serving.boundary_cuda_dependency_review import review
 trace=tmp_path/'trace.json';trace.write_text('{}');side=tmp_path/'side.jsonl';side.write_text('');receipt=tmp_path/'links.json'
 receipt.write_text(json.dumps({'inputs':{str(trace):{'sha256':'wrong'},str(side):{'sha256':'wrong'}}}))
 with pytest.raises(AssertionError):review(trace,side,receipt)
