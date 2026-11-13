from full_serving.boundary_cuda_event_analysis import checked_record,api_for_scope

def record():return dict(event_token=1,call_id=1,operation='record',success=True,overlap=False,start_wall_ns=1,end_wall_ns=2)
def wait():return dict(event_token=1,record_call_id=1,success=True,overlap=False,start_wall_ns=3,end_wall_ns=4)
def test_completed_generation_links():assert checked_record(wait(),{1:record()})[1] is None
def test_unknown_and_identity_mismatch():
 assert checked_record(wait(),{})[1]=='record_not_observed'
 r=record();r['event_token']=2;assert checked_record(wait(),{1:r})[1]=='identity_mismatch'
def test_overlapping_and_failed_record_rejected():
 for key in ('overlap','success'):
  r=record();r[key]=key=='overlap';assert checked_record(wait(),{1:r})[1]=='failed_or_concurrent'
def test_reuse_rejects_stale_generation():
 r2=record();r2.update(call_id=2,start_wall_ns=2.1,end_wall_ns=2.5)
 assert checked_record(wait(),{1:record(),2:r2})[1]=='later_or_overlapping_record'
def test_wait_during_record_unknown():
 r=record();r['end_wall_ns']=3.5;assert checked_record(wait(),{1:r})[1]=='record_not_completed_before_wait'
def test_api_requires_thread_containment_and_uniqueness():
 s=dict(ts=10,dur=10,pid=1,tid=2);a=dict(ts=11,dur=2,pid=1,tid=2,name='cudaEventRecordWithFlags')
 assert api_for_scope(s,[a],'record')==a
 assert api_for_scope(s,[a,a],'record') is None
 assert api_for_scope(s,[dict(a,tid=3)],'record') is None
 assert api_for_scope(s,[dict(a,ts=19)],'record') is None

def test_capture_crossing_unknown():
 r=record();r['end_outside_window']=True;assert checked_record(wait(),{1:r})[1]=='capture_boundary_crossing'

def test_cuda_sync_API_intervals_are_not_GPU_work():
 from full_serving.boundary_cuda_event_analysis import gpu_work
 es=[dict(cat=c,ph='X') for c in ('cuda_sync','cuda_runtime','kernel','gpu_memcpy','gpu_memset')]
 assert [e['cat'] for e in gpu_work(es)]==['kernel','gpu_memcpy','gpu_memset']

def test_direct_stream_identity_unknown_and_conflict():
 from full_serving.boundary_cuda_event_analysis import stream_identity_map
 a=dict(status='resolved',stream_device=1,stream_handle=123,device=1,context=2,stream=9,identity_boundary='start')
 b=dict(a,identity_boundary='end')
 assert stream_identity_map([a,b])[(1,123)]=={(1,2,9)}
 assert not stream_identity_map([a])
 assert not stream_identity_map([a,dict(b,status='unknown')])
 assert len(stream_identity_map([a,dict(b,stream=10)])[(1,123)])==2
