from full_serving.boundary_cuda_event_analysis import analyze_values

def fixture():
 links=[dict(call_id=1,operation='record',event_token=1,success=True,overlap=False,start_wall_ns=10,end_wall_ns=11,stream_device=0,stream_handle=101,annotation='record'),dict(call_id=2,operation='wait',event_token=1,record_call_id=1,success=True,overlap=False,start_wall_ns=20,end_wall_ns=21,stream_device=0,stream_handle=202,annotation='wait')]
 for handle,stream in ((101,3),(202,4)):
  for bound in ('start','end'):links.append(dict(kind='cuda_stream_identity',stream_device=0,stream_handle=handle,status='resolved',device=0,context=7,stream=stream,identity_boundary=bound))
 e=[dict(cat='user_annotation',ph='X',name='record',pid=1,tid=1,ts=10,dur=2),dict(cat='user_annotation',ph='X',name='wait',pid=1,tid=1,ts=20,dur=2),dict(cat='cuda_runtime',ph='X',name='cudaEventRecord',pid=1,tid=1,ts=10.5,dur=.5,args={'correlation':1}),dict(cat='cuda_runtime',ph='X',name='cudaStreamWaitEvent',pid=1,tid=1,ts=20.5,dur=.5,args={'correlation':2})]
 return e,links

def test_two_bound_queries_and_API_link_without_sync_activity():
 e,l=fixture();r=analyze_values(e,l);assert r['cross_stream_links']==1
 assert r['edges'][0]['native_sync_CPU_interval_us'] is None and r['edges'][0]['linkage_basis']=='Python_generation_API_and_stream_queries'

def test_missing_destination_boundary_remains_unknown():
 e,l=fixture();l.pop();r=analyze_values(e,l);assert r['linked_waits']==0

def test_present_conflicting_native_wait_is_not_ignored():
 e,l=fixture();e.append(dict(cat='cuda_sync',ph='X',name='Stream Wait Event',ts=20.5,dur=.5,args=dict(correlation=2,device=0,context=7,stream=9)));assert analyze_values(e,l)['linked_waits']==0
