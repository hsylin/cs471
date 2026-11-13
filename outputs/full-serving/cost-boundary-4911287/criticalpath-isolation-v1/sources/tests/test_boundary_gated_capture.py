from full_serving.boundary_gated_capture import slice_events,gated_capture_class

def event(name,cat,ts,dur=1,**args):return dict(name=name,cat=cat,ts=ts,dur=dur,args=args,ph='X',pid=7)
def test_slice_keeps_overhang_and_unknown_activity():
 e=[event('a','user_annotation',10),event('api','cuda_runtime',12,1,correlation=8),event('b','user_annotation',20),event('overhang','kernel',21,5,correlation=8),event('unknown','kernel',15,2,correlation=99),event('outside','kernel',23,2,correlation=99),event('before','cpu_op',1)]
 out,_=slice_events(e,'a','b');assert {x['name'] for x in out}=={'a','b','api','overhang','unknown'}
def test_slice_rejects_missing_anchor():
 import pytest
 with pytest.raises(AssertionError):slice_events([event('a','user_annotation',10)],'a','b')
def test_gpu_annotation_not_cpu_anchor():
 import pytest
 with pytest.raises(AssertionError):slice_events([event('a','gpu_user_annotation',10),event('b','user_annotation',20)],'a','b')
def test_duplicate_correlation_across_processes_not_linked():
 a=event('api1','cuda_runtime',12,1,correlation=8);b=event('api2','cuda_runtime',12,1,correlation=8);b['pid']=9
 out,_=slice_events([event('a','user_annotation',10),event('b','user_annotation',20),a,b,event('overhang','kernel',23,1,correlation=8)],'a','b');assert 'overhang' not in {e['name'] for e in out}

def test_one_session_three_windows_and_final_tick_no_reentry(tmp_path,monkeypatch):
 import types,contextlib
 import full_serving.boundary_gated_capture as module
 from full_serving.boundary_critical_plugin import Capture
 calls=[]
 class Prof:
  def start(self):calls.append('start')
  def stop(self):calls.append('stop')
  def toggle_collection_dynamic(self,on,activities):calls.append(('toggle',on))
  def export_chrome_trace(self,path):calls.append('export')
 torch=types.SimpleNamespace(profiler=types.SimpleNamespace(ProfilerActivity=types.SimpleNamespace(CPU='cpu',CUDA='gpu'),profile=lambda **kw:Prof(),record_function=lambda n:contextlib.nullcontext()),_C=types.SimpleNamespace(_profiler=types.SimpleNamespace(_ExperimentalConfig=lambda **kw:kw)),cuda=types.SimpleNamespace(synchronize=lambda:calls.append('sync')))
 c=gated_capture_class(Capture)(tmp_path,0,torch,group=1);c.barrier=lambda:None;c.plan={'windows':[{'name':x} for x in ('a','b','c')]}
 monkeypatch.setattr(module,'export_slices',lambda root,rank,session,windows:{w['name']:{'path':w['name']} for w in windows})
 for i,name in enumerate(('a','b','c')):
  c.ordinal=1024+i*64;c.start(dict(name=name,start_ordinal=c.ordinal,end_ordinal=c.ordinal+64));c.ordinal+=64;c.stop()
 assert calls.count('start')==calls.count('stop')==calls.count('export')==1
 assert calls.count('sync')==3 and c.session_closed
 assert not any(isinstance(x,tuple) and x[0]=='toggle' for x in calls)
 before=len(calls);c.tick();assert len(calls)==before

def test_partial_windows_do_not_export(tmp_path,monkeypatch):
 import types,contextlib
 from full_serving.boundary_critical_plugin import Capture
 calls=[]
 p=types.SimpleNamespace(start=lambda:None,toggle_collection_dynamic=lambda *a:None,stop=lambda:calls.append('stop'))
 t=types.SimpleNamespace(profiler=types.SimpleNamespace(ProfilerActivity=types.SimpleNamespace(CPU=1,CUDA=2),profile=lambda **kw:p,record_function=lambda n:contextlib.nullcontext()),_C=types.SimpleNamespace(_profiler=types.SimpleNamespace(_ExperimentalConfig=lambda **kw:kw)),cuda=types.SimpleNamespace(synchronize=lambda:None))
 c=gated_capture_class(Capture)(tmp_path,0,t);c.barrier=lambda:None;c.plan={'windows':[{'name':'a'},{'name':'b'}]};c.ordinal=64;c.start(dict(name='a',start_ordinal=64,end_ordinal=128));c.ordinal=128;c.stop()
 assert not c.session_closed and not calls and c.profile is p

def test_no_native_sync_activity_changes_only_observation_flag():
 from pathlib import Path
 code=(Path(__file__).parents[1]/'full_serving/boundary_gated_capture.py').read_text()
 assert 'enable_cuda_sync_events=False' in code and 'enable_cuda_sync_events=True' not in code
 assert 'profile_all_threads=False' in code
 assert 'toggle_collection_dynamic' not in code
