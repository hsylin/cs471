import types
from full_serving.boundary_cupti_streams import StreamIdentity

def resolver():
 r=object.__new__(StreamIdentity);r.path='loaded-cupti'
 def ctx(p):p._obj.value=123;return 0
 def device(p):p._obj.value=1;return 0
 def cid(c,p):p._obj.value=7;return 0
 def sid(c,s,t,p):assert c.value==123 and s.value==456 and t==0;p._obj.value=19;return 0
 r.cuda=types.SimpleNamespace(cuCtxGetCurrent=ctx,cuCtxGetDevice=device)
 r.cupti=types.SimpleNamespace(cuptiGetContextId=cid,cuptiGetStreamIdEx=sid)
 return r

def test_direct_stream_query_binds_context_device_and_handle():
 v=resolver().resolve(1,456);assert v['status']=='resolved' and (v['device'],v['context'],v['stream'])==(1,7,19)
def test_mismatched_current_device_unknown():assert resolver().resolve(0,456)['status']=='unknown'
def test_special_handle_does_not_guess_semantics():assert resolver().resolve(1,0)['reason']=='special_stream_semantics_unresolved'
def test_failed_query_remains_unknown():
 r=resolver();r.cupti.cuptiGetStreamIdEx=lambda *args:3;assert r.resolve(1,456)['reason']=='CUPTI_query_failed'
