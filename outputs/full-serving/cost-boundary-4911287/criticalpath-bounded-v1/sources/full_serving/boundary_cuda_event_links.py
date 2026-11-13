"""Python-issued CUDA-event identity ledger; native/graph-internal edges unknown.

Never changes event/stream flags, inserts synchronization, or retries CUDA calls.
Concurrent operations on one event are rejected for linkage, not serialized.
"""
import functools,itertools,threading,time,weakref

class EventLedger:
 def __init__(self):
  self.lock=threading.RLock();self.objects=weakref.WeakKeyDictionary();self.tokens=itertools.count(1);self.calls=itertools.count(1);self.streams=set()
 def begin(self,event,kind):
  with self.lock:
   s=self.objects.setdefault(event,dict(token=next(self.tokens),active={},latest=None,ambiguous=False))
   call=next(self.calls);overlap=bool(s['active'])
   if overlap:
    for x in s['active'].values():x['overlap']=True
    s['ambiguous']=True
   row=dict(event_token=s['token'],call_id=call,operation=kind,overlap=overlap,
            record_call_id=s['latest'] if kind!='record' and not overlap and not s['ambiguous'] else None)
   s['active'][call]=row
   return row
 def end(self,event,row,success):
  with self.lock:
   s=self.objects[event];s['active'].pop(row['call_id'])
   if row['operation']=='record':
    s['latest']=row['call_id'] if success else None
    s['ambiguous']=not success or row['overlap']
   row['success']=success
   if row['overlap'] or not success:row['record_call_id']=None
   return dict(row)

def install(torch,active,emit):
 """active()->window name or None; emit receives private per-call dictionaries."""
 ledger=EventLedger();originals={}
 for kind in ('record','wait','synchronize'):
  old=getattr(torch.cuda.Event,kind);originals[kind]=old
  def factory(old,kind):
   @functools.wraps(old)
   def call(event,*args,**kwargs):
    row=ledger.begin(event,kind);window=active();name=f"cs471::cuda_event::{row['event_token']}::{row['call_id']}::{kind}"
    stream=None
    # Resolve exactly the default stream the original Python API would use.
    if kind in ('record','wait') and len(args)<=1 and set(kwargs)<= {'stream'} and not (args and kwargs):
     stream=args[0] if args else kwargs.get('stream')
     if stream is None:stream=torch.cuda.current_stream()
     args=(stream,);kwargs={}
    if stream is not None:
     with ledger.lock:ledger.streams.add((int(stream.device.index),int(stream.cuda_stream)))
    fields=dict(window=window,annotation=name,thread=threading.get_native_id(),
      stream_handle=int(stream.cuda_stream) if stream is not None else None,
      stream_device=int(stream.device.index) if stream is not None else None,
      start_wall_ns=time.time_ns())
    success=False
    try:
     if window is None:out=old(event,*args,**kwargs)
     else:
      with torch.profiler.record_function(name):out=old(event,*args,**kwargs)
     success=True
     return out
    finally:
     fields['end_wall_ns']=time.time_ns();fields.update(ledger.end(event,row,success))
     if window is not None:emit(fields)
   return call
  setattr(torch.cuda.Event,kind,factory(old,kind))
 return originals,ledger
