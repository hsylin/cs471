"""Sample actual native batch metadata and nonblocking GPU stream intervals.

No token IDs, prompts, request IDs or output arrays are recorded. CPU sequence
lengths are native upper bounds, not exact KV occupancy. GPU stream spans include
peer waiting and are neither kernel sums nor a causal critical-path decomposition.
"""
import functools,inspect,json,os,time
from pathlib import Path
import threading
from full_serving.boundary_sync_shape_contract import selected,empty_summary

def summarize_batch(batch):
 import numpy as np
 n=int(batch.num_reqs)
 def array(name,dtype):
  value=getattr(batch,name)
  # Reject GPU/torch objects rather than silently copying or synchronizing.
  assert isinstance(value,np.ndarray),name
  assert value.ndim==1 and len(value)==n,name
  return value.astype(dtype,copy=False)
 q=array("num_scheduled_tokens",np.int64)
 computed=array("num_computed_tokens_np",np.int64)
 prefilling=array("is_prefilling_np",np.bool_)
 upper=computed+q
 assert np.all(q>=0) and np.all(computed>=0)
 assert int(q.sum())==int(batch.num_tokens) and int(batch.num_draft_tokens)==0
 return dict(real_requests=n,real_tokens=int(batch.num_tokens),
  padded_tokens=int(batch.num_tokens_after_padding),padded_requests=int(batch.num_reqs_after_padding),
  prefill_requests=int(prefilling.sum()),decode_requests=int((~prefilling).sum()),
  prefill_tokens=int(q[prefilling].sum()),decode_tokens=int(q[~prefilling].sum()),
  max_scheduled_tokens=int(q.max()) if n else 0,
  computed_tokens_cpu_upper_sum=int(computed.sum()),
  sequence_cpu_upper_sum=int(upper.sum()),sequence_cpu_upper_max=int(upper.max()) if n else 0,
  attention_pairs_cpu_upper_proxy=int((q*upper).sum()),
  cpu_sequence_lengths_are_upper_bounds=True,exact_kv_occupancy=False)

class PendingSpans:
 def __init__(self):self.items=[]
 def add(self,start,end,record):self.items.append((start,end,record))
 def poll(self,emit):
  remaining=[]
  for start,end,record in self.items:
   if end.query():emit("gpu_span_observed",gpu_stream_ms=start.elapsed_time(end),**record)
   else:remaining.append((start,end,record))
  self.items=remaining
 def censored(self):return [record for _,_,record in self.items]

_INSTALLED=False
def install():
 global _INSTALLED
 if _INSTALLED:return
 if os.environ.get("FULL_SERVING_BOUNDARY_PROBE")!="shapes":return
 assert os.environ.get("CS471_SUSTAINED_ARM")=="b4"
 import torch,vllm,hashlib
 from vllm.distributed import get_ep_group
 from vllm.v1.worker.gpu import model_runner as module
 pins=json.loads(Path(os.environ["CS471_GATE_NATIVE_PINS"]).read_text())
 # The hook points and all GPU/CPU metadata assumptions are source-pinned.
 for path in ("v1/worker/gpu/model_runner.py","v1/worker/gpu/input_batch.py","v1/worker/gpu/dp_utils.py"):
  assert hashlib.sha256((Path(vllm.__file__).parent/path).read_bytes()).hexdigest()==pins[path]
 runner=module.GPUModelRunner
 original=runner.execute_model;signature=inspect.signature(original)
 old_prepare=runner.prepare_inputs;old_dispatch=module.dispatch_cg_and_sync_dp
 root=Path(os.environ["FULL_SERVING_RUN_DIR"]);context=threading.local()
 pending=PendingSpans();calls=0;sync_calls=0
 def emit(kind,**fields):
  rank=get_ep_group().rank_in_group
  with (root/f"boundary-shapes-rank{rank}.jsonl").open("a") as f:
   f.write(json.dumps(dict(kind=kind,rank=rank,wall_ns=time.time_ns(),**fields),sort_keys=True)+"\n")
 @functools.wraps(old_prepare)
 def prepare(self,*args,**kwargs):
  batch=old_prepare(self,*args,**kwargs)
  record=getattr(context,"record",None)
  if record is not None:
   # This method is used only by real request execution in the pinned source.
   assert not record["dummy"] and not record["profile"]
   emit("actual_batch",**record,**summarize_batch(batch))
  return batch
 @functools.wraps(old_dispatch)
 def dispatch(*args,**kwargs):
  nonlocal sync_calls
  sync_calls+=1
  start=time.perf_counter_ns();out=old_dispatch(*args,**kwargs)
  record=getattr(context,"record",None)
  if record is not None:
   descriptor,sync=out
   assert record["expected_sync_call"]==sync_calls
   record["sync_call"]=sync_calls
   sizes=None
   if sync is not None:
    assert sync.num_tokens_across_dp.device.type=="cpu"
    sizes=sync.num_tokens_across_dp.tolist()
   emit("agreed_batch",**record,cg_mode=str(descriptor.cg_mode),
        agreed_tokens=int(descriptor.num_tokens),agreed_requests=descriptor.num_reqs,
        padded_tokens_across_dp=sizes,host_sync_dispatch_ns=time.perf_counter_ns()-start,
        warning="DP agreed padded sizes, not unpadded active work or collective bandwidth")
   if record["dummy"]:
    emit("explicit_empty_batch",**record,**empty_summary(int(descriptor.num_tokens),descriptor.num_reqs))
  return out
 @functools.wraps(original)
 def execute(self,scheduler_output,*args,**kwargs):
  nonlocal calls
  calls+=1
  bound=signature.bind(self,scheduler_output,*args,**kwargs);bound.apply_defaults()
  dummy=bool(bound.arguments["dummy_run"]);profile=bool(bound.arguments["is_profile"])
  # A non-dummy zero-token call returns before native DP dispatch and does not
  # advance the global ordinal. Dummy DP peers execute the padding branch.
  scheduled=int(scheduler_output.total_num_scheduled_tokens)
  dispatches=dummy or scheduled>0
  sample=dispatches and selected(sync_calls+1,profile,torch.cuda.is_current_stream_capturing())
  # Query only; no synchronize/wait for pending events.
  if sample:pending.poll(emit)
  record=None;start=end=None;failed=True
  if sample:
   if not profile:
    record=dict(execute_call=calls,dummy=dummy,profile=profile,execute_start_wall_ns=time.time_ns(),
                expected_sync_call=sync_calls+1,sample_policy="native_dp_sync_ordinal",
                scheduler_token_count=scheduled,
                scheduler_count_scope="Dummy scheduler counts are padding descriptors, not actual request work")
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
  context.record=record
  try:
   result=original(self,scheduler_output,*args,**kwargs);failed=False;return result
  finally:
   context.record=None
   if record is not None:
    end.record()
    pending.add(start,end,dict(record,failed=failed,
      scope="current GPU stream around execute_model; excludes later EPLB step, includes preparation/forward/host gaps/peer waiting"))
    emit("gpu_span_enqueued",**record,failed=failed)
    emit("unobserved_span_snapshot",execute_call=calls,
         pending_execute_calls=[x["execute_call"] for x in pending.censored()],
         warning="Snapshot only; remaining events at termination are censored, never zero")
 runner.prepare_inputs=prepare;module.dispatch_cg_and_sync_dp=dispatch;runner.execute_model=execute
 _INSTALLED=True
