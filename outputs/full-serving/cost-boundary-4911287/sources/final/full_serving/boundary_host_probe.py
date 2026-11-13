"""Host control attribution only. No synchronization, policy or map changes.

Intervals nest and overlap across threads/ranks. They are NOT additive GPU cost.
Probe-on/off serving controls must measure instrumentation overhead.
"""
import atexit
import functools
import inspect
import json
import os
from pathlib import Path
import threading
import time

class HostLedger:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock=clock
        self.lock=threading.Lock()
        self.values={}
    def add(self, name, duration, failed=False):
        with self.lock:
            v=self.values.setdefault(name,dict(calls=0,total_ns=0,max_ns=0,failed_calls=0))
            v["calls"]+=1;v["total_ns"]+=duration;v["max_ns"]=max(v["max_ns"],duration)
            v["failed_calls"]+=int(failed)
    def wrap(self,name,fn):
        @functools.wraps(fn)
        def measured(*args,**kwargs):
            start=self.clock();failed=True
            try:
                result=fn(*args,**kwargs);failed=False;return result
            finally:
                self.add(name,self.clock()-start,failed)
        return measured
    def take(self):
        with self.lock:
            values=self.values;self.values={}
        return values

_INSTALLED=False
def install():
    global _INSTALLED
    if _INSTALLED:return
    if os.environ.get("FULL_SERVING_BOUNDARY_PROBE")!="host":
        _INSTALLED=True;return
    if os.environ.get("CS471_SUSTAINED_ARM")!="b4":
        raise RuntimeError("Host control probe currently admitted only for B4")
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb import eplb_state as state,async_worker as worker
    from full_serving.logical_history_gpu import LogicalHistory
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    ledger=HostLedger();record_lock=threading.Lock()
    root=Path(os.environ["FULL_SERVING_RUN_DIR"]);rank=None
    def emit(kind,**fields):
        nonlocal rank
        if rank is None:rank=get_ep_group().rank_in_group
        record=dict(kind=kind,rank=rank,wall_ns=time.time_ns(),monotonic_ns=time.perf_counter_ns(),**fields)
        with record_lock:
            with (root/f"boundary-host-rank{rank}.jsonl").open("a") as f:
                f.write(json.dumps(record,sort_keys=True)+"\n")
    def flush(reason):
        values=ledger.take()
        if values:emit("host_intervals",reason=reason,intervals=values)
    # Wrappers call each original exactly once and preserve return/exception.
    # They time CPU enqueue/wait work, not completed CUDA kernel execution.
    for obj,name,label in [
        (state.EplbState,"step","eplb_step_inclusive"),
        (state.EplbState,"_all_ranks_result_ready","all_rank_ready_cpu_collective"),
        (state.EplbState,"_allreduce_list","gate_window_allreduce_host"),
        (state,"_move_to_workspace","move_including_observers"),
        (state,"_commit_eplb_maps_for_layer","inverse_map_commit_host"),
        (worker,"run_rebalance_experts","plan_including_gate_and_json"),
        (worker,"transfer_layer","transfer_enqueue_host"),
        (LogicalHistory,"update","logical_history_update_enqueue"),
        (LogicalHistory,"snapshot","logical_history_snapshot_enqueue")]:
        setattr(obj,name,ledger.wrap(label,getattr(obj,name)))
    original=GPUModelRunner.execute_model
    original_signature=inspect.signature(original)
    count=0
    @functools.wraps(original)
    def execute(self,scheduler_output,*args,**kwargs):
        nonlocal count
        count+=1;start=time.perf_counter_ns()
        try:
            return original(self,scheduler_output,*args,**kwargs)
        finally:
            ledger.add("execute_model_host_inclusive",time.perf_counter_ns()-start)
            if count%64==0:
                bound=original_signature.bind(self,scheduler_output,*args,**kwargs)
                bound.apply_defaults()
                scheduled=getattr(scheduler_output,"num_scheduled_tokens",{})
                emit("scheduler_shape",execute_call=count,
                    scheduled_tokens=getattr(scheduler_output,"total_num_scheduled_tokens",None),
                    scheduled_requests=len(scheduled),
                    max_tokens_per_request=max(scheduled.values(),default=0),
                    dummy_kwarg=bound.arguments["dummy_run"],
                    profile_kwarg=bound.arguments["is_profile"],
                    scope="CPU scheduler shape, not padded MoE shape, GPU duration, or exact cross-rank clock alignment")
            if count%256==0:flush("256_execute_calls")
    GPUModelRunner.execute_model=execute
    # Worker termination may skip atexit. Unflushed tails are missing, not zero.
    atexit.register(lambda:flush("normal_python_exit"))
    old_add=state.EplbState.add_model
    @functools.wraps(old_add)
    def add(self,*args,**kwargs):
        result=old_add(self,*args,**kwargs)
        emit("bound",observer="host_ledger_v2_signature_bound",sample_shape_every=64,flush_every=256,
            policy_changed=False,gpu_synchronization_added=False,
            warning="Nested host durations overlap; probe-on timing not final speed evidence; unflushed shutdown tail censored")
        return result
    state.EplbState.add_model=add
    _INSTALLED=True
