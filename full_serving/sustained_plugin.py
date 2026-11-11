"""Unchanged policies plus explicit proposal/transfer/commit observation.

The OFF reference does not load this plugin. Observation overhead is charged
to the enabled arms. Event completion is observed, never inferred from drain.
"""
import hashlib
import json
import os
from pathlib import Path
import time

_INSTALLED=False


def register():
    global _INSTALLED
    if _INSTALLED:return
    arm=os.environ['CS471_SUSTAINED_ARM']
    if arm=='resident_v2':
        from full_serving.proposal_plugin import register as policy
        policy();_INSTALLED=True;return
    if arm in ('b3','b4'):
        from full_serving.monitoring_gate_plugin import register as policy
        policy()
    if arm not in ('b1','b2','b3','b4'):raise RuntimeError('Unsupported sustained arm')
    import torch
    import vllm
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb import eplb_state as state, async_worker as worker
    from vllm.distributed.eplb.eplb_utils import device_stream
    pins=json.loads(Path(os.environ['CS471_GATE_NATIVE_PINS']).read_text())
    for name,sha in pins.items():
        assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==sha,name
    old_step,old_rearrange=state.EplbState.step,state.EplbState.rearrange
    old_add=state.EplbState.add_model
    old_plan,old_move,old_transfer=worker.run_rebalance_experts,state._move_to_workspace,worker.transfer_layer
    clock=0;pending=[];cycle=0
    def emit(kind,**fields):
        rank=get_ep_group().rank_in_group
        path=Path(os.environ['FULL_SERVING_RUN_DIR'])/f'migration-events-rank{rank}.jsonl'
        with path.open('a') as f:f.write(json.dumps(dict(kind=kind,arm=arm,rank=rank,wall_ns=time.time_ns(),**fields),sort_keys=True)+'\n')
    def step(self,is_dummy=False,is_profile=False,log_stats=False):
        nonlocal clock
        if not is_profile:
            clock+=1
            ready=[]
            for start,end,record in pending:
                if end.query():emit('commit_completed',gpu_copy_ms=start.elapsed_time(end),observed_step=clock,**record)
                else:ready.append((start,end,record))
            pending[:]=ready
        return old_step(self,is_dummy=is_dummy,is_profile=is_profile,log_stats=log_stats)
    def add(self,*args,**kwargs):
        result=old_add(self,*args,**kwargs)
        assert self.is_async and self.expert_rearrangement_step_interval==3000
        emit('bound',native_interval=3000,async_path=True,observer_cost_charged=True)
        return result
    def rearrange(self,is_profile=False,rank_mapping=None,use_last_expert_load=False):
        if not is_profile:
            for ms in self.model_states.values():ms.cs471_observed_step=clock
        return old_rearrange(self,is_profile=is_profile,rank_mapping=rank_mapping,use_last_expert_load=use_last_expert_load)
    def plan(ms,self,current,stream):
        nonlocal cycle
        start=time.perf_counter_ns();new=old_plan(ms,self,current,stream)
        cycle=getattr(ms,'cs471_observed_step',clock)
        # Diagnostic branch is opt-in and never used in the main comparison.
        branch=os.environ.get('CS471_PAYBACK_BRANCH')
        target=int(os.environ.get('CS471_PAYBACK_STEP','3750'))
        if branch:
            assert arm=='b2' and branch in ('allow','skip')
            admitted=new if branch=='allow' and cycle==target else current.clone()
        else:admitted=new
        emit('proposal',step=cycle,old_map=current.tolist(),candidate=new.tolist(),admitted=admitted.tolist(),
             changed_slots=int((admitted!=current).sum()),host_plan_ns=time.perf_counter_ns()-start,
             old_map_sha256=hashlib.sha256(current.numpy().tobytes()).hexdigest(),branch=branch)
        ms.cs471_observed_cycle=cycle
        return admitted
    def transfer(*args,**kwargs):
        start=time.perf_counter_ns();result=old_transfer(*args,**kwargs)
        # Host return may precede GPU completion. This is enqueue/host time,
        # not transfer latency; commit timestamps give an observed upper bound.
        emit('transfer_enqueued',cycle_step=cycle,layer=kwargs.get('layer_idx'),
             host_call_ns=time.perf_counter_ns()-start,received_experts=int(result.recv_count))
        return result
    def move(ms,rank):
        result=ms.pending_result;layer=result.layer_idx
        record=dict(cycle_step=getattr(ms,'cs471_observed_cycle',None),layer=layer,
            remote_received_tensor_bytes=int(result.transfer_metadata.recv_count)*sum(w[0].numel()*w.element_size() for w in ms.model.expert_weights[layer]),
            map_sha256=hashlib.sha256(result.new_physical_to_logical_map.numpy().tobytes()).hexdigest())
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();old_move(ms,rank);end.record();pending.append((start,end,record))
        emit('commit_enqueued',step=clock,**record)
    state.EplbState.add_model=add;state.EplbState.step=step;state.EplbState.rearrange=rearrange
    worker.run_rebalance_experts=plan;worker.transfer_layer=transfer;state._move_to_workspace=move
    _INSTALLED=True
