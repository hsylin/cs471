"""Candidate: elide only redundant inverse-map commits, preserving native retirement.

This requires the pinned B4 planner's all-rank old/candidate/admitted-map agreement.
No worker flag, transfer, readiness collective or consumed event is bypassed.
CPU tests alone do not admit this candidate for final serving measurements.
"""
import functools
import json
import os
from pathlib import Path
import time

class CycleProof:
    def __init__(self,cycle,old,admitted):
        if len(old)!=len(admitted) or any(len(a)!=len(b) for a,b in zip(old,admitted)):
            raise ValueError("Map geometry changed")
        self.cycle=cycle
        self.rows={i:tuple(a) for i,(a,b) in enumerate(zip(old,admitted)) if tuple(a)==tuple(b)}
    def eligible(self,layer,target):
        return layer in self.rows and self.rows[layer]==tuple(target)
    def consume(self,layer,target):
        expected=self.rows.pop(layer,None)
        return expected is not None and expected==tuple(target)
    def invalidate(self):
        self.rows.clear()

def commit_with_proof(original,ms,new,layer,emit,shadow=None):
    proof=getattr(ms,"cs471_noop_commit_proof",None)
    # Unexpected GPU-sourced commits make the CPU snapshot proof stale.
    if proof is not None and not new.is_cpu:proof.invalidate()
    skip=proof is not None and new.is_cpu and proof.consume(layer,new.tolist())
    if not skip:return original(ms,new,layer)
    if shadow is not None:shadow(ms,new,layer)
    emit("map_commit_elided",cycle_step=proof.cycle,layer=layer,
         meaning="Identical metadata retained; native move retirement and event recording still required")
    return None

def invalidate_then_call(original,state,*args,**kwargs):
    for ms in state.model_states.values():
        proof=getattr(ms,"cs471_noop_commit_proof",None)
        if proof is not None:proof.invalidate()
    return original(state,*args,**kwargs)

_INSTALLED=False
def install():
    global _INSTALLED
    if _INSTALLED:return
    if os.environ.get("FULL_SERVING_BOUNDARY_FIX")!="noop_maps":
        _INSTALLED=True;return
    assert os.environ.get("CS471_SUSTAINED_ARM")=="b4"
    import torch
    from types import SimpleNamespace
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb import eplb_state as state,async_worker as worker
    old_plan=worker.run_rebalance_experts
    old_layer=state._commit_eplb_maps_for_layer
    old_full=state._commit_eplb_maps
    old_move=state._move_to_workspace
    shadow=os.environ.get("FULL_SERVING_BOUNDARY_FIX_SHADOW")=="1"
    root=Path(os.environ["FULL_SERVING_RUN_DIR"])
    def emit(kind,**fields):
        rank=get_ep_group().rank_in_group
        with (root/f"noop-commit-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(dict(kind=kind,rank=rank,wall_ns=time.time_ns(),**fields),sort_keys=True)+"\n")
    @functools.wraps(old_plan)
    def plan(ms,eplb,current,stream):
        # The wrapped pinned B4 gate has already agreed all input/output maps.
        admitted=old_plan(ms,eplb,current,stream)
        assert current.is_cpu and admitted.is_cpu
        assert tuple(current.shape)==tuple(admitted.shape)==(48,160)
        proof=CycleProof(getattr(ms,"cs471_gate_cycle"),current.tolist(),admitted.tolist())
        ms.cs471_noop_commit_proof=proof
        emit("unchanged_proof",cycle_step=proof.cycle,unchanged_layers=sorted(proof.rows),
             provenance="pinned B4 all-rank map/demand digest agreement",
             shadow=shadow)
        return admitted
    def map_shadow(ms,new,layer):
        reference=SimpleNamespace(
            physical_to_logical_map=ms.physical_to_logical_map[layer:layer+1].clone(),
            logical_to_physical_map=ms.logical_to_physical_map[layer:layer+1].clone(),
            logical_replica_count=ms.logical_replica_count[layer:layer+1].clone())
        old_layer(reference,new,0)
        for name in ("physical_to_logical_map","logical_to_physical_map","logical_replica_count"):
            assert torch.equal(getattr(ms,name)[layer],getattr(reference,name)[0]),name
        emit("map_shadow_passed",cycle_step=ms.cs471_noop_commit_proof.cycle,layer=layer,
             comparison="all elements of all three native metadata tensors")
    @functools.wraps(old_layer)
    def layer_commit(model_state,new_physical_to_logical_map,layer):
        return commit_with_proof(old_layer,model_state,new_physical_to_logical_map,layer,emit,
                                 map_shadow if shadow else None)
    @functools.wraps(old_full)
    def full_commit(model_state,new_physical_to_logical_map):
        proof=getattr(model_state,"cs471_noop_commit_proof",None)
        if proof is not None:proof.invalidate()
        return old_full(model_state,new_physical_to_logical_map)
    def invalidate_method(original):
        @functools.wraps(original)
        def wrapped(self,*args,**kwargs):
            return invalidate_then_call(original,self,*args,**kwargs)
        return wrapped
    # Any non-layer map write or a censored drain invalidates snapshot proofs.
    for method in ("drain_async","update_mapping","reconfigure_physical_expert_slots"):
        setattr(state.EplbState,method,invalidate_method(getattr(state.EplbState,method)))
    worker.run_rebalance_experts=plan
    state._commit_eplb_maps_for_layer=layer_commit
    state._commit_eplb_maps=full_commit
    if shadow:
        checked=set()
        @functools.wraps(old_move)
        def checked_move(model_state,ep_rank):
            ms=model_state;result=ms.pending_result;layer=result.layer_idx
            proof=getattr(ms,"cs471_noop_commit_proof",None)
            eligible=(proof is not None and result.new_physical_to_logical_map.is_cpu
                      and proof.eligible(layer,result.new_physical_to_logical_map.tolist()))
            before=None
            if eligible and layer not in checked:
                before=[w.detach().clone() for w in ms.model.expert_weights[layer]]
            out=old_move(ms,ep_rank)
            if before is not None:
                for expected,actual in zip(before,ms.model.expert_weights[layer]):
                    assert torch.equal(expected,actual),"No-op changed expert weights"
                checked.add(layer)
                emit("weight_shadow_passed",layer=layer,
                     compared_elements=sum(w.numel() for w in before),
                     pending_result_retired=ms.pending_result is None)
            return out
        state._move_to_workspace=checked_move
    _INSTALLED=True
