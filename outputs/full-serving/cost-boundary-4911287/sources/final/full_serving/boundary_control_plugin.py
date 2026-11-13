"""Full-model cost interventions; never final policy replacements.

All fixed controls load the same balanced32 map. They retain actual native
communication and full experts. History contrast includes recording and native
window copies; gate-only contrast removes only the extra sustained observer.
"""
import hashlib,json,os,time
from pathlib import Path

CONTROLS=("fixed32_idle","fixed32_record","fixed32_history","native_hash",
          "static_home","resident_charged","b4_gate_only","b4_observed")
_INSTALLED=False

def register():
 global _INSTALLED
 if _INSTALLED:return
 control=os.environ["CS471_BOUNDARY_CONTROL"];assert control in CONTROLS
 assert os.environ.get("FULL_SERVING_BOUNDARY_FIX","native")=="native"
 import torch,vllm
 from vllm.distributed import get_ep_group
 from vllm.distributed.eplb import eplb_state as state
 from vllm.v1.worker.gpu.eplb_utils import EPLBController
 from full_serving.resident_policy import balanced_map
 pins=json.loads(Path(os.environ["CS471_GATE_NATIVE_PINS"]).read_text())
 for name,digest in pins.items():
  assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==digest,name
 root=Path(os.environ["FULL_SERVING_RUN_DIR"])
 def emit(kind,**fields):
  rank=get_ep_group().rank_in_group
  with (root/f"boundary-control-rank{rank}.jsonl").open("a") as f:
   f.write(json.dumps(dict(kind=kind,rank=rank,control=control,wall_ns=time.time_ns(),**fields),sort_keys=True)+"\n")
 if control.startswith("fixed32_"):
  os.environ["CS471_EXPERIMENT_ARM"]="balanced32_off" if control=="fixed32_idle" else "balanced32_record"
  os.environ["CS471_COST_PINS"]=os.environ["CS471_GATE_NATIVE_PINS"]
  # Scratch map/record oracle is a pilot-only admission test.
  os.environ["CS471_COST_DIAGNOSTIC"]="1" if os.environ.get("FULL_SERVING_PROPOSAL_PILOT")=="1" else "0"
  from full_serving.cost_plugin import register as base
  base()
  old_add=state.EplbState.add_model
  histories={}
  def add(self,*args,**kwargs):
   out=old_add(self,*args,**kwargs)
   for key,ms in self.model_states.items():
    assert ms.physical_to_logical_map.cpu().tolist()==[balanced_map()]*48
    if control=="fixed32_history" and key not in histories:
     from full_serving.logical_history_gpu import LogicalHistory
     histories[key]=LogicalHistory(ms.expert_load_pass.device,rolling=False)
   if os.environ.get("FULL_SERVING_PROPOSAL_PILOT")=="1":
    from full_serving.resident_diagnostic import tensor_sha,validate_weight_records
    import torch.distributed as dist
    assert len(self.model_states)==1
    ms=next(iter(self.model_states.values()));rank=get_ep_group().rank_in_group
    records=[]
    for layer,tensors in enumerate(ms.model.expert_weights):
     # Pinned RoutedExperts.get_expert_weights exposes flattened per-expert views.
     assert len(tensors)==2 and all(w.ndim==2 and w.shape[0]==20 for w in tensors)
     by_size={w.shape[1]:w for w in tensors}
     assert set(by_size)=={1536*2048,2048*768}
     tensors=(by_size[1536*2048],by_size[2048*768])
     for slot in range(20):
      physical=rank*20+slot
      records.append(dict(layer_index=layer,physical_slot=physical,logical_expert=balanced_map()[physical],
        w13_sha256=tensor_sha(tensors[0][slot]),w2_sha256=tensor_sha(tensors[1][slot])))
    peers=[None]*8
    dist.all_gather_object(peers,dict(rank=rank,weights=records),group=get_ep_group().cpu_group)
    checked=validate_weight_records(peers,balanced_map())
    emit("fixed_full_weight_check",status="passed",check=checked,local_weights=records)
   emit("bound",fixed_map=True,map=balanced_map(),native_bookkeeping=True,
        recording=control!="fixed32_idle",logical_history=control=="fixed32_history",
        planner=False,worker=False,migrations=False)
   return out
  state.EplbState.add_model=add
  if control=="fixed32_history":
   old_step=state.EplbState.step;old_rearrange=state.EplbState.rearrange
   def step(self,is_dummy=False,is_profile=False,log_stats=False):
    if not is_profile:
     for key,ms in self.model_states.items():
      histories[key].update(ms.physical_to_logical_map,ms.expert_load_pass,dummy=is_dummy)
    return old_step(self,is_dummy=is_dummy,is_profile=is_profile,log_stats=log_stats)
   def rearrange(self,is_profile=False,rank_mapping=None,use_last_expert_load=False):
    if not is_profile:
     values=[]
     for key in self.model_states:values.extend(histories[key].snapshot())
     self._allreduce_list(values)
     emit("history_snapshot_reduced",logical_windows=2,policy_decision=False)
    return old_rearrange(self,is_profile=is_profile,rank_mapping=rank_mapping,use_last_expert_load=use_last_expert_load)
   state.EplbState.step=step;state.EplbState.rearrange=rearrange
 elif control in ("native_hash","static_home","resident_charged"):
  from full_serving import resident_policy
  from full_serving.resident_pair import PairMapper
  from full_serving.resident_plugin import register as base
  resident_policy.ResidentMapper=PairMapper
  expected={"native_hash":"native_hash","static_home":"static_home","resident_charged":"resident_v1"}[control]
  assert os.environ["FULL_SERVING_CUSTOM_MODE"]==expected
  base()
  old_register=EPLBController.maybe_register_model
  def registered(self,*args,**kwargs):
   result=old_register(self,*args,**kwargs)
   emit("bound",fixed_map=True,map=balanced_map(),native_bookkeeping=False,recording=False,
        logical_history=False,planner=control=="resident_charged",
        planner_timing="included in each forward",worker=False,migrations=False,mapper=expected)
   return result
  EPLBController.maybe_register_model=registered
 else:
  os.environ["CS471_SUSTAINED_ARM"]="b4"
  os.environ["CS471_GATE_MODE"]="b4";os.environ["CS471_GATE_HISTORY"]="reduction"
  os.environ["FULL_SERVING_BOUNDARY_OBSERVATION"]="gate_only" if control=="b4_gate_only" else "full"
  from full_serving.sustained_plugin import register as base
  base()
  old_add=state.EplbState.add_model
  def add(self,*args,**kwargs):
   result=old_add(self,*args,**kwargs)
   emit("bound",fixed_map=False,gate_policy="unchanged B4 5%/5%",
        gate_commit_observer=True,additional_sustained_observer=control=="b4_observed",
        overhead_scope="Removes only additional sustained observer; original B4 observer remains")
   return result
  state.EplbState.add_model=add
 _INSTALLED=True
