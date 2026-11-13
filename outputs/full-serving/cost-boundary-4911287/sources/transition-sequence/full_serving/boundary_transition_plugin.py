"""Frozen-candidate placement interventions. Diagnostic only; not a final B4 policy.

KEEP and CHARGED warm up on the old pilot map. PREINSTALLED warms up on the
candidate. All retain native recording/window/readiness/transfer/commit paths
and the charged sustained observer, but replace online candidate optimization
with one frozen candidate. No online B4/history decision is claimed here.
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
    from full_serving.boundary_transition_contract import Schedule,canonical,validate_spec,verify_weight_records,native_hash_reference,agreed_marker
    mode=os.environ["CS471_TRANSITION_MODE"]
    path=Path(os.environ["CS471_TRANSITION_SPEC"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ["CS471_TRANSITION_SPEC_SHA256"]
    spec=json.loads(path.read_text());validate_spec(spec)
    root=Path(os.environ["FULL_SERVING_RUN_DIR"])
    pilot=os.environ.get("FULL_SERVING_PROPOSAL_PILOT")=="1"
    shadow=os.environ.get("CS471_TRANSITION_SHADOW")=="1";assert shadow==pilot
    assert os.environ.get("FULL_SERVING_BOUNDARY_FIX","native")=="native"
    assert os.environ.get("FULL_SERVING_BOUNDARY_PROBE","off")=="off"
    assert not os.environ.get("FULL_SERVING_PAYBACK_BRANCH")
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_ep_group
    from vllm.distributed.parallel_state import get_eplb_group
    from vllm.distributed.eplb import eplb_state as state,async_worker as worker
    from vllm.distributed.eplb.eplb_utils import device_stream
    from full_serving.resident_diagnostic import tensor_sha
    original_add=state.EplbState.add_model
    original_move=state._move_to_workspace
    original_rearrange=state.EplbState.rearrange
    schedules={};references={};shadow_layers=set();graph_readers={}
    def gather(value):
        group=get_eplb_group().cpu_group
        peers=[None]*group.size();assert len(peers)==8
        dist.all_gather_object(peers,value,group=group)
        return peers
    def emit(kind,**kw):
        rank=get_ep_group().rank_in_group
        with (root/f"transition-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(dict(kind=kind,rank=rank,mode=mode,wall_ns=time.time_ns(),**kw),sort_keys=True)+"\n")
    def weights(ms,mapping):
        rank=get_ep_group().rank_in_group;records=[]
        assert len(ms.model.expert_weights)==48
        for layer,tensors in enumerate(ms.model.expert_weights):
            assert len(tensors)==2 and all(w.ndim==2 and w.shape[0]==20 for w in tensors)
            by_size={w.shape[1]:w for w in tensors};assert set(by_size)=={1536*2048,2048*768}
            ordered=(by_size[1536*2048],by_size[2048*768])
            for local in range(20):
                slot=rank*20+local
                records.append(dict(layer=layer,slot=slot,expert=mapping[layer][slot],
                                    sha256=[tensor_sha(w[local]) for w in ordered]))
        return gather(dict(rank=rank,weights=records))
    def metadata(ms,target):
        assert ms.physical_to_logical_map.cpu().tolist()==target
        counts=ms.logical_replica_count.cpu().tolist()
        inverse=ms.logical_to_physical_map.cpu().tolist()
        for layer,row in enumerate(target):
            for expert in range(128):
                wanted=[i for i,e in enumerate(row) if e==expert]
                assert counts[layer][expert]==len(wanted)
                assert inverse[layer][expert][:len(wanted)]==wanted
                assert all(x==-1 for x in inverse[layer][expert][len(wanted):])
    def capture_readers(ms):
        from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
        rank=get_ep_group().rank_in_group
        real=0 if rank==0 else 3+rank%5
        device=ms.physical_to_logical_map.device
        inputs=[[((i*8+j+rank)%128) if i<real else -1 for j in range(8)] for i in range(8)]
        readers={}
        for layer in range(48):
            ids=torch.tensor(inputs,dtype=torch.int32,device=device)
            counts=torch.zeros(160,dtype=torch.int32,device=device)
            flag=torch.tensor(False,device=device)
            num=torch.tensor(real,dtype=torch.int32,device=device)
            def call():
                return eplb_map_to_physical_and_record(ids,counts,ms.logical_to_physical_map[layer],
                    ms.logical_replica_count[layer],flag,num)
            call();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):mapped=call()
            readers[layer]=(graph,mapped,inputs,ids,counts,flag,num)
        return readers
    def check_reader(ms,layer,target):
        graph,mapped,inputs,*retained=graph_readers[id(ms)][layer]
        graph.replay()
        assert mapped.cpu().tolist()==native_hash_reference(inputs,target)
        assert not bool(retained[1].any())
    def add(self,*args,**kwargs):

        result=original_add(self,*args,**kwargs)
        assert len(self.model_states)==1 and self.async_worker is None
        assert self.is_async and self.expert_rearrangement_step_interval==3000
        ms=next(iter(self.model_states.values()))
        assert ms.pending_result is None and not ms.rebalanced
        assert (ms.model.num_moe_layers,ms.model.num_logical_experts,ms.model.num_physical_experts)==(48,128,160)
        native_map=ms.physical_to_logical_map.cpu().tolist()
        logical=verify_weight_records(weights(ms,native_map),native_map)
        schedule=Schedule(mode,spec["old_map"],spec["candidate_map"]);target=schedule.initial()
        fingerprints=gather((canonical(native_map),canonical(target),spec["seal"],mode))
        assert len(set(fingerprints))==1
        if pilot:graph_readers[id(ms)]=capture_readers(ms)
        begin=time.perf_counter_ns()
        # Native synchronous migration before any warmup/capture; not a measured cost.
        state.rearrange_expert_weights_inplace(
            ms.physical_to_logical_map,torch.tensor(target,dtype=torch.int64),
            ms.model.expert_weights,ms.expert_buffer,get_ep_group().device_group,ms.communicator)
        state._commit_eplb_maps(ms,torch.tensor(target,dtype=torch.int64))
        completion=torch.cuda.Event();completion.record();completion.synchronize()
        elapsed=time.perf_counter_ns()-begin
        metadata(ms,target)
        if pilot:
            for layer in range(48):check_reader(ms,layer,target[layer])
        verify_weight_records(weights(ms,target),target,reference=logical)
        schedules[id(ms)]=schedule;references[id(ms)]=logical
        emit("bound",spec_sha256=os.environ["CS471_TRANSITION_SPEC_SHA256"],
             initial_map_sha256=canonical(target),full_weight_slots_checked=48*160,
             metadata_all_elements_checked=True,startup_gpu_completion_observed=True,
             startup_graph_reader_layers=48 if pilot else 0,
             startup_transition_wall_ns=elapsed,shadow=shadow,
             bookkeeping="native B2 recording/window/readiness and sustained observer; frozen proposal replaces optimizer",
             timing_scope="startup migration excluded from request latency; all startup transfers and weight checks precede warmup")
        return result
    def rearrange(self,is_profile=False,rank_mapping=None,use_last_expert_load=False):
        if not is_profile:
            marker=root/"measured-phase-start.json";token=None
            if marker.exists():
                obj=json.loads(marker.read_text())
                assert obj["phase"]=="measured" and obj["candidate_seal"]==spec["seal"]
                token=canonical(obj)
            for ms in self.model_states.values():ms.cs471_transition_phase_marker=token
        return original_rearrange(self,is_profile=is_profile,rank_mapping=rank_mapping,use_last_expert_load=use_last_expert_load)
    def plan(ms,self,current,stream):
        schedule=schedules[id(ms)];clock=ms.cs471_observed_step
        tokens=gather(ms.cs471_transition_phase_marker)
        armed=agreed_marker(tokens);present={t for t in tokens if t is not None}
        new,fired=schedule.choose(current.tolist(),armed)
        with device_stream(stream):demand=ms.eplb_stats.global_expert_load_window.cpu().numpy()
        demand_hash=hashlib.sha256(demand.tobytes()).hexdigest()
        signature=(clock,canonical(current.tolist()),canonical(new),demand_hash,tuple(sorted(present)))
        assert len(set(gather(signature)))==1
        if fired:ms.cs471_transition_target_cycle=clock
        emit("proposal",step=clock,triggered=fired,measured_marker_agreed=armed,
             old_map_sha256=signature[1],admitted_map_sha256=signature[2],
             live_demand_sha256=demand_hash,rank_agreement=True,
             changed_slots=sum(a!=b for old,row in zip(current.tolist(),new) for a,b in zip(old,row)))
        return torch.tensor(new,dtype=current.dtype)
    def move(ms,ep_rank):
        pending=ms.pending_result;assert pending is not None
        layer=pending.layer_idx
        target=pending.new_physical_to_logical_map.tolist()
        cycle=getattr(ms,"cs471_observed_cycle",None)
        check=shadow and mode=="charged" and cycle==getattr(ms,"cs471_transition_target_cycle",-1)
        result=original_move(ms,ep_rank)
        if check:
            assert layer not in shadow_layers
            assert ms.physical_to_logical_map[layer].cpu().tolist()==target
            counts=ms.logical_replica_count[layer].cpu().tolist()
            inverse=ms.logical_to_physical_map[layer].cpu().tolist()
            for expert in range(128):
                wanted=[i for i,e in enumerate(target) if e==expert]
                assert counts[expert]==len(wanted) and inverse[expert][:len(wanted)]==wanted
                assert all(x==-1 for x in inverse[expert][len(wanted):])
            tensors=ms.model.expert_weights[layer];by_size={w.shape[1]:w for w in tensors}
            ordered=(by_size[1536*2048],by_size[2048*768])
            for local in range(20):
                key=(layer,target[ep_rank*20+local])
                assert tuple(tensor_sha(w[local]) for w in ordered)==references[id(ms)][key]
            check_reader(ms,layer,target)
            shadow_layers.add(layer)
            emit("charged_weight_shadow_passed",cycle_step=cycle,layer=layer,
                 full_local_experts_checked=20,metadata_all_elements_checked=True,graph_reader_checked=True,scope="Pilot only. Late pending slot may already hold next layer.")
        return result
    state.EplbState.add_model=add
    state.EplbState.rearrange=rearrange
    worker.run_rebalance_experts=plan
    state._move_to_workspace=move
    os.environ["CS471_SUSTAINED_ARM"]="b2"
    from full_serving.sustained_plugin import register as observer
    observer()
    _INSTALLED=True
