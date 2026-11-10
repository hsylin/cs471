"""Pinned opt-in B3/B4 gate on the native asynchronous candidate path.

Native state and transfers stay intact. Rejected layers use the current map, so
native unchanged-slot transport sends no expert tensors, while bookkeeping and
normal completion still run. No worker flag races or fabricated drain commits.
"""
import hashlib
import json
import os
from pathlib import Path
import time

_INSTALLED=False


def register():
    global _INSTALLED
    mode=os.environ.get('CS471_GATE_MODE')
    if not mode or _INSTALLED:return
    if mode not in ('b3','b4'):raise RuntimeError('Unknown gate mode')
    import numpy as np
    import torch
    from vllm.distributed import get_ep_group
    from vllm.distributed.parallel_state import get_eplb_group
    from vllm.distributed.eplb import eplb_state as state_module, async_worker
    from vllm.distributed.eplb.eplb_utils import device_stream
    from full_serving.proposal_gate import decide
    from full_serving.resident_plugin import EXPECTED_NATIVE_SOURCES
    import vllm
    for name in ('distributed/eplb/eplb_state.py','distributed/eplb/async_worker.py','distributed/eplb/rebalance_execute.py'):
        expected=EXPECTED_NATIVE_SOURCES.get(name)
        # The experiment source manifest supplies pins for additional native files.
        pins=json.loads(Path(os.environ['CS471_GATE_NATIVE_PINS']).read_text())
        expected=pins[name]
        if hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Native async source differs: '+name)
    W=int(os.environ.get('CS471_GATE_WINDOW','1000'))
    gain=float(os.environ.get('CS471_GATE_GAIN','.05'));stable=float(os.environ.get('CS471_GATE_TV','.05'))
    if W!=1000 or gain!=.05 or stable!=.05:raise RuntimeError('Unfrozen gate settings')
    State=state_module.EplbState
    original_add,original_step,original_rearrange=State.add_model,State.step,State.rearrange
    original_plan=async_worker.run_rebalance_experts
    original_move=state_module._move_to_workspace
    buffers={};step_count=0;pending_events=[]
    rank=None
    def emit(kind,**fields):
        nonlocal rank
        if rank is None:rank=get_ep_group().rank_in_group
        p=Path(os.environ['FULL_SERVING_RUN_DIR'])/f'gate-events-rank{rank}.jsonl'
        with p.open('a') as f:f.write(json.dumps({'kind':kind,'mode':mode,'rank':rank,'wall_ns':time.time_ns(),**fields},sort_keys=True)+'\n')
    def add(self,*args,**kwargs):
        out=original_add(self,*args,**kwargs)
        if not self.is_async:raise RuntimeError('Gate requires native async path')
        if self.expert_load_window_size!=W or self.expert_rearrangement_step_interval!=3000:raise RuntimeError('Native window/interval changed')
        for key,ms in self.model_states.items():
            if key in buffers:continue
            if (ms.model.num_moe_layers,ms.model.num_logical_experts,ms.model.num_physical_experts)!=(48,128,160):raise RuntimeError('Unexpected gate model geometry')
            buffers[key]={'ring':torch.zeros((2*W,48,129),dtype=torch.int64,device=ms.expert_load_pass.device),'seen':0}
        # Recording is paid by BOTH gates; B2 remains fully native.
        self._update_layer_should_record()
        emit('bound',window=W,native_interval=3000,ring_bytes=2*W*48*129*8,all_layer_logical_collection=True)
        return out
    def step(self,is_dummy=False,is_profile=False,log_stats=False):
        nonlocal step_count
        if not is_profile:
            step_count+=1
            # Counts were accumulated during this forward under the pre-commit map.
            for key,ms in self.model_states.items():
                b=buffers[key];row=b['ring'][b['seen']%(2*W)];row.zero_()
                if not is_dummy:
                    ids=ms.physical_to_logical_map.long().masked_fill(ms.physical_to_logical_map<0,128)
                    row.scatter_add_(1,ids,ms.expert_load_pass.to(torch.int64))
                b['seen']+=1
            ready=[]
            for event,record in pending_events:
                if event.query():emit('commit_completed',**record)
                else:ready.append((event,record))
            pending_events[:]=ready
        return original_step(self,is_dummy=is_dummy,is_profile=is_profile,log_stats=log_stats)
    def rearrange(self,is_profile=False,rank_mapping=None,use_last_expert_load=False):
        if not is_profile:
            if rank_mapping is not None or use_last_expert_load:raise RuntimeError('Elastic/remapped gate unsupported')
            tensors=[];states=[]
            for key,ms in self.model_states.items():
                b=buffers[key];seen=b['seen'];indices=(torch.arange(2*W,device=b['ring'].device)+seen)%(2*W)
                ordered=b['ring'].index_select(0,indices)
                tensors.extend([ordered[:W,:,:128].sum(0),ordered[W:,:,:128].sum(0)])
                states.append((ms,seen))
            tensors=self._allreduce_list(tensors)
            for i,(ms,seen) in enumerate(states):
                # Device snapshots precede rearrange_event.record and the async stream wait.
                ms.cs471_gate_windows=(tensors[2*i],tensors[2*i+1],seen,step_count)
        return original_rearrange(self,is_profile=is_profile,rank_mapping=rank_mapping,use_last_expert_load=use_last_expert_load)
    def plan(ms,self,current,stream):
        start=time.perf_counter_ns();candidate=original_plan(ms,self,current,stream)
        before,recent,seen,clock=ms.cs471_gate_windows
        with device_stream(stream):a=before.cpu().numpy();b=recent.cpu().numpy()
        old,new=current.numpy(),candidate.numpy();accepted=candidate.clone();decisions=[]
        for layer in range(48):
            d=decide(a[layer],b[layer],old[layer],new[layer],require_stability=mode=='b4',recent_complete=seen>=W,previous_complete=seen>=2*W,gain_threshold=gain,stability_threshold=stable)
            if not d['allow']:accepted[layer].copy_(current[layer])
            decisions.append(d)
        digest=hashlib.sha256(a.tobytes()+b.tobytes()+old.tobytes()+new.tobytes()+accepted.numpy().tobytes()).hexdigest()
        group=get_eplb_group().cpu_group;agreement=[None]*group.size()
        torch.distributed.all_gather_object(agreement,(clock,seen,digest),group=group)
        if len(set(agreement))!=1:raise RuntimeError('Gate rank/window/candidate disagreement')
        ms.cs471_gate_cycle=clock
        emit('decision',step=clock,seen_steps=seen,window=W,window_bounds={'previous':[seen-2*W,seen-W],'recent':[seen-W,seen]},rank_agreement=True,digest=digest,
             previous_counts=a.tolist(),recent_counts=b.tolist(),old_map=old.tolist(),native_candidate=new.tolist(),admitted_map=accepted.tolist(),layers=decisions,host_plan_and_gate_ns=time.perf_counter_ns()-start)
        return accepted
    def move(model_state,ep_rank):
        ms=model_state
        result=ms.pending_result;layer=result.layer_idx
        record={'cycle_step':getattr(ms,'cs471_gate_cycle',None),'layer':layer,'map_sha256':hashlib.sha256(result.new_physical_to_logical_map.numpy().tobytes()).hexdigest(),
                'remote_received_experts':int(result.transfer_metadata.recv_count),'unchanged_local_slots':int(result.transfer_metadata.is_unchanged.sum()),
                'remote_received_tensor_bytes':int(result.transfer_metadata.recv_count)*sum(w[0].numel()*w.element_size() for w in ms.model.expert_weights[layer])}
        original_move(ms,ep_rank)
        event=torch.cuda.Event();event.record();pending_events.append((event,record))
        emit('commit_enqueued',**record)
    State.add_model=add;State.step=step;State.rearrange=rearrange
    State._should_record_current_step=lambda self,log_stats=False:True
    async_worker.run_rebalance_experts=plan;state_module._move_to_workspace=move
    _INSTALLED=True
