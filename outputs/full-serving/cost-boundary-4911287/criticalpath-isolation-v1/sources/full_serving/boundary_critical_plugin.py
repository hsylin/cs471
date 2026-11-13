"""Source-pinned instrumentation and fixed-map management controls.

No activation communication is replaced. No synchronization is added to the
forward path except profiler start/stop itself; those boundaries are excluded
from attribution. CPU spans in the background worker are recorded separately
because PyTorch's thread-local profiler may not include them. Raw traces private.
"""
import contextlib,functools,inspect,json,os,socket,threading,time
from pathlib import Path
from full_serving.boundary_critical_contract import selected_window,digest,requested_window,agreed_window

_INSTALLED=False

class Capture:
    def __init__(self, root, rank, torch, group=None):
        self.root,self.rank,self.torch=root,rank,torch
        self.group=group
        self.ordinal=0;self.profile=None;self.active=None;self.used=set()
        self.plan=None;self.proposal_pending=False;self.rows=[];self.local=threading.local();self.anchor_id=0
        self.root.mkdir(exist_ok=True)
        self.log=root/f'critical-rank{rank}.jsonl'
        self.emit('bound',hostname=socket.gethostname(),pid=os.getpid(),
                  native_thread_id=threading.get_native_id(),diagnostic_only=True)
    def emit(self,kind,**fields):
        row=dict(kind=kind,rank=self.rank,wall_ns=time.time_ns(),mono_ns=time.perf_counter_ns(),
                 ordinal=self.ordinal,thread=threading.get_native_id(),**fields)
        # Append is atomic under CPython's GIL. File I/O is outside capture windows.
        if self.active is not None:self.rows.append(row)
        else:
            with self.log.open('a') as f:f.write(json.dumps(row)+'\n')
    def flush(self):
        rows,self.rows=self.rows,[]
        with self.log.open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    def load_plan(self):
        if self.plan is None:
            p=self.root/'capture-plan.json'
            if p.exists():self.plan=json.loads(p.read_text());self.emit('plan_loaded',plan=self.plan)
    def anchor(self):
        self.anchor_id+=1;name=f'cs471::clock_anchor::{self.anchor_id}'
        before=time.time_ns()
        with self.torch.profiler.record_function(name):middle=time.time_ns()
        after=time.time_ns()
        self.emit('clock_anchor',name=name,before_ns=before,middle_ns=middle,after_ns=after,
                  window=self.active['name'])
    def boundary(self,kind,**fields):
        # Durable boundary evidence survives a failed stop/export; outside sampled work.
        with (self.root/f'critical-boundary-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind,rank=self.rank,ordinal=self.ordinal,
                wall_ns=time.time_ns(),mono_ns=time.perf_counter_ns(),**fields))+'\n')
    def barrier(self):
        assert self.group is not None
        self.torch.distributed.barrier(group=self.group)
    def start(self,window):
        assert self.profile is None and self.ordinal==window['start_ordinal']
        self.boundary('start_agreed',window=window['name'],end_ordinal=window['end_ordinal'])
        self.barrier()  # all previous dispatches submitted before profiler's GPU sync
        self.used.add(window['name']);self.active=window
        self.emit('profiler_start_begin',window=window['name'])
        self.profile=self.torch.profiler.profile(
            activities=[self.torch.profiler.ProfilerActivity.CPU,self.torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,
            experimental_config=self.torch._C._profiler._ExperimentalConfig(
                enable_cuda_sync_events=True,profile_all_threads=True))
        self.profile.start();self.anchor()
        self.boundary('start_complete',window=window['name'])
        self.emit('profiler_start_end',window=window['name'])
    def stop(self):
        assert self.ordinal==self.active['end_ordinal']
        name=self.active['name'];self.boundary('stop_agreed',window=name)
        self.barrier()  # diagnostic boundary, no forward scopes include this barrier
        self.anchor();self.emit('profiler_stop_begin',window=name)
        self.profile.stop()
        path=self.root/f'trace-{name}-rank{self.rank}.json.gz'
        assert not path.exists();self.profile.export_chrome_trace(str(path))
        self.profile=None;self.active=None
        self.boundary('export_complete',window=name,trace=path.name,bytes=path.stat().st_size)
        self.emit('profiler_stop_end',window=name,trace=path.name,bytes=path.stat().st_size)
        self.flush();self.barrier()  # no rank launches next forward while a peer exports
    def tick(self):
        self.ordinal+=1
        if self.active is not None:
            if self.ordinal==self.active['end_ordinal']:
                self.stop()
                if getattr(self,'session_closed',False):return
            else:
                assert self.ordinal<self.active['end_ordinal']
                return  # no coordination collectives inside sampled iterations
        if self.ordinal%64:return
        self.load_plan()
        assert self.group is not None
        choice=requested_window(self.plan,time.time_ns(),self.used,self.proposal_pending) if self.rank==0 else 0
        command=self.torch.tensor([self.ordinal,-self.ordinal,choice],dtype=self.torch.int64)
        self.torch.distributed.all_reduce(command,op=self.torch.distributed.ReduceOp.MAX,group=self.group)
        hi,neglo,choice=command.tolist()
        assert hi==-neglo==self.ordinal,'Native DP ordinal disagreement'
        if choice:
            self.load_plan();assert self.plan is not None,'Capture plan unavailable on agreeing rank'
            window=agreed_window(self.plan,choice,self.ordinal)
            self.boundary('capture_command',window=window['name'],command=command.tolist())
            self.start(window)
    def proposal(self):
        # Called before the native snapshot. Selection happens only at next all-rank poll.
        if self.plan is not None and time.time_ns()>=self.plan['epoch_wall_ns']:
            self.proposal_pending=True
    @contextlib.contextmanager
    def span(self,name,**fields):
        w=self.active
        if w is None:
            yield
            return
        a=time.time_ns();m=time.perf_counter_ns();ordinal=self.ordinal
        with self.torch.profiler.record_function('cs471::'+name):
            try:yield
            finally:
                self.emit('host_span',name=name,start_wall_ns=a,end_wall_ns=time.time_ns(),
                          duration_ns=time.perf_counter_ns()-m,start_ordinal=ordinal,
                          window=w['name'],end_outside_window=self.active is not w,**fields)

def register():
    global _INSTALLED
    if _INSTALLED:return
    assert os.environ.get('CS471_CRITICAL_DIAGNOSTIC')=='1'
    control=os.environ['CS471_CRITICAL_CONTROL']
    assert control in ('off','b4','fixed_idle','fixed_charged')
    import torch,vllm
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb import eplb_state as state,async_worker as worker,eplb_utils
    from vllm.v1.worker.gpu import model_runner as runner_module
    from full_serving.resident_policy import balanced_map
    pins=json.loads(Path(os.environ['CS471_CRITICAL_PINS']).read_text())
    for name,value in pins.items():assert digest(Path(vllm.__file__).parent/name)==value,name
    root=Path(os.environ['FULL_SERVING_RUN_DIR']);holder={}
    def cap():
        if 'capture' not in holder:
            ep=get_ep_group()
            assert len(ep.ranks)==8 and torch.distributed.get_world_size()==8
            # Separate communicator: never interleave diagnostic coordination with native worker/readiness collectives.
            group=torch.distributed.new_group(ranks=ep.ranks,backend='gloo')
            from full_serving.boundary_gated_capture import gated_capture_class
            holder['capture']=gated_capture_class(Capture)(root,ep.rank_in_group,torch,group=group)
            from full_serving.boundary_cuda_event_links import install as install_event_links
            c=holder['capture']
            def event_link(fields):
                fields=dict(fields);fields.pop('thread',None)
                fields['end_outside_window']=c.active is None or c.active['name']!=fields['window']
                c.emit('cuda_event_link',**fields)
            _,c.event_ledger=install_event_links(torch,lambda:c.active['name'] if c.active is not None else None,event_link)
            holder['capture'].boundary('diagnostic_group_created',ranks=ep.ranks,backend='gloo')
            if holder['capture'].rank in (0,4):
                def clock_server():
                    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                        sock.bind(('0.0.0.0',19471))
                        while True:
                            value,address=sock.recvfrom(256)
                            if value.startswith(b'cs471-clock:'):
                                sock.sendto(json.dumps(dict(nonce=value.decode(),wall_ns=time.time_ns(),
                                    mono_ns=time.perf_counter_ns(),rank=holder['capture'].rank)).encode(),address)
                threading.Thread(target=clock_server,daemon=True).start()
        return holder['capture']
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from full_serving.boundary_output_wait_isolation import install as install_output_wait
    install_output_wait(AsyncOutput,cap)
    if control=='fixed_idle':
        os.environ.update(CS471_EXPERIMENT_ARM='balanced32_idle',CS471_COST_PINS=os.environ['CS471_GATE_NATIVE_PINS'],CS471_COST_DIAGNOSTIC='0')
        from full_serving.cost_plugin import register as base
        base()
    elif control in ('b4','fixed_charged'):
        if control=='fixed_charged':
            def initial(logical,redundant):
                assert (logical,redundant)==(128,32)
                return balanced_map()
            state.EplbState.build_initial_global_physical_to_logical_map=staticmethod(initial)
        os.environ.update(CS471_SUSTAINED_ARM='b4',CS471_GATE_MODE='b4',CS471_GATE_HISTORY='reduction')
        from full_serving.sustained_plugin import register as base
        base()
        if control=='fixed_charged':
            old_plan=worker.run_rebalance_experts
            def unchanged(ms,ep,current,stream):
                proposed=old_plan(ms,ep,current,stream)
                assert current.tolist()==[balanced_map()]*48
                cap().emit('diagnostic_proposal_override',cycle=getattr(ms,'cs471_gate_cycle',None),
                    proposed_changed_slots=int((proposed!=current).sum()),actual_changed_slots=0,
                    actual_admitted_sha256=__import__('hashlib').sha256(current.numpy().tobytes()).hexdigest(),
                    meaning='Earlier B4 decision/proposal logs describe computed candidate, NOT executed diagnostic map.')
                return current.clone()
            worker.run_rebalance_experts=unchanged
    if control!='off':
        old_add=state.EplbState.add_model
        def add(self,*args,**kwargs):
            out=old_add(self,*args,**kwargs);c=cap()
            for ms in self.model_states.values():
                actual=ms.physical_to_logical_map.cpu().tolist()
                if control.startswith('fixed_'):assert actual==[balanced_map()]*48
                c.emit('map_bound',mapping=actual,control=control,native_routing_kernel=True)
                if os.environ.get('CS471_CRITICAL_PILOT')=='1' and control.startswith('fixed_'):
                    from full_serving.cost_probe import check_native_mapping
                    c.emit('routing_oracle',result=check_native_mapping(ms))
                    from full_serving.resident_diagnostic import tensor_sha,validate_weight_records
                    records=[]
                    for layer,tensors in enumerate(ms.model.expert_weights):
                        by_size={w.shape[1]:w for w in tensors}
                        assert set(by_size)=={1536*2048,2048*768}
                        for slot in range(20):
                            physical=c.rank*20+slot
                            records.append(dict(layer_index=layer,physical_slot=physical,logical_expert=balanced_map()[physical],
                                w13_sha256=tensor_sha(by_size[1536*2048][slot]),w2_sha256=tensor_sha(by_size[2048*768][slot])))
                    peers=[None]*8
                    torch.distributed.all_gather_object(peers,dict(rank=c.rank,weights=records),group=get_ep_group().cpu_group)
                    c.emit('full_weight_admission',result=validate_weight_records(peers,balanced_map()),weights=records)
            return out
        state.EplbState.add_model=add
        old_commit=state._commit_eplb_maps_for_layer
        def commit(ms,new_physical_to_logical_map,layer):
            if control.startswith('fixed_'):
                assert new_physical_to_logical_map.is_cpu
                assert new_physical_to_logical_map.tolist()==balanced_map()
            return old_commit(ms,new_physical_to_logical_map,layer)
        state._commit_eplb_maps_for_layer=commit
    def wrap(obj,name,label):
        old=getattr(obj,name)
        @functools.wraps(old)
        def call(*args,**kwargs):
            c=cap()
            with c.span(label):return old(*args,**kwargs)
        setattr(obj,name,call)
    old_ready=state.EplbState._all_ranks_result_ready
    def ready(self,model_state):
        c=cap();pending=model_state.pending_result
        with c.span('EPLB._all_ranks_result_ready',local_pending_at_entry=pending is not None,
                    pending_layer_at_entry=pending.layer_idx if pending is not None else None):
            value=old_ready(self,model_state)
        if c.active is not None:c.emit('readiness_result',window=c.active['name'],all_ready=bool(value),
            local_pending_at_entry=pending is not None,pending_layer_at_entry=pending.layer_idx if pending is not None else None)
        return value
    state.EplbState._all_ranks_result_ready=ready
    for name in ('_allreduce_list','prepare_forward','rearrange'):
        wrap(state.EplbState,name,'EPLB.'+name)
    for name in ('_move_to_workspace','_commit_eplb_maps_for_layer'):
        wrap(state,name,'EPLB.'+name)
    for name in ('run_rebalance_experts','transfer_layer'):
        wrap(worker,name,'worker.'+name)
    for name in ('wait','record'):
        old=getattr(eplb_utils.CpuGpuEvent,name)
        def factory(old,name):
            @functools.wraps(old)
            def call(self,*a,**kw):
                with cap().span('event.'+name,event_id=id(self)):return old(self,*a,**kw)
            return call
        setattr(eplb_utils.CpuGpuEvent,name,factory(old,name))
    old_step=state.EplbState.step
    def step(self,is_dummy=False,is_profile=False,log_stats=False):
        if is_profile:return old_step(self,is_dummy,is_profile,log_stats)
        c=cap()
        if self.expert_rearrangement_step+128>=self.expert_rearrangement_step_interval:c.proposal()
        with c.span('EPLB.step',native_step=self.expert_rearrangement_step):
            return old_step(self,is_dummy,is_profile,log_stats)
    state.EplbState.step=step
    Runner=runner_module.GPUModelRunner
    old_execute=Runner.execute_model;sig=inspect.signature(old_execute)
    def execute(self,scheduler_output,*args,**kwargs):
        bound=sig.bind(self,scheduler_output,*args,**kwargs);bound.apply_defaults()
        dummy=bound.arguments['dummy_run'];profile=bound.arguments['is_profile']
        if profile:return old_execute(self,scheduler_output,*args,**kwargs)
        # The pinned native source returns before dispatch in this exact case.
        if not dummy and scheduler_output.total_num_scheduled_tokens==0:
            return old_execute(self,scheduler_output,*args,**kwargs)
        c=cap();c.tick()
        with c.span('execute',dummy=bool(dummy)):
            if c.active is not None:
                c.emit('iteration',window=c.active['name'],dummy=bool(dummy),
                       zero_actual_work=bool(dummy),profile=False)
            return old_execute(self,scheduler_output,*args,**kwargs)
    Runner.execute_model=execute
    old_prepare=Runner.prepare_inputs
    def prepare(self,*args,**kwargs):
        c=cap()
        with c.span('prepare_inputs'):batch=old_prepare(self,*args,**kwargs)
        if c.active is not None:
            from full_serving.boundary_shapes import summarize_batch
            c.emit('actual_batch',window=c.active['name'],**summarize_batch(batch))
        return batch
    Runner.prepare_inputs=prepare
    old_dispatch=runner_module.dispatch_cg_and_sync_dp
    def dispatch(*args,**kwargs):
        c=cap()
        with c.span('DP.dispatch'):out=old_dispatch(*args,**kwargs)
        if c.active is not None:
            desc,sync=out
            c.emit('dispatch',window=c.active['name'],cg_mode=str(desc.cg_mode),padded_tokens=int(desc.num_tokens))
        return out
    runner_module.dispatch_cg_and_sync_dp=dispatch
    for name in ('prepare_attn','prepare_dummy_attn'):
        wrap(Runner,name,name)
    from full_serving.logical_history_gpu import LogicalHistory
    for name in ('update','snapshot'):wrap(LogicalHistory,name,'history.'+name)
    _INSTALLED=True
