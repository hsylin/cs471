"""Diagnostic interventions: fixed placement, recording toggle, native communication.

These are attribution controls, not proposed deployable EPLB replacements.
Never suppress activation communication or return fabricated model outputs.
"""
import functools
import hashlib
import json
import os
from pathlib import Path

_INSTALLED=False


def register():
    global _INSTALLED
    if _INSTALLED:return
    arm=os.environ['CS471_EXPERIMENT_ARM']
    diagnostic=os.environ.get('CS471_COST_DIAGNOSTIC')=='1'
    import torch
    import vllm
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.v1.worker.gpu.eplb_utils import EPLBController
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from full_serving.resident_policy import balanced_map
    pins=json.loads(Path(os.environ['CS471_COST_PINS']).read_text())
    for name,digest in pins.items():
        assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==digest,name
    root=Path(os.environ['FULL_SERVING_RUN_DIR'])
    fixed=arm.startswith(('fixed','balanced'))
    if arm=='resident_v2':
        from full_serving.proposal_plugin import register as base
        base()
    elif arm=='b4_light':
        from full_serving.monitoring_plugin import register as base
        base()
    elif arm!='off' and not fixed:raise RuntimeError('Unknown attribution arm')

    def emit(kind,**kw):
        rank=get_ep_group().rank_in_group
        with (root/f'cost-events-rank{rank}.jsonl').open('a') as f:
            f.write(json.dumps({'kind':kind,'rank':rank,'arm':arm,**kw},sort_keys=True)+'\n')

    if fixed:
        replicas=0 if arm.startswith('fixed0') else 32
        mapping=list(range(128))+list(range(replicas))
        if arm.startswith('balanced32_'):mapping=balanced_map()
        original_initial=EplbState.build_initial_global_physical_to_logical_map
        def initial(logical,redundant):
            assert (logical,redundant)==(128,replicas)
            return list(mapping)
        EplbState.build_initial_global_physical_to_logical_map=staticmethod(initial)
        recording=arm.endswith('_record')
        EplbState._should_record_current_step=lambda self,log_stats=False:recording
        # Keep the native map/record kernel, prepare_forward and step bookkeeping.
        # Both matched arms suppress planning/transfer; neither executes a commit.
        def no_rearrange(self,is_profile=False,rank_mapping=None,use_last_expert_load=False):
            assert rank_mapping is None and not use_last_expert_load
            if not is_profile:
                for ms in self.model_states.values():
                    assert not ms.rebalanced
                emit('suppressed_proposal',recording=recording)
            return None
        EplbState.rearrange=no_rearrange
        EPLBController.maybe_start_async_loop=lambda self,eplb_models_added:None
        original_add=EplbState.add_model
        def add(self,model,model_config):
            out=original_add(self,model,model_config)
            assert (model.num_moe_layers,model.num_logical_experts,model.num_physical_experts)==(48,128,128+replicas)
            self._update_layer_should_record()
            ms=next(iter(self.model_states.values()))
            actual=ms.physical_to_logical_map.detach().cpu().tolist()
            assert actual==[mapping]*48
            assert bool(self.should_record_tensor.item())==recording
            emit('fixed_bound',replicas=replicas,recording=recording,physical_to_logical_map=actual,
                 native_map_record_kernel=True,native_step_bookkeeping=True,
                 native_prepare_forward=True,async_worker_started=False,migration_enabled=False)
            if diagnostic:
                from full_serving.cost_probe import check_native_mapping
                emit('map_record_oracle',**check_native_mapping(ms))
            return out
        EplbState.add_model=add

    if diagnostic:
        # Shape catalogue includes graph construction and eager calls. It is NOT
        # an execution-frequency counter: graph replay bypasses Python hooks.
        seen=set()
        original_dispatch=AgRsAll2AllManager.dispatch
        @functools.wraps(original_dispatch)
        def dispatch(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel=False,extra_tensors=None):
            group=self._get_comm_group(is_sequence_parallel)
            sizes=self._get_sizes(hidden_states.shape[0],group)
            record={'sizes':list(sizes),'hidden_shape':list(hidden_states.shape),'hidden_dtype':str(hidden_states.dtype),
                    'weights_shape':list(topk_weights.shape),'weights_dtype':str(topk_weights.dtype),
                    'ids_shape':list(topk_ids.shape),'ids_dtype':str(topk_ids.dtype),
                    'extra_shapes':[list(t.shape) for t in (extra_tensors or [])]}
            key=json.dumps(record,sort_keys=True)
            if key not in seen:
                seen.add(key);emit('activation_collective_shape',**record)
            with torch.profiler.record_function('cs471::native_AgRs_dispatch'):
                return original_dispatch(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel,extra_tensors)
        AgRsAll2AllManager.dispatch=dispatch
        original_combine=AgRsAll2AllManager.combine
        @functools.wraps(original_combine)
        def combine(self,*a,**kw):
            with torch.profiler.record_function('cs471::native_AgRs_combine'):
                return original_combine(self,*a,**kw)
        AgRsAll2AllManager.combine=combine
        original_step=EplbState.step
        @functools.wraps(original_step)
        def step(self,*a,**kw):
            with torch.profiler.record_function('cs471::EPLB_step'):
                return original_step(self,*a,**kw)
        EplbState.step=step
    _INSTALLED=True
