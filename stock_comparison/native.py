"""Actual pinned Qwen3 sparse block, with explicit per-arm control policies.

GPU imports are lazy. Stock arms use the original router, expert method, default
policy and movement/commit implementations. The facade supplies an eager forward
context, native per-step EPLB calls, finite-input completion, and instrumentation.
Only ``ours`` replaces the physical-mapping instance hook and admission policy.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import numpy as np

PIN = 'db9527a46873454610df6dbedf79a36d6bf1a7f6'
MODEL_REVISION = 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
ARMS = ('stockoff', 'stockon32', 'ours')
SOURCE_SHA256 = {'config/parallel.py': 'eee9418488deb7945562c3127106109bac632471a2c1a08a58169a48cd15684c', 'distributed/device_communicators/cuda_communicator.py': '2a0695d8b46757be83b38fe657df605ba0bfb2ef2dfa690c78c57419bbad0762', 'distributed/eplb/async_worker.py': 'b458e8252f3147982aa71670deae8264587f3ab01e552030641c2f9d8b6dbc43', 'distributed/eplb/eplb_communicator.py': 'ed21ba6a1ab4fa31deae36b5cf54875e78dc16937e503be5a5399e46a3be1b79', 'distributed/eplb/eplb_state.py': 'eaac132acd438074343e5ee56c52e1df46f37a9bf7a303bd3b9373c69b513f17', 'distributed/eplb/policy/default.py': '4b77c5b6075694b882d3f53ca6fa93aa9b5beb823920a69bb9b7333c7ef27270', 'distributed/eplb/rebalance_execute.py': '686eb68f425e7c1450abea9466178d430e95441bd74ea6aaed12c13090654431', 'distributed/nixl_utils.py': '0bffa3c052c9f67472b1c0bc3816114891b010c968bf4b8a292e7da4d687db53', 'distributed/parallel_state.py': 'a33fc846e0f4e682a644ce712d862806ecfa12dca6e5f641806e2e7991cd0087', 'forward_context.py': '8dab5a4dee60e46c4e52821667321c9f05c793578af49cddbdb204cf9324c403', 'model_executor/layers/fused_moe/config.py': '034e5c705d81a37b78911162d3e80e4c3dbbfe57b651f040694942c9e12f44c4', 'model_executor/layers/fused_moe/expert_map_manager.py': 'e39f2fde2a2e23699d09259912fac1c5f37252fd1a9f7ba87aaf599b59ffc2e9', 'model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py': '82e1784f277df2d4e079b61c3af68fa11375e0f71f6cd9a7a95111c885e75963', 'model_executor/layers/fused_moe/experts/triton_moe.py': 'ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab', 'model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py': '9e0aff06802203256093494b578a5aba22a35221440da8ea9d9938d1aa26feb1', 'model_executor/layers/fused_moe/fused_moe.py': '21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7', 'model_executor/layers/fused_moe/layer.py': '0c30a9a56ff22c63a1498f61c72a349e4dba0d1e4fbe42efb333dda45d4bdecf', 'model_executor/layers/fused_moe/modular_kernel.py': 'b8dce8f9af0545c080d8d761bd12a60b607ed84eafbaf89a6f8020040df7d6ce', 'model_executor/layers/fused_moe/oracle/unquantized.py': '64d7731a04872a7365e55112b0b9fa03f3886f0cd418733d8cdb40d80fe4bf43', 'model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py': '263c30ea61e756d389dfbb25927feb3c0fde5936001f1f00215bd15913af0b98', 'model_executor/layers/fused_moe/routed_experts.py': '78bff37911f17cb4cf77a09861dd11fc9bdbf59e7dadb422b4f3fe9e9d526812', 'model_executor/layers/fused_moe/router/base_router.py': 'c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf', 'model_executor/layers/fused_moe/router/custom_routing_router.py': '1a40c7054e3c3a1b3ee2e98f6bc9efa9dfdc11f316a8edf7357dd91533487bfe', 'model_executor/layers/fused_moe/router/fused_moe_router.py': 'd386fce7b5f61abda2d4b2a3a0bc324db53129b5c5158a1a0cbb6fd4ed6b0018', 'model_executor/layers/fused_moe/router/fused_topk_bias_router.py': 'b454bbe4ed7a6ae39a86715077bfb4f022946e25dbdd411436e90c90d9cdec7d', 'model_executor/layers/fused_moe/router/fused_topk_router.py': '1c80f4af4d781fd66b12aa6a7a0fd30082bc521df40f1252987a8ac4c0bef51c', 'model_executor/layers/fused_moe/router/gate_linear.py': '19191d160cb48d79508eb822db795ffe638268f2c585eaffda05966e47b78252', 'model_executor/layers/fused_moe/router/router_factory.py': '794abb7c210758911d349ca6fbd82690f0ef48ba4f8a7ec2ff0f87ca3ad50224', 'model_executor/layers/fused_moe/runner/moe_runner.py': '2c6e1fffba1f4fd2e4c5d3981c4d26f7c0de802fc9c35865748b113c30235da5', 'model_executor/layers/fused_moe/runner/moe_runner_interface.py': '0a52ccb21915d39155f00cffc397630952c5927f00046068421c29eb7a4261c0', 'model_executor/layers/fused_moe/unquantized_fused_moe_method.py': 'dade56ddeef4967aa3ad3448ef46d964e6130a6f3d558e4a77ccce62303816a9', 'model_executor/models/interfaces.py': 'c34635d28191061a46d8c57d178c242dc7bc1993eda23e6a31a72eb1a3759fec', 'model_executor/models/qwen3_moe.py': 'cac0cffc3e5b1ff805e9e15fe1d4bcfd5e55c27378e4a9ab23b0fa6c1b05e5ee', 'v1/worker/workspace.py': '07bcaea79e568a48be9c2415163b31e6605f5670f776d54d5a7cca47bddb57b2'}

SOURCE_SHA256['model_executor/layers/fused_moe/__init__.py'] = '64a32665be6d5afcfbaf7a0915717267278b66b880b908b0263ea58f08e51b5e'


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown stock comparator arm')
    on = arm != 'stockoff'
    return dict(arm=arm, enable_eplb=on, replicas=32 if on else 0,
        physical_slots=160 if on else 128, local_slots=80 if on else 64,
        window_steps=16 if arm == 'ours' else 1000,
        rebalance_interval=64 if arm == 'ours' else 3000,
        initial_counter=0 if arm == 'ours' else 2250 if on else None,
        is_async=arm == 'stockon32')


def _hash_array(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _class_name(value):
    cls = type(value)
    return cls.__module__ + '.' + cls.__name__


def verify_sources(package):
    if not SOURCE_SHA256:
        raise RuntimeError('Stock source inventory missing')
    actual = {name: hashlib.sha256((Path(package) / name).read_bytes()).hexdigest()
              for name in SOURCE_SHA256}
    bad = [name for name in actual if actual[name] != SOURCE_SHA256[name]]
    if bad:
        raise ValueError('Installed stock sources changed: ' + ', '.join(bad))
    return dict(commit=PIN, verified_source_files=len(actual), sha256=actual)


class AsyncPayloadCounter:
    """Transparent sender accounting keyed by the native immutable cycle object.

    A commit and next launch can occur in one main-thread step. Global counter
    differences then race the worker's next sends; EplbStats identity does not.
    Retained references prevent Python object-ID reuse during a trial.
    """
    def __init__(self, inner, model_state):
        self.inner, self.model_state = inner, model_state
        self.reset_counts()

    def __getattr__(self, name):
        return getattr(self.inner,name)

    def reset_counts(self):
        self.sent_bytes = self.sent_messages = 0
        self.cycles = {}

    def add_send(self, tensors, dst, **kwargs):
        tensors = tuple(tensors)
        count = sum(t.numel()*t.element_size() for t in tensors)
        stats = self.model_state.eplb_stats
        if stats is None:
            raise AssertionError('Async transfer has no native cycle identity')
        entry = self.cycles.setdefault(id(stats),dict(stats=stats,bytes=0,messages=0))
        entry['bytes'] += count
        entry['messages'] += 1
        self.sent_bytes += count
        self.sent_messages += 1
        return self.inner.add_send(tensors,dst,**kwargs)

    def cycle_bytes(self, key):
        return self.cycles.get(key,{}).get('bytes',0)


class StockRuntime:
    """One process-group initialization; one reusable layer instance per arm."""
    def __init__(self, model_path, *, cache_dir=None):
        from skewbench.native import initialization_progress
        initialization_progress('stock_import_torch')
        import torch
        import vllm
        from vllm.config import ModelConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment, ensure_model_parallel_initialized,
            get_ep_group, get_eplb_group, get_tp_group)
        from vllm.v1.worker.workspace import init_workspace_manager
        self.torch = torch
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        if self.world_size != 2 or not torch.cuda.is_available():
            raise RuntimeError('Stock comparison requires two CUDA ranks')
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        self.model_path = Path(model_path)
        if not (self.model_path / 'config.json').is_file():
            raise ValueError('model_path must be the exact local Qwen snapshot directory')
        hf = json.loads((self.model_path / 'config.json').read_text())
        for key, expected in dict(num_experts=128, num_experts_per_tok=8,
                                  hidden_size=2048, moe_intermediate_size=768).items():
            if hf.get(key) != expected:
                raise ValueError('Unsupported model geometry: ' + key)
        self.source_attestation = verify_sources(Path(vllm.__file__).parent)
        initialization_progress('stock_sources_verified')
        self.model_config = ModelConfig(model=str(self.model_path),
            tokenizer=str(self.model_path), dtype='bfloat16', max_model_len=8192,
            enforce_eager=True, skip_tokenizer_init=True)
        self.config = self.make_config('stockon32')
        # Bootstrap includes an EPLB group. Off does not use it; all arms share
        # identical TP/EP groups and unmodified native output reduction.
        self._context = set_current_vllm_config(self.config)
        self._context.__enter__()
        initialization_progress('stock_initialize_distributed')
        init_distributed_environment(backend='nccl')
        ensure_model_parallel_initialized(tensor_model_parallel_size=2,
                                          pipeline_model_parallel_size=1)
        self.ep_group, self.eplb_group, self.tp_group = get_ep_group(), get_eplb_group(), get_tp_group()
        init_workspace_manager(self.device)
        self.layers = {}
        self._closed = False
        self.metadata = dict(native_vllm_commit=PIN, native_source_attestation=self.source_attestation,
            world_size=2, dtype='bfloat16', gpu_name=torch.cuda.get_device_name(),
            torch_version=torch.__version__, vllm_version=vllm.__version__, cuda_version=torch.version.cuda,
            torch_num_threads=torch.get_num_threads(),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            dispatch_backend='stock_tp2_ep_replicated_input_native_reduction',
            output_collective='unmodified_tensor_model_parallel_all_reduce',
            native_output_reduction=True, router_in_timing=True, execution_mode='eager_one_layer',
            model_path=str(self.model_path), model_revision=MODEL_REVISION)
        initialization_progress('stock_runtime_ready')

    def make_config(self, arm):
        from vllm.config import VllmConfig, ParallelConfig
        from vllm.config.parallel import EPLBConfig
        spec = arm_spec(arm)
        extra = dict(num_redundant_experts=spec['replicas'])
        if arm == 'ours':
            extra.update(window_size=16, step_interval=64, use_async=False, communicator='torch_nccl')
        pc = ParallelConfig(tensor_parallel_size=2, enable_expert_parallel=True,
            enable_eplb=spec['enable_eplb'], eplb_config=EPLBConfig(**extra),
            distributed_executor_backend='external_launcher')
        cfg = VllmConfig(model_config=self.model_config, parallel_config=pc)
        if cfg.parallel_config.disable_custom_all_reduce or cfg.parallel_config.use_sequence_parallel_moe:
            raise ValueError('Unexpected change to common stock reduction/parallel configuration')
        return cfg

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def barrier(self):
        self.torch.distributed.barrier(group=self.ep_group.device_group)

    def gather_scalars(self, value):
        t = self.torch.tensor([value], device=self.device, dtype=self.torch.float64)
        out = [self.torch.empty_like(t) for _ in range(2)]
        self.torch.distributed.all_gather(out, t, group=self.ep_group.device_group)
        return [float(x.item()) for x in out]

    def gather_objects(self, value):
        out = [None, None]
        self.torch.distributed.all_gather_object(out, value, group=self.ep_group.cpu_group)
        return out

    def close(self):
        if self._closed:
            return
        for layer in self.layers.values():
            layer.complete_pending()
        self.sync()
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
        self._context.__exit__(None, None, None)
        self._closed = True


class StockLayer:
    def __init__(self, rt, weights_or_directory, arm, *, gate_weight=None, batch_tokens=128):
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
        from vllm.distributed.eplb.eplb_state import EplbState
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
        from vllm.model_executor.models.interfaces import MixtureOfExperts
        from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
        from skewbench.native import Weights, CountingCommunicator, NativeLayer, initial_mapping, initialization_progress
        spec = arm_spec(arm)
        if arm in rt.layers:
            raise ValueError('Reuse the existing arm layer; do not create duplicate async workers')
        if batch_tokens not in (128, 1024):
            raise ValueError('Declared batch sizes are128/1024')
        self.rt, self.torch, self.device = rt, rt.torch, rt.device
        self.arm, self.spec, self.batch_tokens = arm, spec, batch_tokens
        self.weights = Weights(weights_or_directory) if isinstance(weights_or_directory, (str, Path)) else weights_or_directory
        if (self.weights.config.get('source_revision') != MODEL_REVISION
                or self.weights.config.get('layer') != 0):
            raise ValueError('Wrong checkpoint export identity')
        self.dtype = self.torch.bfloat16
        self.num_experts, self.hidden_size, self.intermediate_size, self.top_k = 128, 2048, 768, 8
        self.replicas, self.slots = spec['replicas'], spec['local_slots']
        self.initial_map = initial_mapping(128, self.replicas, 2)
        self.config = rt.make_config(arm)
        self.identity = self.config.model_config
        self.window_steps, self.rebalance_interval = spec['window_steps'], spec['rebalance_interval']
        self.state = self.ms = self.layer_state = self.counter = self.mapper = self.guard = None
        self._active = False
        initialization_progress('stock_construct_' + arm)
        previous_dtype = self.torch.get_default_dtype()
        try:
            self.torch.set_default_dtype(self.dtype)
            with set_current_vllm_config(self.config), self.torch.device(self.device):
                self.block = Qwen3MoeSparseMoeBlock(self.config, prefix='model.layers.0.mlp')
        finally:
            self.torch.set_default_dtype(previous_dtype)
        self.block.eval()
        self.runner = self.block.experts
        self.routed = self.runner.routed_experts
        self.router = self.runner.router
        method = self.routed.quant_method
        if method.unquantized_backend.name != 'TRITON':
            raise ValueError('Reviewed checkpoint/map validation requires the actual auto-selected Triton backend')
        if self.runner.moe_config.moe_parallel_config.use_all2all_kernels:
            raise ValueError('Unexpected all-to-all backend in replicated-input experiment')
        self.w13, self.w2 = self.routed.w13_weight, self.routed.w2_weight
        self.expert_map = self.routed.expert_map
        if tuple(self.w13.shape) != (self.slots,1536,2048) or tuple(self.w2.shape) != (self.slots,2048,768):
            raise ValueError('Actual stock weights have unexpected shapes')
        if gate_weight is None:
            from safetensors import safe_open
            with safe_open(str(rt.model_path / 'model-00001-of-00016.safetensors'), framework='pt', device='cpu') as f:
                gate_weight = f.get_tensor('model.layers.0.mlp.gate.weight')
        if gate_weight.dtype != self.dtype or tuple(gate_weight.shape) != (128,2048):
            raise ValueError('Gate must be original BF16[128,2048] checkpoint tensor')
        self._gate_source = gate_weight.detach().cpu().clone()
        self.gate_sha256 = _hash_array(self._gate_source.view(self.torch.uint16).numpy(), '<u2')
        with self.torch.no_grad():
            self.block.gate.weight.copy_(self._gate_source)
            self._reload_experts(self.initial_map)
            with set_current_vllm_config(self.config):
                method.process_weights_after_loading(self.routed)
        self.w13, self.w2 = self.routed.w13_weight, self.routed.w2_weight
        self.mapping_kernel = eplb_map_to_physical_and_record
        self.default_policy = DefaultEplbPolicy
        if spec['enable_eplb']:
            owner = SimpleNamespace(num_moe_layers=1, num_expert_groups=1, num_logical_experts=128,
                num_routed_experts=128, num_shared_experts=0, num_physical_experts=160,
                num_redundant_experts=32, num_local_physical_experts=80,
                moe_layers=[self.runner], expert_weights=[])
            owner.set_eplb_state = lambda load, logical, counts: MixtureOfExperts.set_eplb_state(owner,load,logical,counts)
            self.state = EplbState(self.config.parallel_config, self.device)
            self.state.add_model(owner, self.identity)
            self.ms = self.state.model_states[self.identity.compute_hash()]
            self.layer_state = self.runner.eplb_state
            self.counter = (AsyncPayloadCounter(self.ms.communicator,self.ms) if arm == 'stockon32'
                            else CountingCommunicator(self.ms.communicator))
            self.ms.communicator = self.counter
        self.method = 'SKIP'
        self.epsilon = self.tau = .1
        self.cooldown_steps = 128
        self.collect_history = False
        self.logical_ring = self.torch.zeros((32,128), device=self.device, dtype=self.torch.int64)
        self.logical_ring_timestamps = np.full(32,-1,np.int64)
        self.capture_decision_step = self.captured_candidate = None
        if arm == 'ours':
            from resident_dispatch.online import OnlineMapper
            self.mapper = OnlineMapper(self.torch, self.initial_map, batch_tokens)
            self.mapper.verify_metadata(self.layer_state.logical_to_physical_map,
                self.layer_state.logical_replica_count, self.ms.physical_to_logical_map[0])
            self.mapping_kernel = self.mapper
            owner = self
            def apply_mapping(router, ids):
                ls = router.eplb_state
                return owner.mapper(ids, ls.expert_load_view, ls.logical_to_physical_map,
                    ls.logical_replica_count, ls.should_record_tensor, ls.num_unpadded_tokens_tensors[0])
            self.router._apply_eplb_mapping = MethodType(apply_mapping, self.router)
            NativeLayer._install_policy(self)
        self.reset_trial()
        if arm == 'stockon32':
            self.state.start_async_loop()
        rt.layers[arm] = self
        self.assert_maps_and_weights()
        initialization_progress('stock_ready_' + arm)

    def _reload_experts(self, mapping):
        ids = np.asarray(mapping)[self.rt.rank*self.slots:(self.rt.rank+1)*self.slots]
        for key, actual in [('w13_weight',self.w13),('w2_weight',self.w2)]:
            actual.copy_(self.weights.tensors[key][ids].to(self.device,self.dtype))

    @contextmanager
    def measured(self):
        """Scope ours' shortcut only; stock algorithms have no such adapter."""
        from arrival_fix.noop import unchanged_map_fast_path
        if self._active:
            raise RuntimeError('Nested layer execution scopes are unsupported')
        context = unchanged_map_fast_path(self) if self.arm == 'ours' else nullcontext(None)
        self._active = True
        try:
            with context as guard:
                self.guard = guard
                yield self
        finally:
            self._active = False

    activation_context = measured
    active = measured

    def forward(self, hidden, *, observe=True):
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        if tuple(hidden.shape) != (self.batch_tokens,2048) or hidden.dtype != self.dtype:
            raise ValueError('Stock forward requires declared BF16 hidden batch')
        saved = None
        if self.state is not None:
            self.state.prepare_forward(self.identity, len(hidden))
            if not observe:
                saved = self.state.should_record_tensor.clone()
                self.state.should_record_tensor.fill_(False)
        try:
            with self.torch.no_grad(), set_current_vllm_config(self.config), set_forward_context(None,self.config,num_tokens=len(hidden)):
                output = self.block(hidden)
        finally:
            if saved is not None:
                self.state.should_record_tensor.copy_(saved)
        if observe:
            self.observation_steps += 1
        return output

    def routing(self, hidden):
        """Untimed actual stock gate and logical top-k; never inject HF routes."""
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        with self.torch.no_grad(), set_current_vllm_config(self.config), set_forward_context(None,self.config,num_tokens=len(hidden)):
            logits, _ = self.block.gate(hidden)
            weights, ids = self.router._compute_routing(hidden,logits,
                self.routed.quant_method.topk_indices_dtype)
        return ids, weights

    def _physical_observation_snapshot(self):
        from skewbench.native import NativeLayer
        return NativeLayer._physical_observation_snapshot(self)

    def step(self, step_index):
        if not self._active:
            raise RuntimeError('Use layer.measured() around forwards plus native step calls')
        self.step_index = int(step_index)
        self.measured_step_calls += 1
        if self.state is None:
            return
        sampled = self.state._should_record_current_step(log_stats=False)
        if sampled:
            self.recorded_steps.append(int(step_index))
        if self.arm == 'ours':
            from skewbench.native import NativeLayer
            NativeLayer.step(self,step_index)
            return
        before_counter = self.state.expert_rearrangement_step
        before_bytes = self.counter.sent_bytes
        before_stats, before_pending = self.ms.eplb_stats, self.ms.rebalanced
        if sampled:
            self.native_valid_steps += 1
        self.state.step(is_dummy=False,is_profile=False,log_stats=False)
        # Native may consume a result and start its next cycle in one step.
        # A new stats object distinguishes that case without a policy wrapper.
        launched = self.ms.eplb_stats is not before_stats
        committed = before_pending and (not self.ms.rebalanced or launched)
        if committed:
            self._record_commit(step_index,False)
        if launched:
            self.async_events.append(dict(launch_step=int(step_index),launch_counter_before=before_counter,
                cycle_identity=id(self.ms.eplb_stats),
                window_valid_steps=min(self.window_steps,self.native_valid_steps),
                sampled_step_indices=self.recorded_steps[-self.window_steps:].copy(),
                current_counts_device=self.ms.eplb_stats.global_expert_load_window,
                source_map_device=self.ms.physical_to_logical_map[0].detach().clone(),
                sent_bytes_at_launch=before_bytes,commit_step=None,
                commit_in_completion_tail=False,committed_map_device=None))
        elif self.state.expert_rearrangement_step >= self.rebalance_interval and self.ms.rebalanced:
            self.deferred_boundary_steps.append(int(step_index))

    def _record_commit(self, step_index, tail):
        pending = [e for e in self.async_events if e['commit_step'] is None]
        if len(pending) != 1:
            raise AssertionError('Native async commit does not match exactly one recorded launch')
        event = pending[0]
        event.update(commit_step=int(step_index),commit_in_completion_tail=bool(tail),
            committed_map_device=self.ms.physical_to_logical_map[0].detach().clone(),
            sent_bytes_local=self.counter.cycle_bytes(event['cycle_identity']))

    def complete_pending(self, *, timeout_seconds=120):
        """Normal native commit tail, never the native discard-only drain API."""
        start = time.monotonic()
        if self.state is not None and self.state.is_async:
            from vllm.distributed.eplb.eplb_state import _move_to_workspace
            flag = self.torch.ones(2,dtype=self.torch.int32,device='cpu')
            while self.ms.rebalanced:
                worker = self.state.async_worker
                flag[0] = int(worker is not None and worker.is_alive())
                flag[1] = int(time.monotonic()-start < timeout_seconds)
                self.torch.distributed.all_reduce(flag,op=self.torch.distributed.ReduceOp.MIN,
                                                 group=self.rt.ep_group.cpu_group)
                if not bool(flag.all()):
                    raise RuntimeError('Async worker stopped or normal commit tail timed out')
                if self.state._all_ranks_result_ready(self.ms):
                    _move_to_workspace(self.ms, self.rt.rank)
                    self._record_commit(self.step_index,True)
                else:
                    time.sleep(.001)
            if self.ms.pending_result is not None:
                raise AssertionError('Unconsumed asynchronous result after completion')
        self.rt.sync()
        elapsed = time.monotonic()-start
        self.completion_tail_seconds += elapsed
        return dict(seconds=elapsed,pending_result=False,rebalanced=False,
                    disposition='normal_commit_no_discard')

    close = complete_pending

    def reset_trial(self):
        """Untimed canonical reset after quiescence; worker and storage reused."""
        if self._active:
            raise RuntimeError('Reset must be outside a measurement scope')
        if hasattr(self,'completion_tail_seconds'):
            self.complete_pending()
        with self.torch.no_grad():
            self._reload_experts(self.initial_map)
            if self.state is not None:
                from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
                _commit_eplb_maps(self.ms,self.torch.tensor(self.initial_map[None],dtype=self.torch.int64))
                self.ms.expert_load_pass.zero_(); self.ms.expert_load_window.zero_()
                self.ms.last_expert_load.fill_(1)
                self.ms.eplb_stats = None
                self.state.expert_load_window_step = 0
                self.state.expert_rearrangement_step = self.spec['initial_counter']
                self.state._update_layer_should_record(log_stats=False)
                for tensor in self.ms.num_unpadded_tokens_tensors:
                    tensor.zero_()
                self.counter.sent_bytes = self.counter.sent_messages = 0
                if isinstance(self.counter,AsyncPayloadCounter):
                    self.counter.reset_counts()
        self.step_index = -1
        self.observation_steps = self.native_valid_steps = self.measured_step_calls = 0
        self.last_accepted_step = None
        self.planner_seconds = self.controller_seconds = self.completion_tail_seconds = 0.
        self.decisions, self.async_events = [], []
        self.recorded_steps, self.deferred_boundary_steps = [], []
        self.histogram_capture_count = self.previous_window_timestamp_checks = 0
        self.warming_up = self.end_of_input = False
        self.terminal_proposals_suppressed = 0
        self.logical_ring.zero_(); self.logical_ring_timestamps.fill(-1)
        if self.mapper is not None:
            self.mapper.calls = 0
            self.mapper.verify_metadata(self.layer_state.logical_to_physical_map,
                self.layer_state.logical_replica_count,self.ms.physical_to_logical_map[0])
        self.rt.sync()

    def map_snapshot(self):
        return self.initial_map.copy() if self.ms is None else self.ms.physical_to_logical_map[0].cpu().numpy().copy()

    def assert_maps_and_weights(self):
        mapping = self.map_snapshot()
        maps = self.rt.gather_objects(mapping.tolist())
        if maps[0] != maps[1]:
            raise AssertionError('Committed maps differ across ranks')
        if self.arm != 'stockon32' and not np.array_equal(mapping,self.initial_map):
            raise AssertionError('Fixed-placement arm changed its map')
        if sorted(set(mapping.tolist())) != list(range(128)):
            raise AssertionError('Committed map lost a logical expert')
        ids = mapping[self.rt.rank*self.slots:(self.rt.rank+1)*self.slots]
        for key,actual in [('w13_weight',self.w13),('w2_weight',self.w2)]:
            expected = self.weights.tensors[key][ids].to(self.device,self.dtype)
            if not self.torch.equal(actual,expected):
                raise AssertionError('Committed stock expert differs from checkpoint: '+key)
        if not self.torch.equal(self.block.gate.weight,self._gate_source.to(self.device)):
            raise AssertionError('Stock router weight changed')
        if self.ms is not None:
            logical = self.ms.logical_to_physical_map[0].cpu().numpy()
            counts = self.ms.logical_replica_count[0].cpu().numpy()
            for expert in range(128):
                positions = np.flatnonzero(mapping == expert)
                if counts[expert] != len(positions) or not np.array_equal(logical[expert,:len(positions)],positions):
                    raise AssertionError('Committed inverse/count maps are inconsistent')
        return True

    def reference(self, hidden, logical_ids, combining_weights):
        from skewbench.native import NativeLayer
        return NativeLayer.reference(self,hidden,logical_ids,combining_weights)

    def verify_online_assignment(self, hidden):
        """Untimed current-input GPU selection versus independent CPU oracle."""
        if self.mapper is None:
            raise ValueError('Only the combined arm has an online physical selector')
        from resident_dispatch.online import cpu_greedy_assignment
        ids,weights = self.routing(hidden)
        expected = cpu_greedy_assignment(ids.cpu().numpy(),self.initial_map)
        load = self.torch.zeros_like(self.layer_state.expert_load_view)
        record = self.torch.ones((),device=self.device,dtype=self.torch.bool)
        real = self.torch.tensor(len(hidden),device=self.device,dtype=self.torch.int32)
        actual = self.mapper(ids,load,self.layer_state.logical_to_physical_map,
            self.layer_state.logical_replica_count,record,real)
        physical = actual.cpu().numpy()
        if not np.array_equal(physical,expected.physical_ids) or not np.array_equal(load.cpu().numpy(),expected.diagnostics['physical_counts']):
            raise AssertionError('Online GPU selection or recording differs from CPU oracle')
        digest = _hash_array(physical,'<i8')
        if len(set(self.rt.gather_objects(digest))) != 1:
            raise AssertionError('Online physical assignment differs across ranks')
        return dict(physical_sha256=digest,logical_sha256=_hash_array(ids.cpu().numpy(),'<i8'),
                    weights_sha256=_hash_array(weights.cpu().numpy(),'<f4'),**expected.diagnostics)

    def reduction_state(self):
        communicator = self.rt.tp_group.device_communicator
        names = ('fi_ar_comm','fi_pcie_ipc_ar_comm','qr_comm','aiter_ar_comm',
                 'ca_comm','symm_mem_comm','pynccl_comm')
        return dict(communicator_class=_class_name(communicator),
            candidates={name: None if (value := getattr(communicator,name,None)) is None
                        else dict(class_name=_class_name(value),disabled=bool(getattr(value,'disabled',False)))
                        for name in names},
            per_call_selection_depends_on_tensor=True)

    def resolved_kernel_config(self):
        """Native modular Triton selector arguments, without any override."""
        from vllm.model_executor.layers.fused_moe import get_config
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        if get_config() is not None:
            raise AssertionError('Stock comparison forbids a process-global kernel override')
        experts = self.routed.quant_method.moe_kernel.fused_experts
        name = experts.quant_config.config_name(self.dtype)
        if name is not None or experts.block_shape is not None:
            raise AssertionError('Expected native unquantized BF16 config namespace')
        return dict(try_get_optimal_moe_config(self.w13.size(),self.w2.size(),self.top_k,
            name,self.batch_tokens,block_shape=experts.block_shape))

    def attest_output_reduction(self, hidden):
        """Untimed delegate-only observation; restore all methods before trials.

        This does not force a backend or infer a successful backend from its
        availability alone. An unobserved native torch-op branch stays unknown.
        """
        if self._active or (self.ms is not None and self.ms.rebalanced):
            raise RuntimeError('Output reduction probe requires idle untimed state')
        communicator = self.rt.tp_group.device_communicator
        names = (('fi_ar_comm','all_reduce'),('fi_pcie_ipc_ar_comm','all_reduce'),
                 ('qr_comm','quick_all_reduce'),('aiter_ar_comm','custom_all_reduce'),
                 ('ca_comm','custom_all_reduce'),('symm_mem_comm','all_reduce'),
                 ('pynccl_comm','all_reduce'))
        restorations, calls = [], []
        def observe(obj,name,label):
            original = getattr(obj,name)
            had_local = name in vars(obj)
            local_value = vars(obj).get(name)
            def delegated(*args,**kwargs):
                result = original(*args,**kwargs)
                calls.append(dict(backend=label,returned_output=result is not None))
                return result
            setattr(obj,name,delegated)
            restorations.append((obj,name,had_local,local_value))
        try:
            for name,method in names:
                obj = getattr(communicator,name,None)
                if obj is not None:
                    observe(obj,method,name)
            self.forward(hidden,observe=False)
            self.rt.sync()
        finally:
            for obj,name,had_local,local_value in reversed(restorations):
                if had_local:
                    setattr(obj,name,local_value)
                else:
                    delattr(obj,name)
        result = dict(scope='One untimed actual stock forward; delegate-only backend observation restored before timing',
            batch_tokens=len(hidden),dtype=str(hidden.dtype),calls=calls,
            observed_successful_backends=[c['backend'] for c in calls if c['returned_output']],
            unresolved_native_branch=not any(c['returned_output'] for c in calls),
            backend_forced=False,**self.reduction_state())
        self.rt.metadata['output_reduction_probe'] = result
        return result

    def snapshot(self):
        """Untimed serialization; actual proposal/commit GPU snapshots stay raw until here."""
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
        events = []
        for event in self.async_events:
            row = {k:v for k,v in event.items() if not k.endswith('_device') and k != 'cycle_identity'}
            row['current_counts'] = event['current_counts_device'].cpu().numpy().tolist()
            row['source_map'] = event['source_map_device'].cpu().numpy().tolist()
            value = event['committed_map_device']
            row['committed_map'] = None if value is None else value.cpu().numpy().tolist()
            events.append(row)
        pc = self.config.parallel_config
        kernel = self.routed.quant_method.moe_kernel
        decisions = []
        for decision in self.decisions:
            row = {}
            for key,value in decision.items():
                if key == 'physical_load_window_device':
                    row['physical_window_totals_local'] = value.sum(dim=0).cpu().numpy().tolist()
                elif key.endswith('_device'):
                    row[key.removesuffix('_device')] = value.cpu().numpy().tolist()
                else:
                    row[key] = value.tolist() if isinstance(value,np.ndarray) else value
            decisions.append(row)
        result = dict(**self.spec,schema_version=1,
            native_layer_class=_class_name(self.block),native_runner_class=_class_name(self.runner),
            native_router_class=_class_name(self.router),native_quant_method_class=_class_name(self.routed.quant_method),
            native_experts_class=_class_name(kernel.fused_experts),
            stock_router_unmodified=self.router._apply_eplb_mapping.__func__ is BaseRouter._apply_eplb_mapping,
            stock_policy_unmodified=self.state is None or self.state.policy is DefaultEplbPolicy,
            logical_router_computation_included=True,native_output_reduction=True,
            output_reduction_state=self.reduction_state(),
            resolved_kernel_config=self.resolved_kernel_config(),kernel_override_applied=False,
            custom_all_reduce_disabled=pc.disable_custom_all_reduce,moe_backend_requested=self.config.kernel_config.moe_backend,
            communicator_class=None if self.counter is None else _class_name(self.counter.inner),
            communicator=None if self.state is None else pc.eplb_config.communicator,
            gate_weight_sha256=self.gate_sha256,source_map=self.initial_map.tolist(),final_map=self.map_snapshot().tolist(),
            source_expert_parameter_bytes=self.slots*9437184,controller_step_calls=self.measured_step_calls if self.state is not None else 0,
            native_counter=None if self.state is None else self.state.expert_rearrangement_step,
            native_valid_steps=self.native_valid_steps,async_events=events,
            recorded_step_indices=self.recorded_steps.copy(),deferred_boundary_steps=self.deferred_boundary_steps.copy(),
            decisions=decisions,
            terminal_physical_window_totals_local=None if self.ms is None else self.ms.expert_load_window[:,0,:].sum(dim=0).cpu().numpy().tolist(),
            physical_window_next_write=None if self.state is None else self.state.expert_load_window_step,
            pending_result=self.ms is not None and self.ms.pending_result is not None,
            rebalanced=self.ms is not None and self.ms.rebalanced,
            async_worker_alive=None if self.state is None or not self.state.is_async else self.state.async_worker.is_alive(),
            migration_sent_bytes_local=0 if self.counter is None else self.counter.sent_bytes,
            migration_sent_messages_local=0 if self.counter is None else self.counter.sent_messages,
            completion_tail_seconds=self.completion_tail_seconds,completion_disposition='normal_commit_no_discard',
            mapper_calls=0 if self.mapper is None else self.mapper.calls,
            guard=None if self.guard is None else self.guard.snapshot(),
            terminal_proposals_suppressed=self.terminal_proposals_suppressed)
        return result

    def audit(self):
        self.assert_maps_and_weights()
        result = self.snapshot()
        if result['pending_result'] or result['rebalanced']:
            raise AssertionError('Audit requires normally completed state')
        if self.arm == 'stockon32' and (not result['async_worker_alive'] or any(e['commit_step'] is None for e in result['async_events'])):
            raise AssertionError('Native async work failed to commit')
        result.update(resident_state_verified=True,rank_map_agreement=True,normal_completion_verified=True)
        return result
