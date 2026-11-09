"""Pinned native vLLM one-layer EPLB harness.

The execution path is synchronous expert parallelism with replicated inputs and
an output all-reduce. It uses EplbState's actual observation, planning, movement,
and map commit; it is not a serving engine or a DeepEP/all-to-all benchmark.
GPU dependencies are imported lazily so CPU contract tests need only NumPy.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PINNED_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
METHOD_MODES = {"B0": "skip", "B1": "native", "B2": "native",
                "B3": "improvement", "B4": "stability", "B5": "cooldown",
                "SKIP": "skip"}


def initialization_progress(phase):
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"event": "initialization", "phase": phase}), flush=True)


def harness_fingerprint():
    """Hash code and native-source pin data, independent of outputs/manifests."""
    directory = Path(__file__).parent
    paths = sorted(directory.glob("*.py")) + [directory / "native_sources.json"]
    files = {f"skewbench/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, files


def fill_logical_histogram(row, logical_ids, ones):
    """Fixed-size integer histogram, with no device-valued shape discovery.

    Inputs are validated replay IDs in [0, num_experts). The caller allocates
    ``row`` and all-one int64 ``ones`` before measurement. Unlike CUDA bincount,
    this path does not read an input minimum/maximum back to the host.
    """
    row.zero_()
    row.scatter_add_(0, logical_ids.reshape(-1).long(), ones)


def should_capture_previous_window(window_steps, rebalance_interval, native_counter,
                                   *, warming_up=False):
    """Capture the earlier of the two windows at a fixed synchronous cadence.

    The counter is read before the current forward/step increment. The native
    recorder supplies the immediately preceding window at the decision itself.
    Runtime construction requires rebalance_interval >= 2 * window_steps.
    """
    remaining = rebalance_interval - native_counter
    return warming_up or window_steps < remaining <= 2 * window_steps


def previous_window_indices(row_timestamps, observation_steps, window_steps):
    """Reject stale ring rows before they can become a stability observation."""
    if observation_steps < 2 * window_steps or len(row_timestamps) != 2 * window_steps:
        raise AssertionError("Two complete observation windows are required")
    expected = range(observation_steps - 2 * window_steps, observation_steps - window_steps)
    indices = [step % (2 * window_steps) for step in expected]
    if any(row_timestamps[index] != step for index, step in zip(indices, expected)):
        raise AssertionError("Previous-window ring timestamps are incomplete or stale")
    return indices


def validate_layer_shapes(w13_shape, w2_shape, config):
    e, h, m = (int(config[k]) for k in
               ("num_experts", "hidden_size", "intermediate_size"))
    if tuple(w13_shape) != (e, 2 * m, h) or tuple(w2_shape) != (e, h, m):
        raise ValueError(f"Layer weight shapes do not match config: {w13_shape}, {w2_shape}")


def initial_mapping(num_experts: int, replicas: int, world_size: int) -> np.ndarray:
    if num_experts <= 0 or replicas < 0 or world_size < 2:
        raise ValueError("Require positive experts, nonnegative replicas, >=2 ranks")
    if (num_experts + replicas) % world_size:
        raise ValueError("Physical slots must divide evenly across ranks")
    return np.concatenate((np.arange(num_experts), np.arange(replicas) % num_experts))


def verify_native_sources(package_dir: Path) -> dict:
    """Fail closed if the installed Python sources differ from the pinned commit."""
    manifest = json.loads(Path(__file__).with_name("native_sources.json").read_text())
    mismatches = []
    for rel, expected in manifest["sha256"].items():
        path = package_dir / Path(rel).relative_to("vllm")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"Installed vLLM does not match pinned native sources: {mismatches}")
    return {"commit": manifest["commit"], "verified_source_files": len(manifest["sha256"])}


class Runtime:
    """One distributed initialization shared by a randomized sequence of runs."""
    def __init__(self):
        initialization_progress("import_torch")
        import torch
        initialization_progress("import_vllm")
        import vllm
        initialization_progress("import_vllm_config_and_distributed")
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            init_distributed_environment, ensure_model_parallel_initialized,
            get_ep_group, get_eplb_group, get_tp_group,
        )
        self.torch = torch
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        if self.world_size < 2 or not torch.cuda.is_available():
            raise RuntimeError("Use torchrun with at least two CUDA GPUs")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.source_attestation = verify_native_sources(Path(vllm.__file__).parent)
        initialization_progress("pinned_sources_verified")
        self.config = VllmConfig()
        pc = self.config.parallel_config
        pc.tensor_parallel_size = self.world_size
        pc.enable_expert_parallel = True
        pc.enable_eplb = True
        pc.disable_custom_all_reduce = True  # reproducible torch/NCCL output reduction
        pc.eplb_config.use_async = False
        pc.eplb_config.communicator = "torch_nccl"
        pc.eplb_config.policy = "default"
        self._context = set_current_vllm_config(self.config)
        self._context.__enter__()
        initialization_progress("initialize_distributed_environment")
        init_distributed_environment(backend="nccl")
        initialization_progress("initialize_model_parallel_groups")
        ensure_model_parallel_initialized(tensor_model_parallel_size=self.world_size,
                                          pipeline_model_parallel_size=1)
        self.ep_group = get_ep_group()
        self.eplb_group = get_eplb_group()
        self.tp_group = get_tp_group()
        initialization_progress("runtime_ready")
        self.metadata = {
            "native_vllm_commit": PINNED_COMMIT,
            "native_source_attestation": self.source_attestation,
            "eplb_mode": "synchronous",
            "dispatch_backend": "replicated_input_allreduce",
            "output_collective": "torch_distributed_nccl_all_reduce",
            "weight_communicator": "torch_nccl",
            "reject_execution": "native_noop_mapping",
            "world_size": self.world_size,
            "gpu_name": torch.cuda.get_device_name(),
            "torch_version": torch.__version__, "vllm_version": vllm.__version__,
            "cuda_version": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
        self.metadata["harness_sha256"], self.metadata["harness_file_sha256"] = harness_fingerprint()
        self._closed = False

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def barrier(self):
        self.torch.distributed.barrier(group=self.ep_group.device_group)

    def gather_scalars(self, value):
        t = self.torch.tensor([value], dtype=self.torch.float64, device=self.device)
        out = [self.torch.empty_like(t) for _ in range(self.world_size)]
        self.torch.distributed.all_gather(out, t, group=self.ep_group.device_group)
        return [float(x.item()) for x in out]

    def close(self):
        if self._closed:
            return
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
        self._context.__exit__(None, None, None)
        self._closed = True


class Weights:
    def __init__(self, directory: str | Path):
        from safetensors.torch import load_file
        directory = Path(directory)
        self.config = json.loads((directory / "config.json").read_text())
        initialization_progress("verify_checkpoint_sha256")
        expected_hash = self.config.get("weights_sha256")
        if not expected_hash:
            raise ValueError("Checkpoint export must declare weights_sha256")
        digest = hashlib.sha256()
        with (directory / "layer.safetensors").open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError("Layer checkpoint SHA256 does not match config")
        self.tensors = load_file(str(directory / "layer.safetensors"), device="cpu")
        validate_layer_shapes(self.tensors["w13_weight"].shape,
                              self.tensors["w2_weight"].shape, self.config)
        self.metadata = self.config
        initialization_progress("checkpoint_ready")


class CountingCommunicator:
    """Counts actual payloads posted to the native transport, once on senders."""
    def __init__(self, inner):
        self.inner = inner
        self.sent_bytes = 0
        self.sent_messages = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def add_send(self, tensors, dst, **kwargs):
        self.sent_bytes += sum(t.numel() * t.element_size() for t in tensors)
        self.sent_messages += 1
        return self.inner.add_send(tensors, dst, **kwargs)


@dataclass(frozen=True)
class LayerConfigIdentity:
    model: str

    def compute_hash(self):
        return hashlib.sha256(self.model.encode()).hexdigest()


class NativeLayer:
    def __init__(self, runtime: Runtime, weights: Weights, *, method: str,
                 replicas: int, window_steps: int, rebalance_interval: int,
                 epsilon: float, tau: float, cooldown_steps: int, seed: int):
        from vllm.distributed.eplb.eplb_state import EplbState, EplbLayerState
        from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
        from vllm.model_executor.layers.fused_moe import fused_experts
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            eplb_map_to_physical_and_record,
        )
        if method not in METHOD_MODES:
            raise ValueError(f"Unknown method {method}")
        if rebalance_interval < 2 * window_steps:
            raise ValueError("Core harness requires cadence >= two windows to avoid stale physical history")
        self.rt, self.weights, self.method = runtime, weights, method
        self.torch, self.device = runtime.torch, runtime.device
        self.fused_experts = fused_experts
        self.mapping_kernel = eplb_map_to_physical_and_record
        self.default_policy = DefaultEplbPolicy
        cfg = weights.config
        self.num_experts = int(cfg["num_experts"])
        self.hidden_size = int(cfg["hidden_size"])
        self.intermediate_size = int(cfg["intermediate_size"])
        self.top_k = int(cfg["top_k"])
        self.window_steps = int(window_steps)
        self.rebalance_interval = int(rebalance_interval)
        self.epsilon, self.tau, self.cooldown_steps = epsilon, tau, cooldown_steps
        self.replicas = 0 if method in ("B0", "B1") else replicas
        self.initial_map = initial_mapping(self.num_experts, self.replicas, runtime.world_size)
        self.slots = len(self.initial_map) // runtime.world_size
        start = runtime.rank * self.slots
        local_ids = self.initial_map[start:start + self.slots]
        self.dtype = self.torch.bfloat16
        self.w13 = weights.tensors["w13_weight"][local_ids].to(self.device, self.dtype).contiguous()
        self.w2 = weights.tensors["w2_weight"][local_ids].to(self.device, self.dtype).contiguous()
        self.expert_map = self.torch.full((len(self.initial_map),), -1,
                                        dtype=self.torch.int32, device=self.device)
        self.expert_map[start:start + self.slots] = self.torch.arange(
            self.slots, dtype=self.torch.int32, device=self.device)
        self.layer_state = EplbLayerState()
        layer = SimpleNamespace(eplb_state=self.layer_state)

        def set_state(load, logical_map, replica_count):
            self.layer_state.set_layer_state(0, load, logical_map, replica_count)

        model = SimpleNamespace(
            num_moe_layers=1, num_expert_groups=1,
            num_logical_experts=self.num_experts, num_routed_experts=self.num_experts,
            num_shared_experts=0, num_physical_experts=len(self.initial_map),
            num_redundant_experts=self.replicas, num_local_physical_experts=self.slots,
            moe_layers=[layer], expert_weights=[[self.w13, self.w2]], set_eplb_state=set_state,
        )
        pc = runtime.config.parallel_config
        pc.eplb_config.window_size = window_steps
        pc.eplb_config.step_interval = rebalance_interval
        pc.eplb_config.num_redundant_experts = self.replicas
        self.state = EplbState(pc, self.device)
        self.identity = LayerConfigIdentity("skewbench-one-layer")
        self.state.add_model(model, self.identity)
        self.ms = self.state.model_states[self.identity.compute_hash()]
        if not np.array_equal(self.ms.physical_to_logical_map[0].cpu().numpy(), self.initial_map):
            raise AssertionError("Native initial map differs from loaded weight arrangement")
        self.counter = CountingCommunicator(self.ms.communicator)
        self.ms.communicator = self.counter
        if self.counter.inner._ep_group is not runtime.eplb_group.device_group:
            raise AssertionError("Native weight transport did not reuse the process EPLB group")
        self.step_index = -1
        self.last_accepted_step = None
        self.planner_seconds = 0.0
        self.controller_seconds = 0.0
        self.decisions = []
        self.observation_steps = 0
        self.native_valid_steps = 0
        self.logical_ring = self.torch.zeros((2 * window_steps, self.num_experts),
                                              dtype=self.torch.int64, device=self.device)
        self.logical_ring_timestamps = np.full(2 * window_steps, -1, dtype=np.int64)
        self.histogram_capture_count = 0
        self.previous_window_timestamp_checks = 0
        self.histogram_ones = None
        self.collect_history = method == "B4"
        self.warming_up = False
        self.capture_decision_step = None
        self.captured_candidate = None
        self.end_of_input = False
        self.terminal_proposals_suppressed = 0
        self._install_policy()
        self.rt.sync()

    def _install_policy(self):
        owner = self

        class AdmissionPolicy:
            @classmethod
            def rebalance_experts(cls, weight, num_replicas, num_groups, num_nodes,
                                  num_ranks, old_global_expert_indices=None):
                from .policies import evaluate_admission
                if owner.warming_up:
                    # Common observation warmup: no proposal optimization or migration.
                    return old_global_expert_indices.clone()
                if owner.end_of_input:
                    owner.terminal_proposals_suppressed += 1
                    return old_global_expert_indices.clone()
                t0 = time.perf_counter()
                proposal = owner.default_policy.rebalance_experts(
                    weight, num_replicas, num_groups, num_nodes, num_ranks,
                    old_global_expert_indices)
                owner.planner_seconds += time.perf_counter() - t0
                if owner.method in ("B1", "B2"):
                    # Preserve native decision semantics: no gate validation or threshold.
                    owner.decisions.append({"step": owner.step_index, "allowed": True,
                                            "reason": "native", "tv": None,
                                            "predicted_gain": None,
                                            "current_map": old_global_expert_indices[0].numpy().copy(),
                                            "proposed_map": proposal[0].numpy().copy(),
                                            "current_counts": weight[0].numpy().copy(),
                                            "previous_counts": None,
                                            **owner._physical_observation_snapshot()})
                    return proposal
                t0 = time.perf_counter()
                prev = None
                if owner.collect_history and owner.observation_steps >= 2 * owner.window_steps:
                    indices = previous_window_indices(owner.logical_ring_timestamps,
                                                      owner.observation_steps, owner.window_steps)
                    owner.previous_window_timestamp_checks += 1
                    previous_tensor = owner.logical_ring[indices].sum(dim=0)
                    owner.torch.distributed.all_reduce(
                        previous_tensor, group=owner.rt.ep_group.device_group)
                    prev = previous_tensor.cpu().numpy()
                current = weight[0].numpy()
                old = old_global_expert_indices[0].numpy()
                proposed = proposal[0].numpy()
                decision = evaluate_admission(
                    mode=METHOD_MODES[owner.method], current_counts=current,
                    previous_counts=prev, current_map=old, proposed_map=proposed,
                    world_size=owner.rt.world_size, epsilon=owner.epsilon,
                    tau=owner.tau, step=owner.step_index,
                    last_accepted_step=owner.last_accepted_step,
                    cooldown_steps=owner.cooldown_steps,
                    windows_complete=(owner.observation_steps >= 2 * owner.window_steps,
                                      owner.native_valid_steps >= owner.window_steps),
                )
                intercepted = owner.capture_decision_step == owner.step_index
                if intercepted:
                    owner.captured_candidate = {
                        "current_counts": current.copy(),
                        "previous_counts": None if prev is None else prev.copy(),
                        "current_map": old.copy(), "proposed_map": proposed.copy(),
                        "b3_would_allow": decision.allowed, "b3_reason": decision.reason,
                    }
                if decision.allowed:
                    owner.last_accepted_step = owner.step_index
                owner.decisions.append({
                    "step": owner.step_index, "allowed": decision.allowed,
                    "reason": decision.reason, "tv": decision.tv,
                    "predicted_gain": decision.predicted_gain,
                    "candidate_changed_slots": int(np.count_nonzero(old != proposed)),
                    "current_map": old.copy(), "proposed_map": proposed.copy(),
                    "current_counts": current.copy(),
                    "previous_counts": None if prev is None else prev.copy(),
                    **owner._physical_observation_snapshot(),
                })
                owner.controller_seconds += time.perf_counter() - t0
                return proposal if decision.allowed and not intercepted else old_global_expert_indices.clone()

        self.state.policy = AdmissionPolicy

    def _physical_observation_snapshot(self):
        # The native rolling buffer still refers to the old committed map here.
        # This is shared instrumentation for all monitored methods; no new
        # token histogram is computed. Materialization is deferred past timing.
        return {"physical_load_window_device": self.ms.expert_load_window[:, 0, :].detach().clone(),
                "physical_window_next_write": self.state.expert_load_window_step}

    def map_snapshot(self):
        return self.ms.physical_to_logical_map[0].cpu().numpy().copy()

    def forward(self, hidden, logical_ids, combining_weights, *, observe=True):
        """The same logical token tensor is replicated on all EP ranks."""
        if observe and self.collect_history:
            capture = should_capture_previous_window(
                self.window_steps, self.rebalance_interval, self.state.expert_rearrangement_step,
                warming_up=self.warming_up)
            if capture:
                if self.histogram_ones is None:
                    # First observed step is in the common untimed warmup.
                    self.histogram_ones = self.torch.ones(logical_ids.numel(),
                                                         dtype=self.torch.int64, device=self.device)
                elif self.histogram_ones.numel() != logical_ids.numel():
                    raise ValueError("A run must retain its fixed token-batch shape")
                index = self.observation_steps % len(self.logical_ring)
                fill_logical_histogram(self.logical_ring[index], logical_ids, self.histogram_ones)
                self.logical_ring_timestamps[index] = self.observation_steps
                self.histogram_capture_count += 1
            # Absolute observed-step positions advance even when the histogram
            # is unnecessary. Native current-window recording remains untouched.
            self.observation_steps += 1
        if self.method == "B0":
            physical_ids = logical_ids
        else:
            ls = self.layer_state
            if not observe:
                # Do not perturb the native observation state during correctness checks.
                record = self.torch.zeros((), dtype=self.torch.bool, device=self.device)
            else:
                record = ls.should_record_tensor
            ls.num_unpadded_tokens_tensors[0].fill_(hidden.shape[0])
            physical_ids = self.mapping_kernel(
                logical_ids, ls.expert_load_view, ls.logical_to_physical_map,
                ls.logical_replica_count, record, ls.num_unpadded_tokens_tensors[0])
        output = self.fused_experts(hidden, self.w13, self.w2, combining_weights,
                                    physical_ids, global_num_experts=len(self.initial_map),
                                    expert_map=self.expert_map)
        # Native late-all-reduce semantics, pinned to torch NCCL for reproducibility.
        self.torch.distributed.all_reduce(output, group=self.rt.tp_group.device_group)
        return output

    def step(self, step_index):
        self.step_index = step_index
        if self.method == "B0":
            return
        # Match the native pre-increment recording decision without device synchronization.
        if self.state._should_record_current_step(log_stats=False):
            self.native_valid_steps += 1
        before_decisions = len(self.decisions)
        before_bytes = self.counter.sent_bytes
        self.state.step(is_dummy=False, is_profile=False, log_stats=False)
        if len(self.decisions) > before_decisions:
            # Shared lightweight GPU snapshot; copy/hash/consensus occur after
            # timed completion, avoiding an extra host synchronization here.
            row = self.decisions[-1]
            row["committed_map_device"] = self.ms.physical_to_logical_map[0].detach().clone()
            row["transfer_sent_bytes_local"] = self.counter.sent_bytes - before_bytes

    def assert_maps_and_weights(self):
        """Independent checkpoint identity check, excluded from measured runtime."""
        mapping = self.map_snapshot()
        gathered = [None] * self.rt.world_size
        self.torch.distributed.all_gather_object(gathered, mapping.tolist(),
                                                group=self.rt.ep_group.cpu_group)
        if any(m != gathered[0] for m in gathered):
            raise AssertionError("Ranks disagree on the committed physical mapping")
        start = self.rt.rank * self.slots
        ids = mapping[start:start + self.slots]
        for key, actual in (("w13_weight", self.w13), ("w2_weight", self.w2)):
            expected = self.weights.tensors[key][ids].to(self.device, self.dtype)
            if not self.torch.equal(expected, actual):
                raise AssertionError(f"Committed {key} does not match logical checkpoint expert")
        logical = self.ms.logical_to_physical_map[0].cpu().numpy()
        counts = self.ms.logical_replica_count[0].cpu().numpy()
        for e in range(self.num_experts):
            locations = np.flatnonzero(mapping == e)
            if int(counts[e]) != len(locations) or not np.array_equal(logical[e, :len(locations)], locations):
                raise AssertionError("Logical inverse map or replica count is inconsistent")
        return True

    def reference(self, hidden, logical_ids, combining_weights):
        """Independent dense FP32 SwiGLU reference on a small token sample."""
        torch = self.torch
        result = torch.zeros_like(hidden, dtype=torch.float32)
        ids = logical_ids.cpu().numpy()
        for e in np.unique(ids):
            positions = np.argwhere(ids == e)
            rows = torch.tensor(positions[:, 0], device=self.device)
            choices = torch.tensor(positions[:, 1], device=self.device)
            w13 = self.weights.tensors["w13_weight"][int(e)].to(self.device, torch.float32)
            w2 = self.weights.tensors["w2_weight"][int(e)].to(self.device, torch.float32)
            gate_up = hidden[rows].float() @ w13.T
            gate, up = gate_up.chunk(2, dim=-1)
            output = (torch.nn.functional.silu(gate) * up) @ w2.T
            result.index_add_(0, rows, output * combining_weights[rows, choices, None])
        return result

    def check_outputs(self, hidden, ids, combining_weights, atol=0.02, rtol=0.03):
        # Prefix keeps the native replica hash's token indices unchanged.
        hidden, ids, combining_weights = hidden[:4], ids[:4], combining_weights[:4]
        actual = self.forward(hidden, ids, combining_weights, observe=False)
        expected = self.reference(hidden, ids, combining_weights)
        self.torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
        err = float((actual.float() - expected).abs().max().item())
        return {"correctness_max_abs_error": max(self.rt.gather_scalars(err)),
                "correctness_atol": atol, "correctness_rtol": rtol}

    def force_migration(self, mapping):
        """Diagnostic only: native transfer and native commit for a specified map."""
        from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
        from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
        mapping = self.torch.as_tensor(np.asarray(mapping).copy(), dtype=self.torch.long,
                                       device="cpu").reshape(1, -1)
        rearrange_expert_weights_inplace(
            self.ms.physical_to_logical_map, mapping, self.ms.model.expert_weights,
            self.ms.expert_buffer, self.rt.ep_group.device_group, self.ms.communicator)
        _commit_eplb_maps(self.ms, mapping)
        self.rt.sync()

    def restore_snapshot(self, mapping, local_w13, local_w2):
        """Untimed branch checkpoint restoration, with native map commit."""
        from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
        self.w13.copy_(local_w13)
        self.w2.copy_(local_w2)
        mapping = self.torch.as_tensor(np.asarray(mapping).copy(), dtype=self.torch.long,
                                       device="cpu").reshape(1, -1)
        _commit_eplb_maps(self.ms, mapping)
        self.rt.sync()

    def close(self):
        # Synchronous path has no pending worker. Check instead of silently draining.
        if self.ms.pending_result is not None or self.ms.rebalanced or self.state.is_async:
            raise AssertionError("Unexpected pending or asynchronous work at completion")
        self.rt.sync()
