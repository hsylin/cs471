"""Scoped launch-extent specialization; native alignment allocation is unchanged.

Only the reviewed B128/top8 BF16, two-rank EP path is supported.  The 160-slot
case requires the frozen one-copy-home assignment policy (or exact replay of
its assignments).  No actual padded-count readback occurs on the timed path.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np

VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
SOURCE_SHA256 = {
    "model_executor/layers/fused_moe/experts/triton_moe.py": "ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab",
    "model_executor/layers/fused_moe/fused_moe.py": "21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7",
    "model_executor/layers/fused_moe/moe_align_block_size.py": "2addb33632b611b4c0df21f2b874dc1124d2fc867785f19f05d4ce33d4b872c6",
}
EXPECTED_CONFIG = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                       GROUP_SIZE_M=1, num_warps=4, num_stages=3, SPLIT_K=1)
CAPACITY = 5056
_ACTIVE = None


def padded_capacity_bound(num_routes=1024, active_cap=64, block_size=64):
    """Integer bound on sum(ceil(n_e/B)*B), with <=A positive buckets.

    Each positive bucket contributes at most B-1 padding; the result is a
    multiple of B, so rounding the bound down is safe. No route statistics or
    future workload information enter this bound.
    """
    if any(not isinstance(x, (int, np.integer)) or x < 0
           for x in (num_routes, active_cap)) or block_size < 1:
        raise ValueError("Invalid capacity geometry")
    active_cap = min(int(active_cap), int(num_routes))
    return ((int(num_routes) + active_cap * (block_size - 1)) // block_size) * block_size


def verify_count_bound(counts, *, num_routes=1024, active_cap=64, block_size=64):
    a = np.asarray(counts)
    if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("Counts must be a nonnegative integer vector")
    if int(a.sum()) > num_routes or np.count_nonzero(a) > active_cap:
        raise ValueError("The capacity proof's route/active assumptions fail")
    padded = int((((a + block_size - 1) // block_size) * block_size).sum())
    bound = padded_capacity_bound(num_routes, active_cap, block_size)
    if padded > bound:
        raise AssertionError("Padding bound violated")
    return dict(tokens=int(a.sum()), active=int(np.count_nonzero(a)),
                padded_rows=padded, capacity_bound=bound)


def one_home_active_cap(forced_rank0, forced_rank1, shared_active):
    """Frozen policy's target, not a new assignment algorithm."""
    if not (0 <= forced_rank0 <= 48 and 0 <= forced_rank1 <= 48
            and 0 <= shared_active <= 32):
        raise ValueError("Not the fixed 128+32 residency geometry")
    return max(forced_rank0, forced_rank1,
               (forced_rank0 + forced_rank1 + shared_active + 1) // 2)


def _hash(tensor):
    a = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<i8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def verify_sources():
    import vllm
    root = Path(vllm.__file__).resolve().parent
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise RuntimeError("Native alignment source differs from the reviewed pin")
    return actual


class AlignmentTrim:
    """Use through trim_native_alignment; setup and verify are untimed.

    probe=True additionally delegates through a GEMM-call observer. This is
    correctness instrumentation only and must never be used for timing.
    """
    def __init__(self, layer, *, enabled=True, probe=False, policy=None):
        self.layer, self.enabled, self.probe = layer, bool(enabled), bool(probe)
        self.calls = 0
        self.records = []
        self._installed = False
        self.source_hashes = verify_sources()
        self.module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.triton_moe")
        self.original = self.module._prepare_expert_assignment
        self.original_invoke = self.module.invoke_fused_moe_triton_kernel
        if self.original.__module__ != "vllm.model_executor.layers.fused_moe.fused_moe":
            raise RuntimeError("Another alignment adapter is already installed")
        self._validate_layer(policy)

    def _validate_layer(self, policy):
        layer = self.layer
        if (layer.rt.world_size != 2 or layer.batch_tokens != 128
            or layer.top_k != 8 or str(layer.dtype) != "torch.bfloat16"):
            raise ValueError("Only BF16 B128/top8 is supported")
        self.global_experts = int(len(layer.initial_map))
        if self.global_experts not in (128, 160) or layer.slots != self.global_experts // 2:
            raise ValueError("Expected two equal EP ranks with 128 or 160 physical slots")
        expected = list(range(128)) + (list(range(32)) if self.global_experts == 160 else [])
        if np.asarray(layer.initial_map).tolist() != expected:
            raise ValueError("Changed residency is unsupported")
        if self.global_experts == 160:
            known = policy or getattr(layer, "alignment_policy", None)
            if known is None and getattr(layer, "selector_kind", None) == "fused":
                known = "one_copy_home"
            if known != "one_copy_home":
                raise ValueError("160 slots require frozen one-copy-home policy or its exact replay")
            self.policy = known
        else:
            self.policy = "stockoff_64_resident_experts_per_rank"
        experts = layer.routed.quant_method.moe_kernel.fused_experts
        if (experts.__class__.__name__ != "TritonExperts"
            or experts.quant_dtype is not None or experts.block_shape is not None
            or experts._lora_context is not None):
            raise ValueError("Only native unquantized non-LoRA TritonExperts is supported")
        actual = layer.resolved_kernel_config()
        if any(actual.get(k) != v for k, v in EXPECTED_CONFIG.items()):
            raise ValueError("Native kernel geometry differs from reviewed defaults")
        self.expert_map = layer.expert_map
        if self.expert_map is None or tuple(self.expert_map.shape) != (self.global_experts,):
            raise ValueError("Expected global-to-local EP expert map")
        mapping = self.expert_map.detach().cpu().numpy()
        if np.any(mapping < -1) or sorted(mapping[mapping >= 0].tolist()) != list(range(layer.slots)):
            raise ValueError("Expert map does not map each local slot exactly once")
        self.map_version = self.expert_map._version
        self.map_pointer = self.expert_map.data_ptr()
        self.config = dict(actual)

    def _prepare(self, topk_ids, config, num_tokens, top_k_num,
                 global_num_experts, expert_map, *, use_int8_w8a16=False,
                 use_int4_w4a16=False, block_shape=None,
                 ignore_invalid_experts=False):
        if (tuple(topk_ids.shape) != (128, 8) or not topk_ids.is_cuda
            or not topk_ids.is_contiguous()
            or str(topk_ids.dtype) not in ("torch.int32", "torch.int64")
            or num_tokens != 128 or top_k_num != 8
            or global_num_experts != self.global_experts
            or expert_map is not self.expert_map
            or expert_map._version != self.map_version
            or expert_map.data_ptr() != self.map_pointer
            or use_int8_w8a16 or use_int4_w4a16 or block_shape is not None
            or not ignore_invalid_experts
            or any(config.get(k) != v for k, v in EXPECTED_CONFIG.items())):
            raise RuntimeError("Unsupported or changed alignment call; refusing unsafe trim")
        full, experts, padded = self.original(topk_ids, config, num_tokens,
            top_k_num, global_num_experts, expert_map,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16, block_shape=block_shape,
            ignore_invalid_experts=ignore_invalid_experts)
        expected = 1024 + self.global_experts * 63
        if full is None or tuple(full.shape) != (expected,) or not full.is_contiguous():
            raise RuntimeError("Unexpected native alignment buffer")
        view = full[:CAPACITY] if self.enabled else full
        self.calls += 1
        if self.probe:
            self.records.append(dict(full=full, view=view, experts=experts,
                                     padded=padded, ids=topk_ids.clone(), gemms=[]))
        return view, experts, padded

    def _invoke(self, *args, **kwargs):
        values = inspect.signature(self.original_invoke).bind(*args, **kwargs).arguments
        if not self.records:
            raise AssertionError("GEMM preceded alignment")
        record = self.records[-1]
        if values["sorted_token_ids"] is not record["view"]:
            raise AssertionError("GEMM did not receive this alignment view")
        if values["expert_ids"] is not record["experts"] or values["num_tokens_post_padded"] is not record["padded"]:
            raise AssertionError("GEMM alignment metadata changed")
        cfg = values["config"]
        n = int(values["B"].shape[1])
        extent = int(record["view"].numel())
        record["gemms"].append(dict(n=n, top_k=int(values["top_k"]),
            sorted_extent=extent, m_programs=(extent + cfg["BLOCK_SIZE_M"] - 1) // cfg["BLOCK_SIZE_M"],
            grid_programs=((extent + cfg["BLOCK_SIZE_M"] - 1) // cfg["BLOCK_SIZE_M"])
                          * ((n + cfg["BLOCK_SIZE_N"] - 1) // cfg["BLOCK_SIZE_N"])))
        return self.original_invoke(*args, **kwargs)

    @contextmanager
    def active(self):
        global _ACTIVE
        if _ACTIVE is not None or self._installed:
            raise RuntimeError("Alignment scopes must be sequential and nonnested")
        if not self.enabled and not self.probe:
            yield self
            return
        if self.module._prepare_expert_assignment is not self.original:
            raise RuntimeError("Native alignment hook changed")
        _ACTIVE = self
        self._installed = True
        @wraps(self.original)
        def prepare(*args, **kwargs):
            return self._prepare(*args, **kwargs)
        try:
            self.module._prepare_expert_assignment = prepare
            if self.probe:
                self.module.invoke_fused_moe_triton_kernel = self._invoke
            yield self
        finally:
            self.module._prepare_expert_assignment = self.original
            self.module.invoke_fused_moe_triton_kernel = self.original_invoke
            self._installed = False
            _ACTIVE = None

    def snapshot(self):
        return dict(schema="native_alignment_extent_v1", enabled=self.enabled,
            probe_only=self.probe, alignment_calls=self.calls,
            native_allocation_preserved=True, native_alignment_algorithm_preserved=True,
            policy=self.policy, capacity_bound=CAPACITY, active_local_cap=64,
            global_physical_experts=self.global_experts,
            native_capacity=1024 + self.global_experts * 63,
            config=self.config, native_source_sha256=self.source_hashes,
            runtime_gpu_readback=False, actual_padding_verified=False)

    def probe_assignments(self, physical_ids):
        """Untimed adversarial alignment test without running either GEMM.

        Use actual mapper outputs for the160-slot case; arbitrary160-slot
        assignments need not satisfy the one-home active cap. Actual-layer
        probes separately verify both GEMM calls and numerical outputs.
        """
        if not self.probe or self._installed:
            raise RuntimeError("Use an inactive probe guard")
        with self.active():
            self.module._prepare_expert_assignment(physical_ids, self.config,
                128, 8, self.global_experts, self.expert_map,
                ignore_invalid_experts=True)
        self.records[-1]["alignment_only"] = True
        return self.verify()

    def verify(self):
        """Synchronizing probe checks, strictly outside timed scopes."""
        if self._installed:
            raise RuntimeError("Verify after leaving the execution scope")
        if not self.probe or not self.records:
            raise ValueError("No untimed probe records")
        evidence = []
        for record in self.records:
            padded = int(record["padded"].item())
            if not 0 <= padded <= CAPACITY or padded % 64:
                raise AssertionError("Actual native padding violates proven bound")
            if record["view"].data_ptr() != record["full"].data_ptr():
                raise AssertionError("Trimming copied or moved the backing allocation")
            if not self.layer.torch.equal(record["view"][:padded], record["full"][:padded]):
                raise AssertionError("Valid alignment prefix changed")
            physical = record["ids"].detach().cpu().numpy().reshape(-1)
            mapping = self.expert_map.detach().cpu().numpy()
            logical_valid = (physical >= 0) & (physical < self.global_experts)
            local = np.full(physical.shape, -1, dtype=np.int64)
            local[logical_valid] = mapping[physical[logical_valid]]
            counts = np.bincount(local[local >= 0], minlength=self.layer.slots)
            count_evidence = verify_count_bound(counts)
            if count_evidence["padded_rows"] != padded:
                raise AssertionError("Native padding does not match actual local assignments")
            sorted_ids = record["full"][:padded].detach().cpu().numpy().reshape(-1, 64)
            expert_ids = record["experts"][:padded // 64].detach().cpu().numpy()
            seen = []
            for tokens, expert in zip(sorted_ids, expert_ids):
                real = tokens[tokens < 1024]
                if (expert < 0 or expert >= self.layer.slots
                    or np.any(tokens > 1024)
                    or np.any(real < 0) or np.any(local[real] != expert)):
                    raise AssertionError("Alignment bucket contains a wrong local assignment")
                seen.extend(real.tolist())
            if sorted(seen) != np.flatnonzero(local >= 0).tolist():
                raise AssertionError("Alignment dropped or duplicated a local assignment")
            alignment_only = record.get("alignment_only", False)
            if not alignment_only and [(x["n"], x["top_k"]) for x in record["gemms"]] != [(1536, 8), (2048, 1)]:
                raise AssertionError("Expected exactly the native w13/w2 GEMM pair")
            evidence.append(dict(actual_padded_rows=padded, actual_m_tiles=padded // 64,
                valid_sorted_ids_sha256=_hash(record["full"][:padded]),
                valid_expert_ids_sha256=_hash(record["experts"][:padded // 64]),
                physical_ids_sha256=_hash(record["ids"]),
                backing_pointer_preserved=True, valid_prefix_equal=True,
                independent_local_counts=count_evidence,
                token_conservation_verified=True,
                scope="alignment_only" if alignment_only else "actual_native_gemm_pair",
                gemms=record["gemms"]))
        result = self.snapshot()
        result.update(actual_padding_verified=True, probe_records=evidence)
        return result


@contextmanager
def trim_native_alignment(layer, enabled=True, probe=False, *, policy=None):
    guard = AlignmentTrim(layer, enabled=enabled, probe=probe, policy=policy)
    with guard.active():
        yield guard


def verify_adversarial_alignment(rt, stockoff, previous):
    """Untimed native alignment proof: nine cases, two dtypes, two layouts.

    Both full and trimmed views are tested. The actual-layer probe in the
    runner separately checks both GEMMs and full numerical outputs.
    """
    from resident_dispatch.online import adversarial_inputs
    from resident_fastpath.fused import cpu_reference
    torch = rt.torch
    guards = {(name, enabled): AlignmentTrim(layer, enabled=enabled, probe=True,
                policy="one_copy_home")
              for name, layer in (("stockoff", stockoff), ("previous", previous))
              for enabled in (False, True)}
    before = previous.mapper.verify_context()
    default_real = int(previous.mapper.real_tokens.item())
    if default_real != 128:
        raise AssertionError("Expected unchanged full-batch internal mask")
    evidence = []
    for case in adversarial_inputs(128):
        for name, dtype in (("int32", torch.int32), ("int64", torch.int64)):
            host = case["logical_ids"]
            # The supplied adversarial partial cases explicitly use -1 for
            # every padded row; no valid padded route enters alignment.
            if np.any(host[case["real_tokens"]:] != -1):
                raise AssertionError("Adversarial padding must use invalid IDs")
            ids = torch.from_numpy(host.copy()).to(rt.device, dtype)
            real = torch.tensor(case["real_tokens"], device=rt.device, dtype=torch.int32)
            expected = cpu_reference(host, previous.initial_map,
                                     real_tokens=case["real_tokens"])
            physical = previous.mapper(ids, real)
            if not np.array_equal(physical.cpu().numpy(), expected["physical_ids"]):
                raise AssertionError("Actual physical assignment differs from CPU policy")
            peer_hashes = rt.gather_objects(_hash(physical))
            if len(set(peer_hashes)) != 1:
                raise AssertionError("Ranks disagree on adversarial physical IDs")
            row = dict(case=case["name"], dtype=name, real_tokens=int(case["real_tokens"]),
                       logical_ids_sha256=_hash(ids), physical_ids_sha256=peer_hashes[0],
                       rank_assignment_agreement=True, layouts={})
            for layout, routes in (("stockoff", ids), ("previous", physical)):
                full = guards[layout, False].probe_assignments(routes)["probe_records"][-1]
                trimmed = guards[layout, True].probe_assignments(routes)["probe_records"][-1]
                for key in ("actual_padded_rows", "actual_m_tiles", "physical_ids_sha256", "independent_local_counts"):
                    if full[key] != trimmed[key]:
                        raise AssertionError("Native full/trim assignment semantics differ")
                row["layouts"][layout] = dict(full=full, trimmed=trimmed,
                    canonical_assignment_and_padding_equal=True)
            evidence.append(row)
    if previous.mapper.verify_context() != before or int(previous.mapper.real_tokens.item()) != default_real:
        raise AssertionError("Adversarial alignment altered mapper context/default mask")
    per_rank = rt.gather_objects(evidence)
    return dict(schema="resident_residual_alignment_adversarial_v1", passed=True,
        cases=9, dtypes=["int32", "int64"], layouts=["stockoff128", "one_home160"],
        views=["native_full", "proven_cap"], native_alignment_calls_per_rank=72,
        source_sha256=SOURCE_SHA256, capacity_bound=CAPACITY,
        real_token_default_preserved=True, stream_context_preserved=True,
        per_rank=per_rank,
        scope="Alignment only; full captured-input probes separately check native GEMM pairs and outputs")
