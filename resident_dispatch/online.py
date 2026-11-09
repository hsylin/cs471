"""Current-batch resident-copy selection, with a CPU verification oracle.

The policy is fixed: descending token count (expert ID breaks ties), greedy
token-load placement under exact minimum-active-count capacities. No runtime
optimality is claimed. GPU imports are deliberately deferred to construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


EXPERTS = 128
SLOTS = 160
SLOTS_PER_RANK = 80
SHARED_EXPERTS = 32
HISTOGRAM_CHUNK = 256
POLICY_ID = "current_batch_one_home_lpt_active_caps_v1"


def fixed_map():
    return np.concatenate((np.arange(EXPERTS), np.arange(SHARED_EXPERTS)))


def validate_fixed_map(physical_to_logical_map):
    mapping = np.asarray(physical_to_logical_map)
    if (mapping.shape != (SLOTS,) or not np.issubdtype(mapping.dtype, np.integer)
            or not np.array_equal(mapping, fixed_map())):
        raise ValueError("Online selector requires the frozen [0..127,0..31] map")
    return mapping.astype(np.int64, copy=True)


def validate_native_metadata(logical_map, replica_count, physical_map=None):
    """CPU-only content check, called explicitly outside all measured regions."""
    logical = np.asarray(logical_map)
    replicas = np.asarray(replica_count)
    if (logical.ndim != 2 or logical.shape[0] != EXPERTS or logical.shape[1] < 2
            or not np.issubdtype(logical.dtype, np.integer)):
        raise ValueError("Native logical map has incompatible shape or dtype")
    if replicas.shape != (EXPERTS,) or not np.issubdtype(replicas.dtype, np.integer):
        raise ValueError("Native replica counts have incompatible shape or dtype")
    expected = np.full(logical.shape, -1, dtype=np.int64)
    expected[:, 0] = np.arange(EXPERTS)
    expected[:SHARED_EXPERTS, 1] = np.arange(128, 160)
    expected_replicas = np.ones(EXPERTS, dtype=np.int64)
    expected_replicas[:SHARED_EXPERTS] = 2
    if not np.array_equal(logical, expected) or not np.array_equal(replicas, expected_replicas):
        raise ValueError("Resident map metadata changed; online homes would be stale")
    if physical_map is not None:
        validate_fixed_map(physical_map)
    return True


@dataclass(frozen=True)
class GreedyReference:
    physical_ids: np.ndarray
    homes: np.ndarray
    logical_counts: np.ndarray
    diagnostics: dict


def cpu_greedy_assignment(logical_ids, physical_to_logical_map, *, real_tokens=None):
    """Independent scalar oracle for the GPU policy; never used for online choices.

    Count only real rows. Invalid padded-row IDs map to -1, matching native
    invalid-route handling. Every real row must contain valid logical IDs.
    This function supports small test shapes; the GPU adapter allows B128/B1024
    and top8 only. It does not use the offline subset-sum selector.
    """
    mapping = validate_fixed_map(physical_to_logical_map)
    ids = np.asarray(logical_ids)
    if ids.ndim != 2 or ids.shape[1] == 0 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected integer logical IDs with shape [batch, top_k]")
    real = len(ids) if real_tokens is None else real_tokens
    if (type(real) is not int and not isinstance(real, np.integer)) or isinstance(real, (bool, np.bool_)):
        raise ValueError("real_tokens must be an integer")
    real = int(real)
    if not 0 <= real <= len(ids):
        raise ValueError("real_tokens is outside the batch")
    if np.any(ids[:real] < 0) or np.any(ids[:real] >= EXPERTS):
        raise ValueError("A real route has no eligible resident expert")
    ids = ids.astype(np.int64, copy=False)
    counts = np.bincount(ids[:real].reshape(-1), minlength=EXPERTS)
    forced = [list(range(32, 80)), list(range(80, 128))]
    fixed_active = [sum(counts[e] > 0 for e in experts) for experts in forced]
    loads = [sum(int(counts[e]) for e in experts) for experts in forced]
    shared = sorted((e for e in range(32) if counts[e] > 0),
                    key=lambda e: (-int(counts[e]), e))
    optimum = max(*fixed_active, (sum(fixed_active) + len(shared) + 1) // 2)
    capacities = [int(optimum - n) for n in fixed_active]
    remaining = capacities.copy()
    homes = np.arange(EXPERTS, dtype=np.int64)
    decisions = []
    for expert in shared:
        eligible = [rank for rank in range(2) if remaining[rank] > 0]
        assert eligible, "Active-copy capacities exhausted prematurely"
        rank = min(eligible, key=lambda r: (loads[r], r))
        homes[expert] = expert if rank == 0 else 128 + expert
        remaining[rank] -= 1
        loads[rank] += int(counts[expert])
        decisions.append([expert, rank])
    physical = np.full(ids.shape, -1, dtype=np.int64)
    valid = (ids >= 0) & (ids < EXPERTS)
    physical[valid] = homes[ids[valid]]
    assert np.array_equal(mapping[physical[:real]], ids[:real])
    per_slot = np.bincount(physical[:real].reshape(-1), minlength=SLOTS)
    active = np.count_nonzero(per_slot.reshape(2, 80), axis=1)
    assert int(active.max()) == int(optimum)
    summary = [int(optimum), int(fixed_active[0]), int(fixed_active[1]), len(shared),
               loads[0], loads[1], int(active[0]), int(active[1])]
    for array in (physical, homes, counts):
        array.setflags(write=False)
    return GreedyReference(physical, homes, counts, {
        "policy_id": POLICY_ID, "real_tokens": real,
        "fixed_active_copies": [int(x) for x in fixed_active],
        "shared_active_experts": len(shared),
        "active_copy_capacities": capacities,
        "minimum_possible_max_rank_active_copies": int(optimum),
        "per_rank_tokens": loads, "per_rank_active_copies": active.tolist(),
        "physical_counts": per_slot.tolist(), "greedy_decisions": decisions,
        "summary": summary, "runtime_optimality_claimed": False,
    })


def adversarial_inputs(batch_tokens, top_k=8):
    """Untimed mapping-validation cases; no effect on policy selection."""
    if batch_tokens not in (128, 1024) or top_k != 8:
        raise ValueError("Adversarial GPU cases use the frozen B128/B1024 top8 shapes")
    b = int(batch_tokens)
    shared_ties = np.tile(np.arange(8, dtype=np.int32), (b, 1))
    singleton = np.tile(np.array([32, 33, 34, 35, 80, 81, 82, 83], np.int32), (b, 1))
    asymmetric = np.empty((b, 8), dtype=np.int32)
    for row in range(b):
        asymmetric[row, 0] = row % 32
        asymmetric[row, 1:] = 32 + (row * 7 + np.arange(7)) % 48
    boundary = np.tile(np.arange(32, 40, dtype=np.int32), (b, 1))
    boundary_counts = [63, 64, 65] + ([127, 128, 129] if b == 1024 else [])
    for column, count in enumerate(boundary_counts):
        boundary[:count, column] = column
    hot_shared = np.tile(np.array([0,32,33,34,80,81,82,83], np.int32), (b, 1))
    mixed = shared_ties.copy()
    mixed[1::2] = singleton[1::2]
    partial = shared_ties.copy()
    partial[b // 2:] = -1
    all_experts = (np.arange(b * 8).reshape(b, 8) % 128).astype(np.int32)
    return [
        {"name": "shared_count_ties", "logical_ids": shared_ties, "real_tokens": b},
        {"name": "singleton_only", "logical_ids": singleton, "real_tokens": b},
        {"name": "asymmetric_singleton_owners", "logical_ids": asymmetric, "real_tokens": b},
        {"name": "tile_boundary_counts", "logical_ids": boundary, "real_tokens": b},
        {"name": "hot_shared_with_forced_owners", "logical_ids": hot_shared, "real_tokens": b},
        {"name": "shared_and_singleton_ties", "logical_ids": mixed, "real_tokens": b},
        {"name": "all_logical_experts", "logical_ids": all_experts, "real_tokens": b},
        {"name": "partial_real_rows", "logical_ids": partial, "real_tokens": b // 2},
        {"name": "zero_real_rows", "logical_ids": np.full((b, 8), -1, np.int32), "real_tokens": 0},
    ]


def _signature(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


def _cpu_array(value):
    # Only used by explicitly untimed verification/diagnostic methods.
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _array_digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _check_adversarial_result(reference, snapshot, actual_load, *, sentinel, record_enabled):
    expected = {"physical_ids": reference.physical_ids, "homes": reference.homes,
                "logical_counts": reference.logical_counts,
                "summary": np.asarray(reference.diagnostics["summary"])}
    for field, value in expected.items():
        if not np.array_equal(snapshot[field], value):
            raise AssertionError(f"Adversarial GPU result differs from CPU oracle: {field}")
    expected_load = np.full(SLOTS, sentinel, dtype=np.int64)
    if record_enabled:
        expected_load += np.asarray(reference.diagnostics["physical_counts"], dtype=np.int64)
    if not np.array_equal(actual_load, expected_load):
        raise AssertionError("Adversarial recording flag/real-token mask changed physical counts")
    return {field + "_sha256": _array_digest(snapshot[field]) for field in expected} | {
        "physical_load_sha256": _array_digest(actual_load)}


def verify_adversarial(rt, layer, mapper, batch_tokens):
    """Untimed GPU preflight: nine cases, both recording modes, both ranks.

    Only isolated recording/scalar buffers are passed to the mapper. This checks
    routing and counters, not expert arithmetic; full-batch output checks remain
    the runner's separate responsibility. Do not call inside primary timing.
    """
    if mapper.batch_tokens != batch_tokens or mapper.top_k != 8:
        raise ValueError("Adversarial mapper shape differs from the requested batch")
    torch = rt.torch
    ls = layer.layer_state
    mapping = layer.map_snapshot()
    mapper.verify_metadata(ls.logical_to_physical_map, ls.logical_replica_count, mapping)
    saved_load = _cpu_array(ls.expert_load_view).copy()
    saved_record = _cpu_array(ls.should_record_tensor).copy()
    saved_real = [_cpu_array(t).copy() for t in ls.num_unpadded_tokens_tensors]
    sentinel = 7
    evidence = []
    for case in adversarial_inputs(batch_tokens):
        reference = cpu_greedy_assignment(case["logical_ids"], mapping,
                                          real_tokens=case["real_tokens"])
        ids = torch.from_numpy(case["logical_ids"].copy()).to(rt.device, torch.int32)
        real = torch.tensor(case["real_tokens"], device=rt.device, dtype=torch.int32)
        for enabled in (False, True):
            isolated_load = torch.full_like(ls.expert_load_view, sentinel)
            record = torch.tensor(enabled, device=rt.device, dtype=torch.bool)
            mapper(ids, isolated_load, ls.logical_to_physical_map,
                   ls.logical_replica_count, record, real)
            snapshot = mapper.snapshot_diagnostics()
            hashes = _check_adversarial_result(reference, snapshot, _cpu_array(isolated_load),
                                               sentinel=sentinel, record_enabled=enabled)
            evidence.append({"case": case["name"], "record_enabled": enabled,
                             "real_tokens": case["real_tokens"],
                             "input_ids_sha256": _array_digest(case["logical_ids"]), **hashes})
    mapper.verify_metadata(ls.logical_to_physical_map, ls.logical_replica_count,
                           layer.map_snapshot())
    if (not np.array_equal(_cpu_array(ls.expert_load_view), saved_load)
            or not np.array_equal(_cpu_array(ls.should_record_tensor), saved_record)
            or any(not np.array_equal(_cpu_array(t), saved)
                   for t, saved in zip(ls.num_unpadded_tokens_tensors, saved_real))):
        raise AssertionError("Adversarial preflight mutated native recording state")
    peers = [None] * rt.world_size
    torch.distributed.all_gather_object(peers, evidence, group=rt.ep_group.cpu_group)
    if any(peer != evidence for peer in peers):
        raise AssertionError("Ranks disagree on adversarial physical routes or counters")
    return {"passed": True, "policy_id": POLICY_ID, "batch_tokens": int(batch_tokens),
            "cases_checked": len(evidence) // 2, "recording_modes_per_case": 2,
            "gpu_invocations_checked": len(evidence), "isolated_load_sentinel": sentinel,
            "rank_agreement": True, "native_recording_state_unchanged": True,
            "scope": "physical IDs, homes, logical counts, summary and masked physical recording",
            "cases": evidence,
            "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()}


class OnlineMapper:
    """Three kernels per call; all choices use the current device ID tensor.

    verify_metadata must run before first use and again after measurement. Its
    content check is intentionally outside the timed call; pointer/shape guards
    inside __call__ cannot detect forbidden in-place residency changes.
    """
    def __init__(self, torch, physical_to_logical_map, batch_tokens, top_k=8):
        self.physical_to_logical_map = validate_fixed_map(physical_to_logical_map)
        if batch_tokens not in (128, 1024) or top_k != 8:
            raise ValueError("Online experiment is frozen to B128/B1024, top8")
        from .online_gpu import histogram_partials, choose_homes, map_and_record
        self.torch = torch
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch_tokens, self.top_k = int(batch_tokens), int(top_k)
        self.num_routes = self.batch_tokens * self.top_k
        self.partial_rows = (self.num_routes + HISTOGRAM_CHUNK - 1) // HISTOGRAM_CHUNK
        self.partials = torch.empty((self.partial_rows, EXPERTS), dtype=torch.int32, device=self.device)
        self.homes = torch.empty(EXPERTS, dtype=torch.int32, device=self.device)
        self.logical_counts = torch.empty(EXPERTS, dtype=torch.int32, device=self.device)
        self.summary = torch.empty(8, dtype=torch.int32, device=self.device)
        self._kernels = (histogram_partials, choose_homes, map_and_record)
        self._verified_signature = None
        self.last_physical_ids = None
        self.calls = 0

    def verify_metadata(self, logical_map, replica_count, physical_map=None):
        validate_native_metadata(_cpu_array(logical_map), _cpu_array(replica_count),
                                 None if physical_map is None else _cpu_array(physical_map))
        if logical_map.device != self.device or replica_count.device != self.device:
            raise ValueError("Native maps must be on this mapper's CUDA device")
        self._verified_signature = (_signature(logical_map), _signature(replica_count))
        return True

    def __call__(self, logical_ids, load, logical_map, replica_count, record, real_tokens):
        torch = self.torch
        if self._verified_signature is None:
            raise RuntimeError("Call verify_metadata outside timing before using OnlineMapper")
        if (_signature(logical_map), _signature(replica_count)) != self._verified_signature:
            raise ValueError("Native map storage changed; rerun the untimed residency check")
        if (tuple(logical_ids.shape) != (self.batch_tokens, self.top_k)
                or logical_ids.dtype not in (torch.int32, torch.int64)
                or not logical_ids.is_contiguous()):
            raise ValueError("Current logical routes changed frozen shape/dtype/layout")
        if (tuple(load.shape) != (SLOTS,) or not load.is_contiguous()
                or load.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Physical counter must contain160 contiguous slots")
        if (record.numel() != 1 or record.dtype != torch.bool or
                real_tokens.numel() != 1 or real_tokens.dtype != torch.int32):
            raise ValueError("Native recording/real-token scalar contract changed")
        if any(t.device != self.device for t in (logical_ids, load, record, real_tokens)):
            raise ValueError("Mapper inputs must share its CUDA device")
        output = torch.empty_like(logical_ids)  # Match the native output allocation.
        hist, choose, mapped = self._kernels
        hist[(self.partial_rows,)](logical_ids, self.partials, real_tokens,
            N=self.num_routes, TOPK=self.top_k, CHUNK=HISTOGRAM_CHUNK, num_warps=4)
        choose[(1,)](self.partials, self.homes, self.logical_counts, self.summary,
            PARTIAL_ROWS=self.partial_rows, num_warps=4)
        mapped[(self.partial_rows,)](logical_ids, self.homes, output, load, record, real_tokens,
            N=self.num_routes, TOPK=self.top_k, CHUNK=HISTOGRAM_CHUNK, num_warps=4)
        self.last_physical_ids = output
        self.calls += 1
        return output

    def snapshot_diagnostics(self):
        """Read back the last invocation for untimed correctness checks only."""
        if self.last_physical_ids is None:
            raise RuntimeError("No online invocation is available to inspect")
        return {"physical_ids": _cpu_array(self.last_physical_ids).copy(),
                "homes": _cpu_array(self.homes).copy(),
                "logical_counts": _cpu_array(self.logical_counts).copy(),
                "summary": _cpu_array(self.summary).copy(), "calls": self.calls}
