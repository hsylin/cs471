"""Offline resident-copy assignments for the two-rank headroom experiment.

These CPU selectors are diagnostic, not an online dispatcher or runtime oracle.
They preserve every logical route's position. Combining weights are deliberately
not an input: callers must use the original weights in the original order.

Native selection follows vLLM db9527a46873454610df6dbedf79a36d6bf1a7f6,
vllm/model_executor/layers/fused_moe/router/base_router.py:33-62. Replica lists
are ordered by global physical slot, as in compute_logical_maps at that pin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PINNED_VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
KNUTH_MULTIPLIER = 2654435769
METHODS = ("native", "one_copy_home", "token_balanced")


@dataclass(frozen=True)
class AssignmentResult:
    physical_ids: np.ndarray
    diagnostics: dict[str, Any]


def _inputs(logical_ids, physical_to_logical_map, world_size, block_size_m):
    if (not isinstance(world_size, (int, np.integer)) or
            isinstance(world_size, (bool, np.bool_)) or world_size != 2):
        raise ValueError("This offline study requires exactly two ranks")
    if (not isinstance(block_size_m, (int, np.integer)) or
            isinstance(block_size_m, (bool, np.bool_)) or block_size_m <= 0):
        raise ValueError("block_size_m must be a positive integer")
    ids = np.asarray(logical_ids)
    mapping = np.asarray(physical_to_logical_map)
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("logical_ids must have shape [batch_tokens, positive top_k]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("logical_ids must contain integer expert IDs")
    if mapping.ndim != 1 or not len(mapping) or len(mapping) % world_size:
        raise ValueError("Physical map must be flat with equally sized rank shards")
    if not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0):
        raise ValueError("Physical map must contain nonnegative integer logical IDs")
    # No absent logical expert is allowed; unused physical slots are out of scope.
    experts = np.unique(mapping)
    if not np.array_equal(experts, np.arange(len(experts))):
        raise ValueError("Physical map must cover every logical expert from zero")
    if np.any(ids < 0) or np.any(ids >= len(experts)):
        raise ValueError("Logical route has no eligible resident expert")
    return ids.astype(np.int64, copy=False), mapping.astype(np.int64, copy=False)


def _residency(mapping):
    slots = len(mapping) // 2
    return [tuple(np.flatnonzero(mapping == expert).astype(int).tolist())
            for expert in range(int(mapping.max()) + 1)], slots


def _active_groups(ids, residents, slots):
    counts = np.bincount(ids.reshape(-1), minlength=len(residents))
    fixed = [[], []]
    shared = []
    for expert in np.flatnonzero(counts):
        owners = {slot // slots for slot in residents[int(expert)]}
        if len(owners) == 2:
            shared.append(int(expert))
        else:
            fixed[next(iter(owners))].append(int(expert))
    return counts, fixed, shared


def _active_optimum(fixed, shared_count):
    peaks = [max(len(fixed[0]) + x, len(fixed[1]) + shared_count - x)
             for x in range(shared_count + 1)]
    optimum = min(peaks)
    return optimum, [x for x, peak in enumerate(peaks) if peak == optimum]


def _token_aware_subset(weights, allowed_cardinalities, fixed_tokens):
    """Exact token-peak tie break, restricted to optimal active-copy counts.

    Each integer bitset encodes reachable token sums at one subset cardinality.
    Reconstruction prefers excluding higher-index experts when sums tie. This
    optimizes counts of routed tokens, not measured kernel latency.
    """
    n = len(weights)
    history = [[1] + [0] * n]
    for i, weight in enumerate(weights):
        previous = history[-1]
        current = previous.copy()
        for cardinality in range(1, i + 2):
            current[cardinality] |= previous[cardinality - 1] << int(weight)
        history.append(current)
    total = int(sum(weights) + sum(fixed_tokens))
    target = max(0, min(sum(weights), total // 2 - fixed_tokens[0]))
    best = None
    for cardinality in allowed_cardinalities:
        bits = history[-1][cardinality]
        lower = (bits & ((1 << (target + 1)) - 1)).bit_length() - 1
        upper_bits = bits >> target
        upper = (target + (upper_bits & -upper_bits).bit_length() - 1
                 if upper_bits else None)
        for value in {lower, upper}:
            if value is None or value < 0:
                continue
            load0 = fixed_tokens[0] + value
            key = (max(load0, total - load0), cardinality, value)
            if best is None or key < best:
                best = key
    assert best is not None
    _, cardinality, value = best
    selected = set()
    for i in range(n, 0, -1):
        if (history[i - 1][cardinality] >> value) & 1:
            continue
        selected.add(i - 1)
        cardinality -= 1
        value -= int(weights[i - 1])
        assert cardinality >= 0 and value >= 0
    assert cardinality == 0 and value == 0
    return selected


def _hash_rows(batch_tokens):
    # Multiplication modulo 2**32 is exactly the pinned kernel's unsigned mask.
    rows = np.arange(batch_tokens, dtype=np.uint64)
    return (rows * np.uint64(KNUTH_MULTIPLIER)) & np.uint64(0xFFFFFFFF)


def _native(ids, residents):
    result = np.empty(ids.shape, dtype=np.int64)
    hashes = _hash_rows(ids.shape[0])
    for expert in np.unique(ids):
        rows, columns = np.where(ids == expert)
        copies = np.asarray(residents[int(expert)], dtype=np.int64)
        result[rows, columns] = copies[(hashes[rows] % len(copies)).astype(np.int64)]
    return result


def _one_copy_home(ids, residents, slots):
    counts, fixed, shared = _active_groups(ids, residents, slots)
    _, allowed = _active_optimum(fixed, len(shared))
    fixed_tokens = [int(counts[experts].sum()) for experts in fixed]
    selected = _token_aware_subset([int(counts[e]) for e in shared], allowed,
                                  fixed_tokens)
    homes = np.full(len(residents), -1, dtype=np.int64)
    for rank in range(2):
        for expert in fixed[rank]:
            homes[expert] = min(residents[expert])
    for index, expert in enumerate(shared):
        rank = 0 if index in selected else 1
        homes[expert] = min(slot for slot in residents[expert] if slot // slots == rank)
    return homes[ids]


def _token_balanced(ids, residents, slots):
    """Exact unit-route token balance; active-copy cost is not its objective.

    Every shared route may independently choose either rank. Fixed-owner routes
    are counted first, then the feasible rank-zero load closest to half is used.
    Shared positions are assigned in row-major order, making ties deterministic.
    Copies within an owner use the native row hash restricted to that owner.
    """
    counts, fixed, shared = _active_groups(ids, residents, slots)
    fixed_tokens = [int(counts[experts].sum()) for experts in fixed]
    shared_mask = np.isin(ids, shared)
    shared_positions = np.flatnonzero(shared_mask.reshape(-1))
    target0 = max(fixed_tokens[0], min(ids.size // 2,
                                    fixed_tokens[0] + len(shared_positions)))
    shared_to_zero = target0 - fixed_tokens[0]
    owners = np.empty(ids.shape, dtype=np.int64)
    for rank in range(2):
        owners[np.isin(ids, fixed[rank])] = rank
    owners.reshape(-1)[shared_positions[:shared_to_zero]] = 0
    owners.reshape(-1)[shared_positions[shared_to_zero:]] = 1
    result = np.empty(ids.shape, dtype=np.int64)
    hashes = _hash_rows(ids.shape[0])
    for expert in np.unique(ids):
        for rank in range(2):
            rows, columns = np.where((ids == expert) & (owners == rank))
            if not len(rows):
                continue
            copies = np.asarray([slot for slot in residents[int(expert)]
                                 if slot // slots == rank], dtype=np.int64)
            result[rows, columns] = copies[(hashes[rows] % len(copies)).astype(np.int64)]
    return result


def validate_assignment(logical_ids, physical_ids, physical_to_logical_map, *,
                        world_size=2, block_size_m=64):
    """Independently validate eligibility and count owner-executed routes once.

    Padding is a count-based diagnostic for the specified M tile, not FLOPs,
    latency, or a runtime bound. Replicated input streams must NOT multiply these
    owner counts; each logical route belongs to exactly one physical owner.
    """
    ids, mapping = _inputs(logical_ids, physical_to_logical_map,
                           world_size, block_size_m)
    physical = np.asarray(physical_ids)
    if physical.shape != ids.shape or not np.issubdtype(physical.dtype, np.integer):
        raise ValueError("Physical IDs must be integers with the logical route shape")
    if np.any(physical < 0) or np.any(physical >= len(mapping)):
        raise ValueError("Physical assignment is outside the resident map")
    if not np.array_equal(mapping[physical], ids):
        raise ValueError("Physical assignment changes a logical expert route")
    physical_counts = np.bincount(physical.reshape(-1), minlength=len(mapping))
    logical_counts = np.bincount(ids.reshape(-1), minlength=int(mapping.max()) + 1)
    recovered = np.zeros(len(logical_counts), dtype=np.int64)
    np.add.at(recovered, mapping, physical_counts)
    if not np.array_equal(recovered, logical_counts):
        raise AssertionError("Physical-to-logical route conservation failed")
    slots = len(mapping) // world_size
    ranked = physical_counts.reshape(world_size, slots)
    tiles = (ranked + int(block_size_m) - 1) // int(block_size_m)
    tokens = ranked.sum(axis=1)
    active = (ranked > 0).sum(axis=1)
    tile_counts = tiles.sum(axis=1)
    residents, _ = _residency(mapping)
    _, fixed, shared = _active_groups(ids, residents, slots)
    optimum, _ = _active_optimum(fixed, len(shared))
    return {
        "schema_version": 1,
        "world_size": int(world_size), "slots_per_rank": slots,
        "num_logical_experts": len(logical_counts), "physical_slots": len(mapping),
        "batch_tokens": ids.shape[0], "top_k": ids.shape[1],
        "block_size_m": int(block_size_m), "total_routes": int(ids.size),
        "count_semantics": "one_global_logical_batch_owner_execution",
        "logical_counts": logical_counts.tolist(),
        "physical_counts": physical_counts.tolist(),
        "per_rank_tokens": tokens.tolist(),
        "per_rank_active_copies": active.tolist(),
        "per_rank_m_tiles": tile_counts.tolist(),
        "per_rank_padded_rows": (tile_counts * int(block_size_m)).tolist(),
        "max_rank_tokens": int(tokens.max()),
        "max_rank_active_copies": int(active.max()),
        "max_rank_padded_rows": int(tile_counts.max() * int(block_size_m)),
        "minimum_possible_max_rank_active_copies": int(optimum),
        "route_eligibility_pass": True,
        "route_conservation_pass": True,
    }


def generate_assignment(logical_ids, physical_to_logical_map, *, method="native",
                        world_size=2, block_size_m=64):
    """Generate one offline assignment; no input array is changed or reordered.

    one_copy_home attains the minimum possible maximum active physical copies,
    then the minimum token peak among one-home-per-expert assignments attaining
    that count optimum. Splitting an expert across owners is outside this tie
    break's feasible set, even if doing so would retain the same active peak.
    This lexicographic count optimum is expressly not a runtime optimum.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}")
    ids, mapping = _inputs(logical_ids, physical_to_logical_map,
                           world_size, block_size_m)
    residents, slots = _residency(mapping)
    if method == "native":
        physical = _native(ids, residents)
    elif method == "one_copy_home":
        physical = _one_copy_home(ids, residents, slots)
    else:
        physical = _token_balanced(ids, residents, slots)
    diagnostics = validate_assignment(ids, physical, mapping, world_size=world_size,
                                      block_size_m=block_size_m)
    diagnostics.update({"method": method, "selector_execution": "offline_cpu",
                        "runtime_optimality_claimed": False})
    if method == "one_copy_home":
        assert (diagnostics["max_rank_active_copies"] ==
                diagnostics["minimum_possible_max_rank_active_copies"])
        diagnostics["active_copy_objective_optimal"] = True
        diagnostics["tie_break"] = "one_home_min_token_peak_then_shared_rank0_count_then_token_sum"
    if method == "native":
        diagnostics["native_vllm_commit"] = PINNED_VLLM_COMMIT
    physical.setflags(write=False)
    return AssignmentResult(physical, diagnostics)
