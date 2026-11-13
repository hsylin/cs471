"""Fixed EP8 residency and a global-batch one-home dispatcher.

This is a separate full-model port, not the frozen EP2/B128 implementation.
Placement is workload-independent. Selection sees already gathered logical
routing; it introduces no collective, CPU tensor read or migration.
"""
from __future__ import annotations

POLICY_ID = "ep8_balanced_pairs_one_home_lpt_caps_v1"
LOGICAL, PHYSICAL, RANKS, SLOTS, TOP_K = 128, 160, 8, 20, 8
MAX_GLOBAL_ROWS = 65536


def balanced_map():
    return [e for rank in range(RANKS)
            for e in [*range(rank * 16, rank * 16 + 16),
                      *range((rank ^ 1) * 16,
                             (rank ^ 1) * 16 + 4)]]


def validate_map(mapping):
    values = mapping.detach().cpu().tolist() if hasattr(mapping, "detach") else list(mapping)
    if values != balanced_map() or any(type(x) is not int for x in values):
        raise ValueError("Expected exact workload-independent EP8 balanced-pairs map")
    return values


def cpu_reference(logical_ids, *, mode="resident_v1"):
    """Independent scalar oracle; -1 is padding, other invalid IDs are errors.

    Use a separate active-copy target for each pair of replica-connected ranks.
    Prefer least token load among ranks below that target, then rank ID. Every
    shared expert in a pair is resident on both ranks, making the cap feasible.
    This extends the original two-rank heuristic to four independent pairs; it
    does not optimize arbitrary eight-way placement.
    """
    import numpy as np
    ids = np.asarray(logical_ids)
    if ids.ndim != 2 or ids.shape[1] != TOP_K or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expected integer [tokens,8] logical routing")
    if np.any(ids < -1) or np.any(ids >= LOGICAL):
        raise ValueError("Only -1 padding or logical experts 0..127 are valid")
    mapping = np.asarray(balanced_map(), dtype=np.int64)
    copies = [np.flatnonzero(mapping == e).tolist() for e in range(LOGICAL)]
    counts = np.bincount(ids[ids >= 0], minlength=LOGICAL)
    homes = [rank * SLOTS + index for rank in range(RANKS) for index in range(16)]
    active, loads = [0] * RANKS, [0] * RANKS
    shared = []
    for expert, count in enumerate(counts):
        if not count:
            continue
        if len(copies[expert]) == 1:
            rank = copies[expert][0] // SLOTS
            active[rank] += 1
            loads[rank] += int(count)
        else:
            shared.append(expert)
    target = [max(active[2 * pair], active[2 * pair + 1],
                      (int(np.count_nonzero(counts[32 * pair:32 * pair + 32])) + 1) // 2)
              for pair in range(4)]
    fallback_count = 0
    if mode == "resident_v1":
        for expert in sorted(shared, key=lambda e: (-int(counts[e]), e)):
            eligible = [slot for slot in copies[expert] if active[slot // SLOTS] < target[expert // 32]]
            if eligible:
                chosen = min(eligible, key=lambda slot: (loads[slot // SLOTS], slot // SLOTS))
            else:
                raise AssertionError("Pairwise active-copy capacity exhausted")
            homes[expert] = chosen
            rank = chosen // SLOTS
            active[rank] += 1
            loads[rank] += int(counts[expert])
        result = np.full(ids.shape, -1, dtype=np.int64)
        valid = ids >= 0
        result[valid] = np.asarray(homes)[ids[valid]]
    elif mode == "static_home":
        result = np.full(ids.shape, -1, dtype=np.int64)
        valid = ids >= 0
        result[valid] = (ids[valid] // 16) * SLOTS + ids[valid] % 16
    elif mode == "native_hash":
        result = np.full(ids.shape, -1, dtype=np.int64)
        for row, col in np.ndindex(ids.shape):
            e = int(ids[row, col])
            if e >= 0:
                result[row, col] = copies[e][((row * 2654435769) & 0xFFFFFFFF) % len(copies[e])]
    else:
        raise ValueError("Unknown dispatcher mode")
    valid = ids >= 0
    assert np.array_equal(mapping[result[valid]], ids[valid])
    physical_counts = np.bincount(result[valid], minlength=PHYSICAL)
    return {"physical_ids": result, "logical_counts": counts, "homes": np.asarray(homes),
            "physical_counts": physical_counts, "active_per_rank": np.count_nonzero(physical_counts.reshape(RANKS, SLOTS), axis=1),
            "tokens_per_rank": physical_counts.reshape(RANKS, SLOTS).sum(axis=1),
            "target": target, "fallback_count": fallback_count, "policy_id": "static_canonical_home_v1" if mode=="static_home" else POLICY_ID,
            "runtime_optimality_claimed": False}


class ResidentMapper:
    """Graph-capturable, current-stream mapper with per-call output storage."""
    def __init__(self, device, physical_to_logical_map, *, top_k=TOP_K, mode="resident_v1"):
        import torch
        if top_k != TOP_K or mode not in ("resident_v1", "native_hash", "static_home"):
            raise ValueError("Only top8 resident_v1/native_hash supported")
        validate_map(physical_to_logical_map)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("ResidentMapper requires CUDA")
        self.mode = mode

    def __call__(self, logical_ids):
        import torch
        from vllm.triton_utils import triton
        from full_serving import resident_gpu as gpu
        if (logical_ids.ndim != 2 or logical_ids.shape[1] != TOP_K
                or logical_ids.shape[0] > MAX_GLOBAL_ROWS
                or logical_ids.dtype not in (torch.int32, torch.int64)
                or not logical_ids.is_contiguous()
                or logical_ids.device != self.device):
            raise ValueError("Unsupported full-model routing shape/dtype/device/layout")
        n = logical_ids.numel()
        out = torch.empty_like(logical_ids)
        if n == 0:
            return out
        if self.mode == "static_home":
            from full_serving.boundary_static_gpu import static_home_map
            static_home_map[(triton.cdiv(n,1024),)](logical_ids,out,n,BLOCK=1024)
        elif self.mode == "native_hash":
            gpu.native_hash_map[(triton.cdiv(n, 1024),)](logical_ids, out, n, BLOCK=1024)
        elif n <= 4096:
            gpu.fused_resident_map[(1,)](logical_ids, out, n, BLOCK=triton.next_power_of_2(max(n, 128)), num_warps=4)
        else:
            blocks = triton.cdiv(n, 4096)
            partial = torch.empty((blocks, LOGICAL), dtype=torch.int32, device=logical_ids.device)
            homes = torch.empty(LOGICAL, dtype=torch.int32, device=logical_ids.device)
            gpu.partial_histogram[(blocks,)](logical_ids, partial, n, BLOCK=4096)
            gpu.plan_from_partial[(1,)](partial, homes, blocks, PARTIALS=triton.next_power_of_2(blocks), num_warps=4)
            gpu.apply_homes[(triton.cdiv(n, 1024),)](logical_ids, homes, out, n, BLOCK=1024)
        return out
