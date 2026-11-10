"""Reconstruct dispatch work from actual native routes and committed maps.

This describes token/tile work, not measured GPU utilization or exclusive time.
The native mapping formula is verified against the pinned base_router source
already attested by StockRuntime. A commit after forward j first affects j+1.
"""
import numpy as np

from stock_comparison.inputs import array_digest

NATIVE_SOURCE = dict(
    commit="db9527a46873454610df6dbedf79a36d6bf1a7f6",
    path="model_executor/layers/fused_moe/router/base_router.py",
    sha256="c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf",
    token_hash_multiplier=2654435769,
)


def native_physical_ids(logical_ids, mapping):
    ids = np.asarray(logical_ids)
    mapping = np.asarray(mapping)
    if ids.shape != (128, 8) or ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= 128).any():
        raise ValueError("Expected actual logical IDs[128,8]")
    if mapping.shape not in ((128,), (160,)) or mapping.dtype.kind not in "iu" or set(mapping.tolist()) != set(range(128)):
        raise ValueError("Invalid committed physical-to-logical map")
    inverse = [np.flatnonzero(mapping == expert) for expert in range(128)]
    token_hash = (np.arange(128, dtype=np.uint64) * np.uint64(2654435769)) & np.uint64(0xffffffff)
    result = np.empty_like(ids, dtype=np.int64)
    for expert, copies in enumerate(inverse):
        row, column = np.nonzero(ids == expert)
        result[row, column] = copies[(token_hash[row] % len(copies)).astype(np.int64)]
    if not np.array_equal(mapping[result], ids):
        raise AssertionError("Native reconstruction changed logical routes")
    return result


def rank_work(physical_ids, *, physical_slots):
    ids = np.asarray(physical_ids)
    if physical_slots not in (128, 160) or ids.shape != (128, 8) or (ids < 0).any() or (ids >= physical_slots).any():
        raise ValueError("Invalid physical routes or geometry")
    counts = np.bincount(ids.ravel(), minlength=physical_slots).reshape(2, physical_slots // 2)
    tokens = counts.sum(1)
    active = (counts > 0).sum(1)
    padded = (((counts + 63) // 64) * 64).sum(1)
    return dict(per_rank_tokens=tokens.tolist(), per_rank_active_copies=active.tolist(),
        per_rank_padded_rows=padded.tolist(), per_rank_m_tiles=(padded // 64).tolist(),
        max_to_mean_token_ratio=float(tokens.max() / tokens.mean()),
        max_to_mean_padded_ratio=float(padded.max() / padded.mean()),
        physical_counts=counts.ravel().tolist())


def reconstruct_work(logical_ids, arm, source_map, async_events=(), *, include_counts=False):
    ids = np.asarray(logical_ids)
    if ids.ndim != 3 or ids.shape[1:] != (128, 8):
        raise ValueError("Expected timed logical routing jobs[128,8], excluding warmup")
    if arm not in ("stockoff", "stockon32", "fix"):
        raise ValueError("Unknown arm")
    mapping = np.asarray(source_map, dtype=np.int64)
    if arm == "stockoff" and not np.array_equal(mapping, np.arange(128)):
        raise ValueError("Off must use the original128 physical experts")
    events = list(async_events)
    if arm != "stockon32" and events:
        raise ValueError("Static arm unexpectedly has migration events")
    if any(e["commit_step"] is None or e["committed_map"] is None for e in events):
        raise ValueError("Uncommitted native event")
    commits = {}
    for event in events:
        if not event["commit_in_completion_tail"]:
            step = int(event["commit_step"]) + 1
            if step in commits:
                raise ValueError("Multiple native commits before one next forward")
            commits[step] = np.asarray(event["committed_map"], dtype=np.int64)
    result = []
    for job, logical in enumerate(ids):
        if job in commits:
            mapping = commits[job]
        if arm == "stockoff":
            physical = logical
        elif arm == "fix":
            from resident_dispatch.online import cpu_greedy_assignment
            physical = cpu_greedy_assignment(logical, mapping).physical_ids
        else:
            physical = native_physical_ids(logical, mapping)
        row = rank_work(physical, physical_slots=len(mapping))
        if not include_counts:
            row.pop("physical_counts")
        row.update(job=job, map_sha256=array_digest(mapping), physical_ids_sha256=array_digest(physical))
        result.append(row)
    return dict(schema="sequence_reconstructed_dispatch_work_v1", arm=arm,
        jobs=len(result), records=result, native_mapping_source=NATIVE_SOURCE,
        scope="CPU reconstruction from actual native gate/top-k routing and logged committed maps; not a GPU timing measurement",
        commit_semantics="Commit after job j affects forward j+1; terminal-tail commits affect no admitted job")


def routing_summary(logical_ids):
    ids = np.asarray(logical_ids)
    if ids.ndim != 3 or ids.shape[1:] != (128, 8) or (ids < 0).any() or (ids >= 128).any():
        raise ValueError("Expected valid actual native route jobs")
    counts = np.stack([np.bincount(row.ravel(), minlength=128) for row in ids])
    proportions = counts / 1024
    adjacent_tv = .5 * np.abs(np.diff(proportions, axis=0)).sum(1)
    windows = {}
    for size in (16, 1000):
        n = len(counts) // size
        grouped = counts[:n*size].reshape(n, size, 128).sum(1) / (1024 * size)
        tv = .5 * np.abs(np.diff(grouped, axis=0)).sum(1)
        windows[str(size)] = dict(nonoverlapping_full_windows=n,
            consecutive_window_tv=tv.tolist(), excluded_terminal_jobs=len(counts)-n*size)
    return dict(jobs=len(ids), logical_route_sha256=array_digest(ids),
        mean_distinct_logical_experts_per_job=float((counts > 0).sum(1).mean()),
        mean_largest_expert_fraction=float(proportions.max(1).mean()),
        largest_expert_fraction_max=float(proportions.max()),
        adjacent_job_tv_mean=float(adjacent_tv.mean()) if len(adjacent_tv) else None,
        adjacent_job_tv_p95=float(np.quantile(adjacent_tv,.95)) if len(adjacent_tv) else None,
        total_logical_counts=counts.sum(0).tolist(), windows=windows,
        scope="Actual native logical routing; fixed128 tokens/top8 so absolute assignments per job remain1024")
