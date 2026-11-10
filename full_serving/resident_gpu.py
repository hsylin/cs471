"""CUDA kernels for the fixed-map EP8 full-model port; no host synchronization."""
from vllm.triton_utils import tl, triton


@triton.jit
def _plan(counts):
    experts = tl.arange(0, 128)
    columns = tl.arange(0, 16)
    ranks = tl.arange(0, 8)
    rank_counts = tl.reshape(counts, (8, 16))
    fixed = columns[None, :] >= 4
    active = tl.sum((fixed & (rank_counts > 0)).to(tl.int32), 1)
    loads = tl.sum(tl.where(fixed, rank_counts, 0), 1)
    pair_ids = tl.arange(0, 4)
    pair_targets = tl.maximum(tl.max(tl.reshape(active, (4, 2)), 1),
                              (tl.sum((tl.reshape(counts, (4, 32)) > 0).to(tl.int32), 1) + 1) // 2)
    indices = tl.arange(0, 32)
    shared = (indices // 4) * 16 + indices % 4
    weights = tl.gather(counts, shared, 0)
    ordered = tl.sort(weights * 128 + 127 - shared, descending=True)
    steps = tl.sum((weights > 0).to(tl.int32), 0)
    homes = (experts // 16) * 20 + experts % 16
    for index in range(steps):
        key = tl.sum(tl.where(indices == index, ordered, 0), 0)
        weight, expert = key // 128, 127 - key % 128
        a, b = expert // 16, (expert // 16) ^ 1
        target = tl.sum(tl.where(pair_ids == expert // 32, pair_targets, 0), 0)
        aa = tl.sum(tl.where(ranks == a, active, 0), 0)
        ab = tl.sum(tl.where(ranks == b, active, 0), 0)
        la = tl.sum(tl.where(ranks == a, loads, 0), 0)
        lb = tl.sum(tl.where(ranks == b, loads, 0), 0)
        eligible_a, eligible_b = aa < target, ab < target
        load_tie = (la < lb) | ((la == lb) & (a < b))
        take_a = eligible_a & (~eligible_b | load_tie)
        rank = tl.where(take_a, a, b)
        slot = tl.where(take_a, a * 20 + expert % 16, b * 20 + 16 + expert % 16)
        homes = tl.where(experts == expert, slot, homes)
        active += (ranks == rank).to(tl.int32)
        loads += tl.where(ranks == rank, weight, 0)
    return homes


@triton.jit
def fused_resident_map(ids, output, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    logical = tl.load(ids + offsets, mask=offsets < n, other=-1).to(tl.int32)
    valid = (offsets < n) & (logical >= 0) & (logical < 128)
    counts = tl.histogram(logical, 128, mask=valid)
    homes = _plan(counts)
    physical = tl.gather(homes, tl.where(valid, logical, 0), 0)
    tl.store(output + offsets, tl.where(valid, physical, -1), mask=offsets < n)


@triton.jit
def partial_histogram(ids, partial, n, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(ids + offsets, mask=offsets < n, other=-1).to(tl.int32)
    counts = tl.histogram(logical, 128, mask=(offsets < n) & (logical >= 0) & (logical < 128))
    tl.store(partial + block * 128 + tl.arange(0, 128), counts)


@triton.jit
def plan_from_partial(partial, output_homes, blocks, PARTIALS: tl.constexpr):
    rows, columns = tl.arange(0, PARTIALS), tl.arange(0, 128)
    values = tl.load(partial + rows[:, None] * 128 + columns[None, :], mask=rows[:, None] < blocks, other=0)
    counts = tl.sum(values, 0)
    tl.store(output_homes + columns, _plan(counts))


@triton.jit
def apply_homes(ids, homes, output, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(ids + offsets, mask=offsets < n, other=-1).to(tl.int32)
    valid = (logical >= 0) & (logical < 128)
    home = tl.load(homes + tl.where(valid, logical, 0))
    tl.store(output + offsets, tl.where(valid, home, -1), mask=offsets < n)


@triton.jit
def native_hash_map(ids, output, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(ids + offsets, mask=offsets < n, other=-1).to(tl.int32)
    valid = (logical >= 0) & (logical < 128)
    expert = tl.where(valid, logical, 0)
    base = expert // 16 * 20 + expert % 16
    alternate = ((expert // 16) ^ 1) * 20 + 16 + expert % 16
    shared = expert % 16 < 4
    first = tl.where(shared, tl.minimum(base, alternate), base)
    second = tl.where(shared, tl.maximum(base, alternate), base)
    hashed = ((offsets // 8).to(tl.int64) * 2654435769) & 0xFFFFFFFF
    physical = tl.where((hashed % 2 == 1) & shared, second, first)
    tl.store(output + offsets, tl.where(valid, physical, -1), mask=offsets < n)
