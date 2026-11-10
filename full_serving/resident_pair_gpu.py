"""Equivalent resident planner: four independent replica pairs in parallel.

The original global LPT ordering never exchanges load across pairs. Sorting each
pair independently preserves its exact decision order, with eight rather than
up to32 serial iterations. Selection, tie-breaking and copy budget are unchanged.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def pair_plan(counts):
    pair = tl.arange(0, 4)
    column = tl.arange(0, 32)
    shared_column = tl.arange(0, 8)
    matrix = tl.reshape(counts, (4, 32))
    fixed = column % 16 >= 4
    left = column < 16
    active_a = tl.sum((fixed[None, :] & left[None, :] & (matrix > 0)).to(tl.int32), 1)
    active_b = tl.sum((fixed[None, :] & ~left[None, :] & (matrix > 0)).to(tl.int32), 1)
    load_a = tl.sum(tl.where(fixed[None, :] & left[None, :], matrix, 0), 1)
    load_b = tl.sum(tl.where(fixed[None, :] & ~left[None, :], matrix, 0), 1)
    target = tl.maximum(tl.maximum(active_a, active_b), (tl.sum((matrix > 0).to(tl.int32), 1) + 1) // 2)
    shared = shared_column // 4 * 16 + shared_column % 4
    weights = tl.gather(matrix, tl.broadcast_to(shared[None, :], (4, 8)), 1)
    ordered = tl.sort(weights * 32 + 31 - shared[None, :], descending=True, dim=1)
    experts = pair[:, None] * 32 + column[None, :]
    homes = experts // 16 * 20 + experts % 16
    # Eight shared experts per pair. Mask zero-demand assignments explicitly.
    for index in range(8):
        key = tl.sum(tl.where(shared_column[None, :] == index, ordered, 0), 1)
        weight, expert = key // 32, 31 - key % 32
        valid = weight > 0
        ea, eb = active_a < target, active_b < target
        take_a = ea & (~eb | (load_a <= load_b))
        rank = pair * 2 + tl.where(take_a, 0, 1)
        own_side = expert // 16
        slot = rank * 20 + tl.where(own_side == (rank % 2), expert % 16, 16 + expert % 16)
        homes = tl.where(valid[:, None] & (column[None, :] == expert[:, None]), slot[:, None], homes)
        active_a += (valid & take_a).to(tl.int32)
        active_b += (valid & ~take_a).to(tl.int32)
        load_a += tl.where(valid & take_a, weight, 0)
        load_b += tl.where(valid & ~take_a, weight, 0)
    return tl.reshape(homes, (128,))


@triton.jit
def fused_resident_map(ids, output, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    logical = tl.load(ids + offsets, mask=offsets < n, other=-1).to(tl.int32)
    valid = (offsets < n) & (logical >= 0) & (logical < 128)
    counts = tl.histogram(logical, 128, mask=valid)
    homes = pair_plan(counts)
    physical = tl.gather(homes, tl.where(valid, logical, 0), 0)
    tl.store(output + offsets, tl.where(valid, physical, -1), mask=offsets < n)


@triton.jit
def plan_from_partial(partial, output_homes, blocks, PARTIALS: tl.constexpr):
    rows, columns = tl.arange(0, PARTIALS), tl.arange(0, 128)
    values = tl.load(partial + rows[:, None] * 128 + columns[None, :], mask=rows[:, None] < blocks, other=0)
    counts = tl.sum(values, 0)
    tl.store(output_homes + columns, pair_plan(counts))
