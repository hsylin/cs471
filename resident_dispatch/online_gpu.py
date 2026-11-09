"""Fixed Triton kernels; imported lazily by the online adapter.

All choices are integer deterministic. No host readback, global histogram reset
or future routes are involved. The fixed layout is checked by OnlineMapper.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def histogram_partials(ids, partials, real_tokens,
                       N: tl.constexpr, TOPK: tl.constexpr, CHUNK: tl.constexpr):
    chunk = tl.program_id(0)
    offsets = chunk * CHUNK + tl.arange(0, CHUNK)
    values = tl.load(ids + offsets, offsets < N, other=-1).to(tl.int32)
    real = tl.load(real_tokens)
    valid = (offsets < N) & (offsets < real * TOPK) & (values >= 0) & (values < 128)
    counts = tl.histogram(values, 128, mask=valid)
    tl.store(partials + chunk * 128 + tl.arange(0, 128), counts)


@triton.jit
def choose_homes(partials, homes, logical_counts, summary, PARTIAL_ROWS: tl.constexpr):
    experts = tl.arange(0, 128)
    rows = tl.arange(0, PARTIAL_ROWS)
    counts = tl.sum(tl.load(partials + rows[:, None] * 128 + experts[None, :]), axis=0)
    tl.store(logical_counts + experts, counts)
    owner0 = (experts >= 32) & (experts < 80)
    owner1 = experts >= 80
    f0 = tl.sum((owner0 & (counts > 0)).to(tl.int32), axis=0)
    f1 = tl.sum((owner1 & (counts > 0)).to(tl.int32), axis=0)
    d = tl.sum(((experts < 32) & (counts > 0)).to(tl.int32), axis=0)
    optimum = tl.maximum(tl.maximum(f0, f1), (f0 + f1 + d + 1) // 2)
    cap0, cap1 = optimum - f0, optimum - f1
    tokens0 = tl.sum(tl.where(owner0, counts, 0), axis=0)
    tokens1 = tl.sum(tl.where(owner1, counts, 0), axis=0)
    shared_ids = tl.arange(0, 32)
    shared_counts = tl.sum(tl.load(partials + rows[:, None] * 128 + shared_ids[None, :]), axis=0)
    # Unique packed keys implement count descending, then expert ID ascending.
    keys = shared_counts * 128 + (127 - shared_ids)
    sorted_keys = tl.sort(keys, dim=0, descending=True)
    home_values = experts
    for index in range(32):
        key = tl.sum(tl.where(shared_ids == index, sorted_keys, 0), axis=0)
        weight = key // 128
        expert = 127 - key % 128
        active = weight > 0
        choose0 = (cap0 > 0) & ((cap1 <= 0) | (tokens0 <= tokens1))
        take0 = active & choose0
        take1 = active & ~choose0
        home = tl.where(choose0, expert, 128 + expert)
        home_values = tl.where(active & (experts == expert), home, home_values)
        tokens0 += tl.where(take0, weight, 0)
        tokens1 += tl.where(take1, weight, 0)
        cap0 -= take0.to(tl.int32)
        cap1 -= take1.to(tl.int32)
    tl.store(homes + experts, home_values)
    # One final vector store avoids conflicting scalar/vector writes to homes.
    diag = tl.arange(0, 8)
    values = tl.where(diag == 0, optimum,
             tl.where(diag == 1, f0,
             tl.where(diag == 2, f1,
             tl.where(diag == 3, d,
             tl.where(diag == 4, tokens0,
             tl.where(diag == 5, tokens1,
             tl.where(diag == 6, optimum - cap0, optimum - cap1)))))))
    tl.store(summary + diag, values)


@triton.jit
def map_and_record(ids, homes, output, load, record, real_tokens,
                   N: tl.constexpr, TOPK: tl.constexpr, CHUNK: tl.constexpr):
    offsets = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    logical = tl.load(ids + offsets, offsets < N, other=-1).to(tl.int32)
    eligible = (offsets < N) & (logical >= 0) & (logical < 128)
    safe = tl.where(eligible, logical, 0)
    physical = tl.load(homes + safe, mask=eligible, other=-1)
    tl.store(output + offsets, physical, mask=offsets < N)
    enabled = tl.load(record) != 0
    real = tl.load(real_tokens)
    valid = eligible & (offsets < real * TOPK) & (physical >= 0) & (physical < 160)
    tl.atomic_add(load + tl.maximum(physical, 0), 1, mask=valid & enabled)
