"""GPU recorder for an explicitly precomputed physical assignment.

This is an optimistic diagnostic, not an online selector. The same adapter is
used for precomputed native and alternative assignments, so their comparison
isolates the assignment rather than removing the native mapping kernel.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_and_record(physical, output, load, record, real_tokens,
                     N: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(physical + offsets, offsets < N, other=-1)
    tl.store(output + offsets, ids, offsets < N)
    enabled = tl.load(record) != 0
    real = tl.load(real_tokens)
    valid = (offsets < N) & (offsets < real * TOPK) & (ids >= 0) & (ids < SLOTS)
    tl.atomic_add(load + tl.maximum(ids, 0), 1, mask=valid & enabled)


class PrecomputedMapper:
    def __init__(self, torch, physical_ids):
        self.torch = torch
        self.physical_ids = physical_ids
        self.raw_index = 0

    def __call__(self, logical_ids, load, logical_map, replica_count, record, real_tokens):
        physical = self.physical_ids[self.raw_index]
        if physical.shape != logical_ids.shape or physical.dtype != logical_ids.dtype:
            raise ValueError('Precomputed physical-route shape/dtype mismatch')
        output = self.torch.empty_like(logical_ids)
        _copy_and_record[(triton.cdiv(logical_ids.numel(), 256),)](
            physical, output, load, record, real_tokens,
            N=logical_ids.numel(), TOPK=logical_ids.shape[-1], SLOTS=load.numel(), BLOCK=256)
        return output
