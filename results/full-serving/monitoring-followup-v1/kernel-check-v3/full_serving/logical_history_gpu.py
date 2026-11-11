"""Exact logical demand windows, fused outside the captured model graph.

One CTA owns one layer, including reduction of duplicate physical copies into a logical ring row.
Counts are observed BEFORE native async commits. No sampling or clock changes.
"""
from vllm.triton_utils import triton, tl


@triton.jit(do_not_specialize=['seen'])
def update_history(Ring, Map, Counts, seen,
                   W: tl.constexpr, L: tl.constexpr, E: tl.constexpr,
                   P: tl.constexpr, DUMMY: tl.constexpr,
                   BLOCK: tl.constexpr):
    layer = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    logical = columns < E + 1
    row = Ring + ((seen % (2 * W)) * L + layer) * (E + 1)
    tl.store(row + columns, 0, logical)
    tl.debug_barrier()
    if not DUMMY:
        ids = tl.load(Map + layer * P + columns, columns < P, E)
        ids = tl.where(ids < 0, E, ids)
        values = tl.load(Counts + layer * P + columns, columns < P, 0).to(tl.int64)
        tl.atomic_add(row + ids, values, columns < P, sem='relaxed')


@triton.jit(do_not_specialize=['seen'])
def reduce_history(Ring, Output, seen, W: tl.constexpr, L: tl.constexpr,
                   E: tl.constexpr, ROWS: tl.constexpr, C: tl.constexpr):
    layer = tl.program_id(0)
    columns = tl.program_id(1) * C + tl.arange(0, C)
    half = tl.program_id(2)
    offsets = tl.arange(0, ROWS)
    positions = (seen + half * W + offsets) % (2 * W)
    values = tl.load(Ring + (positions[:, None] * L + layer) * (E + 1) + columns[None, :],
                     (offsets[:, None] < W) & (columns[None, :] < E), 0)
    sums = tl.sum(values, axis=0)
    tl.store(Output + (half * L + layer) * E + columns, sums, columns < E)


class LogicalHistory:
    def __init__(self, device, *, window=1000, layers=48, experts=128, physical=160, rolling=False):
        import torch
        self.window, self.layers, self.experts, self.physical = window, layers, experts, physical
        if rolling:raise ValueError("Rolling prototype was rejected by the exact GPU oracle")
        self.ring = torch.zeros((2 * window, layers, experts + 1), dtype=torch.int64, device=device)
        self.seen = 0

    def update(self, mapping, counts, dummy=False):
        assert mapping.shape == counts.shape == (self.layers, self.physical)
        assert mapping.is_contiguous() and counts.is_contiguous()
        update_history[(self.layers,)](self.ring, mapping, counts,
                                     self.seen, self.window, self.layers, self.experts,
                                     self.physical, dummy,
                                     triton.next_power_of_2(max(self.physical, self.experts + 1)), num_warps=4)
        self.seen += 1

    def snapshot(self):
        import torch
        out = torch.empty((2, self.layers, self.experts), dtype=torch.int64, device=self.ring.device)
        reduce_history[(self.layers, triton.cdiv(self.experts, 8), 2)](
            self.ring, out, self.seen, self.window, self.layers, self.experts,
            triton.next_power_of_2(self.window), 8, num_warps=4)
        return out[0], out[1]
