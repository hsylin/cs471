"""Bit-preserving coalesced AgRs payload; component candidate, not yet admitted.

The pinned BF16/FP32/int32 geometry has 1040 int32 words per token. Views
reinterpret bytes; they do not quantize hidden states, routing weights or IDs.
Native grouped gathering is the reference. Both packing kernels are charged.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _pack(H, W, I, P, N:tl.constexpr, BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//1040;col=x%1040;valid=row<N
    h=tl.load(H+row*1024+col,valid&(col<1024),other=0)
    w=tl.load(W+row*8+col-1024,valid&(col>=1024)&(col<1032),other=0)
    ids=tl.load(I+row*8+col-1032,valid&(col>=1032),other=0)
    tl.store(P+x,h|w|ids,valid)


@tr.jit
def _unpack(P,H,W,I,N:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=x//1040;col=x%1040;valid=row<N
    bits=tl.load(P+x,valid,other=0)
    tl.store(H+row*1024+col,bits,valid&(col<1024))
    tl.store(W+row*8+col-1024,bits,valid&(col>=1024)&(col<1032))
    tl.store(I+row*8+col-1032,bits,valid&(col>=1032))


def packed_gather(group,inputs,sizes):
    hidden,weights,ids=inputs;n=hidden.shape[0];total=sum(sizes)
    assert hidden.dtype==torch.bfloat16 and weights.dtype==torch.float32 and ids.dtype==torch.int32
    assert hidden.shape==(n,2048) and weights.shape==ids.shape==(n,8)
    assert all(t.is_contiguous() and t.device==hidden.device for t in inputs)
    assert len(sizes)==8 and n==sizes[group.rank_in_group]
    packed=torch.empty((n,1040),device=hidden.device,dtype=torch.int32)
    if n:_pack[(tr.cdiv(n*1040,256),)](hidden.view(torch.int32),weights.view(torch.int32),ids,packed,n,256)
    gathered=group.all_gatherv(packed,dim=0,sizes=sizes)
    h=torch.empty((total,2048),device=hidden.device,dtype=torch.bfloat16)
    w=torch.empty((total,8),device=hidden.device,dtype=torch.float32)
    i=torch.empty((total,8),device=hidden.device,dtype=torch.int32)
    if total:_unpack[(tr.cdiv(total*1040,256),)](gathered,h.view(torch.int32),w.view(torch.int32),i,total,256)
    return [h,w,i]
