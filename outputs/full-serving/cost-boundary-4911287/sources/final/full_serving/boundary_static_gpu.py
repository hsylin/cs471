"""Diagnostic static-home mapper for the identical balanced32 placement."""
from vllm.triton_utils import triton,tl

@triton.jit
def static_home_map(ids,out,n:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 expert=tl.load(ids+i,i<n,other=-1)
 physical=(expert//16)*20+expert%16
 tl.store(out+i,tl.where(expert>=0,physical,-1),i<n)
