"""One-launch B128 implementation of the frozen resident-home policy.

This is a static-placement dispatcher, not stock EPLB-off. No recording,
planning, history or movement is performed. GPU imports are deferred until
construction, leaving the independent scalar reference usable on CPU.
"""
from __future__ import annotations

import numpy as np


POLICY_ID = 'current_batch_one_home_lpt_active_caps_v1'
IMPLEMENTATION_ID = 'b128_single_cta_static_resident_v1'


def fixed_map():
    return np.r_[np.arange(128),np.arange(32)].astype(np.int64)


def validate_map(mapping):
    value=np.asarray(mapping)
    if (value.shape!=(160,) or not np.issubdtype(value.dtype,np.integer)
            or not np.array_equal(value,fixed_map())):
        raise ValueError('Fused selector requires the exact frozen160-slot map')
    value=value.astype(np.int64,copy=True)
    value.setflags(write=False)
    return value


def cpu_reference(logical_ids,physical_to_logical_map,*,real_tokens=None):
    """Scalar policy specification independent of the GPU implementation.

    Real rows must contain valid experts. Invalid padded IDs map to -1; valid
    padded IDs retain the chosen home but contribute nothing to the histogram.
    The policy is the existing greedy heuristic, not a runtime optimum.
    """
    mapping=validate_map(physical_to_logical_map)
    ids=np.asarray(logical_ids)
    if ids.ndim!=2 or ids.shape[1]<1 or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('Expected an integer [tokens,topk] array')
    real=len(ids) if real_tokens is None else real_tokens
    if isinstance(real,(bool,np.bool_)) or not isinstance(real,(int,np.integer)) or not 0<=real<=len(ids):
        raise ValueError('Invalid real-token count')
    real=int(real)
    if np.any(ids[:real]<0) or np.any(ids[:real]>=128):
        raise ValueError('Invalid logical ID in a real row')
    counts=np.zeros(128,np.int64)
    for value in ids[:real].flat:
        counts[int(value)]+=1
    active=[sum(counts[e]>0 for e in range(32,80)),sum(counts[e]>0 for e in range(80,128))]
    fixed_active=active.copy()
    loads=[sum(int(counts[e]) for e in range(32,80)),sum(int(counts[e]) for e in range(80,128))]
    shared=[e for e in range(32) if counts[e]>0]
    target=max(active[0],active[1],(sum(active)+len(shared)+1)//2)
    homes=np.arange(128,dtype=np.int64)
    for expert in sorted(shared,key=lambda e:(-int(counts[e]),e)):
        available=[rank for rank in (0,1) if active[rank]<target]
        if not available:
            raise AssertionError('Active-copy capacity exhausted')
        rank=min(available,key=lambda rank:(loads[rank],rank))
        homes[expert]=expert if rank==0 else expert+128
        active[rank]+=1
        loads[rank]+=int(counts[expert])
    physical=np.full(ids.shape,-1,np.int64)
    for index in np.ndindex(ids.shape):
        expert=int(ids[index])
        if 0<=expert<128:
            physical[index]=homes[expert]
    slot_counts=np.bincount(physical[:real].ravel(),minlength=160)
    if not np.array_equal(mapping[physical[:real]],ids[:real]):
        raise AssertionError('Physical assignment lost logical identity')
    if active!=np.count_nonzero(slot_counts.reshape(2,80),axis=1).tolist():
        raise AssertionError('Active-copy accounting mismatch')
    summary=np.asarray([target,*fixed_active,len(shared),*loads,*active],np.int64)
    for array in (physical,homes,counts,summary,slot_counts):
        array.setflags(write=False)
    return dict(physical_ids=physical,homes=homes,logical_counts=counts,
                summary=summary,physical_counts=slot_counts,real_tokens=real,
                runtime_optimality_claimed=False,policy_id=POLICY_ID)


def _cpu(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


class FusedMapper:
    """One CTA, one launch, fixed map and one sequential CUDA stream.

    Output storage is reused: consume it on the constructing inference stream
    before the next call, or explicitly clone outside timing if retaining it.
    Multi-stream/concurrent calls and changing placement are unsupported.
    verify_context/verify_metadata are untimed pre/post checks; call-time checks
    cover only shape/layout/device/type, with no host tensor-value readback.
    """
    def __init__(self,torch,physical_to_logical_map,batch_tokens=128,top_k=8):
        if batch_tokens!=128 or top_k!=8:
            raise ValueError('Fused prototype supports exactly B128/top8')
        self.physical_to_logical_map=validate_map(physical_to_logical_map)
        from .fused_gpu import fused_home_map
        self.torch=torch
        self.device=torch.device('cuda',torch.cuda.current_device())
        self.stream_handle=int(torch.cuda.current_stream(self.device).cuda_stream)
        self.batch_tokens,self.top_k=128,8
        self.kernel=fused_home_map
        self.real_tokens=torch.tensor(128,dtype=torch.int32,device=self.device)
        self.homes=torch.empty(128,dtype=torch.int32,device=self.device)
        self.logical_counts=torch.empty(128,dtype=torch.int32,device=self.device)
        self.summary=torch.empty(8,dtype=torch.int32,device=self.device)
        self._output_i32=torch.empty((128,8),dtype=torch.int32,device=self.device)
        self._output_i64=torch.empty((128,8),dtype=torch.int64,device=self.device)
        self.last_physical_ids=None
        self.calls=0

    def verify_context(self):
        if (self.torch.cuda.current_device()!=self.device.index or
                int(self.torch.cuda.current_stream(self.device).cuda_stream)!=self.stream_handle):
            raise ValueError('Fused output buffer requires the constructing CUDA device/stream')
        return dict(device=str(self.device),stream_handle=self.stream_handle,
                    sequential_single_stream_required=True,reused_output_storage=True)

    def verify_metadata(self,logical_map,replica_count,physical_map=None):
        """Check actual native map contents outside timed calls."""
        mapping=validate_map(self.physical_to_logical_map if physical_map is None else _cpu(physical_map))
        logical,replicas=_cpu(logical_map),_cpu(replica_count)
        if (logical.ndim!=2 or logical.shape[0]!=128 or logical.shape[1]<2
                or replicas.shape!=(128,)
                or not np.issubdtype(logical.dtype,np.integer)
                or not np.issubdtype(replicas.dtype,np.integer)):
            raise ValueError('Wrong native static residency metadata shapes or dtypes')
        for expert in range(128):
            slots=np.flatnonzero(mapping==expert)
            if (replicas[expert]!=len(slots) or not np.array_equal(logical[expert,:len(slots)],slots)
                    or np.any(logical[expert,len(slots):]!=-1)):
                raise ValueError('Native residency changed; fused hard-coded homes would be stale')
        self.verify_context()
        return True

    def __call__(self,logical_ids,real_tokens=None):
        torch=self.torch
        if (tuple(logical_ids.shape)!=(128,8) or logical_ids.dtype not in (torch.int32,torch.int64)
                or not logical_ids.is_contiguous() or logical_ids.device!=self.device):
            raise ValueError('Fused selector input shape/layout/device/type changed')
        real=self.real_tokens if real_tokens is None else real_tokens
        if real_tokens is not None and (real.numel()!=1 or real.dtype!=torch.int32 or real.device!=self.device):
            raise ValueError('real_tokens must be an int32 scalar on this CUDA device')
        output=self._output_i32 if logical_ids.dtype==torch.int32 else self._output_i64
        self.kernel[(1,)](logical_ids,real,output,self.homes,self.logical_counts,self.summary,
                          N=1024,TOPK=8,num_warps=4)
        self.last_physical_ids=output
        self.calls+=1
        return output

    def snapshot_diagnostics(self):
        if self.last_physical_ids is None:
            raise RuntimeError('No fused invocation has completed')
        return {name:_cpu(value).copy() for name,value in
                [('physical_ids',self.last_physical_ids),('homes',self.homes),
                 ('logical_counts',self.logical_counts),('summary',self.summary)]} | {'calls':self.calls}

    def metadata(self):
        return dict(policy_id=POLICY_ID,implementation_id=IMPLEMENTATION_ID,
            kernels_per_call=1,ctas_per_call=1,batch_tokens=128,top_k=8,
            native_recording=False,history=False,cpu_readback_in_selection=False,
            output_lifetime='Consumed on constructing stream before next call; reused buffer',
            **self.verify_context())
