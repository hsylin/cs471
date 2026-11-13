"""Untimed exact native mapping/recording checks on actual loaded metadata."""


def check_native_mapping(ms):
    import numpy as np
    import torch
    from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
    rng=np.random.default_rng(4711112)
    device=ms.physical_to_logical_map.device
    mapping=ms.physical_to_logical_map[0].cpu().numpy()
    cases=0
    for rows,real in ((0,0),(1,1),(8,0),(8,3),(17,17),(257,253)):
        ids=rng.integers(0,128,size=(rows,8),dtype=np.int32)
        if rows:ids[real:]=-1
        inp=torch.tensor(ids,device=device)
        expected=np.full_like(ids,-1)
        for i in range(rows):
            for j in range(8):
                e=int(ids[i,j])
                if e>=0:
                    slots=np.flatnonzero(mapping==e)
                    expected[i,j]=slots[((i*2654435769)&0xffffffff)%len(slots)]
        for flag in (False,True):
            counts=torch.zeros(mapping.size,dtype=torch.int32,device=device)
            result=eplb_map_to_physical_and_record(inp,counts,ms.logical_to_physical_map[0],ms.logical_replica_count[0],torch.tensor(flag,device=device),torch.tensor(real,dtype=torch.int32,device=device))
            np.testing.assert_array_equal(result.cpu().numpy(),expected)
            wanted=np.bincount(expected[:real][expected[:real]>=0],minlength=mapping.size) if flag else np.zeros(mapping.size,dtype=np.int64)
            np.testing.assert_array_equal(counts.cpu().numpy(),wanted)
            cases+=1
    return {'status':'passed','cases':cases,'padding_preserved':True,'physical_mapping_exact':True,'recorded_counts_exact':True,
            'scope':'Scratch counters using actual model metadata; no model weights or live counters modified; not full-model numerical equivalence.'}
