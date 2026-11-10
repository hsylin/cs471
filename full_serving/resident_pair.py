"""Parallel pair planner, same placement and exact assignment as resident_v1."""
from full_serving.resident_policy import TOP_K, LOGICAL, MAX_GLOBAL_ROWS, validate_map

class PairMapper:
    """Graph-capturable, current-stream mapper with per-call output storage."""
    def __init__(self, device, physical_to_logical_map, *, top_k=TOP_K, mode="resident_v1"):
        import torch
        if top_k != TOP_K or mode not in ("resident_v1", "native_hash"):
            raise ValueError("Only top8 resident_v1/native_hash supported")
        validate_map(physical_to_logical_map)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("ResidentMapper requires CUDA")
        self.mode = mode

    def __call__(self, logical_ids):
        import torch
        from vllm.triton_utils import triton
        from full_serving import resident_gpu as gpu
        from full_serving import resident_pair_gpu as pair_gpu
        if (logical_ids.ndim != 2 or logical_ids.shape[1] != TOP_K
                or logical_ids.shape[0] > MAX_GLOBAL_ROWS
                or logical_ids.dtype not in (torch.int32, torch.int64)
                or not logical_ids.is_contiguous()
                or logical_ids.device != self.device):
            raise ValueError("Unsupported full-model routing shape/dtype/device/layout")
        n = logical_ids.numel()
        out = torch.empty_like(logical_ids)
        if n == 0:
            return out
        if self.mode == "native_hash":
            gpu.native_hash_map[(triton.cdiv(n, 1024),)](logical_ids, out, n, BLOCK=1024)
        elif n <= 4096:
            pair_gpu.fused_resident_map[(1,)](logical_ids, out, n, BLOCK=triton.next_power_of_2(max(n, 128)), num_warps=4)
        else:
            blocks = triton.cdiv(n, 4096)
            partial = torch.empty((blocks, LOGICAL), dtype=torch.int32, device=logical_ids.device)
            homes = torch.empty(LOGICAL, dtype=torch.int32, device=logical_ids.device)
            gpu.partial_histogram[(blocks,)](logical_ids, partial, n, BLOCK=4096)
            pair_gpu.plan_from_partial[(1,)](partial, homes, blocks, PARTIALS=triton.next_power_of_2(blocks), num_warps=4)
            gpu.apply_homes[(triton.cdiv(n, 1024),)](logical_ids, homes, out, n, BLOCK=1024)
        return out
