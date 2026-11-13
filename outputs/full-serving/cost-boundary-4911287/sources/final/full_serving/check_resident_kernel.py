"""GPU development gate: independent oracle, native hash parity and graph replay.

This uses synthetic routing only and cannot certify full-model numerical output.
It runs on one allocated GPU and never loads final content or alters vLLM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from full_serving.resident_policy import ResidentMapper, balanced_map, cpu_reference
    from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(4711110)
    mapping = np.asarray(balanced_map())
    logical_map = np.full((128, 2), -1, dtype=np.int64)
    counts = np.bincount(mapping, minlength=128)
    for e in range(128):
        logical_map[e, :counts[e]] = np.flatnonzero(mapping == e)
    gpu_map = torch.tensor(logical_map, device=device)
    gpu_counts = torch.tensor(counts, device=device)
    load = torch.zeros(160, dtype=torch.int32, device=device)
    enabled = torch.zeros((), dtype=torch.bool, device=device)
    cases, timings = [], []
    start = time.monotonic()
    for dtype in (torch.int32, torch.int64):
        for size in (0, 1, 3, 8, 17, 128, 512, 513, 2048, 8192, 65536):
            rng = np.random.default_rng(471 + size)
            values = rng.integers(0, 128, size=(size, 8), dtype=np.int64)
            # Masks have holes and padded tails, not one global real-token prefix.
            values[::3] = -1
            if size:
                values[-1] = -1
            ids = torch.tensor(values, device=device, dtype=dtype)
            for mode in ("resident_v1", "native_hash"):
                mapper = ResidentMapper(device, balanced_map(), mode=mode)
                out = mapper(ids)
                expected = cpu_reference(values, mode=mode)["physical_ids"]
                assert np.array_equal(out.cpu().numpy(), expected), (size, dtype, mode, "oracle")
                if mode == "native_hash":
                    native = eplb_map_to_physical_and_record(ids, load, gpu_map, gpu_counts, enabled)
                    assert torch.equal(out, native), (size, dtype, "native parity")
                # Capture after compilation; change IDs in place and replay twice.
                if size:
                    for _ in range(2):
                        mapper(ids)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = mapper(ids)
                    for variant in ("all_padding", "new_hotspot"):
                        modified = np.full_like(values, -1)
                        if variant == "new_hotspot":
                            modified[1::2] = np.array([0, 1, 2, 3, 112, 113, 114, 115])
                        ids.copy_(torch.tensor(modified, device=device, dtype=dtype))
                        graph.replay()
                        expected = cpu_reference(modified, mode=mode)["physical_ids"]
                        assert np.array_equal(captured.cpu().numpy(), expected), (size, dtype, mode, variant)
                    ids.copy_(torch.tensor(values, device=device, dtype=dtype))
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(100):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timings.append({"rows": size, "dtype": str(dtype), "mode": mode,
                                    "graph_replay_mean_us": begin.elapsed_time(end) * 10})
                cases.append({"rows": size, "dtype": str(dtype), "mode": mode,
                              "cpu_oracle_exact": True, "padding_and_conservation": True,
                              "changed_input_graph_replays": 2 if size else 0,
                              "native_hash_exact": True if mode == "native_hash" else None})
    assert not torch.any(load).item()
    result = {"schema": "full_resident_kernel_gate_v1", "passed": True,
              "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "elapsed_s": time.monotonic() - start,
              "cases": cases, "timings": timings, "native_recording_unchanged_zero": True,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ("resident_policy.py", "resident_gpu.py", "check_resident_kernel.py")},
              "scope": "Synthetic routing/oracle/native-hash/capture correctness. No model weights or final data; timings exclude planning Python and full inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
