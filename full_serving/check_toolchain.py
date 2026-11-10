"""Exercise native FlashInfer sampling on one allocated GPU, without a model."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != "4882284":
        raise RuntimeError("This diagnostic is restricted to existing allocation 4882284")
    toolkit = Path(os.environ["CUDA_HOME"])
    manifest = toolkit / "install-manifest.json"
    nvcc = toolkit / "bin/nvcc"
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "V12.9.86" not in version:
        raise RuntimeError("Expected the pinned CUDA compiler")
    import torch
    import flashinfer
    from flashinfer.sampling import get_sampling_module, top_k_mask_logits

    torch.cuda.set_device(0)
    start = time.monotonic()
    module = get_sampling_module()
    module_seconds = time.monotonic() - start
    # Distinct values avoid unspecified tie selection. Exercise scalar and
    # per-row top-k paths, including k=1 and identity k=vocabulary size.
    logits = torch.arange(4096, dtype=torch.float32, device="cuda").repeat(4, 1)
    cases = []
    for top_k in (5, torch.tensor([1, 5, 32, 4096], dtype=torch.int32, device="cuda")):
        actual = top_k_mask_logits(logits, top_k)
        ks = [top_k] * 4 if isinstance(top_k, int) else top_k.cpu().tolist()
        expected = torch.stack([row.masked_fill(row < 4096 - k, -float("inf"))
                                for row, k in zip(logits, ks)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cases.append({"top_k": ks, "exact_mask_match": True})
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1, "status": "passed", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ.get("SLURM_STEP_ID"),
        "rank": int(os.environ.get("SLURM_PROCID", "0")), "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count(), "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)), "gpu_total_memory": props.total_memory,
        "cuda_home": str(toolkit), "nvcc_version": version,
        "nvcc_sha256": hashlib.sha256(nvcc.read_bytes()).hexdigest(),
        "install_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "flashinfer_version": flashinfer.__version__, "sampling_module_type": type(module).__name__,
        "get_sampling_module_elapsed_s": module_seconds, "checks": cases,
        "scope": "Native FlashInfer module compilation/loading and exact top-k masks; no full-model serving or speedup measurement",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (socket.gethostname() + ".json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
