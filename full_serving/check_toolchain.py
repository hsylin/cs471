"""Exercise native FlashInfer sampling on one allocated GPU, without a model."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def verified_workspace():
    node = os.environ.get("SLURMD_NODENAME", socket.gethostname().split(".")[0])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", node) or node != socket.gethostname().split(".")[0]:
        raise RuntimeError("FlashInfer workspace node must identify this host")
    expected = (Path(os.environ["SKB_ROOT"]) / "cache/flashinfer-jit"
                / os.environ["SLURM_JOB_ID"] / node).resolve()
    actual = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if not actual or Path(actual).resolve() != expected:
        raise RuntimeError("Set the isolated job/node FLASHINFER_WORKSPACE_BASE before importing FlashInfer")
    return node, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != "4882284":
        raise RuntimeError("This diagnostic is restricted to existing allocation 4882284")
    node, workspace = verified_workspace()
    toolkit = Path(os.environ["CUDA_HOME"])
    manifest = toolkit / "install-manifest.json"
    nvcc = toolkit / "bin/nvcc"
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "V12.9.86" not in version:
        raise RuntimeError("Expected the pinned CUDA compiler")
    import torch
    import flashinfer
    from flashinfer.sampling import (get_sampling_module, top_k_mask_logits,
                                     top_k_top_p_sampling_from_logits,
                                     top_p_sampling_from_probs)
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import get_tmpdir

    if jit_env.FLASHINFER_BASE_DIR.resolve() != workspace:
        raise RuntimeError("FlashInfer did not adopt the isolated workspace")

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
    sampled = top_k_top_p_sampling_from_logits(logits, 1, 0.75, seed=471110,
                                               offset=0, check_nan=True)
    torch.testing.assert_close(sampled, torch.full_like(sampled, 4095), rtol=0, atol=0)
    cases.append({"operation": "top_k_top_p_sampling_from_logits", "top_k": 1,
                  "top_p": 0.75, "exact_sample_match": True})
    probs = torch.zeros_like(logits)
    targets = torch.tensor([0, 17, 2048, 4095], dtype=torch.int32, device="cuda")
    probs.scatter_(1, targets.long().unsqueeze(1), 1.0)
    sampled = top_p_sampling_from_probs(probs, 0.9, seed=471110, offset=0, check_nan=True)
    torch.testing.assert_close(sampled, targets, rtol=0, atol=0)
    cases.append({"operation": "top_p_sampling_from_probs", "top_p": 0.9,
                  "distribution": "one-hot", "exact_sample_match": True})
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 2, "status": "passed", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "flashinfer_workspace_base": str(workspace),
        "flashinfer_jit_dir": str(jit_env.FLASHINFER_JIT_DIR.resolve()),
        "sampling_lock_path": str(get_tmpdir() / "sampling.lock"),
        "flashinfer_source_sha256": {name: hashlib.sha256((Path(flashinfer.__file__).parent / name).read_bytes()).hexdigest()
                                    for name in ("jit/env.py", "jit/core.py", "jit/cpp_ext.py", "sampling.py")},
        "sampling_module_sha256": {str(path.relative_to(jit_env.FLASHINFER_JIT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in jit_env.FLASHINFER_JIT_DIR.glob("sampling/*.so")},
        "get_sampling_module_elapsed_s": module_seconds, "checks": cases,
        "scope": "Native FlashInfer compilation/loading, exact top-k masks and deterministic top-k/top-p samples; no full-model serving or speedup measurement",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (node + ".json")
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(out)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
