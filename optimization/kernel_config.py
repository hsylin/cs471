"""Bounded BF16 fused-MoE tile choices for a separate optimization experiment.

Uses the native process-global override only outside concurrent execution.
Neither this module nor its source attestation changes the frozen studies.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path


PINNED_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
SOURCE_SHA256 = {
    "model_executor/layers/fused_moe/__init__.py":
        "64a32665be6d5afcfbaf7a0915717267278b66b880b908b0263ea58f08e51b5e",
    "model_executor/layers/fused_moe/fused_moe.py":
        "21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7",
}


def _config(block_m, block_n, warps=4):
    return {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
            "num_warps": warps, "num_stages": 3}


CANDIDATES = {"DEFAULT": None,
              **{f"M{m}_N{n}_K64_W4_S3": _config(m, n)
                 for m in (16, 32, 64) for n in (64, 128)},
              "M128_N64_K64_W8_S3": _config(128, 64, 8)}


def config_fingerprint(config):
    """Digest of exact explicit settings; None denotes unchanged native lookup."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_config(config):
    if config is None:
        return
    expected = set(_config(16, 64))
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Explicit kernel config must contain exactly the seven declared tile keys")
    if any(type(v) is not int for v in config.values()):
        raise ValueError("Kernel configuration values must be integers")
    if config not in [value for value in CANDIDATES.values() if value is not None]:
        raise ValueError("Kernel configuration is outside the bounded declared candidate set")


@contextmanager
def kernel_override(config, *, native_module=None):
    """Restore the native override even if a candidate raises during execution.

Pinned native override_config lacks a try/finally around its yield. Exit it
normally from our finally block so restoration executes on both success and
failure. Original candidate exceptions still propagate to the caller. The
optional module injection is exclusively for CPU contract tests.
    """
    validate_config(config)
    native = native_module or importlib.import_module("vllm.model_executor.layers.fused_moe")
    previous = native.get_config()
    installed = None if config is None else dict(config)
    context = native.override_config(installed)
    context.__enter__()
    try:
        if native.get_config() != installed:
            raise AssertionError("Native fused-MoE override was not installed")
        yield
    finally:
        context.__exit__(None, None, None)
        if native.get_config() is not previous:
            raise AssertionError("Native fused-MoE override was not restored")


def verify_kernel_sources(vllm_root=None):
    """Attest the native selector/context API and GEMM implementation."""
    if vllm_root is None:
        vllm_root = Path(importlib.import_module("vllm").__file__).parent
    root = Path(vllm_root)
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise RuntimeError("Installed fused-MoE selector or implementation differs from pinned source")
    return {"commit": PINNED_COMMIT, "sha256": actual,
            "override_scope": "process global, single-threaded use; explicit exception-safe restoration",
            "candidate_table_sha256": config_fingerprint(CANDIDATES)}


def resolve_kernel_config(layer, batch_tokens, *, native_module=None, implementation_module=None):
    """Resolve and verify the same config selected by unquantized BF16 forward.

Call inside kernel_override, before warmup/timing. This experiment is deliberately
limited to its declared128/1024-token batches; a changed shape must be reviewed.
    """
    if batch_tokens not in (128, 1024) or str(layer.dtype) != "torch.bfloat16":
        raise ValueError("Kernel attestation requires declared128/1024-token BF16 inputs")
    native = native_module or importlib.import_module("vllm.model_executor.layers.fused_moe")
    implementation = implementation_module or importlib.import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe")
    # Match fused_experts_impl's configuration dtype and selector arguments.
    dtype = implementation._get_config_dtype_str(
        dtype=layer.dtype, use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False)
    if dtype is not None:
        raise AssertionError("Unquantized BF16 selected a quantized configuration namespace")
    actual = dict(implementation.try_get_optimal_moe_config(
        w1_shape=tuple(layer.w13.shape), w2_shape=tuple(layer.w2.shape),
        top_k=layer.top_k, dtype=dtype, M=batch_tokens, block_shape=None))
    override = native.get_config()
    expected = override if override is not None else (
        _config(64, 128, 4) if batch_tokens == 128 else _config(128, 128, 8))
    if actual != expected:
        raise AssertionError("Resolved kernel configuration differs from declared override/fallback")
    return actual
