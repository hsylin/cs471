import json
from pathlib import Path
import shlex
import struct
import subprocess
from types import SimpleNamespace

import pytest

from full_serving import preflight as p


def write_tensor(path, *, offsets=(0, 8), shape=(2, 2), payload=b"\x00" * 8):
    header = {"weight": {"dtype": "BF16", "shape": list(shape), "data_offsets": list(offsets)}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_header_and_dense_ranges(tmp_path):
    path = tmp_path / "shard.safetensors"
    write_tensor(path)
    tensors, size = p.inspect_safetensors(path)
    assert tensors["weight"]["shape"] == [2, 2]
    assert size == 8
    write_tensor(path, payload=b"\0" * 7)
    with pytest.raises(p.PreflightError, match="Truncated/trailing"):
        p.inspect_safetensors(path)


@pytest.mark.parametrize("offsets,shape,error", [
    ((0, 6), (2, 2), "Tensor size mismatch"),
    ((2, 10), (2, 2), "noncontiguous"),
    ((-1, 7), (2, 2), "Invalid offsets"),
    ((0, 8), (-2, -2), "Bad shape"),
])
def test_reject_invalid_tensor_layout(tmp_path, offsets, shape, error):
    path = tmp_path / "shard.safetensors"
    write_tensor(path, offsets=offsets, shape=shape)
    with pytest.raises(p.PreflightError, match=error):
        p.inspect_safetensors(path)


def small_index():
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        keys += [prefix + "mlp.gate.weight", prefix + "self_attn.q_proj.weight"]
        keys += [f"{prefix}mlp.experts.{expert}.{projection}_proj.weight"
                 for expert in range(2) for projection in ("down", "gate", "up")]
    return {"weight_map": dict.fromkeys(keys, "one.safetensors")}


def test_all_layers_and_projections_are_required():
    index = small_index()
    assert len(p.validate_weight_index(index, layers=2, experts=2)) == 19
    del index["weight_map"]["model.layers.1.mlp.experts.1.up_proj.weight"]
    with pytest.raises(p.PreflightError, match="layers.1.mlp.experts.1.up_proj"):
        p.validate_weight_index(index, layers=2, experts=2)


def test_index_does_not_escape_snapshot():
    index = small_index()
    index["weight_map"]["lm_head.weight"] = "../escape.safetensors"
    with pytest.raises(p.PreflightError, match="Unsafe shard path"):
        p.validate_weight_index(index, layers=2, experts=2)


def cached_fixture(tmp_path, monkeypatch):
    root = tmp_path / p.MODEL_REVISION
    root.mkdir()
    asset, shard = root / "config.json", root / "one.safetensors"
    asset.write_text('{}\n')
    write_tensor(shard)
    monkeypatch.setattr(p, "EXPECTED_ASSETS", {asset.name: p.sha256(asset)})
    monkeypatch.setattr(p, "EXPECTED_SHARDS", {shard.name: {"sha256": p.sha256(shard), "bytes": shard.stat().st_size}})
    manifest = {"schema": 1, "complete": True, "kind": "full_checkpoint_sha256",
                "snapshot": str(root), "model_revision": p.MODEL_REVISION,
                "assets": {asset.name: {"sha256": p.sha256(asset), "stat": p.fingerprint(asset)}},
                "shards": {shard.name: {"sha256": p.sha256(shard), "stat": p.fingerprint(shard)}}}
    return root, manifest


def test_cached_verification_detects_same_size_content_change(tmp_path, monkeypatch):
    root, manifest = cached_fixture(tmp_path, monkeypatch)
    p.verify_cached_checkpoint(root, manifest)
    shard = root / "one.safetensors"
    contents = bytearray(shard.read_bytes())
    contents[-1] ^= 1
    shard.write_bytes(contents)
    with pytest.raises(p.PreflightError, match="File changed since full verification"):
        p.verify_cached_checkpoint(root, manifest)


@pytest.mark.parametrize("mutation", ["missing", "hash", "revision", "kind"])
def test_cached_manifest_must_represent_the_pinned_full_verification(tmp_path, monkeypatch, mutation):
    root, manifest = cached_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        manifest["shards"] = {}
    elif mutation == "hash":
        manifest["shards"]["one.safetensors"]["sha256"] = "0" * 64
    elif mutation == "revision":
        manifest["model_revision"] = "other"
    else:
        manifest["kind"] = "nonempty-files-only"
    with pytest.raises(p.PreflightError):
        p.verify_cached_checkpoint(root, manifest)


@pytest.mark.parametrize("arm", ["off", "on"])
@pytest.mark.parametrize("role", ["primary", "worker"])
def test_launcher_dry_run_pins_native_multinode_semantics(arm, role):
    script = Path(__file__).parents[1] / "full_serving/launch_native_draft.sh"
    result = subprocess.run(["bash", str(script), arm, role, "/snapshot with space", "10.1.2.3"],
                            text=True, capture_output=True, check=True)
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[:3] == ["vllm", "serve", "/snapshot with space"]
    def value(option):
        return command[command.index(option) + 1]
    assert value("--data-parallel-size") == "8"
    assert value("--data-parallel-size-local") == "4"
    assert value("--tensor-parallel-size") == "1"
    assert value("--data-parallel-backend") == "mp"
    assert value("--all2all-backend") == "allgather_reducescatter"
    assert ("--enable-eplb" in command) == (arm == "on")
    assert "--enforce-eager" not in command
    assert "--data-parallel-rank" not in command
    if role == "primary":
        assert value("--api-server-count") == "1"
        assert value("--host") == "0.0.0.0"
        assert "--data-parallel-start-rank" not in command
    else:
        assert "--headless" in command
        assert value("--data-parallel-start-rank") == "4"
        assert "--api-server-count" not in command


def test_toolchain_admission_fails_before_model_loading_on_missing_or_changed_compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with pytest.raises(p.PreflightError, match="install manifest"):
        p.verify_toolchain()
    names = ("bin/nvcc", "bin/ptxas", "include/cuda_runtime.h", "include/cub/cub.cuh", "include/curand.h")
    for name in names:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
    records = [{"path": name, "sha256": p.sha256(tmp_path / name)} for name in names]
    (tmp_path / "install-manifest.json").write_text(json.dumps({
        "status": "installed_verified", "release": "12.9.1", "prefix": str(tmp_path), "files": records}))
    calls = []
    monkeypatch.setattr(p.subprocess, "check_output", lambda command, **kwargs:
                        calls.append(command) or "Cuda compilation tools, release 12.9, V12.9.86")
    assert len(p.verify_toolchain()["critical_file_sha256"]) == 5
    assert calls == [[str(tmp_path / "bin/nvcc"), "--version"]]
    (tmp_path / "bin/nvcc").write_text("changed compiler")
    with pytest.raises(p.PreflightError, match="file changed"):
        p.verify_toolchain()
    assert len(calls) == 1


def test_sampler_admission_requires_same_step_isolated_intact_prebuild(tmp_path, monkeypatch):
    node = "kh074"
    workspace = tmp_path / "cache/flashinfer-jit" / p.JOB_ID / node
    jit_dir = workspace / ".cache/flashinfer/0.7.0.post1/90a/cached_ops"
    module = jit_dir / "sampling/sampling.so"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"compiled sampler")
    package_root = tmp_path / "package"
    sources = {}
    for name in ("jit/env.py", "jit/core.py", "jit/cpp_ext.py", "sampling.py"):
        path = package_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = p.sha256(path)
    monkeypatch.setattr(p.socket, "gethostname", lambda: node)
    monkeypatch.setattr(p.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(package_root / "__init__.py")))
    for key, value in {"SKB_ROOT": str(tmp_path), "SLURM_STEP_ID": "99", "SLURMD_NODENAME": node,
                       "FLASHINFER_WORKSPACE_BASE": str(workspace), "FULL_SERVING_RUN_DIR": str(tmp_path / "run")}.items():
        monkeypatch.setenv(key, value)
    toolchain = {"install_manifest_sha256": "manifest", "critical_file_sha256": {"bin/nvcc": "nvcc"}}
    report = {"schema_version": 2, "status": "passed", "job_id": p.JOB_ID, "step_id": "99", "hostname": node,
              "flashinfer_workspace_base": str(workspace), "flashinfer_jit_dir": str(jit_dir),
              "sampling_lock_path": str(jit_dir / "tmp/sampling.lock"),
              "diagnostic_source_sha256": p.sha256(Path(p.__file__).with_name("check_toolchain.py")),
              "install_manifest_sha256": "manifest", "nvcc_sha256": "nvcc", "flashinfer_source_sha256": sources,
              "sampling_module_sha256": {"sampling/sampling.so": p.sha256(module)},
              "get_sampling_module_elapsed_s": 1,
              "checks": [{"exact_mask_match": True}, {"exact_mask_match": True},
                         {"operation": "top_k_top_p_sampling_from_logits", "exact_sample_match": True},
                         {"operation": "top_p_sampling_from_probs", "exact_sample_match": True}]}
    report_path = tmp_path / "run/flashinfer-prebuild" / (node + ".json")
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report))
    assert p.verify_flashinfer_prebuild(toolchain)["workspace_base"] == str(workspace)
    monkeypatch.setenv("SLURM_STEP_ID", "100")
    with pytest.raises(p.PreflightError, match="Stale or wrong-node"):
        p.verify_flashinfer_prebuild(toolchain)
    monkeypatch.setenv("SLURM_STEP_ID", "99")
    module.write_bytes(b"unexpected replacement")
    with pytest.raises(p.PreflightError, match="module changed"):
        p.verify_flashinfer_prebuild(toolchain)
