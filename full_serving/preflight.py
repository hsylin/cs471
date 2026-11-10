"""Pinned full-checkpoint and per-launch hardware/source preflight.

The checkpoint command hashes all 61 GB once. Runtime checks reuse that record
only when every asset and shard has the same resolved path/stat fingerprint.
This is accidental-change detection, not a replacement for immutable storage.
Only the runtime subcommand imports CUDA packages; checkpoint checks use stdlib.
No preflight establishes successful collectives, numerical correctness, normal
EPLB commit completion, or an end-to-end performance result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
from datetime import datetime, timezone


class PreflightError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PreflightError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {"resolved_path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino}


def inspect_safetensors(path):
    """Validate dense tensor ranges without loading weights into RAM."""
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, f"Truncated shard: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        require(2 <= header_size <= 64 << 20, f"Invalid header size: {path.name}")
        raw = handle.read(header_size)
        require(len(raw) == header_size, f"Truncated header: {path.name}")
        header = json.loads(raw)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    require(bool(tensors), f"Empty shard: {path.name}")
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
             "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    ranges = []
    for name, entry in tensors.items():
        shape, dtype = entry["shape"], entry["dtype"]
        require(dtype in sizes, f"Unsupported checkpoint dtype: {dtype}")
        require(all(type(dim) is int and dim >= 0 for dim in shape), f"Bad shape: {name}")
        start, end = entry["data_offsets"]
        require(type(start) is int and type(end) is int and 0 <= start <= end,
                f"Invalid offsets: {name}")
        require(end - start == math.prod(shape) * sizes[dtype], f"Tensor size mismatch: {name}")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        require(start == cursor, f"Overlapping or noncontiguous data: {path.name}")
        cursor = end
    require(8 + header_size + cursor == path.stat().st_size, f"Truncated/trailing data: {path.name}")
    return tensors, cursor


def validate_weight_index(index, *, layers=48, experts=128):
    weights = index["weight_map"]
    require(isinstance(weights, dict) and weights, "Missing weight map")
    for name in weights.values():
        require(isinstance(name, str) and Path(name).name == name
                and name.endswith(".safetensors"), "Unsafe shard path")
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        required = [prefix + "mlp.gate.weight", prefix + "self_attn.q_proj.weight"]
        required += [f"{prefix}mlp.experts.{expert}.{projection}_proj.weight"
                     for expert in range(experts) for projection in ("down", "gate", "up")]
        for name in required:
            require(name in weights, f"Missing full-model weight: {name}")
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        require(name in weights, f"Missing full-model weight: {name}")
    return weights


def verify_checkpoint(snapshot):
    root = Path(snapshot).absolute()
    require(root.name == MODEL_REVISION, "Checkpoint path must use pinned snapshot revision")
    assets = {}
    for name, expected in EXPECTED_ASSETS.items():
        path = root / name
        before = fingerprint(path)
        digest = sha256(path)
        require(digest == expected, f"Pinned asset hash mismatch: {name}")
        require(fingerprint(path) == before, f"Asset changed during verification: {name}")
        assets[name] = {"sha256": digest, "stat": before}
    config = json.loads((root / "config.json").read_text())
    for key, expected in {"num_hidden_layers": 48, "num_experts": 128,
                          "num_experts_per_tok": 8, "hidden_size": 2048,
                          "moe_intermediate_size": 768}.items():
        require(config[key] == expected, f"Model geometry mismatch: {key}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weights = validate_weight_index(index)
    require(set(weights.values()) == set(EXPECTED_SHARDS), "Pinned shard set mismatch")
    shards, total_payload, tensor_count = {}, 0, 0
    for name, expected in EXPECTED_SHARDS.items():
        path = root / name
        before = fingerprint(path)
        require(before["bytes"] == expected["bytes"], f"Incomplete/wrong shard size: {name}")
        tensors, payload_bytes = inspect_safetensors(path)
        require(set(tensors) == {key for key, shard in weights.items() if shard == name},
                f"Index/header tensor mismatch: {name}")
        for tensor_name, tensor in tensors.items():
            if ".mlp.experts." in tensor_name:
                expected_shape = [2048, 768] if ".down_proj." in tensor_name else [768, 2048]
                require(tensor["shape"] == expected_shape and tensor["dtype"] == "BF16",
                        f"Expert shape/dtype mismatch: {tensor_name}")
        digest = sha256(path)
        require(digest == expected["sha256"], f"Pinned shard hash mismatch: {name}")
        require(fingerprint(path) == before, f"Shard changed during verification: {name}")
        shards[name] = {"sha256": digest, "stat": before, "tensor_count": len(tensors)}
        total_payload += payload_bytes
        tensor_count += len(tensors)
    require(total_payload == index["metadata"]["total_size"], "Total tensor byte count mismatch")
    return {"schema": 1, "kind": "full_checkpoint_sha256", "complete": True,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "snapshot": str(root), "model_revision": MODEL_REVISION,
            "assets": assets, "shards": shards, "tensor_count": tensor_count,
            "tensor_bytes": total_payload, "expert_layers": list(range(48)),
            "experts_per_layer": 128, "expert_projections": ["down", "gate", "up"]}


def verify_cached_checkpoint(snapshot, manifest):
    root = Path(snapshot).absolute()
    require(manifest.get("schema") == 1 and manifest.get("complete") is True
            and manifest.get("kind") == "full_checkpoint_sha256", "Not a full checkpoint verification")
    require(manifest.get("model_revision") == MODEL_REVISION and root.name == MODEL_REVISION,
            "Cached checkpoint revision mismatch")
    require(manifest.get("snapshot") == str(root), "Cached snapshot path mismatch")
    for kind, expected in (("assets", EXPECTED_ASSETS), ("shards", EXPECTED_SHARDS)):
        records = manifest.get(kind, {})
        require(set(records) == set(expected), f"Cached {kind} set mismatch")
        for name, reference in expected.items():
            digest = reference if kind == "assets" else reference["sha256"]
            require(records[name].get("sha256") == digest, f"Cached hash mismatch: {name}")
            # Linux st_dev identifies this host's mount, not the shared file.
            previous = {k: v for k, v in records[name].get("stat", {}).items() if k != "device"}
            current = {k: v for k, v in fingerprint(root / name).items() if k != "device"}
            require(previous == current,
                    f"File changed since full verification: {name}; rerun checkpoint preflight")


def verify_toolchain():
    require(bool(os.environ.get("CUDA_HOME")), "Set the verified private CUDA_HOME before serving")
    root = Path(os.environ["CUDA_HOME"]).resolve()
    manifest_path = root / "install-manifest.json"
    require(manifest_path.is_file(), "Missing private CUDA install manifest")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "installed_verified" and manifest.get("release") == "12.9.1",
            "Expected the verified CUDA 12.9.1 compiler components")
    require(manifest.get("prefix") == str(root), "Toolchain manifest path mismatch")
    records = {r["path"]: r for r in manifest["files"]}
    checked = {}
    for name in ("bin/nvcc", "bin/ptxas", "include/cuda_runtime.h", "include/cub/cub.cuh", "include/curand.h"):
        require((root / name).is_file(), f"Missing CUDA toolchain component: {name}")
        actual = sha256(root / name)
        require(records.get(name, {}).get("sha256") == actual, f"CUDA toolchain file changed: {name}")
        checked[name] = actual
    version = subprocess.check_output([str(root / "bin/nvcc"), "--version"], text=True, timeout=30)
    require("V12.9.86" in version, "Unexpected CUDA compiler version")
    return {"cuda_home": str(root), "install_manifest_sha256": sha256(manifest_path),
            "critical_file_sha256": checked, "nvcc_version": version,
            "runtime_library_environment_changed": False}


def runtime_preflight(snapshot, checkpoint_manifest, role, primary_ip):
    require(os.environ.get("SLURM_JOB_ID") == JOB_ID, f"Expected allocation {JOB_ID}")
    require(os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES")) == "2",
            "Expected two-node allocation")
    toolchain = verify_toolchain()
    checkpoint_record = json.loads(Path(checkpoint_manifest).read_text())
    verify_cached_checkpoint(snapshot, checkpoint_record)
    if role == "primary":
        # HTTP binds to all local interfaces; the DP master address must still
        # belong to this host, and be reachable from the other node.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((primary_ip, 0))
    import torch
    import vllm
    distribution_version = importlib.metadata.version("vllm")
    require(distribution_version == VLLM_VERSION, f"Unexpected vLLM distribution: {distribution_version}")
    # The official CUDA wheel's Python module exposes the base version while
    # distribution metadata retains the CUDA build suffix. Check both, plus
    # the exact source bytes below; do not mistake packaging for a different pin.
    require(vllm.__version__ in {VLLM_VERSION, VLLM_VERSION.split("+")[0]},
            f"Unexpected vLLM module: {vllm.__version__}")
    package_root = Path(vllm.__file__).parent
    source_hashes = {}
    for name, expected in EXPECTED_VLLM_SOURCES.items():
        digest = sha256(package_root / name)
        require(digest == expected, f"Pinned vLLM source mismatch: {name}")
        source_hashes[name] = digest
    require(torch.cuda.device_count() == 4, "Expected four visible CUDA GPUs per node")
    devices = []
    for index in range(4):
        props = torch.cuda.get_device_properties(index)
        require("H100" in props.name, f"Not an H100: {props.name}")
        devices.append({"cuda_index": index, "name": props.name,
                        "total_memory": props.total_memory, "uuid": str(getattr(props, "uuid", "unknown"))})
    def command_output(command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        return result.stdout
    return {"schema": 1, "kind": "runtime_preflight", "checked_at": datetime.now(timezone.utc).isoformat(),
            "job_id": JOB_ID, "hostname": socket.gethostname(), "role": role,
            "primary_ip": primary_ip, "checkpoint_manifest": str(Path(checkpoint_manifest).absolute()),
            "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
            "checkpoint_verification": "full SHA256 record plus unchanged file stat fingerprints",
            "model_revision": MODEL_REVISION, "vllm_commit": VLLM_COMMIT,
            "cuda_compiler_toolchain": toolchain,
            "vllm_version": vllm.__version__, "vllm_distribution_version": distribution_version,
            "torch_version": torch.__version__,
            "selected_vllm_source_sha256": source_hashes, "devices": devices,
            "slurm_nodes": os.environ.get("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "transport_environment": {key: value for key, value in os.environ.items()
                                      if key.startswith(("NCCL_", "GLOO_", "VLLM_", "UCX_"))
                                      and not any(term in key.upper() for term in ("TOKEN", "KEY", "SECRET", "PASSWORD"))},
            "optional_dependencies": {name: importlib.util.find_spec(name) is not None
                                      for name in ("deep_ep", "nixl", "ray")},
            "gpu_inventory": command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv"]),
            "topology": command_output(["nvidia-smi", "topo", "-m"]),
            "network_interfaces": command_output(["ip", "-json", "address", "show"]),
            "limits": "Does not validate collective execution, generation, rank agreement, or EPLB commits."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("checkpoint", "runtime"):
        sub = commands.add_parser(name)
        sub.add_argument("--snapshot", required=True)
        sub.add_argument("--output", required=True)
        if name == "runtime":
            sub.add_argument("--checkpoint-manifest", required=True)
            sub.add_argument("--role", choices=("primary", "worker"), required=True)
            sub.add_argument("--primary-ip", required=True)
    args = parser.parse_args()
    if args.command == "checkpoint":
        result = verify_checkpoint(args.snapshot)
    else:
        result = runtime_preflight(args.snapshot, args.checkpoint_manifest, args.role, args.primary_ip)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"complete": True, "kind": result["kind"], "output": str(output)}))

MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
VLLM_VERSION = "0.31.0+cu129"
JOB_ID = "4882284"
# HF API /api/models/Qwen/Qwen3-30B-A3B/revision/<revision>?blobs=true,
# retrieved 2026-11-10; byte hashes are Git LFS SHA256, not Git blob IDs.
EXPECTED_SHARDS = {'model-00001-of-00016.safetensors': {'bytes': 3999417504,
                                      'sha256': '454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5'},
 'model-00002-of-00016.safetensors': {'bytes': 3999974192,
                                      'sha256': '47f015d6e5bb1782a834d75c06113c5e8c77ddca1cb7daf89686a4ec0deda19e'},
 'model-00003-of-00016.safetensors': {'bytes': 3997360832,
                                      'sha256': 'ac0bf5990f2da995c1e8b77a3149dee71900b4fe1b8a614231dea9647193d96b'},
 'model-00004-of-00016.safetensors': {'bytes': 3999975056,
                                      'sha256': '89b01fd34a683c70fdb7f22c2e7090538d65063335ba0db46f3654642b17d1e5'},
 'model-00005-of-00016.safetensors': {'bytes': 3999975400,
                                      'sha256': '9849eb3584d928e35bf5956ccfd91b64207467e1fad8304aae68dc4d92d6d6d9'},
 'model-00006-of-00016.safetensors': {'bytes': 3999975400,
                                      'sha256': 'f7a0f1525557d740158fb692986ad43506e1b5901e7406ae65808c6457bc53ad'},
 'model-00007-of-00016.safetensors': {'bytes': 3999975472,
                                      'sha256': '920702a50f27a009e5f676da3b02978dbd794f6ad7854f301d709136154c1a84'},
 'model-00008-of-00016.safetensors': {'bytes': 3997362064,
                                      'sha256': 'a85bf0cc8a8047c116172d4c08cf4792eb59d4375032708ba3e1a7ffca0f708a'},
 'model-00009-of-00016.safetensors': {'bytes': 3999975408,
                                      'sha256': '25cd8aaed86b8e17685efb152912f92448ea3fffb1bdc247f257f04d05907604'},
 'model-00010-of-00016.safetensors': {'bytes': 3999975400,
                                      'sha256': '7c3307ee214797c4bc61c0994f81d690e768150df5f14d18edf84e53ba4810dc'},
 'model-00011-of-00016.safetensors': {'bytes': 3999975408,
                                      'sha256': 'c658cad2842d36fa4c7c7f726f8515dc5670a3d80c94e13257d4e322ffe61988'},
 'model-00012-of-00016.safetensors': {'bytes': 3987400496,
                                      'sha256': '8cb898bc5e78492600053d1105c9ff61d7a719f5cc2802df63e41535a652cc26'},
 'model-00013-of-00016.safetensors': {'bytes': 3997353632,
                                      'sha256': '599594421f314cb8e8ab4474db0eb490cc6310585067e82d73821254b93319ce'},
 'model-00014-of-00016.safetensors': {'bytes': 3999975400,
                                      'sha256': '66d3294e9976b5f01d117a0a0bed768f128a8898eb6937e224d051de2961d363'},
 'model-00015-of-00016.safetensors': {'bytes': 3999975400,
                                      'sha256': '2fe000da4fc7399ff6303bb0daec8c415a5d850f5ea53863392758c77fc37664'},
 'model-00016-of-00016.safetensors': {'bytes': 1087928584,
                                      'sha256': '0c979e314faf06e94a5e1841abd285d4e26e763c79c1abe7710d543192d9da48'}}
EXPECTED_ASSETS = {'config.json': '2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297',
 'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
 'model.safetensors.index.json': 'df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24',
 'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
 'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}
# Exact raw GitHub bytes at VLLM_COMMIT; selected sources, not an entire wheel closure.
EXPECTED_VLLM_SOURCES = {'config/parallel.py': 'eee9418488deb7945562c3127106109bac632471a2c1a08a58169a48cd15684c',
 'distributed/device_communicators/all2all.py': 'e1671a092d64fa75e370680a4951d00c64a7d839e1e954ff16c6cf5d82350c8d',
 'distributed/eplb/eplb_state.py': 'eaac132acd438074343e5ee56c52e1df46f37a9bf7a303bd3b9373c69b513f17',
 'engine/arg_utils.py': '95be1adb96f55933f6a52260c8cb2181d04707401ca21d97c33108c5ab58f3ac',
 'entrypoints/cli/serve.py': 'c182c88347f1eb3440a816986f1c2025a2ab2a13236cd68b0ad0146250d031ff',
 'entrypoints/launchers/api_server/entry.py': '4f60dee4693159b79016c6a84f016eb0ade024433807ff393ea782a31bb9f7e4',
 'entrypoints/launchers/api_server/routers.py': 'c2f1bb2708f44f0fe24f3d8ab8f884c7262e551c6557ebc606263abc4d7b1db1',
 'entrypoints/launchers/launcher.py': 'd5ab9febbe139885c95b03b804680f3a404f74d6bfef7f3d433d506319b490e1',
 'entrypoints/openai/api_server.py': '470ff555bdc19f8e17edf4ee7b6ec3f49ff5af0f773e2a7f91bab536e2239109',
 'model_executor/layers/fused_moe/layer.py': '0c30a9a56ff22c63a1498f61c72a349e4dba0d1e4fbe42efb333dda45d4bdecf',
 'model_executor/models/qwen3_moe.py': 'cac0cffc3e5b1ff805e9e15fe1d4bcfd5e55c27378e4a9ab23b0fa6c1b05e5ee',
 'v1/engine/utils.py': '2f5b1b802ca31f19545ba00c68afc83cbbec4e55399e311e9102ce7a483254ab'}


if __name__ == "__main__":
    main()
