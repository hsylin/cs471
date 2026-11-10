"""Read-only runtime admission for one contemporaneous OFF/ON/resident triple.

This checks recorded source, hardware, backend and build identity. It does not
infer numerical correctness, migration commits or speedup from startup logs.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from full_serving.audit_pilots import (AuditError, paired_build_evidence,
                                      paired_configuration, require)
from full_serving.data import digest, file_digest
from full_serving.preflight import EXPECTED_VLLM_SOURCES
from full_serving.resident_plugin import (EXPECTED_NATIVE_SOURCES, MODEL_REVISION,
                                          PIN, balanced_map)

ARMS = ("off", "on", "custom")
ROLES = ("primary", "worker")
SCOPE = {"dp": 8, "tp": 1, "ep": 8, "dtype": "bfloat16",
         "activation_backend": "allgather_reducescatter",
         "vllm_commit": PIN, "model_revision": MODEL_REVISION}


def logged_arguments(text):
    entries = [line.split("non-default args: ", 1)[1]
               for line in text.split("\n") if "non-default args: " in line]
    require(len(entries) == 1, "absent or ambiguous native argument record")

    class SafeConfig(ast.NodeTransformer):
        def visit_Call(self, node):
            require(isinstance(node.func, ast.Name) and node.func.id == "EPLBConfig"
                    and not node.args and all(k.arg is not None for k in node.keywords),
                    "unsupported constructor in native configuration")
            return ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],
                            values=[k.value for k in node.keywords])

    value = ast.literal_eval(SafeConfig().visit(ast.parse(entries[0], mode="eval")))
    require(isinstance(value, dict), "native arguments must be a dictionary")
    return value


def engine_configuration(text):
    entries = [line.split("with config: ", 1)[1] for line in text.split("\n")
               if "Initializing a V1 LLM engine" in line and "with config: " in line]
    require(len(entries) == 1, "absent or ambiguous initialized engine configuration")
    config = entries[0]
    for token in ("dtype=torch.bfloat16", "max_seq_len=32768", "tensor_parallel_size=1",
                  "pipeline_parallel_size=1", "data_parallel_size=8", "quantization=None",
                  "enforce_eager=False", "served_model_name=qwen3-full", "FULL_AND_PIECEWISE"):
        require(token in config, "initialized engine differs from frozen native scope")
    return config


def runtime_identity(value, role):
    require(value.get("kind") == "runtime_preflight" and value.get("role") == role,
            "wrong runtime preflight role/type")
    require(value.get("model_revision") == MODEL_REVISION and value.get("vllm_commit") == PIN,
            "runtime source/checkpoint pin differs")
    require(value.get("vllm_version") == "0.31.0"
            and value.get("vllm_distribution_version") == "0.31.0+cu129"
            and value.get("torch_version") == "2.13.0+cu129", "runtime package versions differ")
    require(value.get("selected_vllm_source_sha256") == EXPECTED_VLLM_SOURCES,
            "runtime native source bytes differ from exact pinned sources")
    devices = value.get("devices", [])
    require(len(devices) == 4 and [d.get("cuda_index") for d in devices] == list(range(4))
            and all(d.get("name") == "NVIDIA H100" and d.get("total_memory") == 99981918208
                    and isinstance(d.get("uuid"), str) and d["uuid"] for d in devices),
            "four expected H100 94GB devices are required on each node")
    # CPU affinity can vary across Slurm steps. Compare GPU-to-GPU links only.
    topology = re.sub(r"\x1b\[[0-9;]*m", "", value.get("topology", ""))
    links = [line.split()[1:5] for line in topology.split("\n")
             if re.match(r"^GPU[0-3]\s", line)]
    require(links == [["X" if i == j else "NV6" for j in range(4)] for i in range(4)],
            "recorded within-node GPU topology differs from four-way NV6")
    require(re.fullmatch(r"[0-9a-f]{64}", value.get("checkpoint_manifest_sha256", "")),
            "checkpoint verification manifest hash missing")
    return {k: value[k] for k in ("hostname", "model_revision", "devices", "torch_version",
            "vllm_commit", "vllm_version", "vllm_distribution_version", "selected_vllm_source_sha256",
            "checkpoint_manifest_sha256", "transport_environment")} | {"gpu_links": links}


def audit_runtime_triplet(block_dir, protocol):
    """Return passed/failed/pending; missing runtime evidence never admits a block."""
    root = Path(block_dir)
    result = {"schema": "cs471_custom_runtime_triplet_v1", "status": "pending",
              "scope": "Recorded runtime/configuration/build identity only; HTTP correctness and source/data admission are separate. Native async launch or drain is not proof of a committed map."}
    try:
        require(all(protocol.get("scope", {}).get(k) == v for k, v in SCOPE.items()),
                "protocol runtime scope differs from reviewed implementation")
        required = [root / arm / name for arm in ARMS for name in
                    ("server-0.log", "server-1.log", "server-0.err", "server-1.err",
                     "preflight-primary.json", "preflight-worker.json")]
        required += [root / "custom" / f"resident-bindings-rank{rank}.json" for rank in range(8)]
        missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
        if missing:
            result.update(reason="runtime evidence is not complete", missing=missing)
            return result
        native_pair = paired_configuration(root)
        result["native_off_on"] = native_pair
        require(native_pair["status"] == "passed", "native OFF/ON configuration admission failed")
        configs, preflights, identities, engines, backends = {}, {}, {}, {}, {}
        for arm in ARMS:
            directory = root / arm
            texts = [(directory / f"server-{node}.log").read_text() for node in range(2)]
            configs[arm] = logged_arguments(texts[0])
            require(configs[arm].get("served_model_name") == ["qwen3-full"], "wrong served model identity")
            engines[arm] = [engine_configuration(text) for text in texts]
            require(engines[arm][0] == engines[arm][1], "node engine configurations disagree")
            joined = "\n".join(texts)
            markers = ("Using AgRsAll2AllManager", "Using V2 Model Runner",
                       "Using FlashAttention version 3", "Using TRITON Unquantized MoE")
            require(all(marker in joined for marker in markers), "actual runner/backend evidence missing")
            for node, rank in enumerate((0, 4)):
                expected = (f"rank {rank} in world size 8 is assigned as DP rank {rank}, PP rank 0, "
                            f"PCP rank 0, TP rank 0, ETP rank 0, EP rank {rank}, EPLB rank ")
                require(expected in texts[node], "actual distributed rank topology evidence missing")
            if arm == "on":
                require("TorchDistGlooStagedEplbCommunicator" in joined,
                        "native ON weight-transport backend not recorded")
            else:
                require("Initialized EPLB communicator:" not in joined and "Rearranging experts" not in joined,
                        "control-disabled arm recorded native EPLB activity")
            backends[arm] = {"markers": list(markers), "representative_logged_ep_ranks": [0, 4],
                             "native_weight_transport": "TorchDistGlooStagedEplbCommunicator" if arm == "on" else None}
            preflights[arm], identities[arm] = {}, []
            for role in ROLES:
                value = json.loads((directory / f"preflight-{role}.json").read_bytes())
                preflights[arm][role] = value
                identities[arm].append(runtime_identity(value, role))
            require(len({n["hostname"] for n in identities[arm]}) == 2
                    and len({d["uuid"] for n in identities[arm] for d in n["devices"]}) == 8,
                    "runtime does not identify two nodes and eight distinct H100s")
        require(configs["custom"] == configs["on"], "custom native initialization flags differ from native ON")
        require(engines["off"] == engines["on"] == engines["custom"],
                "engine initialization differs across the three arms")
        require(identities["off"] == identities["on"] == identities["custom"],
                "three-arm hardware/checkpoint/source/environment fingerprints differ")
        for rank in range(8):
            binding = json.loads((root / "custom" / f"resident-bindings-rank{rank}.json").read_bytes())
            require(binding.get("ep_rank") == rank and binding.get("hostname") == identities["custom"][rank // 4]["hostname"]
                    and binding.get("status") == "bound" and binding.get("layer_count") == 48,
                    "custom rank/host/layer binding differs")
            require(binding.get("native_source_sha256") == EXPECTED_NATIVE_SOURCES,
                    "custom pinned native source hashes differ")
            require(binding.get("mode") == protocol["custom"]["mode"] == "resident_v1"
                    and binding.get("config_sha256") == protocol["custom"]["config_sha256"]
                    and binding.get("source_manifest_sha256") == protocol["custom"]["source_manifest_sha256"]
                    and binding.get("physical_to_logical_map") == balanced_map(), "custom treatment identity differs")
            require(all(binding.get(k) is False for k in ("native_eplb_state_created", "native_observation_history_active",
                    "native_profile_migration_active", "native_async_worker_active", "native_control_step_active",
                    "normal_commit_applicable")) and "startup_diagnostic" in binding and binding["startup_diagnostic"] is None,
                    "custom timed run has recurring control or startup diagnostic instrumentation")
        builds = {"off_on": paired_build_evidence({"off": preflights["off"], "on": preflights["on"]}),
                  "off_custom": paired_build_evidence({"off": preflights["off"], "on": preflights["custom"]})}
        require(all(x["flashinfer_prebuild"]["status"] == "matched" for x in builds.values()),
                "three-arm FlashInfer build/cache evidence is incomplete")
        result.update(status="passed", common_runtime_fingerprint_sha256=digest(identities["off"]),
                      actual_backends=backends, native_build_evidence=builds,
                      native_engine_configuration_sha256=digest(engines["off"]),
                      evidence_file_sha256={str(p.relative_to(root)): file_digest(p) for p in required},
                      intended_differences=["OFF: no replicas or EPLB", "ON: native 32 replicas and native async EPLB",
                          "custom: 32 replicas in frozen paired map, resident dispatch, no recurring EPLB"])
    except (AttributeError, KeyError, TypeError, ValueError, OSError, SyntaxError) as exc:
        result.update(status="failed", reason=str(exc) if isinstance(exc, AuditError) else type(exc).__name__)
    return result
