"""Pinned full-model fixed-resident adapter, loaded by vLLM general plugins.

The native checkpoint loader installs a declared balanced map before inference.
Only resident_v1 moves logical-to-physical selection past AgRs all-gather.
No upstream EPLB state, profiling migration, history or async worker is created.
This adapter is intentionally limited to the reviewed Qwen3/EP8/TRITON path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
from types import MethodType

PIN = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
CONFIG_SCHEMA = "cs471_full_resident_v1"
SOURCES_SCHEMA = "cs471_full_resident_sources_v1"
MODES = {"resident_v1", "native_hash"}
ROOT = Path(__file__).resolve().parents[1]
_INSTALLED = None
_BOUND = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_map():
    return [expert for rank in range(8)
            for expert in (*range(16 * rank, 16 * rank + 16),
                           *range(16 * (rank ^ 1), 16 * (rank ^ 1) + 4))]


def validate_config(config):
    if (config.get("schema") != CONFIG_SCHEMA or config.get("mode") not in MODES
            or config.get("vllm_commit") != PIN or config.get("model_revision") != MODEL_REVISION
            or config.get("physical_to_logical_map") != balanced_map()
            or any(type(v) is not int for v in config.get("physical_to_logical_map", []))
            or type(config.get("startup_diagnostic", False)) is not bool):
        raise RuntimeError("Unreviewed resident configuration, checkpoint, or fixed placement")
    return config


def verify_source_manifest(manifest, root=ROOT):
    if manifest.get("schema") != SOURCES_SCHEMA or not isinstance(manifest.get("files"), dict):
        raise RuntimeError("Missing custom source manifest")
    required = {"full_serving/resident_plugin.py", "full_serving/custom_entrypoint.py",
                "full_serving/resident_policy.py", "full_serving/resident_gpu.py",
                "full_serving/resident_diagnostic.py",
                "full_serving/bootstrap/cs471_resident-0.1.dist-info/entry_points.txt",
                "full_serving/bootstrap/cs471_resident-0.1.dist-info/METADATA"}
    if not required.issubset(manifest["files"]):
        raise RuntimeError("Incomplete custom source manifest")
    root = Path(root).resolve()
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Custom source hash mismatch or unsafe path: {name}")


def load_admission():
    config_path = Path(os.environ["FULL_SERVING_CUSTOM_CONFIG"])
    sources_path = Path(os.environ["FULL_SERVING_CUSTOM_SOURCE_MANIFEST"])
    if sha(config_path) != os.environ.get("FULL_SERVING_CUSTOM_CONFIG_SHA256"):
        raise RuntimeError("Custom configuration bytes changed")
    if sha(sources_path) != os.environ.get("FULL_SERVING_CUSTOM_SOURCE_MANIFEST_SHA256"):
        raise RuntimeError("Custom source manifest bytes changed")
    config = validate_config(json.loads(config_path.read_text()))
    if config["mode"] != os.environ.get("FULL_SERVING_CUSTOM_MODE"):
        raise RuntimeError("Custom mode does not match admitted configuration")
    verify_source_manifest(json.loads(sources_path.read_text()))
    return config, {"config_sha256": sha(config_path), "source_manifest_sha256": sha(sources_path)}


def validate_parallel(parallel):
    expected = {"data_parallel_size": 8, "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1, "enable_expert_parallel": True,
                "enable_eplb": True, "data_parallel_backend": "mp",
                "all2all_backend": "allgather_reducescatter",
                "expert_placement_strategy": "linear", "enable_dbo": False,
                "enable_elastic_ep": False}
    for key, value in expected.items():
        if getattr(parallel, key, None) != value:
            raise RuntimeError(f"Unsupported custom parallel configuration: {key}")
    if parallel.eplb_config.num_redundant_experts != 32:
        raise RuntimeError("Custom mode requires the fixed 32-replica budget")


def _wrap_prepare(original, mapper):
    def prepare(self, *args, **kwargs):
        result = original(*args, **kwargs)
        hidden, scales, metadata, logical_ids, weights = result
        physical_ids = mapper(logical_ids)
        return hidden, scales, metadata, physical_ids, weights
    return prepare


def register():
    """Supported vllm.general_plugins entry point; harmless outside custom mode."""
    global _INSTALLED
    if not os.environ.get("FULL_SERVING_CUSTOM_MODE"):
        return
    config, admission = load_admission()
    if _INSTALLED is not None:
        if _INSTALLED != admission:
            raise RuntimeError("Resident adapter admission changed in this process")
        return

    import torch
    import vllm
    import vllm.envs as envs
    from vllm.distributed import get_ep_group
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import MoEPrepareAndFinalizeNaiveDPEPModular
    from vllm.v1.worker.gpu.eplb_utils import EPLBController
    from full_serving.resident_policy import ResidentMapper

    package = Path(vllm.__file__).parent
    for name, digest in EXPECTED_NATIVE_SOURCES.items():
        if sha(package / name) != digest:
            raise RuntimeError(f"Pinned native source changed: {name}")
    if not envs.VLLM_MOE_SKIP_PADDING:
        raise RuntimeError("Native per-rank padding masks must remain enabled")

    def initial_map(num_routed_experts, num_redundant_experts):
        if (num_routed_experts, num_redundant_experts) != (128, 32):
            raise RuntimeError("Resident checkpoint-loader geometry changed")
        return list(config["physical_to_logical_map"])

    def forbid_native_state(*args, **kwargs):
        raise RuntimeError("Native EPLB state/transport is forbidden in fixed-resident mode")

    def prepare_load(self):
        validate_parallel(self.parallel_config)
        self.state = None
        self.suppressed = True
        self._has_registered_models = False

    def maybe_register_model(self, model, model_config, load_dummy_weights):
        validate_parallel(self.parallel_config)
        hf = model_config.hf_text_config
        if (load_dummy_weights or model_config.dtype != torch.bfloat16
                or getattr(hf, "model_type", None) != "qwen3_moe"
                or (hf.num_hidden_layers, hf.num_experts, hf.num_experts_per_tok,
                    hf.hidden_size, hf.moe_intermediate_size) != (48, 128, 8, 2048, 768)):
            raise RuntimeError("Only the pinned full BF16 Qwen3 MoE geometry is supported")
        if get_ep_group().world_size != 8:
            raise RuntimeError("Resident adapter requires eight actual EP ranks")
        rank = get_ep_group().rank_in_group
        layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, MoERunner)]
        if len(layers) != 48 or _BOUND:
            raise RuntimeError("Expected exactly 48 previously unbound MoE layers")
        records = []
        for name, layer in layers:
            router = layer.router
            routed = layer.routed_experts
            kernel = routed.quant_method.moe_kernel
            if (type(router) is not FusedTopKRouter or router.top_k != 8
                    or router.scoring_func != "softmax" or not router.renormalize
                    or type(kernel.prepare_finalize) is not MoEPrepareAndFinalizeNaiveDPEPModular
                    or kernel.fused_experts.__class__.__name__ != "TritonExperts"
                    or routed.quant_method.__class__.__name__ != "UnquantizedFusedMoEMethod"
                    or layer.do_naive_dispatch_combine or layer.enable_dbo or layer.gate is None
                    or layer.moe_config.num_experts != 160
                    or layer.moe_config.num_local_experts != 20
                    or layer.moe_config.is_sequence_parallel):
                raise RuntimeError(f"Unsupported resident layer/backend: {name}")
            expert_map = routed.expert_map.detach().cpu().tolist()
            expected_map = [-1] * 160
            expected_map[rank * 20:(rank + 1) * 20] = list(range(20))
            if expert_map != expected_map:
                raise RuntimeError(f"Unexpected physical-slot ownership: {name}")
            if tuple(routed.w13_weight.shape) != (20, 1536, 2048) or tuple(routed.w2_weight.shape) != (20, 2048, 768):
                raise RuntimeError(f"Unexpected loaded expert tensor geometry: {name}")
            if routed.w13_weight.dtype != torch.bfloat16 or routed.w2_weight.dtype != torch.bfloat16:
                raise RuntimeError(f"Expected full BF16 expert weights: {name}")
            mapper = ResidentMapper(routed.w13_weight.device, config["physical_to_logical_map"],
                                    top_k=8, mode=config["mode"])
            router.eplb_state = None
            if config["mode"] == "resident_v1":
                prepare = kernel.prepare_finalize
                prepare.prepare = MethodType(_wrap_prepare(prepare.prepare, mapper), prepare)
                insertion = "after native AgRs gather of logical IDs"
            else:
                def local_mapping(self, ids, mapper=mapper):
                    return mapper(ids)
                router._apply_eplb_mapping = MethodType(local_mapping, router)
                insertion = "native-hash mapping on local IDs before gather; no recording"
            _BOUND.append((name, layer, mapper))
            records.append({"layer": name, "insertion": insertion,
                            "physical_slots": list(range(rank * 20, (rank + 1) * 20)),
                            "resident_logical_experts": config["physical_to_logical_map"][rank * 20:(rank + 1) * 20],
                            "w13_shape": list(routed.w13_weight.shape), "w2_shape": list(routed.w2_weight.shape)})
        diagnostic = None
        if config.get("startup_diagnostic", False):
            from full_serving.resident_diagnostic import run_startup_diagnostics
            diagnostic = run_startup_diagnostics(_BOUND, config, admission)
        report = {"schema": "cs471_full_resident_bindings_v1", "status": "bound",
                  "mode": config["mode"], "hostname": socket.gethostname(), "ep_rank": rank,
                  "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID"),
                  "native_vllm_commit": PIN, "model_revision": MODEL_REVISION,
                  "physical_to_logical_map": config["physical_to_logical_map"],
                  "layers": records, "layer_count": len(records),
                  "native_eplb_state_created": False, "native_observation_history_active": False,
                  "native_profile_migration_active": False, "native_async_worker_active": False,
                  "native_control_step_active": False, "normal_commit_applicable": False,
                  "padding_source": "Native top-k -1 IDs from V2 per-rank is_padding buffers, gathered unchanged",
                  "native_source_sha256": EXPECTED_NATIVE_SOURCES, **admission}
        report["startup_diagnostic"] = diagnostic
        directory = Path(os.environ["FULL_SERVING_RUN_DIR"])
        output = directory / f"resident-bindings-rank{rank}.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        print(f"[cs471-resident] rank={rank} mode={config['mode']} bound_layers={len(records)} fixed_map=true native_control=false", flush=True)
        return True

    def maybe_register_speculator(self, speculator, speculative_config, load_dummy_weights):
        if speculator is not None or speculative_config is not None:
            raise RuntimeError("Speculative execution is unsupported by this resident adapter")
        return False

    def inactive(self, *args, **kwargs):
        return None

    EplbState.build_initial_global_physical_to_logical_map = staticmethod(initial_map)
    EplbState.__init__ = forbid_native_state
    EPLBController.prepare_load = prepare_load
    EPLBController.maybe_register_model = maybe_register_model
    EPLBController.maybe_register_speculator = maybe_register_speculator
    EPLBController.maybe_start_async_loop = inactive
    EPLBController.prepare_forward = inactive
    EPLBController.step = inactive
    EPLBController.setup_from_mapping = forbid_native_state
    _INSTALLED = admission


# Exact Python source bytes are checked before any worker/model construction.
EXPECTED_NATIVE_SOURCES = {'distributed/device_communicators/all2all.py': 'e1671a092d64fa75e370680a4951d00c64a7d839e1e954ff16c6cf5d82350c8d',
 'distributed/device_communicators/cuda_communicator.py': '2a0695d8b46757be83b38fe657df605ba0bfb2ef2dfa690c78c57419bbad0762',
 'distributed/device_communicators/pynccl.py': '4c51becc2ebfd41910526b93d7bd8e9e36312df444fdfb55fa2fbc92b0f8f88e',
 'distributed/eplb/eplb_state.py': 'eaac132acd438074343e5ee56c52e1df46f37a9bf7a303bd3b9373c69b513f17',
 'forward_context.py': '8dab5a4dee60e46c4e52821667321c9f05c793578af49cddbdb204cf9324c403',
 'model_executor/layers/fused_moe/all2all_utils.py': '829d16c4dedbd69dffdb8c2bf1b68caf23f095b320cbf83795149cfccb9f1906',
 'model_executor/layers/fused_moe/fused_moe_method_base.py': 'f8defdd0f8aa1f7a4cb68aca64f9d8e737eb9c65f00fa2d66571704aa1532916',
 'model_executor/layers/fused_moe/layer.py': '0c30a9a56ff22c63a1498f61c72a349e4dba0d1e4fbe42efb333dda45d4bdecf',
 'model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py': '5cc53791999bc0116f8d9d5a4658f8b39f53f59c1b7d644219cf4655011e29c4',
 'model_executor/layers/fused_moe/routed_experts.py': '78bff37911f17cb4cf77a09861dd11fc9bdbf59e7dadb422b4f3fe9e9d526812',
 'model_executor/layers/fused_moe/router/base_router.py': 'c909924fcfa461ccdaa6290d215d4d35bd4062fb02feeb23619b234da993f6bf',
 'model_executor/layers/fused_moe/router/fused_topk_router.py': '1c80f4af4d781fd66b12aa6a7a0fd30082bc521df40f1252987a8ac4c0bef51c',
 'model_executor/layers/fused_moe/runner/moe_runner.py': '2c6e1fffba1f4fd2e4c5d3981c4d26f7c0de802fc9c35865748b113c30235da5',
 'model_executor/layers/fused_moe/unquantized_fused_moe_method.py': 'dade56ddeef4967aa3ad3448ef46d964e6130a6f3d558e4a77ccce62303816a9',
 'model_executor/models/qwen3_moe.py': 'cac0cffc3e5b1ff805e9e15fe1d4bcfd5e55c27378e4a9ab23b0fa6c1b05e5ee',
 'plugins/__init__.py': '06942d64e526c2c7407265e1903d625c4d2118d8d7fc39c616b1cc6aa6fc2c8f',
 'v1/worker/gpu/cudagraph_utils.py': '8e6ac6f655dd013dd2a3d3561bc88fe05ceeea2b060e5cb6bc85decb0bb2f813',
 'v1/worker/gpu/eplb_utils.py': '98aaa1e7763faa3ae6fbd012f5a222f050ab2d5a6d8942193341f2e5716228ae',
 'v1/worker/gpu/input_batch.py': '96163c486368f2cf0e539fe68eaa06f4df3ac814da8f7a92af2b04242e01bc6d',
 'v1/worker/gpu/model_runner.py': '725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c',
 'v1/worker/worker_base.py': '41cdc6fb72d5cb2352093de7fdf35bf829fbb9e097dc6113d4597cd9a88a46aa'}
