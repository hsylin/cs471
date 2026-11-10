"""Runtime identity corruption must reject a complete-looking timing triple."""
import copy
import json

import pytest

from full_serving.audit_pilots import CUDA_CRITICAL_FILES, FLASHINFER_SOURCES
from full_serving.custom_runtime import SCOPE, audit_runtime_triplet
from full_serving.preflight import EXPECTED_VLLM_SOURCES
from full_serving.resident_plugin import EXPECTED_NATIVE_SOURCES, MODEL_REVISION, PIN, balanced_map


def fixture(tmp_path):
    protocol = {"scope": dict(SCOPE), "custom": {"mode": "resident_v1", "config_sha256": "a"*64,
                                                "source_manifest_sha256": "b"*64}}
    base_args = {"data_parallel_size": 8, "data_parallel_size_local": 4,
                 "enable_expert_parallel": True, "dtype": "bfloat16", "max_model_len": 32768,
                 "model": "/checkpoint/" + MODEL_REVISION, "served_model_name": ["qwen3-full"]}
    engine = ("dtype=torch.bfloat16, max_seq_len=32768, tensor_parallel_size=1, pipeline_parallel_size=1, "
              "data_parallel_size=8, quantization=None, enforce_eager=False, served_model_name=qwen3-full, FULL_AND_PIECEWISE")
    for arm in ("off", "on", "custom"):
        directory = tmp_path / arm
        directory.mkdir()
        args = copy.deepcopy(base_args)
        if arm != "off":
            args.update(enable_eplb=True, eplb_config={"num_redundant_experts": 32, "use_async": True})
        for index, role in enumerate(("primary", "worker")):
            rank = index * 4
            text = (f"non-default args: {args!r}\n" if index == 0 else "")
            text += "Initializing a V1 LLM engine (v0.31.0) with config: " + engine + "\n"
            text += (f"rank {rank} in world size 8 is assigned as DP rank {rank}, PP rank 0, PCP rank 0, "
                     f"TP rank 0, ETP rank 0, EP rank {rank}, EPLB rank " + (str(rank) if arm != "off" else "N/A") + "\n")
            text += "Using AgRsAll2AllManager\nUsing V2 Model Runner\nUsing FlashAttention version 3\nUsing TRITON Unquantized MoE\n"
            if arm == "on":
                text += "Initialized EPLB communicator: TorchDistGlooStagedEplbCommunicator\n"
            (directory / f"server-{index}.log").write_text(text)
            (directory / f"server-{index}.err").write_text("")
            value = {"kind": "runtime_preflight", "role": role, "hostname": f"host{index}",
                     "model_revision": MODEL_REVISION, "vllm_commit": PIN, "vllm_version": "0.31.0",
                     "vllm_distribution_version": "0.31.0+cu129", "torch_version": "2.13.0+cu129",
                     "selected_vllm_source_sha256": EXPECTED_VLLM_SOURCES,
                     "checkpoint_manifest_sha256": "c"*64, "transport_environment": {},
                     "devices": [{"cuda_index": j, "name": "NVIDIA H100", "total_memory": 99981918208,
                                  "uuid": f"uuid-{rank+j}"} for j in range(4)],
                     "topology": "\n".join("GPU"+str(i)+"\t"+"\t".join("X" if i == j else "NV6" for j in range(4))+"\tSYS" for i in range(4)),
                     "cuda_compiler_toolchain": {"cuda_home": "/tools/cuda", "install_manifest_sha256": "d"*64,
                         "critical_file_sha256": {k: "e"*64 for k in CUDA_CRITICAL_FILES},
                         "nvcc_version": "V12.9.86", "runtime_library_environment_changed": False},
                     "flashinfer_prebuild": {"workspace_base": f"/cache/host{index}", "jit_dir": f"/cache/host{index}/ops",
                         "source_sha256": {k: "f"*64 for k in FLASHINFER_SOURCES},
                         "sampling_module_sha256": {"sampling/sampling.so": str(index)*64},
                         "checks": [{"exact_mask_match": True}, {"exact_mask_match": True},
                                    {"exact_sample_match": True, "operation": "top_k_top_p_sampling_from_logits"},
                                    {"exact_sample_match": True, "operation": "top_p_sampling_from_probs"}]}}
            (directory / f"preflight-{role}.json").write_text(json.dumps(value))
    for rank in range(8):
        binding = {"ep_rank": rank, "hostname": f"host{rank//4}", "status": "bound", "layer_count": 48,
                   "native_source_sha256": EXPECTED_NATIVE_SOURCES, **protocol["custom"],
                   "physical_to_logical_map": balanced_map(), "startup_diagnostic": None}
        binding.update({k: False for k in ("native_eplb_state_created", "native_observation_history_active",
                       "native_profile_migration_active", "native_async_worker_active", "native_control_step_active",
                       "normal_commit_applicable")})
        (tmp_path / "custom" / f"resident-bindings-rank{rank}.json").write_text(json.dumps(binding))
    return protocol


def change_json(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    path.write_text(json.dumps(value))


def test_complete_matching_triplet_passes_with_explicit_evidence_limits(tmp_path):
    result = audit_runtime_triplet(tmp_path, fixture(tmp_path))
    assert result["status"] == "passed", result
    assert result["native_build_evidence"]["off_custom"]["flashinfer_prebuild"]["status"] == "matched"
    assert "not proof of a committed map" in result["scope"]
    assert len(result["evidence_file_sha256"]) == 26


def test_missing_custom_runtime_log_cannot_pass_with_complete_native_pair(tmp_path):
    protocol = fixture(tmp_path)
    (tmp_path / "custom/server-1.log").unlink()
    result = audit_runtime_triplet(tmp_path, protocol)
    assert result["status"] == "pending" and "custom/server-1.log" in result["missing"]


@pytest.mark.parametrize("corruption", ["gpu", "source", "checkpoint", "compiler", "cache", "missing_cache", "topology"])
def test_custom_preflight_changes_reject_three_arm_admission(tmp_path, corruption):
    protocol = fixture(tmp_path)
    def mutate(value):
        if corruption == "gpu": value["devices"][0]["uuid"] = "different-physical-gpu"
        elif corruption == "source": value["selected_vllm_source_sha256"]["config/parallel.py"] = "0"*64
        elif corruption == "checkpoint": value["checkpoint_manifest_sha256"] = "0"*64
        elif corruption == "compiler": value["cuda_compiler_toolchain"]["critical_file_sha256"]["bin/nvcc"] = "0"*64
        elif corruption == "cache": value["flashinfer_prebuild"]["sampling_module_sha256"]["sampling/sampling.so"] = "9"*64
        elif corruption == "missing_cache": del value["flashinfer_prebuild"]
        else: value["topology"] = value["topology"].replace("NV6", "SYS")
    change_json(tmp_path / "custom/preflight-primary.json", mutate)
    assert audit_runtime_triplet(tmp_path, protocol)["status"] == "failed"


@pytest.mark.parametrize("old,new", [("Using AgRsAll2AllManager", "Using DeepEP"),
                                       ("enforce_eager=False", "enforce_eager=True"),
                                       ("'max_model_len': 32768", "'max_model_len': 16384")])
def test_changed_actual_backend_or_engine_not_hidden_by_same_preflight(tmp_path, old, new):
    protocol = fixture(tmp_path)
    for p in (tmp_path / "custom").glob("server-*.log"):
        p.write_text(p.read_text().replace(old, new))
    assert audit_runtime_triplet(tmp_path, protocol)["status"] == "failed"


def test_custom_native_control_or_diagnostic_cannot_leak_into_timing(tmp_path):
    protocol = fixture(tmp_path)
    change_json(tmp_path / "custom/resident-bindings-rank3.json", lambda r: r.update(native_control_step_active=True))
    assert audit_runtime_triplet(tmp_path, protocol)["status"] == "failed"


def test_native_log_constructor_is_not_executed(tmp_path):
    protocol = fixture(tmp_path)
    path = tmp_path / "custom/server-0.log"
    path.write_text(path.read_text().replace("'dtype': 'bfloat16'", "'dtype': __import__('os').system('false')"))
    result = audit_runtime_triplet(tmp_path, protocol)
    assert result["status"] == "failed" and "unsupported constructor" in result["reason"]


def test_malformed_custom_preflight_returns_failed_instead_of_crashing_report(tmp_path):
    protocol = fixture(tmp_path)
    (tmp_path / "custom/preflight-primary.json").write_text("null\n")
    assert audit_runtime_triplet(tmp_path, protocol)["status"] == "failed"


@pytest.mark.parametrize("arm", ["off", "on", "custom"])
def test_generic_pre_shutdown_error_rejects_each_final_arm(tmp_path, arm):
    protocol = fixture(tmp_path)
    path = tmp_path / arm / "server-0.log"
    with path.open("a") as output:
        output.write("ERROR 11-10 19:40:57 inference failed\nINFO 11-10 19:40:57 [shutdown] stopping\n")
    result = audit_runtime_triplet(tmp_path, protocol)
    assert result["status"] == "failed" and "runtime error" in result["reason"]
    assert result["runtime_error_evidence"][arm]["before_shutdown_or_ambiguous"] == 1


def test_final_triplet_retains_true_post_shutdown_error_counts(tmp_path):
    protocol = fixture(tmp_path)
    path = tmp_path / "custom/server-0.log"
    # Same ordering/timestamp as the real timed-resident-v1 teardown.
    with path.open("a") as output:
        output.write("(APIServer pid=52256) INFO 11-10 19:40:57 [launcher.py:128] [shutdown] API server: shutdown triggered\n"
                     "(APIServer pid=52256) ERROR 11-10 19:40:57 [async_llm.py:871] AsyncLLM output_handler failed.\n"
                     "(APIServer pid=52256) ERROR 11-10 19:40:57 [async_llm.py:871] Traceback (most recent call last):\n"
                     "(APIServer pid=52256) ERROR 11-10 19:40:57 [async_llm.py:871] vllm.v1.engine.exceptions.EngineDeadError\n")
    result = audit_runtime_triplet(tmp_path, protocol)
    assert result["status"] == "passed", result
    assert result["runtime_error_evidence"]["custom"] == {"before_shutdown_or_ambiguous": 0, "after_shutdown": 3, "total": 3}


def test_final_triplet_rejects_stderr_traceback_without_shutdown_order(tmp_path):
    protocol = fixture(tmp_path)
    (tmp_path / "custom/server-1.err").write_text("Traceback (most recent call last):\n")
    result = audit_runtime_triplet(tmp_path, protocol)
    assert result["status"] == "failed"
    assert result["runtime_error_evidence"]["custom"]["before_shutdown_or_ambiguous"] == 1
