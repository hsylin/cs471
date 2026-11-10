import copy
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from full_serving import custom_compare as compare


ORDERS = [
    [["custom", "off", "on"], ["custom", "on", "off"], ["off", "custom", "on"], ["on", "off", "custom"]],
    [["off", "custom", "on"], ["custom", "on", "off"], ["on", "off", "custom"], ["on", "custom", "off"]],
    [["on", "off", "custom"], ["on", "custom", "off"], ["off", "on", "custom"], ["custom", "off", "on"]],
    [["custom", "off", "on"], ["off", "on", "custom"], ["on", "custom", "off"], ["custom", "on", "off"]],
    [["custom", "off", "on"], ["off", "on", "custom"], ["on", "custom", "off"], ["off", "custom", "on"]],
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return compare.native.sha(path)


def fixture_protocol(tmp_path, *, payloads=False):
    config = {"physical_to_logical_map": [e for rank in range(8) for e in
        [*range(rank * 16, rank * 16 + 16), *range((rank ^ 1) * 16, (rank ^ 1) * 16 + 4)]],
        "vllm_commit": "pinned-vllm", "model_revision": "pinned-model"}
    config_hash = write_json(tmp_path / "config.json", config)
    source_hash = write_json(tmp_path / "sources.json", {"files": {}})
    admission = {"status": "passed", "pilot_only": True, "correctness_gate_passed": True, "performance_gate_passed": True,
                 "custom_config_sha256": config_hash, "custom_source_manifest_sha256": source_hash}
    admission_hash = write_json(tmp_path / "admission.json", admission)
    protocol = {"schema": "resident_compare_protocol_v1", "status": "frozen", "final_execution_allowed": True,
        "custom": {"entrypoint": "full_serving.custom_entrypoint", "mode": "resident_v1", "bootstrap_path": "full_serving/bootstrap",
                   "config_file": "config.json", "config_sha256": config_hash, "source_manifest_file": "sources.json",
                   "source_manifest_sha256": source_hash},
        "pilot_admission": {"file": "admission.json", "sha256": admission_hash},
        "execution": {"repeats": 1, "warmup_passes": 2, "max_inflight": 64},
        "generation": {"max_tokens": 2048, "max_model_len": 32768, "temperature": 0, "enable_thinking": False,
                       "ignore_eos": False, "truncate_prompts": False}, "source_admission": {},
        "acceptance": {"useful_gain_percent": 5}, "datasets": []}
    for dataset_index, name in enumerate(compare.DATASETS):
        dataset = {"dataset": name, "blocks": []}
        for index, count in [("warmup", 8), *[(str(i), 16) for i in range(4)]]:
            ids = [f"{name}-{index}-{i}" for i in range(count)]
            item = {"request_count": count, "request_ids": ids, "requests_file": f"{name}/{index}.jsonl",
                    "requests_sha256": "1" * 64, "split": "pilot" if index == "warmup" else "final"}
            if payloads:
                path = tmp_path / "private" / item["requests_file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps({"id": rid, "messages": [{"role": "user", "content": rid + "\u2028real separator"}],
                                                    "input_tokens": 5}, ensure_ascii=False) + "\n" for rid in ids))
                item["requests_sha256"] = compare.native.sha(path)
            if index == "warmup":
                dataset["warmup"] = item
            else:
                item.update(block_id="block-" + index, arm_order=ORDERS[dataset_index][int(index)], calendar_file=f"calendars/{name}-{index}.json")
                item["calendar_sha256"] = write_json(tmp_path / item["calendar_file"], {
                    "request_ids": ids, "arrival_offsets_ns": [i * 100 for i in range(count)], "horizon_ns": count * 100})
                dataset["blocks"].append(item)
        protocol["datasets"].append(dataset)
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    return protocol, path


def test_explicit_balanced_schedule_is_sixty_fresh_sessions(tmp_path):
    protocol, _ = fixture_protocol(tmp_path)
    jobs = compare.schedule(protocol)
    assert len(jobs) == len({job["session_id"] for job in jobs}) == 60
    assert [j["arm"] for j in jobs[:3]] == ORDERS[0][0]
    for start in range(0, 60, 3):
        assert {j["arm"] for j in jobs[start:start+3]} == set(compare.ARMS)
        assert len({(j["dataset"], j["block_id"]) for j in jobs[start:start+3]}) == 1
    broken = copy.deepcopy(protocol)
    for block in broken["datasets"][0]["blocks"]:
        block["arm_order"] = ["off", "on", "custom"]
    with pytest.raises(ValueError, match="every order position"):
        compare.schedule(broken)


def test_draft_schedule_never_admits_final_execution(tmp_path):
    protocol, path = fixture_protocol(tmp_path)
    protocol.update(status="draft", final_execution_allowed=False)
    protocol["custom"] = {k: None for k in protocol["custom"]}
    protocol["pilot_admission"] = {"file": None, "sha256": None}
    for dataset in protocol["datasets"]:
        for block in dataset["blocks"]:
            block["requests_sha256"] = None
    write_json(path, protocol)
    assert len(compare.schedule(protocol, allow_draft=True)) == 60
    with pytest.raises(ValueError, match="frozen"):
        compare.load_admitted_protocol(path)


def test_failed_pilot_gate_stops_before_any_final_payload_access(tmp_path, monkeypatch):
    protocol, path = fixture_protocol(tmp_path)
    admission = json.loads((tmp_path / "admission.json").read_text())
    admission["performance_gate_passed"] = False
    protocol["pilot_admission"]["sha256"] = write_json(tmp_path / "admission.json", admission)
    write_json(path, protocol)
    def forbidden(*args, **kwargs):
        raise AssertionError("final data opened before pilot admission")
    monkeypatch.setattr(compare, "admit_and_copy_inputs", forbidden)
    with pytest.raises(ValueError, match="both custom pilot gates"):
        compare.run(SimpleNamespace(protocol=path, execute=True))


def test_generation_contract_cannot_change_even_when_all_arms_would_match(tmp_path):
    protocol, path = fixture_protocol(tmp_path)
    protocol["generation"]["max_tokens"] = 128
    write_json(path, protocol)
    with pytest.raises(ValueError, match="generation contract"):
        compare.load_admitted_protocol(path)


def mock_source_metadata(protocol, monkeypatch):
    metadata = {"status": "passed", "by_id": {rid: {"id": rid} for dataset in protocol["datasets"]
                for item in [dataset["warmup"], *dataset["blocks"]] for rid in item["request_ids"]}}
    monkeypatch.setattr(compare, "load_source_metadata", lambda *args: metadata)
    monkeypatch.setattr(compare, "verify_private_request", lambda row, meta: {"status": "passed", "id": row["id"]})
    return metadata


def test_copied_payloads_preserve_unicode_and_pair_one_immutable_calendar(tmp_path, monkeypatch):
    protocol, path = fixture_protocol(tmp_path, payloads=True)
    mock_source_metadata(protocol, monkeypatch)
    compare.load_admitted_protocol(path)
    private, public = tmp_path / "output-private", tmp_path / "output-public"
    inputs = compare.admit_and_copy_inputs(protocol, path, tmp_path / "private", private, public)
    item = inputs["wildchat"]["block-0"]
    assert compare.native.sha(item["requests"]) == item["requests_sha256"]
    assert len(compare.native.read_jsonl(item["requests"])) == 16
    assert item["requests"].stat().st_mode & 0o777 == 0o600
    assert compare.native.sha(item["calendar"]) == item["calendar_sha256"]
    assert set(inputs) == set(compare.DATASETS)
    with pytest.raises(ValueError, match="confined"):
        compare.private_path(tmp_path / "private", "../secret")


def test_native_on_arguments_match_the_unchanged_native_dry_run(tmp_path):
    snapshot = tmp_path / "snapshot with spaces"
    for role in ("primary", "worker"):
        output = subprocess.check_output(["bash", str(compare.ROOT / "full_serving/launch_native_draft.sh"),
                                          "on", role, str(snapshot), "10.0.0.1"], text=True)
        command = shlex.split(output.split("\n")[1])
        assert command == ["vllm", *compare.native_on_arguments(snapshot, "10.0.0.1", role)]


def test_custom_environment_does_not_leak_into_native_controls(tmp_path):
    protocol, _ = fixture_protocol(tmp_path)
    env = compare.clean_server_environment({"SLURM_JOB_ID": compare.native.JOB, "SLURM_GPUS": "0", "CUDA_VISIBLE_DEVICES": "",
        "SLURM_STEP_ID": "controller-step", "SLURM_CPU_BIND": "controller-binding",
        "FULL_SERVING_CUSTOM_MODE": "resident_v1", "FULL_SERVING_CUSTOM_CONFIG": "secret", "VLLM_PLUGINS": "cs471_resident,other",
        "PYTHONPATH": str(compare.ROOT / "full_serving/bootstrap") + ":/some/normal/path"}, protocol, for_srun=True)
    assert env["SLURM_JOB_ID"] == compare.native.JOB
    assert "SLURM_GPUS" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert "SLURM_STEP_ID" not in env and "SLURM_CPU_BIND" not in env
    assert "FULL_SERVING_CUSTOM_MODE" not in env and "FULL_SERVING_CUSTOM_CONFIG" not in env
    assert env["VLLM_PLUGINS"] == "other" and env["PYTHONPATH"] == "/some/normal/path"


@pytest.mark.parametrize("arm", ["off", "on", "custom"])
def test_in_step_launch_preserves_allocation_and_preflight_identity(tmp_path, monkeypatch, arm):
    protocol, path = fixture_protocol(tmp_path)
    run_dir = tmp_path / "server-step"
    run_dir.mkdir()
    allocation = {"SLURM_JOB_ID": compare.native.JOB, "SLURM_STEP_ID": "41", "SLURM_STEP_NUM_TASKS": "2",
                  "SLURM_STEP_GPUS": "0,1,2,3", "SLURM_PROCID": "1", "SLURM_LOCALID": "0",
                  "SLURMD_NODENAME": "kh077", "SLURM_JOB_NODELIST": "kh[074,077]",
                  "SLURM_CPUS_PER_TASK": "56", "SLURM_MEM_PER_NODE": "307200", "SLURM_GPUS_PER_NODE": "4",
                  "SLURM_CPU_BIND": "quiet,mask_cpu:0x00ff", "SLURM_GPU_BIND": "none",
                  "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    for key, value in allocation.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FULL_SERVING_PROTOCOL_SHA256", compare.native.sha(path))
    monkeypatch.setenv("FULL_SERVING_RUN_DIR", str(run_dir))
    monkeypatch.setenv("FULL_SERVING_CHECKPOINT_MANIFEST", str(tmp_path / "checkpoint.json"))
    monkeypatch.setenv("FULL_SERVING_CUSTOM_MODE", "stale-mode")
    monkeypatch.setenv("FULL_SERVING_CUSTOM_CONFIG", "stale-config")
    monkeypatch.setenv("FULL_SERVING_CUSTOM_UNREVIEWED", "must-be-removed")
    monkeypatch.setenv("VLLM_PLUGINS", "cs471_resident,other")
    monkeypatch.setenv("PYTHONPATH", str(compare.ROOT / "full_serving/bootstrap") + ":/normal/path")
    observed = []
    def check_common(env):
        assert {key: env.get(key) for key in allocation} == allocation
        assert env.get("FULL_SERVING_CUSTOM_UNREVIEWED") is None
        assert env.get("VLLM_PLUGINS") == "other"
    def preflight(command, **kwargs):
        env = kwargs["env"]
        check_common(env)
        assert not any(key.startswith("FULL_SERVING_CUSTOM_") for key in env)
        assert env["PYTHONPATH"] == "/normal/path"
        # These are the values required by the native sampler-report guard.
        sampler_report = {"job_id": compare.native.JOB, "step_id": "41", "hostname": "kh077"}
        assert sampler_report["job_id"] == env["SLURM_JOB_ID"]
        assert sampler_report["step_id"] == env["SLURM_STEP_ID"]
        assert sampler_report["hostname"] == env["SLURMD_NODENAME"]
        observed.append("preflight")
        return SimpleNamespace(returncode=0, stdout="preflight passed\n")
    class ExecCaptured(Exception):
        pass
    def execute(file, command, env):
        check_common(env)
        if arm == "custom":
            assert env["FULL_SERVING_CUSTOM_MODE"] == "resident_v1"
            assert env["FULL_SERVING_CUSTOM_CONFIG"] == str(tmp_path / "config.json")
            assert env["FULL_SERVING_CUSTOM_CONFIG_SHA256"] == protocol["custom"]["config_sha256"]
        else:
            assert not any(key.startswith("FULL_SERVING_CUSTOM_") for key in env)
            assert env["PYTHONPATH"] == "/normal/path"
        observed.append("exec")
        raise ExecCaptured
    monkeypatch.setattr(compare.subprocess, "run", preflight)
    monkeypatch.setattr(compare.os, "execvpe", execute)
    args = SimpleNamespace(arm=arm, role="worker", snapshot=tmp_path / "snapshot",
                           primary_ip="10.91.54.74", protocol=path)
    with pytest.raises(ExecCaptured):
        compare.launch_node(args)
    assert observed == (["preflight", "exec"] if arm == "custom" else ["exec"])


def write_bindings(directory, protocol, path):
    directory.mkdir()
    (directory / "server-step-primary.txt").write_text(compare.native.JOB + ".42")
    (directory / "server-step-worker.txt").write_text(compare.native.JOB + ".42")
    config = json.loads((path.parent / "config.json").read_text())
    for rank in range(8):
        report = {"schema": "cs471_full_resident_bindings_v1", "status": "bound", "ep_rank": rank, "mode": "resident_v1",
            "job_id": compare.native.JOB, "step_id": "42", "config_sha256": protocol["custom"]["config_sha256"],
            "source_manifest_sha256": protocol["custom"]["source_manifest_sha256"], "physical_to_logical_map": config["physical_to_logical_map"],
            "native_vllm_commit": config["vllm_commit"], "model_revision": config["model_revision"],
            "native_eplb_state_created": False, "native_observation_history_active": False, "native_profile_migration_active": False,
            "native_async_worker_active": False, "native_control_step_active": False, "normal_commit_applicable": False,
            "layer_count": 48, "native_source_sha256": {"native.py": "hash"}, "startup_diagnostic": None,
            "layers": [{"layer": f"model.layers.{i}.mlp", "physical_slots": list(range(rank*20,rank*20+20)),
                        "resident_logical_experts": config["physical_to_logical_map"][rank*20:rank*20+20],
                        "w13_shape": [20,1536,2048], "w2_shape": [20,2048,768]} for i in range(48)]}
        write_json(directory / f"resident-bindings-rank{rank}.json", report)


@pytest.mark.parametrize("damage", [None, "missing_rank", "wrong_map", "wrong_step", "missing_layer", "native_control", "wrong_hash"])
def test_custom_requires_all_ranks_all_layers_and_exact_static_source_binding(tmp_path, damage):
    protocol, path = fixture_protocol(tmp_path)
    directory = tmp_path / "custom"
    write_bindings(directory, protocol, path)
    target = directory / "resident-bindings-rank7.json"
    if damage == "missing_rank":
        target.unlink()
    elif damage:
        report = json.loads(target.read_text())
        if damage == "wrong_map": report["physical_to_logical_map"][0] = 99
        if damage == "wrong_step": report["step_id"] = "1"
        if damage == "missing_layer": report["layers"].pop()
        if damage == "native_control": report["native_control_step_active"] = True
        if damage == "wrong_hash": report["config_sha256"] = "wrong"
        write_json(target, report)
    if damage:
        with pytest.raises((ValueError, FileNotFoundError)):
            compare.verify_custom_bindings(directory, protocol, path)
    else:
        result = compare.verify_custom_bindings(directory, protocol, path)
        assert result["status"] == "passed" and len(result["reports"]) == 8
        assert result["all_layer_numerical_equivalence_proved"] is False


@pytest.mark.parametrize("damage", [None, "failed_check", "changed_diagnostic", "probe_in_timing_mode"])
def test_startup_probe_evidence_is_hash_bound_and_never_silently_timed(tmp_path, damage):
    protocol, path = fixture_protocol(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["startup_diagnostic"] = damage != "probe_in_timing_mode"
    protocol["custom"]["config_sha256"] = write_json(config_path, config)
    directory = tmp_path / "custom"
    write_bindings(directory, protocol, path)
    flags = ("logical_identity_exact", "physical_cpu_oracle_exact", "one_home_policy_checked_by_cpu_oracle",
             "routing_weights_unchanged", "token_conservation_exact", "gathered_rank_agreement",
             "native_expert_outputs_close", "direct_fp32_outputs_close")
    for rank in range(8):
        probe = {"schema": "cs471_full_resident_startup_diagnostic_v1", "status": "passed", "ep_rank": rank,
            "job_id": compare.native.JOB, "step_id": "42", "config_sha256": protocol["custom"]["config_sha256"],
            "source_manifest_sha256": protocol["custom"]["source_manifest_sha256"],
            "weight_check": {"physical_layer_experts": 7680, "logical_layer_experts": 6144,
                "duplicated_logical_layer_experts": 1536, "full_bf16_replica_hashes_equal": True},
            "local_weight_records": [{} for _ in range(960)],
            "native_expert_numerical_checks": [{key: True for key in flags} for _ in range(9)]}
        if damage == "failed_check" and rank == 7:
            probe["native_expert_numerical_checks"][0]["logical_identity_exact"] = False
        probe_path = directory / f"resident-startup-diagnostic-rank{rank}.json"
        digest = write_json(probe_path, probe)
        binding_path = directory / f"resident-bindings-rank{rank}.json"
        binding = json.loads(binding_path.read_text())
        binding["startup_diagnostic"] = {"path": str(probe_path), "sha256": digest, "status": "passed", "timed_serving_measurement": False}
        write_json(binding_path, binding)
        if damage == "changed_diagnostic" and rank == 7:
            probe_path.write_text(probe_path.read_text() + " ")
    if damage:
        with pytest.raises(ValueError):
            compare.verify_custom_bindings(directory, protocol, path)
    else:
        assert compare.verify_custom_bindings(directory, protocol, path)["status"] == "passed"


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_controller_fresh_servers_shared_calendar_and_failure_cleanup(tmp_path, monkeypatch, fail_warmup):
    protocol, path = fixture_protocol(tmp_path, payloads=True)
    mock_source_metadata(protocol, monkeypatch)
    selected = compare.schedule(protocol)[:3]
    monkeypatch.setattr(compare, "schedule", lambda _: selected)
    monkeypatch.setenv("SLURM_JOB_ID", compare.native.JOB)
    monkeypatch.setattr(compare.socket, "gethostbyname", lambda _: "127.0.0.1")
    monkeypatch.setattr(compare.native, "wait_server", lambda *args: {"data": [{"id": "qwen3-full"}]})
    monkeypatch.setattr(compare.native, "get_url", lambda *_: b"# synthetic test metrics\n")
    monkeypatch.setattr(compare, "verify_custom_bindings", lambda *args: {"status": "passed"})
    starts, stops, clients = [], [], []
    def popen(command, **kwargs):
        process = SimpleNamespace(index=len(starts))
        starts.append((process, command, kwargs["env"]))
        return process
    monkeypatch.setattr(compare.subprocess, "Popen", popen)
    monkeypatch.setattr(compare.native, "stop_server", lambda process, directory: stops.append((process, directory)))
    def command(argv, output=None, timeout=None):
        if argv[:3] == ["scontrol", "show", "job"]:
            value = json.dumps({"jobs": [{"job_state": "RUNNING", "nodes": "n[0-1]"}]})
        elif argv[:3] == ["scontrol", "show", "hostnames"]:
            value = "n0 n1"
        elif "full_serving.preflight" in argv:
            write_json(Path(argv[argv.index("--output") + 1]), {"complete": True})
            value = "synthetic preflight"
        else:
            assert "full_serving.client" in argv
            clients.append(argv)
            phase = argv[argv.index("--load-id") + 1]
            if fail_warmup and phase == "warmup-01":
                raise RuntimeError("synthetic client failure")
            target = Path(argv[argv.index("--output") + 1])
            target.write_text(json.dumps({"type": "run_summary", "valid": True,
                                          "completed_requests": 16 if phase == "measured" else 8}) + "\n")
            value = "synthetic client completed"
        if output:
            Path(output).write_text(value)
        return value
    monkeypatch.setattr(compare.native, "command", command)
    args = SimpleNamespace(protocol=path, private_data=tmp_path / "private", snapshot=tmp_path / "snapshot",
        output=tmp_path / "output", private_output=tmp_path / "private-output", source_revision="a" * 40, execute=True)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="synthetic client failure"):
            compare.run(args)
        assert len(starts) == len(stops) == 1
        assert all("--calendar" not in client for client in clients)
        state = json.loads((args.output / "status.json").read_text())
        assert state["state"] == "failed" and state["completed_sessions"] == []
    else:
        compare.run(args)
        assert len(starts) == len(stops) == 3
        measured = [client for client in clients if "--calendar" in client]
        assert len(measured) == 3
        assert len({client[client.index("--calendar") + 1] for client in measured}) == 1
        assert len({client[client.index("--requests") + 1] for client in measured}) == 1
        assert [client[client.index("--arm") + 1] for client in measured] == ORDERS[0][0]
        for i in range(0, 9, 3):
            assert "--logprobs" in clients[i] and "--logprobs" not in clients[i + 1]
            assert "--logprobs" not in clients[i + 2]
        assert len({env["FULL_SERVING_RUN_DIR"] for _, _, env in starts}) == 3
        assert all("FULL_SERVING_CUSTOM_MODE" not in env for _, _, env in starts)
