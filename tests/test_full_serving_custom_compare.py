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


def test_copied_payloads_preserve_unicode_and_pair_one_immutable_calendar(tmp_path):
    protocol, path = fixture_protocol(tmp_path, payloads=True)
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
        "FULL_SERVING_CUSTOM_MODE": "resident_v1", "FULL_SERVING_CUSTOM_CONFIG": "secret", "VLLM_PLUGINS": "cs471_resident,other",
        "PYTHONPATH": str(compare.ROOT / "full_serving/bootstrap") + ":/some/normal/path"}, protocol)
    assert env["SLURM_JOB_ID"] == compare.native.JOB
    assert "SLURM_GPUS" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert "FULL_SERVING_CUSTOM_MODE" not in env and "FULL_SERVING_CUSTOM_CONFIG" not in env
    assert env["VLLM_PLUGINS"] == "other" and env["PYTHONPATH"] == "/some/normal/path"


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
