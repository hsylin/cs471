import json
import math
from pathlib import Path

import pytest

from full_serving.analyze import summarize_run
from full_serving import custom_report as report
from test_full_serving_custom_compare import fixture_protocol, write_json, mock_source_metadata


def write_run(path, dataset, block, arm, phase, factor=1, calendar=None):
    warmup = phase != "measured"
    item = dataset["warmup"] if warmup else block
    diagnostics = phase == "warmup-00"
    if warmup:
        calendar = {"kind": "synthetic_pilot_burst", "rate_requests_per_s": 0.0,
                    "arrival_offsets_ns": [0] * 8, "request_ids": item["request_ids"]}
        calendar_hash = report.hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest()
    else:
        calendar_hash = block["calendar_sha256"]
    header = {"type": "run_header", "request_count": item["request_count"], "arm": arm,
        "requests_sha256": item["requests_sha256"], "calendar_sha256": calendar_hash,
        "request_ids_sha256": report.compare.canonical_hash(item["request_ids"]), "block_id": dataset["dataset"] + "-" + block["block_id"],
        "load_id": phase, "pair_id": dataset["dataset"] + "-" + block["block_id"] + "-" + phase,
        "model": "qwen3-full", "max_tokens": 2048, "max_model_len": 32768, "temperature": 0,
        "enable_thinking": False, "ignore_eos": False, "logprobs": diagnostics, "max_inflight": 64,
        "http_timeout_s": 1800, "client_source_sha256": "client-hash",
        "arrival_kind": "synthetic_pilot_burst" if warmup else "external_calendar",
        "synthetic_rate_requests_per_s": 0.0 if warmup else None}
    epoch = 10**9
    records = []
    for i, rid in enumerate(item["request_ids"]):
        offset = calendar["arrival_offsets_ns"][i]
        scheduled = epoch + offset
        send = scheduled + 100
        first = send + int(2e6 * factor)
        last = first + int(1e6 * factor)
        end = scheduled + int((10 + i) * 1e6 * factor)
        row = {"type": "request", "request_id": rid, "request_index": i, "success": True,
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, "expected_input_tokens": 5,
            "input_sha256": report.compare.canonical_hash([{"role": "user", "content": rid}]),
            "payload_sha256": "payload-" + rid, "output_sha256": "output-" + rid,
            "finish_reason": "stop", "scheduled_ns": scheduled, "send_ns": send, "first_content_ns": first,
            "last_content_ns": last, "end_ns": end, "arrival_offset_ns": offset,
            "response_from_schedule_ns": end - scheduled, "response_from_send_ns": end - send,
            "ttft_from_schedule_ns": first - scheduled, "ttft_from_send_ns": first - send,
            "client_semaphore_wait_ns": 0, "client_schedule_lag_ns": 0, "client_send_delay_ns": 100,
            "tpot_stream_boundary_estimate_ns": (last - first) / 2, "chunks": [{"content_chars": 3}],
            "diagnostic_requested": diagnostics, "http_status": 200, "saw_done": True, "usage_events": 1,
            "error_type": None, "error_message": None}
        if diagnostics:
            row.update(diagnostic_passed=True, prompt_token_ids_count=5, completion_token_ids_count=3, diagnostic_logprob_count=3)
        records.append(row)
    last_end = max(r["end_ns"] for r in records)
    horizon = calendar.get("horizon_ns", 0)
    completion = max(last_end, epoch + horizon)
    timing = {"epoch_monotonic_ns": epoch, "last_request_end_ns": last_end, "completion_ns": completion,
              "completed_elapsed_ns": completion - epoch, "offered_horizon_ns": horizon,
              "drain_ns": max(0, last_end - epoch - horizon), "client_finished_ns": completion + 100}
    summary = {"type": "run_summary", **timing, **summarize_run(records, timing)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in [header, *records, summary]) + "\n")


def write_experiment(tmp_path, monkeypatch):
    protocol, protocol_path = fixture_protocol(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "frozen-protocol.json", protocol)
    schedule = report.compare.schedule(protocol)
    write_json(root / "schedule.json", schedule)
    write_json(root / "controller-source.json", {"files": {"full_serving/client.py": "client-hash"}})
    (root / "implementation").mkdir()
    for source, dest in (("config.json", "config.json"), ("sources.json", "source-manifest.json"), ("admission.json", "pilot-admission.json")):
        (root / "implementation" / dest).write_bytes((tmp_path / source).read_bytes())
    metadata = mock_source_metadata(protocol, monkeypatch)
    admission = {"status": "passed", "requests": {rid: {"status": "passed", "source_metadata_sha256": report.compare.canonical_hash(meta),
        "client_input_sha256": report.compare.canonical_hash([{"role": "user", "content": rid}])} for rid, meta in metadata["by_id"].items()}}
    admission_hash = write_json(root / "request-admission.json", admission)
    write_json(root / "status.json", {"state": "comparison_completed", "completed_sessions": [j["session_id"] for j in schedule],
        "protocol_sha256": report.compare.native.sha(root / "frozen-protocol.json"), "request_admission_sha256": admission_hash})
    monkeypatch.setattr(report.compare, "verify_custom_bindings", lambda *args: {"status": "passed"})
    monkeypatch.setattr(report, "runtime_triplet", lambda *args: {"status": "passed", "common_runtime_fingerprint_sha256": "a" * 64,
                                                                "native_engine_configuration_sha256": "b" * 64})
    monkeypatch.setattr(report, "audit_counters", lambda *args: {"status": "matched"})
    for job in schedule:
        directory = root / job["session_id"]
        write_json(directory / "launch.json", {**job, "started_utc": f'2026-11-10T00:{job["ordinal"]:02d}:00+00:00',
            "command": ["srun", "--jobid=" + report.compare.native.JOB, "--nodes=2", "--ntasks=2", "--gpus-per-node=4",
                        "bash", "/repo/full_serving/custom_compare_node.sh", job["arm"], "snapshot", "127.0.0.1", "protocol"]})
        for role in ("primary", "worker"):
            (directory / f"server-step-{role}.txt").write_text(report.compare.native.JOB + "." + str(job["ordinal"] + 1))
        write_json(directory / "status.json", {**job, "state": "session_completed"})
    for dataset in protocol["datasets"]:
        for block in dataset["blocks"]:
            calendar = json.loads((tmp_path / block["calendar_file"]).read_bytes())
            calendar_path = root / "calendars" / dataset["dataset"] / (block["block_id"] + ".json")
            calendar_path.parent.mkdir(parents=True, exist_ok=True)
            calendar_path.write_bytes((tmp_path / block["calendar_file"]).read_bytes())
            for arm, factor in (("off", 1), ("on", 1.2), ("custom", .9)):
                for phase in ("warmup-00", "warmup-01", "measured"):
                    write_run(root / dataset["dataset"] / block["block_id"] / arm / f"{phase}.jsonl", dataset, block, arm, phase, factor, calendar)
    return root, protocol


def test_intervals_use_four_paired_blocks_and_are_explicitly_exploratory():
    result = report.paired_log_interval([.8, .9, 1.1, 1.2])
    assert result["independent_block_count"] == 4
    assert result["equal_block_geometric_mean_ratio"] == pytest.approx((.8 * .9 * 1.1 * 1.2) ** .25)
    assert result["pointwise_95_percent_interval"][0] < 1 < result["pointwise_95_percent_interval"][1]
    assert "Exploratory" in result["interpretation"] and "df=3" in result["method"]
    assert not report.paired_log_interval([1, 1, 1])["available"]
    assert not report.paired_log_interval([1, None, 1, 1])["available"]
    assert report.ratio(0, 1) == 0 and not report.paired_log_interval([0, 1, 1, 1])["available"]


def test_complete_three_arm_report_keeps_sources_and_blocks_separate(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    result = report.build_report(root)
    assert result["complete"] and result["final_speedup_claim"] is False
    assert len(result["datasets"]) == 5
    for dataset in result["datasets"]:
        assert len(dataset["blocks"]) == 4 and dataset["complete"]
        primary = dataset["summary"]["custom/off"]["mean_response_ms"]
        assert primary["equal_block_geometric_mean_ratio"] == pytest.approx(.9)
        assert primary["useful_gain_margin_met_by_point_only"] is True
        assert primary["confirmatory_acceptance"] is False
        secondary = dataset["summary"]["custom/on"]["mean_response_ms"]
        assert secondary["equal_block_geometric_mean_ratio"] == pytest.approx(.75)
    assert "aggregate" not in result


@pytest.mark.parametrize("damage", ["missing_arm", "failed_request", "calendar_mismatch", "timestamp_identity", "bad_custom_binding", "worker_error"])
def test_invalid_triple_withholds_all_contrasts_and_four_block_interval(tmp_path, monkeypatch, damage):
    root, _ = write_experiment(tmp_path, monkeypatch)
    path = root / "wildchat/block-0/on/measured.jsonl"
    if damage == "missing_arm":
        path.unlink()
    elif damage in ("failed_request", "calendar_mismatch", "timestamp_identity"):
        rows = [json.loads(line) for line in path.read_text().split("\n") if line]
        if damage == "failed_request": rows[1]["success"] = False
        if damage == "calendar_mismatch": rows[0]["calendar_sha256"] = "different"
        if damage == "timestamp_identity": rows[1]["end_ns"] += 1000
        path.write_text("\n".join(json.dumps(row) for row in rows))
    elif damage == "bad_custom_binding":
        def failed(*args): raise ValueError("bad source binding")
        monkeypatch.setattr(report.compare, "verify_custom_bindings", failed)
    elif damage == "worker_error":
        (path.parent / "server-0.err").write_text("ERROR async loop error: transfer failed\n")
    result = report.build_report(root)
    block = result["datasets"][0]["blocks"][0]
    assert not result["complete"] and not block["valid"] and not block["contrasts"]
    assert not result["datasets"][0]["summary"]["custom/off"]["mean_response_ms"]["available"]


@pytest.mark.parametrize("damage", ["elapsed", "drain", "shared_wrong_calendar", "warmup_generation", "warmup_calendar",
                                  "missing_runtime", "missing_counters", "reused_server_step", "wrong_launch_order"])
def test_new_corruption_cases_cannot_survive_metadata_only_pairing(tmp_path, monkeypatch, damage):
    root, _ = write_experiment(tmp_path, monkeypatch)
    directory = root / "wildchat/block-0/off"
    if damage in ("elapsed", "drain", "warmup_generation", "warmup_calendar"):
        phase = "warmup-01" if damage.startswith("warmup") else "measured"
        path = directory / f"{phase}.jsonl"
        rows = [json.loads(line) for line in path.read_text().split("\n") if line]
        if damage == "elapsed": rows[-1]["completed_elapsed_ns"] //= 10
        if damage == "drain": rows[-1]["drain_ns"] = 0
        if damage == "warmup_generation": rows[0]["max_tokens"] = 128
        if damage == "warmup_calendar": rows[1]["arrival_offset_ns"] = 123
        path.write_text("\n".join(json.dumps(row) for row in rows))
    elif damage == "shared_wrong_calendar":
        for arm in report.compare.ARMS:
            path = root / "wildchat/block-0" / arm / "measured.jsonl"
            rows = [json.loads(line) for line in path.read_text().split("\n") if line]
            requests = [r for r in rows if r["type"] == "request"]
            for r in requests:
                new_offset = r["request_index"] * 10**6
                shift = new_offset - r["arrival_offset_ns"]
                r["arrival_offset_ns"] = new_offset
                for key in ("scheduled_ns", "send_ns", "first_content_ns", "last_content_ns", "end_ns"):
                    r[key] += shift
            stored = rows[-1]
            last_end = max(r["end_ns"] for r in requests)
            stored.update(offered_horizon_ns=16*10**6, last_request_end_ns=last_end, completion_ns=last_end,
                          completed_elapsed_ns=last_end-stored["epoch_monotonic_ns"],
                          drain_ns=last_end-stored["epoch_monotonic_ns"]-16*10**6, client_finished_ns=last_end+100)
            path.write_text("\n".join(json.dumps(row) for row in rows))
    elif damage == "missing_runtime":
        from full_serving.custom_runtime import audit_runtime_triplet
        monkeypatch.setattr(report, "runtime_triplet", audit_runtime_triplet)
    elif damage == "missing_counters":
        monkeypatch.setattr(report, "audit_counters", lambda *args: {"status": "unavailable"})
    elif damage == "reused_server_step":
        value = (directory / "server-step-primary.txt").read_text()
        for role in ("primary", "worker"):
            (root / "wildchat/block-0/on" / f"server-step-{role}.txt").write_text(value)
    elif damage == "wrong_launch_order":
        path = directory / "launch.json"
        value = json.loads(path.read_text())
        value["position"] = 0
        write_json(path, value)
    result = report.build_report(root)
    block = result["datasets"][0]["blocks"][0]
    assert not block["valid"] and not block["contrasts"] and not result["complete"]
    assert not result["datasets"][0]["summary"]["custom/off"]["mean_response_ms"]["available"]
    if damage == "missing_counters":
        assert block["arms"]["off"]["http_timing_valid"] is True
        assert block["arms"]["off"]["native_completion_counters"]["status"] == "unavailable"


def test_zero_drain_still_charges_the_entire_frozen_offered_horizon(tmp_path):
    path = tmp_path / "phase.jsonl"
    rid = "one"
    calendar = {"request_ids": [rid], "arrival_offsets_ns": [0], "horizon_ns": 1000}
    header = {"type": "run_header", "calendar_sha256": "frozen", "request_ids_sha256": report.compare.canonical_hash([rid]),
              "max_tokens": 2048, "max_model_len": 32768}
    row = {"type": "request", "request_id": rid, "arrival_offset_ns": 0, "scheduled_ns": 100, "send_ns": 110,
           "end_ns": 200, "success": True, "http_status": 200, "saw_done": True, "usage_events": 1,
           "error_type": None, "error_message": None, "response_from_schedule_ns": 100, "response_from_send_ns": 90,
           "finish_reason": "stop", "usage": {"completion_tokens": 1, "total_tokens": 2}}
    summary = {"type": "run_summary", "epoch_monotonic_ns": 100, "offered_horizon_ns": 1000,
               "last_request_end_ns": 200, "completion_ns": 1100, "completed_elapsed_ns": 1000, "drain_ns": 0,
               "client_finished_ns": 1110}
    path.write_text("\n".join(json.dumps(r) for r in [header, row, summary]))
    result = report.replay_evidence(path, [row], calendar, "frozen")
    assert result["valid"] and result["completed_elapsed_ns"] == 1000 and result["drain_ns"] == 0


def test_hardware_identity_must_remain_fixed_across_blocks(tmp_path, monkeypatch):
    root, _ = write_experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(report, "runtime_triplet", lambda directory, _: {
        "status": "passed", "common_runtime_fingerprint_sha256": ("c" if directory.name == "block-1" else "a") * 64,
        "native_engine_configuration_sha256": "b" * 64})
    result = report.build_report(root)
    assert not result["complete"]
    changed = result["datasets"][0]["blocks"][1]
    assert not changed["valid"] and not changed["contrasts"]
    assert "changed across experiment blocks" in " ".join(changed["issues"])


def test_cache_and_preemption_counters_preserve_labels_and_missing_not_zero(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    name = "vllm:prefix_cache_hits_total"
    before.write_text(f'{name}{{engine="0",model_name="qwen"}} 10\n{name}{{engine="1",model_name="qwen"}} 5\n')
    after.write_text(f'{name}{{engine="0",model_name="qwen"}} 20\n{name}{{engine="1",model_name="qwen"}} 8\n')
    result = report.cache_counter_deltas(before, after)
    assert result["available"]
    assert [r["delta"] for r in result["series"]] == [10, 3]
    assert [r["labels"]["engine"] for r in result["series"]] == ["0", "1"]
    after.write_text(f'{name}{{engine="1",model_name="qwen"}} 4\n')
    broken = report.cache_counter_deltas(before, after)
    assert not broken["available"] and all(r["delta"] is None for r in broken["series"])
