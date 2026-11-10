import copy
import json
from pathlib import Path

import pytest

from full_serving import amend_two_workloads, audit_final_statistics, custom_compare, custom_inputs, custom_report, render_custom_report
from test_full_serving_custom_report import write_experiment
from test_full_serving_custom_compare import write_json

PARENT = Path(__file__).resolve().parents[1] / "results/full-serving/custom-compare-v1/frozen-v2/protocol.json"


def test_actual_amendment_has_six_sessions_and_unchanged_reserved_inputs(tmp_path):
    output = tmp_path / "amended" / "protocol.json"
    result = amend_two_workloads.amend(PARENT, output)
    protocol = json.loads(output.read_bytes())
    assert len(result["sessions"]) == 6 and not result["private_payloads_read"]
    assert len(custom_inputs.load_source_metadata(protocol, output)["by_id"]) == 640
    assert custom_compare.generation_for(protocol, "longbench")["max_tokens"] == 16
    assert custom_compare.generation_for(protocol, "numinamath")["max_tokens"] == 2048
    assert protocol["analysis"]["inferential_intervals"] is False
    # An amended seal cannot authorize selecting a more favorable final block.
    protocol["datasets"][0]["blocks"][0]["request_ids"].reverse()
    protocol["protocol_sha256"] = custom_compare.canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    with pytest.raises(ValueError, match="first reserved group"):
        custom_inputs.load_source_metadata(protocol, output)


def compact_experiment(tmp_path, monkeypatch):
    root, parent = write_experiment(tmp_path, monkeypatch)
    protocol = copy.deepcopy(parent)
    protocol["schema"] = custom_compare.COMPACT_SCHEMA
    protocol["generation_by_dataset"] = {"longbench": {"max_tokens": 16}, "numinamath": {"max_tokens": 2048}}
    protocol["analysis"] = {"formal_hypothesis_tests": False, "inferential_intervals": False}
    protocol["datasets"] = [copy.deepcopy(next(d for d in parent["datasets"] if d["dataset"] == n)) for n in custom_compare.COMPACT_DATASETS]
    for index, d in enumerate(protocol["datasets"]):
        d["blocks"] = d["blocks"][:1]
        d["blocks"][0]["arm_order"] = ["custom", "off", "on"] if index == 0 else ["off", "on", "custom"]
    protocol["protocol_sha256"] = custom_compare.canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
    protocol_hash = write_json(root / "frozen-protocol.json", protocol)
    jobs = custom_compare.schedule(protocol)
    write_json(root / "schedule.json", jobs)
    for job in jobs:
        directory = root / job["session_id"]
        launch = json.loads((directory / "launch.json").read_bytes())
        launch.update(job)
        launch["started_utc"] = f'2026-11-10T00:{job["ordinal"]:02d}:00+00:00'
        write_json(directory / "launch.json", launch)
        write_json(directory / "status.json", {**job, "state": "session_completed"})
        for role in ("primary", "worker"):
            (directory / f"server-step-{role}.txt").write_text("4882284." + str(job["ordinal"] + 1))
        if job["dataset"] == "longbench":
            for phase in ("warmup-00", "warmup-01", "measured"):
                path = directory / (phase + ".jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["max_tokens"] = 16
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    admission = json.loads((root / "request-admission.json").read_bytes())
    ids = {rid for d in protocol["datasets"] for b in [d["warmup"], *d["blocks"]] for rid in b["request_ids"]}
    admission["requests"] = {k: v for k, v in admission["requests"].items() if k in ids}
    ahash = write_json(root / "request-admission.json", admission)
    write_json(root / "status.json", {"state": "comparison_completed", "completed_sessions": [j["session_id"] for j in jobs],
                                      "protocol_sha256": protocol_hash, "request_admission_sha256": ahash})
    return root


def test_both_statistics_paths_use_single_group_without_confidence_intervals(tmp_path, monkeypatch):
    root = compact_experiment(tmp_path, monkeypatch)
    report = custom_report.build_report(root)
    assert report["complete"] and len(report["datasets"]) == 2
    for d in report["datasets"]:
        value = d["summary"]["custom/off"]["mean_response_ms"]
        assert value["available"] and value["independent_block_count"] == 1
        assert value["pointwise_95_percent_interval"] is None
        notes = " ".join(render_custom_report.comparison_notes(d, "custom/off"))
        assert "No confidence interval" in notes
        assert "supports lower response latency" not in notes
    path = tmp_path / "report.json"
    write_json(path, report)
    audit = audit_final_statistics.audit(root, path)
    assert audit["status"] == "passed" and audit["reconstructed_sessions"] == 6
    assert len(render_custom_report.table_rows(report)) == 6


def test_longbench_wrong_generation_budget_invalidates_pair(tmp_path, monkeypatch):
    root = compact_experiment(tmp_path, monkeypatch)
    path = root / "longbench/block-0/off/measured.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["max_tokens"] = 2048
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = custom_report.build_report(root)
    assert not report["complete"] and not report["datasets"][0]["blocks"][0]["valid"]
    assert report["datasets"][0]["blocks"][0]["contrasts"] == {}


def test_phase_label_uses_actual_durations_and_complete_counts():
    rows = [{"metric": "vllm:request_prefill_time_seconds", "available": True, "delta_count": 16, "delta_sum_seconds": 8},
            {"metric": "vllm:request_decode_time_seconds", "available": True, "delta_count": 16, "delta_sum_seconds": 2}]
    result = custom_report.phase_balance({"series": rows})
    assert result["dominant_request_phase"] == "prefill" and result["prefill_fraction"] == .8
    rows[1]["delta_sum_seconds"] = 12
    assert custom_report.phase_balance({"series": rows})["dominant_request_phase"] == "decode"
    rows[1]["delta_count"] = 15
    assert not custom_report.phase_balance({"series": rows})["available"]
