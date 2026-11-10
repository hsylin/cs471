import hashlib
import json

import pytest

from full_serving import custom_pilot_report as report
from test_full_serving_custom_report import write_run


def rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def save(path, data):
    path.write_text("\n".join(json.dumps(row) for row in data) + "\n")


def pilot_fixture(tmp_path):
    dirs = {arm: tmp_path / arm for arm in report.ARMS}
    for arm, directory in dirs.items():
        directory.mkdir()
        (directory / "status.json").write_text(json.dumps({"state": "pilot_completed", "arm": arm, "pilot_only": True,
            "final_inputs_opened": False, "completed_phases": [{"dataset": d, "phase": n} for d in report.DATASETS for n in (0, 1)]}))
        for dataset in report.DATASETS:
            item = {"request_ids": [dataset + str(i) for i in range(8)], "request_count": 8, "requests_sha256": "same-inputs"}
            d = {"dataset": dataset, "warmup": item}
            block = {**item, "block_id": "resident-pilot"}
            for number in (0, 1):
                path = directory / f"{dataset}-pass{number}.jsonl"
                write_run(path, d, block, arm, f"warmup-0{number}", {"off": 1, "resident": .9, "placement_only": 1.1}[arm])
                data = rows(path)
                header, records, summary = data[0], data[1:-1], data[-1]
                calendar = {"kind": "synthetic_regular_pilot", "rate_requests_per_s": 2.0,
                            "arrival_offsets_ns": [i * 500_000_000 for i in range(8)], "request_ids": item["request_ids"]}
                header.update(load_id=f"pass{number}", pair_id=dataset + f"-resident-pilot:pass{number}",
                    calendar_sha256=hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest(),
                    arrival_kind="synthetic_regular_pilot", synthetic_rate_requests_per_s=2.0)
                for i, r in enumerate(records):
                    offset = calendar["arrival_offsets_ns"][i]
                    for key in ("scheduled_ns", "send_ns", "first_content_ns", "last_content_ns", "end_ns"):
                        r[key] += offset
                    r["arrival_offset_ns"] = offset
                    r["output_sha256"] = ("r" if arm == "resident" and i == 0 else "b") * 64
                last = max(r["end_ns"] for r in records)
                summary.update(offered_horizon_ns=3_500_000_000, last_request_end_ns=last, completion_ns=last,
                               completed_elapsed_ns=last - summary["epoch_monotonic_ns"],
                               drain_ns=last - summary["epoch_monotonic_ns"] - 3_500_000_000, client_finished_ns=last+100)
                save(path, data)
    return dirs


def test_separate_diagnostic_and_ordinary_passes_and_mechanism_pair(tmp_path):
    result = report.build_report(pilot_fixture(tmp_path))
    assert result["complete"] and result["final_speedup_claim"] is False
    assert result["inferential_intervals"] is None
    assert len(result["datasets"]) == 5
    for d in result["datasets"]:
        assert [p["diagnostic_payload"] for p in d["passes"]] == [True, False]
        for p in d["passes"]:
            c = p["contrasts"]["resident/placement_only"]
            assert c["valid"] and c["equal_generated_texts"] == 7
            assert len(c["requests"]) == 8
            assert c["ratios"]["mean_response_ms"] == pytest.approx(.9/1.1)
            assert not p["arms"]["off"]["cache"]["available"]
            assert p["arms"]["off"]["metrics"]["completed_requests"] == 8
            assert "speed_claim_allowed" not in p["arms"]["off"]["summary"]


@pytest.mark.parametrize("damage", ["elapsed", "wrong_shared_calendar", "diagnostics_disabled", "failed_request", "missing_request"])
def test_corrupt_phase_cannot_make_descriptive_claim(tmp_path, damage):
    dirs = pilot_fixture(tmp_path)
    for arm in report.ARMS if damage == "wrong_shared_calendar" else ("resident",):
        path = dirs[arm] / "wildchat-pass0.jsonl"
        data = rows(path)
        if damage == "elapsed": data[-1]["completed_elapsed_ns"] //= 10
        if damage == "wrong_shared_calendar": data[2]["arrival_offset_ns"] += 10
        if damage == "diagnostics_disabled": data[0]["logprobs"] = False
        if damage == "failed_request": data[1]["success"] = False
        if damage == "missing_request": data.pop(1)
        save(path, data)
    result = report.build_report(dirs)
    assert not result["complete"]
    pair = result["datasets"][0]["passes"][0]["contrasts"]["resident/placement_only"]
    assert not pair["valid"] and pair["ratios"] == {} and pair["requests"] == []


def test_identical_ids_with_different_payloads_fail_pair_and_missing_arm_remains_pending(tmp_path):
    dirs = pilot_fixture(tmp_path)
    path = dirs["resident"] / "wildchat-pass1.jsonl"
    data = rows(path)
    data[1]["payload_sha256"] = "different"
    save(path, data)
    result = report.build_report(dirs)
    assert not result["complete"]
    pair = result["datasets"][0]["passes"][1]["contrasts"]["resident/off"]
    assert not pair["valid"] and "payload" in pair["issues"][0]
    dirs["placement_only"] = tmp_path / "not-yet-collected"
    result = report.build_report(dirs)
    assert not result["complete"]
    assert result["datasets"][0]["passes"][0]["arms"]["placement_only"]["metrics"] is None


def test_controller_phase_closure_and_private_read_guard(tmp_path):
    dirs = pilot_fixture(tmp_path)
    p = dirs["off"] / "status.json"
    status = json.loads(p.read_text())
    status["completed_phases"].pop()
    p.write_text(json.dumps(status))
    assert not report.build_report(dirs)["complete"]
    status["final_inputs_opened"] = True
    p.write_text(json.dumps(status))
    result = report.build_report(dirs)
    assert not result["datasets"][0]["passes"][0]["arms"]["off"]["valid"]


def test_report_cannot_overwrite_evidence(tmp_path):
    dirs = pilot_fixture(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        report.main(["--off", str(dirs["off"]), "--resident", str(dirs["resident"]),
                     "--placement-only", str(dirs["placement_only"]), "--output", str(dirs["off"] / "status.json")])
