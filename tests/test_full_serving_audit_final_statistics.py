import json
import math

import pytest

from full_serving import audit_final_statistics as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def fixture(tmp_path):
    protocol = {"schema": "resident_compare_protocol_v1", "status": "frozen", "final_execution_allowed": True, "datasets": []}
    report = {"schema": "resident_compare_report_v1", "complete": True, "datasets": []}
    for dataset in audit.DATASETS:
        item, reported = {"dataset": dataset, "blocks": []}, {"dataset": dataset, "blocks": [], "summary": {}}
        for index in range(4):
            bid = "block-"+str(index)
            ids = [f"{dataset}-{index}-{i}" for i in range(16)]
            offsets = [i*1_000_000 for i in range(16)]
            calendar = {"request_ids": ids, "arrival_offsets_ns": offsets, "horizon_ns": 1_000_000_000}
            calendar_path = tmp_path/"calendars"/dataset/(bid+".json")
            write(calendar_path, calendar)
            block = {"block_id": bid, "request_ids": ids, "requests_sha256": "a"*64,
                     "calendar_sha256": audit.sha(calendar_path)}
            item["blocks"].append(block)
            rb = {"block_id": bid, "valid": True, "arms": {}, "contrasts": {}}
            for arm, factor in (("off", 1), ("on", 2), ("custom", .5)):
                header = {"type": "run_header", "arm": arm, "request_count": 16, "requests_sha256": "a"*64,
                    "calendar_sha256": block["calendar_sha256"], "request_ids_sha256": audit.canonical(ids),
                    "max_tokens": 2048, "max_model_len": 32768, "logprobs": False, "temperature": 0,
                    "enable_thinking": False, "ignore_eos": False}
                epoch = 10_000_000_000
                records = []
                for i, (rid, offset) in enumerate(zip(ids, offsets)):
                    start, elapsed = epoch+offset, int((10+i)*factor*1_000_000)
                    tokens = 2048 if i == 0 else 10
                    records.append({"type": "request", "request_id": rid, "request_index": i,
                        "success": True, "http_status": 200, "saw_done": True,
                        "scheduled_ns": start, "send_ns": start, "end_ns": start+elapsed,
                        "arrival_offset_ns": offset, "response_from_schedule_ns": elapsed,
                        "response_from_send_ns": elapsed, "expected_input_tokens": 100,
                        "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100+tokens},
                        "finish_reason": "length" if i == 0 else "stop", "input_sha256": audit.canonical(rid),
                        "payload_sha256": audit.canonical({"request": rid})})
                summary = {"type": "run_summary", "epoch_monotonic_ns": epoch,
                    "last_request_end_ns": records[-1]["end_ns"], "completion_ns": epoch+1_000_000_000,
                    "completed_elapsed_ns": 1_000_000_000, "offered_horizon_ns": 1_000_000_000,
                    "drain_ns": 0, "client_finished_ns": epoch+1_000_000_100}
                path = tmp_path/dataset/bid/arm/"measured.jsonl"
                path.parent.mkdir(parents=True)
                # HTTP completion order is deliberately different from input order.
                path.write_text("".join(json.dumps(row)+"\n" for row in [header, *records[::-1], summary]))
                rb["arms"][arm] = {"summary": {"response_from_schedule_ms": {"mean": 17.5*factor, "p95": 24.25*factor},
                    "completed_elapsed_s": 1., "drain_s": 0., "completed_requests": 16,
                    "completed_output_tokens": 2198, "output_cap_completions": 1}}
            for base, ratio in (("off", .5), ("on", .25)):
                rb["contrasts"]["custom/"+base] = {"ratios": {"mean_response_ms": ratio,
                    "p95_response_ms": ratio, "completed_elapsed_s": 1.}}
            reported["blocks"].append(rb)
        for base, ratio in (("off", .5), ("on", .25)):
            reported["summary"]["custom/"+base] = {}
            for metric in audit.METRICS:
                value = 1. if metric == "completed_elapsed_s" else ratio
                reported["summary"]["custom/"+base][metric] = {"available": True, "block_ratios": [value]*4,
                    "independent_block_count": 4, "equal_block_geometric_mean_ratio": value,
                    "pointwise_95_percent_interval": [value, value]}
        protocol["datasets"].append(item)
        report["datasets"].append(reported)
    write(tmp_path/"frozen-protocol.json", protocol)
    report["protocol_sha256"] = audit.sha(tmp_path/"frozen-protocol.json")
    write(tmp_path/"report.json", report)
    return tmp_path/"report.json"


def test_independent_reconstruction_matches_all_sixty_sessions_and_horizon(tmp_path):
    report = audit.audit(tmp_path, fixture(tmp_path))
    assert report["status"] == "passed" and report["reconstructed_sessions"] == 60
    assert report["differences"] == report["incomplete"] == []
    arm = report["datasets"][0]["blocks"][0]["arms"]["custom"]
    assert arm["mean_response_ms"] == 8.75 and arm["p95_response_ms"] == 12.125
    assert arm["completed_elapsed_s"] == 1 and arm["drain_s"] == 0
    assert arm["completed_output_tokens"] == 2198 and arm["output_cap_completions"] == 1
    assert report["private_payloads_read"] is False


def test_interval_uses_four_log_ratios_and_sample_variance():
    result = audit.interval([math.exp(x) for x in (-3, -1, 1, 3)])
    half_width = 3.182446305284263*math.sqrt(5/3)
    assert result["equal_block_geometric_mean_ratio"] == pytest.approx(1)
    assert result["pointwise_95_percent_interval"] == pytest.approx([math.exp(-half_width), math.exp(half_width)])


@pytest.mark.parametrize("damage", ["reported_interval", "paired_input", "elapsed"])
def test_mismatches_are_identified_without_a_successful_subset(tmp_path, damage):
    report_path = fixture(tmp_path)
    if damage == "reported_interval":
        report = json.loads(report_path.read_text())
        report["datasets"][0]["summary"]["custom/off"]["mean_response_ms"]["pointwise_95_percent_interval"][1] = .9
        write(report_path, report)
    else:
        path = tmp_path/"wildchat/block-0/custom/measured.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "paired_input": rows[1]["input_sha256"] = "b"*64
        if damage == "elapsed": rows[-1]["completed_elapsed_ns"] //= 10
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    report = audit.audit(tmp_path, report_path)
    assert report["status"] == "failed" and report["differences"]
    if damage != "reported_interval":
        assert report["datasets"][0]["summary"] == {}


def test_unfinished_session_is_incomplete_and_never_yields_four_block_interval(tmp_path):
    report_path = fixture(tmp_path)
    (tmp_path/"wildchat/block-0/custom/measured.jsonl").unlink()
    report = audit.audit(tmp_path, report_path)
    assert report["status"] == "incomplete" and not report["differences"]
    assert report["datasets"][0]["summary"] == {}
    assert report["reconstructed_sessions"] == 59
