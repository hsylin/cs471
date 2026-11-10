import json

import pytest

from full_serving.analyze import summarize_run
from full_serving.report_pilots import (build_report, eplb_log_evidence, main,
                                        metric_deltas, parse_histograms)


def write_cohort(root, dataset="wildchat", on_multiplier=1.1):
    root.mkdir(parents=True)
    (root / "status.json").write_text(json.dumps({"state": "pilot_completed", "dataset": dataset}))
    splits = {}
    for stage in ("smoke", "calibration", "external"):
        ids = [f"{dataset}:{stage}:{i}" for i in range(2)]
        splits[stage] = {"ids": ids, "request_file_sha256": f"requests-{dataset}-{stage}"}
        for arm, factor in (("off", 1), ("on", on_multiplier)):
            (root / arm).mkdir(exist_ok=True)
            header = {"type": "run_header", "request_count": 2, "arm": arm,
                "requests_sha256": splits[stage]["request_file_sha256"], "calendar_sha256": "same-calendar",
                "request_ids_sha256": f"ids-{dataset}-{stage}", "block_id": f"{dataset}-{stage}",
                "pair_id": f"{dataset}-{stage}", "load_id": stage, "model": "qwen3-full",
                "max_tokens": 2048, "max_model_len": 32768, "temperature": 0, "enable_thinking": False,
                "ignore_eos": False, "logprobs": stage == "smoke", "max_inflight": 64,
                "http_timeout_s": 1800, "client_source_sha256": "same-client"}
            records = []
            for index, rid in enumerate(ids):
                row = {"type": "request", "request_id": rid, "request_index": index, "input_sha256": rid,
                    "expected_input_tokens": 4, "arrival_offset_ns": index * 1000, "success": True,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    "finish_reason": "stop", "output_sha256": f"output-{rid}", "first_content_ns": 2,
                    "chunks": [{"content_chars": 4}], "client_semaphore_wait_ns": 0,
                    "client_schedule_lag_ns": 0, "client_send_delay_ns": 0,
                    "response_from_schedule_ns": factor * 10000, "response_from_send_ns": factor * 10000,
                    "ttft_from_schedule_ns": factor * 1000, "ttft_from_send_ns": factor * 1000,
                    "tpot_stream_boundary_estimate_ns": factor * 5000,
                    "diagnostic_requested": stage == "smoke"}
                if stage == "smoke":
                    row.update(diagnostic_passed=True, prompt_token_ids_count=4, completion_token_ids_count=2,
                               diagnostic_logprob_count=2, completion_token_ids_sha256=f"tokens-{rid}")
                records.append(row)
            timing = {"completed_elapsed_ns": factor * 10000, "drain_ns": factor * 9000, "offered_horizon_ns": 1000}
            summary = {"type": "run_summary", **timing, **summarize_run(records, timing)}
            (root / arm / f"{stage}.jsonl").write_text("\n".join(json.dumps(r) for r in [header, *records, summary]) + "\n")
    (root / "pilot-splits.json").write_text(json.dumps(splits))


def test_metric_deltas_preserve_exact_engine_labels_without_pooling(tmp_path):
    before, after = tmp_path / "before.txt", tmp_path / "after.txt"
    name = "vllm:request_inference_time_seconds"
    before.write_text(f'{name}_sum{{model_name="a,b\\\"c",engine="0"}} 10\n'
                     f'{name}_count{{model_name="a,b\\\"c",engine="0"}} 2\n'
                     f'{name}_sum{{model_name="a,b\\\"c",engine="1"}} 8\n'
                     f'{name}_count{{model_name="a,b\\\"c",engine="1"}} 2\n')
    after.write_text(f'{name}_sum{{engine="0",model_name="a,b\\\"c"}} 14\n'
                    f'{name}_count{{engine="0",model_name="a,b\\\"c"}} 4\n'
                    f'{name}_sum{{engine="1",model_name="a,b\\\"c"}} 20\n'
                    f'{name}_count{{engine="1",model_name="a,b\\\"c"}} 6\n')
    result = metric_deltas(before, after)
    assert result["available"] and result["all_series_valid"]
    rows = {r["labels"]["engine"]: r for r in result["series"]}
    assert set(rows) == {"0", "1"}
    assert rows["0"]["mean_seconds"] == 2 and rows["1"]["mean_seconds"] == 3
    assert rows["0"]["labels"]["model_name"] == 'a,b"c'
    assert "mean_seconds" not in result
    assert "not isolated GPU service" in rows["0"]["semantics"]


def test_metric_reset_missing_series_and_nonfinite_values_fail_closed(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    name = "vllm:request_queue_time_seconds"
    before.write_text(f'{name}_sum{{engine="0"}} 8\n{name}_count{{engine="0"}} 4\n')
    after.write_text(f'{name}_sum{{engine="0"}} 2\n{name}_count{{engine="0"}} 1\n'
                     f'{name}_sum{{engine="1"}} 5\n{name}_count{{engine="1"}} 1\n')
    result = metric_deltas(before, after)
    assert not result["available"] and not result["all_series_valid"]
    assert "reset" in result["series"][0]["issue"]
    assert "zero baseline is not assumed" in result["series"][1]["issue"]
    with pytest.raises(ValueError):
        parse_histograms(f"{name}_sum NaN")
    assert not metric_deltas(before, tmp_path / "missing")["available"]


def test_async_launch_logs_are_separate_from_unobservable_commits(tmp_path):
    assert eplb_log_evidence(tmp_path)["ordinary_async_launch_log_entries"] is None
    (tmp_path / "server-0.err").write_text(
        "INFO Rearranging experts (async mode) (profile)...\n"
        "INFO Rearranged experts (profile) in 0.1 s.\n"
        "INFO Rearranging experts (async mode) ...\n"
        "INFO Rearranging experts (async mode) ...\n"
        "ERROR async loop error (Rank 0): unavailable\n"
        "WARNING async worker (rank=0): layer 12 coordinated stop\n"
        "INFO Async EPLB worker drained for model model0\n")
    report = eplb_log_evidence(tmp_path)
    assert report["ordinary_async_launch_log_entries"] == 2
    assert report["profile_launch_log_entries"] == report["discarding_drain_log_entries"] == 1
    assert report["sync_completion_log_entries"] == 0
    assert report["normal_async_commits"] is None
    assert report["committed_state_consistency_verified"] is False


def test_datasets_and_stages_remain_separate_with_no_pilot_uncertainty(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_cohort(a, "wildchat", 1.2)
    write_cohort(b, "sharegpt", .8)
    result = build_report([a, b])
    assert result["complete"] and not result["final_speedup_claim"]
    assert len(result["cohorts"]) == 2
    for cohort, ratio in zip(result["cohorts"], [1.2, .8]):
        assert set(cohort["stages"]) == {"smoke", "calibration", "external"}
        for stage in cohort["stages"].values():
            pair = stage["pair"]
            assert pair["valid_for_descriptive_pair"]
            assert pair["uncertainty"]["available"] is False
            assert pair["uncertainty"]["independent_blocks"] == 1
            assert pair["on_over_off_ratios_descriptive_only"]["completed_elapsed_s"] == pytest.approx(ratio)
            assert pair["final_speedup_claim"] is False
    assert "aggregate" not in result
    with pytest.raises(ValueError, match="duplicate"):
        build_report([a, a])


def test_background_eplb_failure_does_not_disappear_behind_successful_requests(tmp_path):
    root = tmp_path / "cohort"
    write_cohort(root)
    (root / "on/server-0.err").write_text("ERROR async loop error (Rank 0): transfer failed\n")
    report = build_report([root])
    assert not report["complete"]
    assert "native asynchronous worker error" in " ".join(report["cohorts"][0]["issues"])
    assert not report["cohorts"][0]["stages"]["external"]["pair"]["valid_for_descriptive_pair"]
    assert "on_over_off_ratios_descriptive_only" not in report["cohorts"][0]["stages"]["external"]["pair"]


@pytest.mark.parametrize("damage", ["missing_run", "failed_request", "calendar_mismatch", "usage_mismatch", "arrival_mismatch"])
def test_partial_failures_and_identity_mismatches_withhold_comparison(tmp_path, damage):
    root = tmp_path / "cohort"
    write_cohort(root)
    path = root / "on/external.jsonl"
    if damage == "missing_run":
        path.unlink()
    else:
        rows = [json.loads(line) for line in path.read_text().split("\n") if line]
        if damage == "failed_request":
            rows[1]["success"] = False
        elif damage == "calendar_mismatch":
            rows[0]["calendar_sha256"] = "different-arrivals"
        elif damage == "usage_mismatch":
            rows[1]["usage"]["total_tokens"] = 7
        elif damage == "arrival_mismatch":
            rows[1]["arrival_offset_ns"] += 1
        path.write_text("\n".join(json.dumps(r) for r in rows))
    report = build_report([root])
    assert not report["complete"]
    pair = report["cohorts"][0]["stages"]["external"]["pair"]
    assert not pair["valid_for_descriptive_pair"]
    assert "on_over_off_ratios_descriptive_only" not in pair
    assert main([str(root), "--output", str(tmp_path / "failed.json")]) == 1
    assert main([str(root), "--output", str(tmp_path / "partial.json"), "--allow-partial"]) == 0
    assert json.loads((tmp_path / "partial.json").read_text())["complete"] is False
