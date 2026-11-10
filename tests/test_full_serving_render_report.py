import copy
import csv
import json

import pytest

from full_serving.render_report import comparison_rows, main, render_html


def fixture_report():
    cohorts = []
    for dataset, ratio in (("wildchat", 1.1), ("sharegpt", .8)):
        stages = {}
        for stage in ("external", "calibration", "smoke"):
            arms = {}
            for arm, multiplier in (("off", 1), ("on", ratio)):
                arms[arm] = {
                    "valid": True, "issues": [],
                    "header": {"max_tokens": 2048, "block_id": f"{dataset}-{stage}", "arrival_kind": "external_calendar"},
                    "summary": {
                        "offered_requests": 2, "completed_requests": 2, "failed_requests": 0,
                        "response_from_schedule_ms": {"mean": 1000 * multiplier, "p95": 2000 * multiplier},
                        "ttft_from_send_ms": {"mean": 30, "p95": 50},
                        "completed_elapsed_s": 4 * multiplier, "drain_s": 2 * multiplier,
                        "completed_output_tokens_per_s": 250 / multiplier, "completed_output_tokens": 1000,
                        "output_cap_completions": 0, "completed_input_tokens": 1200,
                    },
                    "native_metrics": {"available": False, "series": [], "issue": "snapshots missing"},
                }
            stages[stage] = {"arms": arms, "pair": {
                "valid_for_descriptive_pair": True, "issues": [], "completion_token_counts_equal": True,
                "generated_text_hashes_equal": False, "generated_token_ids_equal": False,
                "diagnostic_token_identity_available": stage == "smoke", "numerical_review_required": True,
                "on_over_off_ratios_descriptive_only": {"response_from_schedule_ms": {"mean": ratio, "p95": ratio},
                    "completed_elapsed_s": ratio, "completed_output_tokens_per_s": 1 / ratio},
            }}
        cohorts.append({"dataset": dataset, "result_directory": f"results/{dataset}", "controller_state": "pilot_completed",
                        "complete": True, "issues": [], "stages": stages, "native_eplb": {}})
    return {"schema": "full_serving_pilot_report_v1", "complete": True, "final_speedup_claim": False,
            "created_utc": "2026-11-10T00:00:00Z", "cohorts": cohorts}


def test_separate_datasets_stages_and_arms_without_inference_or_pooling():
    report = fixture_report()
    rows = comparison_rows(report)
    assert len(rows) == 12
    assert {(r["dataset"], r["stage"], r["arm"]) for r in rows} == {
        (dataset, stage, arm) for dataset in ("wildchat", "sharegpt")
        for stage in ("external", "calibration", "smoke") for arm in ("off", "on")}
    on = [r for r in rows if r["stage"] == "external" and r["arm"] == "on"]
    assert [r["on_vs_off_elapsed_change_pct"] for r in on] == pytest.approx([10, -20])
    assert all(r["final_speedup_claim"] is False for r in rows)
    assert all("unavailable" in r["independent_run_interval"] for r in rows)
    html = render_html(report, rows, "report.json", "hash")
    assert html.index("External-arrival workload") < html.index("Calibration workload")
    assert "Smoke is a correctness diagnostic, not a performance headline" in html
    assert "not equal-work speedup estimates" in html


def test_invalid_arm_preserves_errors_but_withholds_successful_subset_timings():
    report = fixture_report()
    phase = report["cohorts"][0]["stages"]["external"]
    phase["arms"]["on"]["valid"] = False
    phase["arms"]["on"]["issues"] = ["one failed request"]
    phase["arms"]["on"]["summary"].update(completed_requests=1, failed_requests=1)
    phase["pair"]["valid_for_descriptive_pair"] = False
    phase["pair"]["issues"] = ["both complete failure-free arms required"]
    # Retained stale ratios cannot accidentally appear in the rendered comparison.
    row = next(r for r in comparison_rows(report) if r["dataset"] == "wildchat" and r["stage"] == "external" and r["arm"] == "on")
    assert row["failed_requests"] == row["completed_requests"] == 1
    for key in ("response_mean_ms", "completed_elapsed_s", "output_tokens_per_s", "on_vs_off_elapsed_change_pct"):
        assert row[key] is None
    assert "one failed request" in row["issues"]


def test_missing_arm_and_identity_mismatch_are_exposed_without_comparison():
    report = fixture_report()
    report["complete"] = False
    phase = report["cohorts"][0]["stages"]["external"]
    phase["arms"].pop("on")
    phase["pair"].update(valid_for_descriptive_pair=False, issues=["calendar mismatch"])
    rows = comparison_rows(report)
    missing = next(r for r in rows if r["dataset"] == "wildchat" and r["stage"] == "external" and r["arm"] == "on")
    assert missing["completed_requests"] is None and not missing["arm_valid"]
    assert "arm record missing" in missing["issues"] and "calendar mismatch" in missing["issues"]
    html = render_html(report, rows, "report.json", "hash")
    assert "INCOMPLETE / INVALID pilot report" in html
    assert "withheld / unavailable" in html


def test_native_histograms_keep_exact_labels_and_missing_not_zero_and_escape_html():
    report = fixture_report()
    cohort = report["cohorts"][0]
    cohort["dataset"] = '<script>alert("x")</script>'
    native = cohort["stages"]["external"]["arms"]["off"]["native_metrics"]
    native.update(available=True, series=[
        {"metric": "vllm:request_inference_time_seconds", "labels": {"engine": str(engine), "model_name": 'm<a,"b'},
         "available": True, "delta_count": count, "delta_sum_seconds": total, "mean_seconds": mean}
        for engine, count, total, mean in ((0, 2, 4, 2), (1, 4, 20, 5), (2, 0, 0, None))])
    html = render_html(report, comparison_rows(report), "<source>", "hash")
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert '&quot;engine&quot;: &quot;0&quot;' in html and '&quot;engine&quot;: &quot;1&quot;' in html
    assert 'm&lt;a,' in html
    assert '<td>2.0000</td>' in html and '<td>5.0000</td>' in html
    assert '<td>0</td><td>0.0000</td><td>unavailable</td>' in html
    assert "not isolated GPU service" in html and "snapshots missing" in html
    assert 'Normal async commits</th>' in html


def test_cli_writes_public_html_and_csv_with_correct_units_and_no_formula_execution(tmp_path):
    report = fixture_report()
    report["cohorts"][0]["dataset"] = '=HYPERLINK("bad")'
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    prefix = tmp_path / "outputs" / "comparison"
    assert main([str(path), "--output-prefix", str(prefix)]) == 0
    assert prefix.with_suffix(".html").exists()
    with prefix.with_suffix(".csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert rows[0]["dataset"].startswith("'=")
    assert float(rows[0]["response_mean_ms"]) == 1000
    assert rows[0]["on_vs_off_elapsed_change_pct"] == ""
    assert rows[0]["final_speedup_claim"] == "False"


def test_rejects_wrong_schema_and_causal_claim_report():
    report = fixture_report()
    bad = copy.deepcopy(report)
    bad["schema"] = "other"
    with pytest.raises(ValueError, match="expected"):
        comparison_rows(bad)
    report["final_speedup_claim"] = True
    with pytest.raises(ValueError, match="descriptive"):
        comparison_rows(report)
