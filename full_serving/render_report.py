"""Render an explicit full_serving.report_pilots JSON as separate pilot comparisons.

python -m full_serving.render_report REPORT.json --output-prefix outputs/full-serving/native-comparison
Only consumes the public report; never loads private prompts or live run directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path


STAGES = ("external", "calibration", "smoke")
METRICS = {
    "vllm:request_queue_time_seconds": "WAITING",
    "vllm:request_inference_time_seconds": "RUNNING",
}
FIELDS = (
    "dataset", "cohort", "block_id", "stage", "arm", "arm_valid", "pair_valid", "issues",
    "arrival_kind", "offered_requests", "completed_requests", "failed_requests",
    "response_mean_ms", "response_p95_ms", "ttft_from_send_mean_ms", "ttft_from_send_p95_ms",
    "completed_elapsed_s", "drain_s", "output_tokens_per_s", "completed_output_tokens",
    "output_cap_completions", "max_tokens", "completed_input_tokens",
    "on_vs_off_response_mean_change_pct", "on_vs_off_response_p95_change_pct",
    "on_vs_off_elapsed_change_pct", "on_vs_off_output_tokens_per_s_change_pct",
    "generated_text_hashes_equal", "completion_token_counts_equal", "numerical_review_required",
    "independent_run_interval", "final_speedup_claim",
)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _at(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _issues(*groups):
    return "; ".join(dict.fromkeys(str(item) for group in groups for item in (group or [])))


def comparison_rows(report):
    """Keep every arm, including missing/invalid ones; never select successes."""
    if report.get("schema") != "full_serving_pilot_report_v1":
        raise ValueError("expected full_serving_pilot_report_v1")
    if report.get("final_speedup_claim") is not False:
        raise ValueError("renderer accepts descriptive pilot reports only")
    rows = []
    for cohort in report.get("cohorts", []):
        for stage in STAGES:
            phase = cohort.get("stages", {}).get(stage, {})
            pair = phase.get("pair", {})
            pair_valid = pair.get("valid_for_descriptive_pair") is True
            for arm in ("off", "on"):
                data = phase.get("arms", {}).get(arm, {})
                valid = data.get("valid") is True
                summary = data.get("summary", {})
                header = data.get("header", {})
                row = dict.fromkeys(FIELDS)
                row.update(dataset=cohort.get("dataset", "unknown"), cohort=cohort.get("result_directory", "unknown"),
                           stage=stage, arm=arm, arm_valid=valid, pair_valid=pair_valid,
                           issues=_issues(cohort.get("issues"), pair.get("issues"), data.get("issues")),
                           block_id=header.get("block_id", "unavailable"),
                           arrival_kind=header.get("arrival_kind", "unavailable"), max_tokens=header.get("max_tokens"),
                           independent_run_interval="unavailable: single-block fixed-order pilot",
                           final_speedup_claim=False)
                if not valid and not data:
                    row["issues"] = _issues([row["issues"]] if row["issues"] else [], ["arm record missing"])
                # Counts remain visible on failure; successful-subset timings do not.
                for key in ("offered_requests", "completed_requests", "failed_requests"):
                    row[key] = _number(summary.get(key))
                if valid:
                    for dest, source in (("response_mean_ms", ("response_from_schedule_ms", "mean")),
                                         ("response_p95_ms", ("response_from_schedule_ms", "p95")),
                                         ("ttft_from_send_mean_ms", ("ttft_from_send_ms", "mean")),
                                         ("ttft_from_send_p95_ms", ("ttft_from_send_ms", "p95"))):
                        row[dest] = _number(_at(summary, *source))
                    for key in ("completed_elapsed_s", "drain_s", "completed_output_tokens", "output_cap_completions", "completed_input_tokens"):
                        row[key] = _number(summary.get(key))
                    row["output_tokens_per_s"] = _number(summary.get("completed_output_tokens_per_s"))
                for key in ("generated_text_hashes_equal", "completion_token_counts_equal", "numerical_review_required"):
                    row[key] = pair.get(key)
                if pair_valid and valid and arm == "on":
                    ratios = pair.get("on_over_off_ratios_descriptive_only", {})
                    for dest, source in (
                        ("on_vs_off_response_mean_change_pct", ("response_from_schedule_ms", "mean")),
                        ("on_vs_off_response_p95_change_pct", ("response_from_schedule_ms", "p95")),
                        ("on_vs_off_elapsed_change_pct", ("completed_elapsed_s",)),
                        ("on_vs_off_output_tokens_per_s_change_pct", ("completed_output_tokens_per_s",)),
                    ):
                        ratio = _number(_at(ratios, *source))
                        row[dest] = (ratio - 1) * 100 if ratio is not None else None
                rows.append(row)
    return rows


def _cell(value, digits=2, scale=1):
    number = _number(value)
    return f"{number / scale:,.{digits}f}" if number is not None else "unavailable"


def _boolean(value):
    return "yes" if value is True else "no" if value is False else "unavailable"


def _table(headers, records):
    head = "".join(f"<th scope=\"col\">{escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in records)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _arm_table(rows):
    records = []
    for r in rows:
        records.append([
            f'{r["dataset"]} ({r["block_id"]})', r["arm"].upper(),
            f'{_cell(r["completed_requests"], 0)}/{_cell(r["offered_requests"], 0)}; errors {_cell(r["failed_requests"], 0)}',
            _cell(r["response_mean_ms"], scale=1000), _cell(r["response_p95_ms"], scale=1000),
            _cell(r["ttft_from_send_mean_ms"]), _cell(r["ttft_from_send_p95_ms"]),
            _cell(r["completed_elapsed_s"]), _cell(r["drain_s"]), _cell(r["output_tokens_per_s"], 1),
            _cell(r["completed_output_tokens"], 0),
            f'{_cell(r["output_cap_completions"], 0)} / {_cell(r["max_tokens"], 0)}',
            "valid descriptive pair" if r["arm_valid"] and r["pair_valid"] else "INCOMPLETE / INVALID",
        ])
    return _table(["Dataset", "EPLB", "Completed/offered; errors", "Response mean (s)", "Response p95 (s)",
                   "TTFT mean (ms)", "TTFT p95 (ms)", "Elapsed (s)", "Drain (s)", "Output tokens/s",
                   "Output tokens", "Cap finishes / token limit", "Gate"], records)


def _change_table(rows):
    records = []
    for row in rows:
        if row["arm"] != "on":
            continue
        changes = []
        for field in ("on_vs_off_response_mean_change_pct", "on_vs_off_response_p95_change_pct",
                      "on_vs_off_elapsed_change_pct", "on_vs_off_output_tokens_per_s_change_pct"):
            value = _number(row[field])
            changes.append(f"{value:+.2f}%" if value is not None else "withheld / unavailable")
        records.append([f'{row["dataset"]} ({row["block_id"]})', *changes, _boolean(row["completion_token_counts_equal"]),
                        _boolean(row["generated_text_hashes_equal"])])
    return _table(["Dataset", "Response mean change", "Response p95 change", "Elapsed change", "Output tokens/s change",
                   "Per-request output counts match?", "Generated texts match?"], records)


def _native_details(cohort):
    result = ['<details><summary>Native WAITING / RUNNING histograms, with exact engine/model labels</summary>',
              '<p>These are request phase durations, not isolated GPU service or client queue delay. '
              'Each row retains its exact labels; no engine averages are pooled. Missing snapshots and zero-observation '
              'series have no latency estimate. Publication lag can make a phase snapshot incomplete.</p>']
    for stage in ("external", "calibration"):
        records = []
        for arm in ("off", "on"):
            data = _at(cohort, "stages", stage, "arms", arm, "native_metrics") or {}
            relevant = [r for r in data.get("series", []) if r.get("metric") in METRICS]
            if not relevant:
                records.append([arm.upper(), "unavailable", "unavailable", "unavailable", "unavailable", "unavailable",
                                data.get("issue", "requested series missing")])
            for series in relevant:
                valid = series.get("available") is True
                records.append([arm.upper(), METRICS[series["metric"]], json.dumps(series.get("labels", {}), sort_keys=True),
                                _cell(series.get("delta_count") if valid else None, 0),
                                _cell(series.get("delta_sum_seconds") if valid else None, 4),
                                _cell(series.get("mean_seconds") if valid else None, 4),
                                "available" if valid else series.get("issue", "unavailable")])
        result.append(f'<h4>{escape(stage)}</h4>' + _table(
            ["EPLB", "Phase", "Exact labels", "Observations", "Sum (s)", "Mean (s)", "Coverage"], records))
    result.append("</details>")
    return "".join(result)


def render_html(report, rows, source_name, source_sha256):
    status = "Complete pilot records" if report.get("complete") is True else "INCOMPLETE / INVALID pilot report"
    parts = ["<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
             "<title>Full-model native EPLB pilot comparison</title><style>body{font:15px/1.5 system-ui,sans-serif;color:#17212b;margin:2rem auto;padding:0 1.2rem;max-width:1480px}h1{font-size:1.7rem}h2{margin-top:2rem}p{max-width:1100px}.notice{padding:1rem;background:#fff4d4;border-left:4px solid #9c6900}.scroll{overflow-x:auto;margin:1rem 0}table{border-collapse:collapse;font-size:13px;width:100%}th,td{text-align:left;padding:.55rem;border-bottom:1px solid #d9e0e6;vertical-align:top}th{background:#edf2f6}td{font-variant-numeric:tabular-nums}details{margin:1rem 0;padding:.65rem;border:1px solid #d9e0e6}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}a{color:#174f92}</style><body>",
             f"<h1>Full-model native EPLB: OFF versus ON</h1><p><strong>{escape(status)}.</strong> Created {escape(str(report.get('created_utc', 'unavailable')))}.</p>",
             '<p class="notice">Descriptive fixed-order, single-block pilots only. Datasets and stages are kept separate. '
             'No independent-run confidence interval or final speedup claim is available. Requests are not independent experimental repetitions. '
             'A complete report means the input/calendar and failure gates passed; it does not prove numerical or committed-state correctness.</p>',
             '<p>OFF means native vLLM with EPLB disabled; ON means native vLLM with EPLB enabled. '
             'This comparison does not test the custom resident dispatcher. Response time runs from the scheduled arrival to stream completion; '
             'TTFT runs from client submission to the first nonempty content. Completed elapsed time runs from the shared arrival epoch '
             'through final request completion; it excludes server startup and earlier phases. Output tokens/s uses actual generated tokens divided by completed elapsed time.</p>',
             '<h2>External-arrival workload</h2><p>These are full-model text-generation requests on the recorded external arrival calendar. '
             'Prompt content and arrival data are separate workload dimensions. Tail values are descriptive within this small block.</p>',
             _arm_table([r for r in rows if r["stage"] == "external"]),
             '<p>ON relative to OFF, computed only for valid pairs. Negative latency changes mean lower observed latency; positive throughput changes mean higher observed throughput. '
             'Different generated answers and lengths change the executed workload, so these are not equal-work speedup estimates.</p>',
             _change_table([r for r in rows if r["stage"] == "external"]),
             '<h2>Calibration workload — separate pilot</h2><p>Calibration uses its own content and arrival condition; do not compare its numbers directly with the external-arrival stage.</p>',
             _arm_table([r for r in rows if r["stage"] == "calibration"]),
             '<h2>Evidence gates and diagnostic coverage</h2>']
    for cohort in report.get("cohorts", []):
        parts += [f'<h3>{escape(str(cohort.get("dataset", "unknown")))}</h3>',
                  f'<p>Controller: {escape(str(cohort.get("controller_state", "unknown")))}. Cohort: <code>{escape(str(cohort.get("result_directory", "unavailable")))}</code>.</p>']
        gates = [r for r in rows if r["cohort"] == cohort.get("result_directory", "unknown")]
        parts.append(_table(["Stage", "EPLB", "Arrival kind", "Arm valid?", "Pair valid?", "Issues"], [
            [r["stage"], r["arm"].upper(), r["arrival_kind"], _boolean(r["arm_valid"]), _boolean(r["pair_valid"]), r["issues"] or "none reported"] for r in gates]))
        smoke = cohort.get("stages", {}).get("smoke", {})
        pair = smoke.get("pair", {})
        parts.append('<p>Smoke is a correctness diagnostic, not a performance headline. '
                     f'Generated token identities available: {_boolean(pair.get("diagnostic_token_identity_available"))}; '
                     f'generated token sequences equal: {_boolean(pair.get("generated_token_ids_equal"))}; '
                     f'numerical review required: {_boolean(pair.get("numerical_review_required"))}. '
                     'Finite selected-token logprobs do not prove all-layer numerical correctness. After token divergence, '
                     'later logprobs have different conditioning and cannot be compared as matching-token evidence.</p>')
        logs = cohort.get("native_eplb", {})
        parts.append(_table(["EPLB", "Ordinary async launch log entries", "Async error log entries", "Normal async commits", "Committed state verified?"], [
            [arm.upper(), _cell(logs.get(arm, {}).get("ordinary_async_launch_log_entries"), 0),
             _cell(logs.get(arm, {}).get("async_error_log_entries"), 0),
             _cell(logs.get(arm, {}).get("normal_async_commits"), 0),
             _boolean(logs.get(arm, {}).get("committed_state_consistency_verified"))] for arm in ("off", "on")]))
        parts.append('<p>Launch entries are neither unique verified cycles nor commit counts. Profiling launches are excluded. '
                     'Native asynchronous logs do not attest successful weight/map commits; draining work is not commit evidence.</p>')
        parts.append(_native_details(cohort))
    parts += ['<h2>Source and limits</h2><p>CSV retains separate rows for each cohort, stage and arm. Invalid-arm performance cells and invalid-pair comparisons are withheld; '
              'request/error counts remain visible. No successful subset is promoted into a valid comparison. '
              'The report contains public summaries only, without prompt or generated text.</p>',
              f'<p>Input: <code>{escape(str(source_name))}</code><br>SHA-256: <code>{escape(source_sha256)}</code></p>', '</body></html>']
    return "".join(parts)


def _csv_value(value):
    # Keep untrusted dataset/path/issue strings from becoming spreadsheet formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_outputs(report_path, output_prefix):
    report_path, output_prefix = Path(report_path), Path(output_prefix)
    raw = report_path.read_bytes()
    report = json.loads(raw)
    rows = comparison_rows(report)
    html_path, csv_path = Path(str(output_prefix) + ".html"), Path(str(output_prefix) + ".csv")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(report, rows, report_path, hashlib.sha256(raw).hexdigest()), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    return html_path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("outputs/full-serving/native-comparison"))
    args = parser.parse_args(argv)
    for path in write_outputs(args.report, args.output_prefix):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
