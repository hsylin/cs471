"""Render a public three-arm held-out report without reading experiment inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path

from full_serving.custom_report import METRICS, at

FIELDS = ("dataset", "block_id", "arm", "block_valid", "arm_valid", "issues", "completed_requests", "failed_requests",
          *METRICS, "completed_input_tokens", "completed_output_tokens", "output_cap_completions",
          "native_completion_counters", "architecture_metrics_available", "cache_metrics_available",
          "normal_async_commits", "custom_over_off_mean_response", "custom_over_on_mean_response",
          "custom_off_equal_generated_texts", "custom_on_equal_generated_texts")


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def fmt(value, digits=3):
    n = number(value)
    return f"{n:,.{digits}f}" if n is not None else "unavailable"


def comparison_notes(dataset, contrast):
    """Describe the frozen endpoints; never turn exploratory data into acceptance."""
    estimates = dataset.get("summary", {}).get(contrast, {})
    response = estimates.get("mean_response_ms", {})
    if dataset.get("complete") is not True or response.get("available") is not True:
        return ["Dataset comparison pending: all four valid paired blocks are required."]
    point = response["equal_block_geometric_mean_ratio"]
    low, high = response["pointwise_95_percent_interval"]
    notes = [f"Mean response changes by {100 * (point - 1):+.2f}% "
             f"(exploratory pointwise 95% interval {100 * (low - 1):+.2f}% to {100 * (high - 1):+.2f}%)."]
    notes.append("The interval supports lower response latency in these episodes."
                 if high < 1 else "The interval supports higher response latency in these episodes."
                 if low > 1 else "The interval includes no change; the direction remains inconclusive.")
    notes.append("The point estimate meets the frozen 5% useful-gain margin."
                 if point <= .95 else "The point estimate does not meet the frozen 5% useful-gain margin.")
    for metric, label in (("p95_response_ms", "p95 response"), ("completed_elapsed_s", "completed elapsed time")):
        estimate = estimates.get(metric, {})
        if estimate.get("available") is not True:
            notes.append(f"The {label} guard is unavailable.")
            continue
        value = estimate["equal_block_geometric_mean_ratio"]
        lo, hi = estimate["pointwise_95_percent_interval"]
        notes.append(f"The {label} changes by {100 * (value - 1):+.2f}% "
                     f"(pointwise interval {100 * (lo - 1):+.2f}% to {100 * (hi - 1):+.2f}%); "
                     + ("its point estimate reaches the 2% regression margin." if value >= 1.02
                        else "its point estimate stays below the 2% regression margin."))
    details = [b["contrasts"][contrast] for b in dataset["blocks"]]
    matched = sum(d["equal_generated_texts"] for d in details)
    count = sum(d["paired_requests"] for d in details)
    baseline_tokens = sum(d["baseline_output_tokens"] for d in details)
    custom_tokens = sum(d["custom_output_tokens"] for d in details)
    notes.append(f"Generated work: {matched}/{count} answers have identical output hashes; "
                 f"the baseline generated {baseline_tokens:,} tokens and our method generated {custom_tokens:,}.")
    if matched != count:
        notes.append("Different generated answers prevent interpreting this as an equal-generated-work speedup; semantic answer quality was not evaluated.")
    else:
        notes.append("All generated answer hashes match in this comparison; this supports equal-output timing, not general model-quality equivalence.")
    notes.append("These are warmed full-model HTTP response times, including waiting and generation. Startup and warmup are separate. "
                 "Four-block pointwise intervals do not give family-wise guarantees, steady-state capacity, or confirmatory acceptance.")
    return notes


def table_rows(report):
    if report.get("schema") != "resident_compare_report_v1" or report.get("final_speedup_claim") is not False:
        raise ValueError("expected an audited resident_compare_report_v1 exploratory report")
    rows = []
    for dataset in report.get("datasets", []):
        for block in dataset.get("blocks", []):
            for arm in ("off", "on", "custom"):
                item = block.get("arms", {}).get(arm, {})
                summary = item.get("summary", {})
                valid = block.get("valid") is True and item.get("valid") is True
                row = dict.fromkeys(FIELDS)
                row.update(dataset=dataset.get("dataset"), block_id=block.get("block_id"), arm=arm,
                           block_valid=block.get("valid") is True, arm_valid=item.get("valid") is True,
                           issues="; ".join(dict.fromkeys(block.get("issues", []) + item.get("issues", []))),
                           native_completion_counters=item.get("native_completion_counters", {}).get("status", "unavailable"),
                           architecture_metrics_available=item.get("architecture_metrics_available", False),
                           cache_metrics_available=item.get("cache_preemption_deltas", {}).get("available", False),
                           normal_async_commits=item.get("native_eplb", {}).get("normal_async_commits"))
                for key in ("completed_requests", "failed_requests"):
                    row[key] = number(summary.get(key))
                if valid:
                    row.update({name: number(at(summary, keys)) for name, keys in METRICS.items()})
                    row.update({name: number(summary.get(name)) for name in ("completed_input_tokens", "completed_output_tokens", "output_cap_completions")})
                    if arm == "custom":
                        for baseline in ("off", "on"):
                            contrast = block.get("contrasts", {}).get("custom/" + baseline, {})
                            row[f"custom_over_{baseline}_mean_response"] = number(contrast.get("ratios", {}).get("mean_response_ms"))
                            row[f"custom_{baseline}_equal_generated_texts"] = contrast.get("equal_generated_texts")
                rows.append(row)
    return rows


def render_html(report, rows, source_hash):
    esc = lambda x: escape(str(x))
    sections = ["<!doctype html><html lang=en><meta charset=utf-8><title>Full-model resident comparison</title>",
        "<style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#18212b}table{border-collapse:collapse;width:100%;margin:16px 0 28px}th,td{text-align:right;padding:7px;border-bottom:1px solid #d4dbe2}th:first-child,td:first-child{text-align:left}th{background:#eef2f6}p{max-width:1050px;line-height:1.55}.warn{background:#fff2d0;padding:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}h2{margin-top:34px}</style>",
        "<h1>Full-model resident comparison</h1>",
        "<p class=warn>" + ("Protocol complete; this does not declare superiority." if report.get("complete") else "Incomplete or invalid experiment. Missing comparisons are withheld.") + "</p>",
        "<p>Primary: resident method / native EPLB OFF. Secondary: resident method / native EPLB ON. A latency ratio below 1 is lower latency. No datasets are pooled.</p>"]
    for key in ("scope", "uncertainty", "warmup_state", "commit_and_numerical_limit", "source_dependence"):
        sections.append("<p>" + esc(report.get(key, "unavailable")) + "</p>")
    for dataset in report.get("datasets", []):
        name = dataset.get("dataset")
        sections.append("<h2>" + esc(name) + "</h2>")
        sections.append("<table><tr><th>Contrast</th><th>Mean-response ratio</th><th>Exploratory 95% interval</th><th>Independent blocks</th></tr>")
        for contrast in ("custom/off", "custom/on"):
            estimate = dataset.get("summary", {}).get(contrast, {}).get("mean_response_ms", {})
            available = dataset.get("complete") is True and estimate.get("available") is True
            interval = estimate.get("pointwise_95_percent_interval")
            values = [contrast, fmt(estimate.get("equal_block_geometric_mean_ratio")) if available else "withheld",
                      " to ".join(fmt(v) for v in interval) if available and isinstance(interval, list) and len(interval) == 2 else "unavailable",
                      str(estimate.get("independent_block_count")) if available else "incomplete"]
            sections.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
        sections.append("</table><p>The interval is pointwise and assumption-dependent with four blocks. A point estimate meeting a 5% gain or 2% regression margin is not a confirmatory acceptance decision.</p>")
        for contrast in ("custom/off", "custom/on"):
            sections.append("<h3>" + esc(contrast) + " — interpretation</h3><ul>" +
                            "".join("<li>" + esc(note) + "</li>" for note in comparison_notes(dataset, contrast)) + "</ul>")
        headers = ("Block / arm", "Completed / failed", "Mean response ms", "p95 response ms", "TTFT ms", "Elapsed s", "Drain s", "Output tok/s", "Output tokens", "Length caps")
        sections.append("<table><tr>" + "".join("<th>" + x + "</th>" for x in headers) + "</tr>")
        for row in (r for r in rows if r["dataset"] == name):
            values = [str(row["block_id"]) + " / " + row["arm"], f'{row["completed_requests"]} / {row["failed_requests"]}',
                      *[fmt(row[k]) for k in ("mean_response_ms", "p95_response_ms", "mean_ttft_schedule_ms", "completed_elapsed_s", "drain_s", "completed_output_tokens_per_s")],
                      fmt(row["completed_output_tokens"], 0), fmt(row["output_cap_completions"], 0)]
            sections.append("<tr>" + "".join("<td>" + esc(v) + "</td>" for v in values) + "</tr>")
            if row["issues"]:
                sections.append('<tr><td colspan="10">Withheld: ' + esc(row["issues"]) + "</td></tr>")
        sections.append("</table>")
        for block in dataset.get("blocks", []):
            notes = {"block_id": block.get("block_id"), "valid": block.get("valid"), "issues": block.get("issues"),
                     "runtime_status": block.get("runtime_admission", {}).get("status"), "generation": {}, "native_metrics": {}}
            for contrast, details in block.get("contrasts", {}).items():
                notes["generation"][contrast] = {k: v for k, v in details.items() if k != "ratios"}
            for arm, item in block.get("arms", {}).items():
                notes["native_metrics"][arm] = {k: item.get(k) for k in ("native_completion_counters", "native_metrics", "cache_preemption_deltas", "native_eplb")}
            sections.append("<details><summary>" + esc(block.get("block_id")) + " — generation, counters, cache and exact-label native phase records</summary><pre>" + esc(json.dumps(notes, indent=2, allow_nan=False)) + "</pre></details>")
    sections.append("<p>Generated outputs can differ under normal EOS, so output throughput alone is not a speedup measure. TTFT is first nonempty streamed content; native RUNNING duration is not isolated GPU service. Async launches and drained work do not attest normal EPLB commits.</p>")
    sections.append("<p>Public report SHA-256: <code>" + esc(source_hash) + "</code>. This page reads only that report; no private prompts or outputs are embedded.</p></html>")
    return "\n".join(sections)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.report.read_bytes()  # one immutable snapshot of an atomically replaced report
    report = json.loads(raw)
    rows = table_rows(report)
    csv_path = args.output_prefix.with_suffix(".csv")
    html_path = args.output_prefix.with_suffix(".html")
    if args.report.resolve() in {csv_path.resolve(), html_path.resolve()}:
        raise ValueError("output must not overwrite its source report")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    html_path.write_text(render_html(report, rows, hashlib.sha256(raw).hexdigest()))
    print(json.dumps({"html": str(html_path), "csv": str(csv_path), "rows": len(rows), "complete": report.get("complete"), "final_speedup_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
