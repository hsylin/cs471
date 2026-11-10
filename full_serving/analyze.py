"""Descriptive full-serving summaries and paired *block-level* uncertainty.

Requests within one replay are not independent experimental units. This tool
never bootstraps requests. Comparisons use matched arm runs, then cluster the
paired log ratios by independent block_id. A failed or missing run invalidates
the relevant comparison rather than silently selecting its successful subset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def describe(values) -> dict:
    values = [float(v) for v in values if v is not None]
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "p50": quantile(values, .5), "p95": quantile(values, .95),
            "p99": quantile(values, .99), "max": max(values) if values else None}


def summarize_run(records: list[dict], timing: dict) -> dict:
    good = [r for r in records if r.get("success")]
    elapsed_ns = timing["completed_elapsed_ns"]
    completed = len(good)
    total_output = sum(r["usage"]["completion_tokens"] for r in good)
    total_input = sum(r["usage"]["prompt_tokens"] for r in good)
    summary = {
        "valid": bool(records) and completed == len(records) and elapsed_ns > 0,
        "offered_requests": len(records), "completed_requests": completed,
        "failed_requests": len(records) - completed,
        "error_counts": dict(Counter(r.get("error_type") or "unknown" for r in records if not r.get("success"))),
        "finish_reason_counts": dict(Counter(r.get("finish_reason") or "missing" for r in records)),
        "output_cap_completions": sum(r.get("finish_reason") == "length" for r in good),
        "empty_content_completions": sum(r.get("first_content_ns") is None for r in good),
        "completed_elapsed_s": elapsed_ns / 1e9,
        "drain_s": timing["drain_ns"] / 1e9,
        "offered_horizon_s": timing["offered_horizon_ns"] / 1e9,
        "completed_requests_per_s": completed / (elapsed_ns / 1e9) if elapsed_ns > 0 else None,
        "completed_output_tokens_per_s": total_output / (elapsed_ns / 1e9) if elapsed_ns > 0 else None,
        "completed_input_tokens": total_input, "completed_output_tokens": total_output,
        "successful_request_metrics_only": True,
        "speed_claim_allowed": bool(records) and completed == len(records) and elapsed_ns > 0,
        "prompt_tokens": describe(r["usage"]["prompt_tokens"] for r in good),
        "completion_tokens": describe(r["usage"]["completion_tokens"] for r in good),
        "expected_prompt_tokens_all_requests": describe(r["expected_input_tokens"] for r in records),
        "content_chunk_counts": describe(sum(c["content_chars"] > 0 for c in r["chunks"]) for r in good),
        "client_queue_over_1ms_requests": sum(r["client_semaphore_wait_ns"] > 1_000_000 for r in records),
        "server_queue_and_service": "not measured by HTTP client",
        "tpot_note": "stream-boundary estimate; SSE chunks may include multiple tokens; not individual-token intervals",
        "uncertainty_unit": "independent content/arrival blocks, not requests",
    }
    metrics = {
        "response_from_schedule_ms": "response_from_schedule_ns",
        "response_from_send_ms": "response_from_send_ns",
        "ttft_from_schedule_ms": "ttft_from_schedule_ns",
        "ttft_from_send_ms": "ttft_from_send_ns",
        "tpot_stream_boundary_estimate_ms": "tpot_stream_boundary_estimate_ns",
    }
    for name, source in metrics.items():
        summary[name] = describe(r[source] / 1e6 if r.get(source) is not None else None for r in good)
    for source in ("client_schedule_lag_ns", "client_semaphore_wait_ns", "client_send_delay_ns"):
        summary[source.removesuffix("_ns") + "_ms"] = describe(r[source] / 1e6 for r in records)
    return summary


def load_run(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    headers = [r for r in rows if r.get("type") == "run_header"]
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    records = [r for r in rows if r.get("type") == "request"]
    if len(headers) != 1:
        raise ValueError(f"{path}: expected exactly one run header")
    header = headers[0]
    if len(summaries) != 1:
        return {"path": str(path), "header": header,
                "summary": {"valid": False, "speed_claim_allowed": False,
                            "failure": "missing/duplicate completion summary", "recorded_requests": len(records)}}
    summary = summarize_run(records, summaries[0])
    if (len(records) != header["request_count"]
            or sorted(r["request_index"] for r in records) != list(range(header["request_count"]))
            or len({r["request_id"] for r in records}) != len(records)):
        summary.update(valid=False, speed_claim_allowed=False, failure="missing/duplicate request records")
    return {"path": str(path), "header": header, "summary": summary}


def _metric(summary: dict, name: str) -> float | None:
    value = summary
    for part in name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def compare_runs(runs: list[dict], baseline: str = "off", bootstrap: int = 10000,
                 seed: int = 471) -> list[dict]:
    metrics = ["completed_elapsed_s", "completed_output_tokens_per_s",
               "response_from_schedule_ms.mean", "response_from_schedule_ms.p95",
               "ttft_from_schedule_ms.mean", "tpot_stream_boundary_estimate_ms.mean"]
    comparisons = []
    for load in sorted({r["header"]["load_id"] for r in runs}):
        subset = [r for r in runs if r["header"]["load_id"] == load]
        for arm in sorted({r["header"]["arm"] for r in subset} - {baseline}):
            matching = [r for r in subset if r["header"]["arm"] in {baseline, arm}]
            pairs = defaultdict(dict)
            issues = []
            for run in matching:
                h = run["header"]
                if h["arm"] in pairs[h["pair_id"]]:
                    issues.append(f"duplicate arm for pair {h['pair_id']}")
                pairs[h["pair_id"]][h["arm"]] = run
            valid_pairs = []
            for pair_id, pair in pairs.items():
                if set(pair) != {baseline, arm}:
                    issues.append(f"incomplete pair {pair_id}")
                    continue
                b, t = pair[baseline], pair[arm]
                if any(not r["summary"]["valid"] for r in (b, t)):
                    issues.append(f"invalid run in pair {pair_id}")
                    continue
                keys = ("requests_sha256", "calendar_sha256", "block_id", "model",
                        "max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos",
                        "logprobs", "max_inflight", "http_timeout_s", "client_source_sha256")
                if any(b["header"].get(k) != t["header"].get(k) for k in keys):
                    issues.append(f"unmatched workload/configuration in pair {pair_id}")
                    continue
                valid_pairs.append((b, t))
            result = {"baseline": baseline, "arm": arm, "load_id": load,
                      "valid": not issues and bool(valid_pairs), "issues": issues,
                      "matched_run_pairs": len(valid_pairs),
                      "independent_blocks": len({b["header"]["block_id"] for b, _ in valid_pairs}),
                      "estimand": "equal-block geometric mean of paired treatment/baseline metric ratios",
                      "uncertainty": "percentile bootstrap of independent blocks; descriptive, not multiplicity-adjusted",
                      "metrics": {}}
            if issues:
                result["inference"] = "withheld: failed/missing/unmatched runs cannot be silently filtered"
                comparisons.append(result)
                continue
            for metric in metrics:
                logs = defaultdict(list)
                for b, t in valid_pairs:
                    bv, tv = _metric(b["summary"], metric), _metric(t["summary"], metric)
                    if bv is not None and tv is not None and bv > 0 and tv > 0:
                        logs[b["header"]["block_id"]].append(math.log(tv / bv))
                if len(logs) != result["independent_blocks"] or not logs:
                    result["metrics"][metric] = {"available": False, "reason": "missing/zero metric in one or more blocks"}
                    continue
                block_means = [statistics.fmean(v) for v in logs.values()]
                value = {"available": True, "treatment_over_baseline": math.exp(statistics.fmean(block_means)),
                         "block_ratios": {key: math.exp(statistics.fmean(v)) for key, v in logs.items()},
                         "ci95": None}
                if len(block_means) >= 2 and bootstrap > 0:
                    rng = random.Random(seed)
                    estimates = [math.exp(statistics.fmean(rng.choices(block_means, k=len(block_means))))
                                 for _ in range(bootstrap)]
                    value["ci95"] = [quantile(estimates, .025), quantile(estimates, .975)]
                    value["bootstrap_draws"] = bootstrap
                else:
                    value["uncertainty_note"] = "at least two independent blocks required; requests are not resampled"
                result["metrics"][metric] = value
            comparisons.append(result)
    return comparisons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", default="off")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    runs = [load_run(path) for path in args.runs]
    result = {"runs": runs, "comparisons": compare_runs(runs, args.baseline, args.bootstrap),
              "scope": "actual full-model HTTP request response; initialization measured separately",
              "success_criteria": "apply the separately frozen protocol; these descriptive intervals do not supply multiplicity control"}
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0 if all(r["summary"]["valid"] for r in runs) and all(c["valid"] for c in result["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
