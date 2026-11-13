"""Report explicitly named full-model pilot directories without pooling datasets.

Usage: python -m full_serving.report_pilots RESULT_DIR [...] --output report.json
Use --allow-partial for an explicitly incomplete monitoring snapshot. The report
never treats request counts as independent repetitions or declares superiority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from full_serving.analyze import load_run


STAGES = ("smoke", "calibration", "external")
METRIC_NAMES = {
    "vllm:request_queue_time_seconds": "request WAITING phase; not client queue delay",
    "vllm:request_inference_time_seconds": "request RUNNING phase; not isolated GPU service",
    "vllm:request_prefill_time_seconds": "request PREFILL phase",
    "vllm:request_decode_time_seconds": "request DECODE phase",
}
REVISION = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
SOURCE = f"https://github.com/vllm-project/vllm/blob/{REVISION}/vllm"
AUDIT_NOTES = {
    "prompt_token_usage": "Pinned vLLM counts the complete chat-templated prompt, including cached tokens. Cache hits are separate optional prompt_tokens_details; do not subtract them from prompt identity checks.",
    "prompt_template": "Pinned Qwen template appends the assistant prefix and an empty think block for enable_thinking=False. Diagnostic prompt-ID hashes check tokenization; absence of structured reasoning alone is not a semantic proof when no parser is enabled.",
    "streaming": "SSE receipt chunks may contain multiple tokens or metadata only. TTFT uses first nonempty content. TPOT is a stream-boundary estimate, not measured individual-token latency.",
    "numerical_scope": "Smoke verifies returned prompt/output token identities and finite selected-token API logprobs. Equal text alone is weaker. After a generated-token divergence, logprobs have different conditioning. Temperature 0 and finite selected-token logprobs do not prove all-layer numerical, routing or committed-weight consistency.",
    "logprob_limits": "Pinned vLLM can clamp unavailable/extreme logprobs to -9999; the client reports these separately. Top-logprobs0 does not expose runner-up margins or all logits.",
    "eplb": "Native async launch logs do not report successful layer/final commits or map versions. A later launch is only conditional evidence that prior work ceased pending; the last cycle remains unverified. drain_async acknowledges but discards transferred weights.",
    "opportunities": "Native counter starts at 2250 for interval 3000. Dummy DP steps increment it, and pending async work may defer launches. Request/output-token totals cannot recover exact opportunity counts. Balancedness logging adds collective/synchronization overhead and is not enabled merely for this report.",
    "metrics": "Native histogram deltas retain exact engine/model labels. Request RUNNING time is not GPU kernel service. Snapshot publication can lag completion; absent or decreasing series are unavailable, not zero. No EPLB commit metric is present in the audited metrics source.",
    "sources": [SOURCE + "/entrypoints/openai/chat_completion/serving.py#L515-L885",
                SOURCE + "/entrypoints/openai/chat_completion/protocol.py#L82-L180",
                SOURCE + "/distributed/eplb/eplb_state.py#L584-L687",
                SOURCE + "/distributed/eplb/eplb_state.py#L985-L1019",
                SOURCE + "/distributed/eplb/eplb_state.py#L1418-L1443",
                SOURCE + "/distributed/eplb/async_worker.py#L104-L159",
                SOURCE + "/v1/metrics/loggers.py#L850-L900",
                "https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/tokenizer_config.json"],
}


def _json(path):
    return json.loads(Path(path).read_text())


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def read_phase(path):
    path = Path(path)
    result = {"path": str(path), "available": path.exists(), "valid": False, "issues": []}
    if not path.exists():
        result["issues"].append("missing request result file")
        return result, []
    try:
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        records = sorted((r for r in rows if r.get("type") == "request"), key=lambda r: r["request_index"])
        loaded = load_run(path)
        result.update(header=loaded["header"], summary=loaded["summary"])
        required_header = {"arm", "requests_sha256", "calendar_sha256", "request_ids_sha256", "block_id",
                           "load_id", "pair_id", "model", "max_tokens", "max_model_len", "temperature",
                           "enable_thinking", "ignore_eos", "logprobs", "max_inflight", "http_timeout_s",
                           "client_source_sha256"}
        missing = sorted(required_header - loaded["header"].keys())
        if missing:
            result["issues"].append("missing identity/configuration fields: " + ", ".join(missing))
        if not loaded["summary"]["valid"]:
            result["issues"].append("run incomplete, failed, or missing/duplicate requests")
        for row in records:
            usage = row.get("usage")
            if not isinstance(usage, dict) or any(not _integer(usage.get(k)) or usage[k] < 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                result["issues"].append(f"invalid usage: {row.get('request_id')}")
                continue
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                result["issues"].append(f"inconsistent token total: {row['request_id']}")
            if usage["prompt_tokens"] != row["expected_input_tokens"]:
                result["issues"].append(f"prompt count mismatch: {row['request_id']}")
            if loaded["header"].get("logprobs") or row.get("diagnostic_requested"):
                if (row.get("diagnostic_passed") is not True
                        or row.get("prompt_token_ids_count") != usage["prompt_tokens"]
                        or row.get("completion_token_ids_count") != usage["completion_tokens"]
                        or row.get("diagnostic_logprob_count") != usage["completion_tokens"]):
                    result["issues"].append(f"diagnostic token metadata incomplete: {row['request_id']}")
                if (row.get("expected_prompt_token_ids_sha256") is not None
                        and row.get("prompt_token_ids_sha256") != row["expected_prompt_token_ids_sha256"]):
                    result["issues"].append(f"diagnostic prompt-ID hash mismatch: {row['request_id']}")
        result["valid"] = not result["issues"]
        result["logprob_clamp_or_placeholder_count"] = sum(r.get("diagnostic_clamped_or_placeholder_logprobs", 0) for r in records)
        return result, records
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["issues"].append(f"unreadable/incomplete records: {type(exc).__name__}")
        return result, []


def _labels(raw):
    if not raw:
        return ()
    body = raw[1:-1]
    result, position = {}, 0
    pattern = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"\s*(,|$)')
    while position < len(body):
        match = pattern.match(body, position)
        if match is None or match[1] in result:
            raise ValueError("invalid/duplicate metric label")
        result[match[1]] = json.loads('"' + match[2] + '"')
        position = match.end()
    return tuple(sorted(result.items()))


def parse_histograms(text):
    """Only exact requested metric sum/count series; never merge engine labels."""
    pattern = re.compile(r'^(vllm:request_(?:queue|inference|prefill|decode)_time_seconds)_(sum|count)(\{.*\})?\s+(\S+)(?:\s+.*)?$')
    samples = {}
    for line in text.split("\n"):
        match = pattern.match(line.strip())
        if match is None:
            continue
        value = float(match[4])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid native histogram sample")
        key = (match[1], _labels(match[3]))
        parts = samples.setdefault(key, {})
        if match[2] in parts:
            raise ValueError("duplicate native histogram series")
        parts[match[2]] = value
    return samples


def metric_deltas(before, after):
    before, after = Path(before), Path(after)
    result = {"available": False, "before": str(before), "after": str(after), "series": [],
              "interpretation": "Native per-request phase duration histograms; exact labels preserved; RUNNING is not isolated GPU service; no pooled-engine mean."}
    if not before.exists() or not after.exists():
        result["issue"] = "one or both native metric snapshots missing"
        return result
    try:
        b, a = parse_histograms(before.read_text()), parse_histograms(after.read_text())
        if not b and not a:
            result["issue"] = "requested native histogram series absent"
            return result
        for key in sorted(set(b) | set(a)):
            name, labels = key
            row = {"metric": name, "labels": dict(labels), "semantics": METRIC_NAMES[name], "available": False}
            left, right = b.get(key, {}), a.get(key, {})
            if set(left) != {"sum", "count"} or set(right) != {"sum", "count"}:
                row["issue"] = "unmatched sum/count or labels; zero baseline is not assumed"
            else:
                total, count = right["sum"] - left["sum"], right["count"] - left["count"]
                if total < -1e-12 or count < 0 or not count.is_integer():
                    row["issue"] = "counter reset/decrease or fractional observation count"
                elif count == 0 and abs(total) > 1e-12:
                    row["issue"] = "sum changed without observations"
                else:
                    total = max(total, 0)
                    row.update(available=True, delta_sum_seconds=total, delta_count=int(count),
                               mean_seconds=total / count if count else None)
            result["series"].append(row)
        result["available"] = any(row["available"] for row in result["series"])
        result["all_series_valid"] = all(row["available"] for row in result["series"])
    except (OSError, ValueError) as exc:
        result["issue"] = f"invalid snapshot: {type(exc).__name__}"
    return result


def eplb_log_evidence(arm_dir):
    arm_dir = Path(arm_dir)
    files = sorted(set(arm_dir.glob("server-*.log")) | set(arm_dir.glob("server-*.err")))
    if (arm_dir / "srun.log").exists():
        files.append(arm_dir / "srun.log")
    counts, per_file = Counter(), []
    for path in files:
        local = Counter()
        for line in path.read_text(errors="replace").split("\n"):
            if "Rearranging experts" in line:
                if "(profile)" in line:
                    local["profile_launch_log_entries"] += 1
                elif "(async mode)" in line:
                    local["ordinary_async_launch_log_entries"] += 1
                else:
                    local["ordinary_sync_launch_log_entries"] += 1
            if "Rearranged experts" in line and "(profile)" not in line:
                local["sync_completion_log_entries"] += 1
            if "async loop error" in line:
                local["async_error_log_entries"] += 1
            if "coordinated stop" in line:
                local["coordinated_stop_log_entries"] += 1
            if "Async EPLB worker drained" in line:
                local["discarding_drain_log_entries"] += 1
        counts.update(local)
        per_file.append({"path": str(path), **dict(local)})
    names = ("ordinary_async_launch_log_entries", "profile_launch_log_entries", "ordinary_sync_launch_log_entries",
             "sync_completion_log_entries", "async_error_log_entries", "coordinated_stop_log_entries", "discarding_drain_log_entries")
    return {"logs_available": bool(files), **{k: counts[k] if files else None for k in names},
            "files": per_file, "normal_async_commits": None, "committed_state_consistency_verified": False,
            "count_semantics": "Raw matching log entries, not unique cycle IDs. No cross-file deduplication or phase attribution is assumed.",
            "commit_verification": "Unavailable from native async success logs/metrics; launch or drain is not a commit attestation."}


def paired_phase(off, on, off_rows, on_rows, expected_ids=None):
    result = {"valid_for_descriptive_pair": False, "issues": [], "final_speedup_claim": False,
              "uncertainty": {"available": False, "independent_blocks": 1,
                              "reason": "single-block fixed-order pilot; requests are not independent repetitions"}}
    if not off["valid"] or not on["valid"]:
        result["issues"].append("both complete, failure-free arm runs are required")
        return result
    bh, th = off["header"], on["header"]
    identity_keys = ("requests_sha256", "calendar_sha256", "request_ids_sha256", "block_id", "load_id", "pair_id",
                     "model", "max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos", "logprobs",
                     "max_inflight", "http_timeout_s", "client_source_sha256")
    mismatches = [key for key in identity_keys if bh.get(key) != th.get(key)]
    if bh.get("arm") != "off" or th.get("arm") != "on":
        mismatches.append("arm_labels")
    ids = [r["request_id"] for r in off_rows]
    if ids != [r["request_id"] for r in on_rows] or (expected_ids is not None and ids != expected_ids):
        mismatches.append("ordered_request_ids")
    if [r.get("arrival_offset_ns") for r in off_rows] != [r.get("arrival_offset_ns") for r in on_rows]:
        mismatches.append("actual_scheduled_offsets")
    if [r["input_sha256"] for r in off_rows] != [r["input_sha256"] for r in on_rows]:
        mismatches.append("actual_input_hashes")
    if [r["usage"]["prompt_tokens"] for r in off_rows] != [r["usage"]["prompt_tokens"] for r in on_rows]:
        mismatches.append("actual_prompt_token_counts")
    result["paired_identity_mismatches"] = mismatches
    if mismatches:
        result["issues"].append("input/calendar/configuration identity mismatch; ratios withheld")
        return result
    result["valid_for_descriptive_pair"] = True
    result["completion_token_counts_equal"] = [r["usage"]["completion_tokens"] for r in off_rows] == [r["usage"]["completion_tokens"] for r in on_rows]
    result["generated_text_hashes_equal"] = all(b["output_sha256"] == t["output_sha256"] for b, t in zip(off_rows, on_rows))
    token_hashes_present = all(r.get("completion_token_ids_sha256") is not None for r in off_rows + on_rows)
    result["generated_token_ids_equal"] = (all(b["completion_token_ids_sha256"] == t["completion_token_ids_sha256"]
        for b, t in zip(off_rows, on_rows)) if token_hashes_present else None)
    result["diagnostic_token_identity_available"] = token_hashes_present
    result["finish_reasons_equal"] = all(b["finish_reason"] == t["finish_reason"] for b, t in zip(off_rows, on_rows))
    result["numerical_review_required"] = (not result["generated_text_hashes_equal"]
        or result["generated_token_ids_equal"] is False
        or off["logprob_clamp_or_placeholder_count"] > 0 or on["logprob_clamp_or_placeholder_count"] > 0)
    metrics = ("completed_elapsed_s", "completed_requests_per_s", "completed_output_tokens_per_s")
    ratios = {}
    for key in metrics:
        b, t = off["summary"][key], on["summary"][key]
        ratios[key] = t / b if b is not None and t is not None and b > 0 else None
    for key in ("response_from_schedule_ms", "ttft_from_send_ms", "tpot_stream_boundary_estimate_ms"):
        ratios[key] = {}
        for statistic in ("mean", "p95", "p99"):
            b, t = off["summary"][key][statistic], on["summary"][key][statistic]
            ratios[key][statistic] = t / b if b is not None and t is not None and b > 0 else None
    result["on_over_off_ratios_descriptive_only"] = ratios
    return result


def report_cohort(directory):
    directory = Path(directory)
    result = {"result_directory": str(directory.resolve()), "dataset": "unknown", "controller_state": "unknown",
              "complete": False, "issues": [], "stages": {}, "native_eplb": {}, "final_speedup_claim": False}
    try:
        status = _json(directory / "status.json")
        result.update(dataset=status.get("dataset", "wildchat"), controller_state=status.get("state", "unknown"))
    except (OSError, ValueError):
        result["issues"].append("missing/unreadable controller status")
    try:
        splits = _json(directory / "pilot-splits.json")
    except (OSError, ValueError):
        splits = {}
        result["issues"].append("missing/unreadable frozen pilot split identities")
    for stage in STAGES:
        arms, records = {}, {}
        for arm in ("off", "on"):
            arms[arm], records[arm] = read_phase(directory / arm / f"{stage}.jsonl")
            arms[arm]["native_metrics"] = metric_deltas(directory / arm / f"{stage}-metrics-before.txt",
                                                        directory / arm / f"{stage}-metrics-after.txt")
            if arms[arm].get("header") and stage in splits:
                if arms[arm]["header"].get("requests_sha256") != splits[stage]["request_file_sha256"]:
                    arms[arm]["valid"] = False
                    arms[arm]["issues"].append("request-file hash differs from frozen pilot split")
        pair = paired_phase(arms["off"], arms["on"], records["off"], records["on"], splits.get(stage, {}).get("ids"))
        result["stages"][stage] = {"arms": arms, "pair": pair}
    native_issues = []
    for arm in ("off", "on"):
        result["native_eplb"][arm] = eplb_log_evidence(directory / arm)
        logs = result["native_eplb"][arm]
        if logs.get("async_error_log_entries"):
            native_issues.append(f"{arm}: native asynchronous worker error recorded")
        if logs.get("coordinated_stop_log_entries"):
            native_issues.append(f"{arm}: coordinated worker stop requires investigation in static pilot")
        if arm == "off" and (logs.get("ordinary_async_launch_log_entries") or logs.get("ordinary_sync_launch_log_entries")):
            native_issues.append("off: unexpected native EPLB launch")
    if native_issues:
        result["issues"].extend(native_issues)
        for stage in result["stages"].values():
            stage["pair"]["valid_for_descriptive_pair"] = False
            stage["pair"]["issues"].append("native worker/configuration issue cannot be assigned safely to a single stage; ratios withheld")
            stage["pair"].pop("on_over_off_ratios_descriptive_only", None)
    result["complete"] = (result["controller_state"] == "pilot_completed" and not result["issues"]
                           and all(stage["pair"]["valid_for_descriptive_pair"] for stage in result["stages"].values()))
    if not result["complete"]:
        result["issues"].append("cohort pilot incomplete or invalid; successful subsets do not establish a complete comparison")
    return result


def build_report(directories):
    paths = [Path(p).resolve() for p in directories]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate result directory; repetitions must not be fabricated")
    cohorts = [report_cohort(path) for path in paths]
    return {"schema": "full_serving_pilot_report_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": bool(cohorts) and all(c["complete"] for c in cohorts), "cohorts": cohorts,
            "complete_semantics": "controller/request records complete with identity/failure gates; not an all-layer numerical or committed-state proof; optional log/metric coverage is reported separately",
            "pooling": "none: each explicit cohort, dataset, stage and arm remains separate",
            "final_speedup_claim": False, "independent_run_uncertainty": "unavailable for these single-block pilots",
            "audit_notes": AUDIT_NOTES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.directories)
    report["partial_snapshot_requested"] = args.allow_partial
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print("dataset\tstage\tOFF elapsed(s)\tON elapsed(s)\tpaired status")
    for cohort in report["cohorts"]:
        for stage, item in cohort["stages"].items():
            values = [item["arms"][a].get("summary", {}).get("completed_elapsed_s") for a in ("off", "on")]
            labels = [f"{v:.3f}" if v is not None else "unavailable" for v in values]
            state = "descriptive pilot" if item["pair"]["valid_for_descriptive_pair"] else "INCOMPLETE/INVALID"
            print("\t".join([cohort["dataset"], stage, *labels, state]))
    print(f"Report: {args.output}; complete={report['complete']}. No pooled result, run-level interval, or final speedup claim.")
    return 0 if report["complete"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
