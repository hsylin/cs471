"""Read public full-serving records and audit runtime evidence without inference.

Launch log entries are not commit certificates. Startup and JIT observations
are kept separate from completed request timings and from semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
PIN = "db9527a46873454610df6dbedf79a36d6bf1a7f6"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def log_time(line, year=2026):
    match = re.search(r"(?:INFO|WARNING|ERROR|DEBUG) (\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if not match:
        return None
    return datetime.strptime(f"{year}-{match[1]}", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("Europe/Zurich")).timestamp()


def read_phase(path):
    if not path.exists():
        return {"available": False}, None
    try:
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    except json.JSONDecodeError:
        return {"available": False, "issue": "partial or malformed snapshot"}, None
    headers = [r for r in rows if r.get("type") == "run_header"]
    summaries = [r for r in rows if r.get("type") == "run_summary"]
    requests = [r for r in rows if r.get("type") == "request"]
    header = headers[0] if len(headers) == 1 else {}
    summary = summaries[0] if len(summaries) == 1 else {}
    usages = [r.get("usage") if isinstance(r.get("usage"), dict) else {} for r in requests]
    completion_counts = [u.get("completion_tokens") for u in usages]
    complete_usage = all(type(n) is int and n >= 0 for n in completion_counts)
    observed_complete = bool(requests) and len(headers) == len(summaries) == 1 and all(
        r.get("success") is True and r.get("saw_done") is True for r in requests)
    observed_complete = observed_complete and complete_usage and all(
        u.get("prompt_tokens") == r.get("expected_input_tokens") and type(u.get("prompt_tokens")) is int
        for r, u in zip(requests, usages)) and all(
        count == len(requests) for count in (header.get("request_count"), summary.get("offered_requests"),
                                           summary.get("completed_requests"))) and summary.get("failed_requests") == 0
    result = {"available": True, "complete": len(summaries) == 1,
              "valid": summary.get("valid") is True, "requests": len(requests),
              "valid_semantics": "reported client summary flag; raw integrity is audited separately",
              "observed_request_completion_coherent": observed_complete,
              "successes": sum(r.get("success") is True for r in requests),
              "failures": sum(r.get("success") is not True for r in requests),
              "all_sse_done": all(r.get("saw_done") is True for r in requests) if requests else None,
              "all_prompt_counts_match": all(u.get("prompt_tokens") == r.get("expected_input_tokens")
                                             and type(u.get("prompt_tokens")) is int
                                             for r, u in zip(requests, usages)) if requests else None,
              "diagnostic_passes": sum(r.get("diagnostic_passed") is True for r in requests),
              "diagnostic_requests": sum(r.get("diagnostic_requested") is True for r in requests),
              "completed_elapsed_s": summary.get("completed_elapsed_s"),
              "finish_reasons": dict(Counter(r.get("finish_reason") for r in requests)),
              "generated_tokens": sum(completion_counts) if complete_usage else None,
              "input_hash": header.get("requests_sha256"), "calendar_hash": header.get("calendar_sha256"),
              "client_source_hash": header.get("client_source_sha256"), "source_sha256": digest(path)}
    if summary and "epoch_wall_ns_approx" in summary:
        start = summary["epoch_wall_ns_approx"] / 1e9
        last = start + (summary["last_request_end_ns"] - summary["epoch_monotonic_ns"]) / 1e9
        result["last_request_end_utc_approx"] = datetime.fromtimestamp(last, timezone.utc).isoformat()
        return result, (header.get("created_wall_ns", summary["epoch_wall_ns_approx"]) / 1e9, last)
    return result, None


def parse_logs(paths, phase_intervals):
    data = {"kv_tokens_per_dp_rank": {}, "engine_init_seconds_per_dp_rank": {},
            "reported_compilation_seconds_per_dp_rank": {}, "graph_capture_rank_records": {},
            "weight_load_seconds_per_dp_rank": {}, "ordinary_async_launches": [],
            "profile_rearrangement_entries": [], "jit_during_inference": [], "errors": [],
            "warnings": [], "shutdown_abort_zero_timeout": False, "forced_engine_shutdown": False,
            "selected_weight_communicators": [], "activation_managers": [], "runner": [],
            "attention": [], "moe": [], "normal_async_commits_observed": None,
            "committed_state_consistency_verified": False,
            "commit_limit": "Native launch/profile/transfer messages and HTTP completion are not normal-commit certificates."}
    for path in paths:
        shutdown_seen = False
        for n, line in enumerate(path.read_text().splitlines(), 1):
            timestamp = log_time(line)
            event = {"file": path.name, "line": n}
            if timestamp is not None:
                event["utc_second"] = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            if "[shutdown]" in line or "signal=SIGTERM" in line:
                shutdown_seen = True
            if "mode=abort timeout=0s" in line:
                data["shutdown_abort_zero_timeout"] = True
            if "force killing remaining process" in line:
                data["forced_engine_shutdown"] = True
            for marker, key in [("Initialized EPLB communicator: ", "selected_weight_communicators"),
                                ("Using AgRsAll2AllManager", "activation_managers"),
                                ("Using V2 Model Runner", "runner"),
                                ("Using FlashAttention version 3", "attention"),
                                ("Using TRITON Unquantized MoE", "moe")]:
                if marker in line:
                    value = line.split(marker, 1)[1].strip(" .") if key == "selected_weight_communicators" else marker
                    if value not in data[key]: data[key].append(value)
            rank = re.search(r"(?:EngineCore|Worker)_DP(\d+)", line)
            if rank:
                rank_id = rank[1]
                match = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
                if match: data["kv_tokens_per_dp_rank"][rank_id] = int(match[1].replace(",", ""))
                match = re.search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s(?: \(compilation: ([\d.]+) s)?", line)
                if match:
                    data["engine_init_seconds_per_dp_rank"][rank_id] = float(match[1])
                    if match[2]: data["reported_compilation_seconds_per_dp_rank"][rank_id] = float(match[2])
                match = re.search(r"Graph capturing finished in ([\d.]+) secs", line)
                if match: data["graph_capture_rank_records"].setdefault(rank_id, []).append(float(match[1]))
                match = re.search(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", line)
                if match: data["weight_load_seconds_per_dp_rank"][rank_id] = float(match[1])
            if "Rearranging experts" in line:
                if "(profile)" in line:
                    data["profile_rearrangement_entries"].append(event)
                elif "(async mode)" in line:
                    data["ordinary_async_launches"].append(event)
            if "Triton kernel JIT compilation during inference:" in line:
                candidates = [name for name, interval in phase_intervals.items()
                              if interval and timestamp is not None and interval[0] - 1 <= timestamp <= interval[1] + 1]
                data["jit_during_inference"].append({**event, "kernel": line.split("during inference:", 1)[1].split(".", 1)[0].strip(),
                                                   "compatible_phases": candidates,
                                                   "time_limit": "Logs have one-second precision; boundary overlap is not exact attribution."})
            if re.search(r"\bERROR\b|Traceback|EngineDeadError|async loop error", line):
                data["errors"].append({**event, "after_shutdown_marker_in_same_file": shutdown_seen,
                                       "text": line[:500]})
            if "WARNING" in line and any(s in line for s in ["default MoE config", "OMP_NUM_THREADS", "out-of-order step"]):
                data["warnings"].append({**event, "text": line[:650]})
    data["ordinary_async_launch_log_entries"] = len(data["ordinary_async_launches"])
    data["async_error_log_entries"] = sum("async loop error" in e["text"] for e in data["errors"])
    data["errors_before_shutdown_marker"] = sum(not e["after_shutdown_marker_in_same_file"] for e in data["errors"])
    return data


def audit_arm(directory):
    phases, intervals = {}, {}
    for phase in ("smoke", "calibration", "external"):
        phases[phase], intervals[phase] = read_phase(directory / f"{phase}.jsonl")
    logs = sorted(p for p in directory.glob("server-*.*") if p.suffix in (".log", ".err"))
    result = {"available": directory.exists(), "phases": phases, "runtime": parse_logs(logs, intervals),
              "complete_requests": all(p.get("complete") and p.get("valid") and p.get("observed_request_completion_coherent")
                                       for p in phases.values()),
              "source_log_sha256": {p.name: digest(p) for p in logs}}
    startup = load_json(directory / "startup.json")
    result["startup_elapsed_s"] = startup.get("startup_elapsed_s") if startup else None
    result["hardware"] = []
    for role in ("primary", "worker"):
        p = directory / f"preflight-{role}.json"
        d = load_json(p)
        if d:
            result["hardware"].append({"role": role, "hostname": d.get("hostname"), "job_id": d.get("job_id"),
                                       "devices": d.get("devices"), "topology": d.get("topology"),
                                       "preflight_sha256": digest(p), "checkpoint_manifest_sha256": d.get("checkpoint_manifest_sha256")})
    return result


def audit_cohort(directory):
    status = load_json(directory / "status.json") or {}
    arms = {name: audit_arm(directory / name) for name in ("off", "on")}
    pairs = {}
    for phase in ("smoke", "calibration", "external"):
        off, on = [arms[a]["phases"][phase] for a in ("off", "on")]
        available = all(p.get("complete") and p.get("valid") and p.get("observed_request_completion_coherent") for p in (off, on))
        pairs[phase] = {"completed_valid_arms": bool(available),
                        "input_calendar_client_hashes_match": all(off.get(k) and off.get(k) == on.get(k)
                                                                    for k in ("input_hash", "calendar_hash", "client_source_hash")) if available else None,
                        "off_generated_tokens": off.get("generated_tokens"), "on_generated_tokens": on.get("generated_tokens")}
    smoke = load_json(directory / "smoke-output-comparison.json")
    return {"dataset": directory.name, "directory": str(directory), "controller_state": status.get("state", "not available"),
            "completed_pair": all(arms[a]["complete_requests"] for a in arms) and status.get("state") == "pilot_completed",
            "custom_dispatcher_installed": status.get("custom_dispatcher_installed"), "arms": arms, "pairs": pairs,
            "smoke_generated_texts_all_equal": smoke.get("all_generated_text_hashes_equal") if smoke else None,
            "smoke_equal_text_count": sum(p.get("same_generated_text_hash") is True for p in smoke.get("pairs", [])) if smoke else None,
            "smoke_pair_count": len(smoke.get("pairs", [])) if smoke else None,
            "numerical_and_committed_state_correctness_proven": False}


def build_audit(root):
    return {"schema": "full_serving_runtime_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "vllm_pin": PIN, "root": str(root), "datasets_kept_separate": True,
            "cohorts": [audit_cohort(root / name) for name in DATASETS],
            "interpretation": [
                "Fixed OFF-then-ON pilots provide descriptive native-configuration comparisons, not independent-run uncertainty or final speedup claims.",
                "Equal input and arrival hashes do not guarantee equal generated work: outputs, termination and token counts can differ at temperature zero.",
                "ON adds 32 physical expert copies per layer and may change KV capacity and local-expert kernel shape. This does not isolate decision overhead.",
                "Full HTTP request timing excludes server startup; startup is reported separately. Smoke is a diagnostic, not a performance headline.",
                "All-layer routing/numerical correctness and normally committed expert-map consistency are not established by finite token logprobs or successful requests.",
                "Errors after a same-file intentional shutdown marker are reported separately, not automatically treated as inference failures. Other-file ordering still requires review.",
            ],
            "startup_explanation": {
                "verified": "The native init-engine timer includes memory profiling, KV allocation and compile_or_warm_up_model. That worker method invokes kernel/autotune warmup, V2 scheduler-realistic eager prefill/decode sampling warmup, final graph capture and lazy compiler initialization. It is not a pure compilation timer.",
                "deduction": "Completed WildChat OFF/ON have similar roughly 329/327-second initialization and later successful requests; shared-memory waiting notices in that interval do not alone show a hang. The long gap after brief FlashInfer autotune and before final graph capture is consistent with intervening V2 kernel warmup.",
                "unresolved": "Logs do not time every startup subphase; do not assign the entire gap to a particular kernel, compiler, storage operation or synchronization.",
                "sources": [f"https://github.com/vllm-project/vllm/blob/{PIN}/vllm/v1/engine/core.py#L286-L378",
                            f"https://github.com/vllm-project/vllm/blob/{PIN}/vllm/v1/worker/gpu_worker.py#L829-L1000",
                            f"https://github.com/vllm-project/vllm/blob/{PIN}/vllm/v1/worker/gpu/warmup.py#L213-L454"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_datasets": [c["dataset"] for c in report["cohorts"] if c["completed_pair"]],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
