"""Report the frozen three-arm experiment by source and independent block.

No request bootstrap, cross-source pooling, fastest-attempt selection or automatic
superiority declaration. Four-block log-ratio intervals are explicitly exploratory.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from full_serving import custom_compare as compare
from full_serving.report_pilots import _labels, eplb_log_evidence, metric_deltas, read_phase
from full_serving.audit_pilots import audit_counters


T_975_DF3 = 3.182446305284263
METRICS = {
    "mean_response_ms": ("response_from_schedule_ms", "mean"),
    "p95_response_ms": ("response_from_schedule_ms", "p95"),
    "mean_ttft_schedule_ms": ("ttft_from_schedule_ms", "mean"),
    "mean_ttft_send_ms": ("ttft_from_send_ms", "mean"),
    "completed_elapsed_s": ("completed_elapsed_s",),
    "drain_s": ("drain_s",),
    "completed_requests_per_s": ("completed_requests_per_s",),
    "completed_output_tokens_per_s": ("completed_output_tokens_per_s",),
}
IDENTITY_KEYS = ("requests_sha256", "calendar_sha256", "request_ids_sha256", "block_id", "load_id", "pair_id",
                 "model", "max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos", "logprobs",
                 "max_inflight", "http_timeout_s", "client_source_sha256", "output_lengths_sha256", "output_length_mode")


def at(value, keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def ratio(numerator, denominator):
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (numerator, denominator)):
        return None
    return numerator / denominator if denominator > 0 and numerator >= 0 else None


def paired_log_interval(values):
    """Exactly four content/arrival block ratios, never individual requests."""
    if len(values) != 4 or any(v is None or v <= 0 or not math.isfinite(v) for v in values):
        return {"available": False, "reason": "four valid positive block ratios required"}
    logs = [math.log(v) for v in values]
    mean = statistics.mean(logs)
    half = T_975_DF3 * statistics.stdev(logs) / math.sqrt(4)
    return {"available": True, "independent_block_count": 4, "block_ratios": values,
            "equal_block_geometric_mean_ratio": math.exp(mean),
            "pointwise_95_percent_interval": [math.exp(mean - half), math.exp(mean + half)],
            "method": "Student-t interval on four paired log ratios, df=3; independent-block and approximate log-normality assumptions",
            "interpretation": "Exploratory, assumption-dependent and pointwise only; no family-wise coverage or confirmatory claim at n=4."}


def descriptive_ratio(values):
    if len(values) != 1 or values[0] is None or not math.isfinite(values[0]) or values[0] < 0:
        return {"available": False, "reason": "one valid block ratio required"}
    return {"available": True, "independent_block_count": 1, "block_ratios": values,
            "equal_block_geometric_mean_ratio": values[0], "pointwise_95_percent_interval": None,
            "method": "Single paired group ratio; no resampling, confidence interval or significance test.",
            "interpretation": "Descriptive episode effect only; no estimate of between-group/run variation."}


def phase_balance(metrics, request_count=16):
    result = {"available": False, "meaning": "Native per-request phase durations, summed across engines; not isolated GPU busy time or memory/compute bottleneck proof."}
    totals = {}
    for phase in ("prefill", "decode"):
        rows = [r for r in metrics.get("series", []) if r.get("metric") == f"vllm:request_{phase}_time_seconds"]
        if not rows or not all(r.get("available") for r in rows) or sum(r["delta_count"] for r in rows) != request_count:
            result["issue"] = "phase counters do not cover exactly the admitted requests"
            return result
        totals[phase] = sum(r["delta_sum_seconds"] for r in rows)
    if sum(totals.values()) <= 0:
        return result
    result.update(available=True, request_duration_sum_s=totals, prefill_fraction=totals["prefill"] / sum(totals.values()),
                  dominant_request_phase="prefill" if totals["prefill"] > totals["decode"] else "decode")
    return result


def cache_counter_deltas(before, after):
    pattern = re.compile(r'^(vllm:(?:prefix_cache_hits_total|prefix_cache_queries_total|num_preemptions_total))(\{.*\})?\s+(\S+)(?:\s+.*)?$')
    result = {"available": False, "series": [], "semantics": "Exact native counter names and labels retained; no pooled-engine cache estimate."}
    if not before.exists() or not after.exists():
        result["issue"] = "cache/preemption snapshot missing"
        return result
    def parse(path):
        data = {}
        for line in path.read_text().split("\n"):
            match = pattern.match(line.strip())
            if match:
                key = (match[1], _labels(match[2]))
                value = float(match[3])
                if key in data or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid counter snapshot")
                data[key] = value
        return data
    try:
        left, right = parse(before), parse(after)
        for key in sorted(set(left) | set(right)):
            valid = key in left and key in right and right[key] >= left[key]
            result["series"].append({"metric": key[0], "labels": dict(key[1]), "available": valid,
                                     "delta": right[key] - left[key] if valid else None,
                                     "issue": None if valid else "missing baseline/series or counter reset"})
        result["available"] = bool(result["series"]) and all(r["available"] for r in result["series"])
        if not result["series"]:
            result["issue"] = "requested cache/preemption counters absent"
    except (OSError, ValueError):
        result["issue"] = "invalid cache/preemption snapshot"
    return result


def timing_issues(records):
    issues = []
    for r in records:
        try:
            if not (r["scheduled_ns"] <= r["send_ns"] <= r["end_ns"]
                    and r["response_from_schedule_ns"] == r["end_ns"] - r["scheduled_ns"]
                    and r["response_from_send_ns"] == r["end_ns"] - r["send_ns"]):
                raise ValueError
            if r.get("first_content_ns") is not None:
                if not (r["send_ns"] <= r["first_content_ns"] <= r["last_content_ns"] <= r["end_ns"]
                        and r["ttft_from_schedule_ns"] == r["first_content_ns"] - r["scheduled_ns"]
                        and r["ttft_from_send_ns"] == r["first_content_ns"] - r["send_ns"]):
                    raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("raw timestamp identity or ordering failed")
            break
    return issues


def runtime_triplet(directory, protocol):
    from full_serving.custom_runtime import audit_runtime_triplet
    return audit_runtime_triplet(directory, protocol)


def replay_evidence(path, records, calendar, expected_calendar_hash):
    """Reconstruct episode completion, not a trusted stored elapsed counter."""
    result = {"valid": False, "issues": []}
    try:
        rows = compare.native.read_jsonl(path)
        header = next(r for r in rows if r.get("type") == "run_header")
        summaries = [r for r in rows if r.get("type") == "run_summary"]
        if len(summaries) != 1 or not records:
            raise ValueError
        stored = summaries[0]
        epoch = stored["epoch_monotonic_ns"]
        horizon = calendar.get("horizon_ns", calendar["arrival_offsets_ns"][-1])
        if (type(epoch) is not int or type(horizon) is not int or epoch < 0 or horizon < 0
                or header.get("calendar_sha256") != expected_calendar_hash
                or [r["request_id"] for r in records] != calendar["request_ids"]
                or header.get("request_ids_sha256") != compare.canonical_hash(calendar["request_ids"])
                or [r["arrival_offset_ns"] for r in records] != calendar["arrival_offsets_ns"]):
            raise ValueError
        for r, offset in zip(records, calendar["arrival_offsets_ns"]):
            if (any(type(r.get(key)) is not int for key in ("scheduled_ns", "send_ns", "end_ns", "arrival_offset_ns"))
                    or r["scheduled_ns"] != epoch + offset or not r.get("success")
                    or r.get("http_status") != 200 or r.get("saw_done") is not True or r.get("usage_events") != 1
                    or r.get("error_type") is not None or r.get("error_message") is not None):
                raise ValueError
            limit = (r.get("requested_output_tokens") if header.get("output_length_mode") == "fixed_previous_off_reference"
                     else header["max_tokens"])
            if (type(limit) is not int or not 0 < limit <= header["max_tokens"]
                    or r.get("finish_reason") not in ("stop", "length")
                    or r["usage"]["completion_tokens"] > header["max_tokens"]
                    or r["usage"]["total_tokens"] > header["max_model_len"]
                    or (r["finish_reason"] == "length" and r["usage"]["completion_tokens"] != limit)):
                raise ValueError
        last_end = max(r["end_ns"] for r in records)
        completion = max(last_end, epoch + horizon)
        expected = {"offered_horizon_ns": horizon, "last_request_end_ns": last_end, "completion_ns": completion,
                    "completed_elapsed_ns": completion - epoch, "drain_ns": max(0, last_end - epoch - horizon)}
        if any(type(stored.get(key)) is not int or stored[key] != value for key, value in expected.items()):
            raise ValueError
        if type(stored.get("client_finished_ns")) is not int or stored["client_finished_ns"] < completion:
            raise ValueError
        result["issues"].extend(timing_issues(records))
        result.update(expected, client_tail_after_completion_ns=stored["client_finished_ns"] - completion)
        result["valid"] = not result["issues"]
    except (OSError, ValueError, TypeError, KeyError, StopIteration):
        result["issues"].append("HTTP completion, frozen calendar, epoch, elapsed or drain arithmetic failed")
    return result


def expected_generation(header, protocol, diagnostics, dataset=None):
    generation = protocol.get("generation", {"max_tokens": 2048, "max_model_len": 32768,
                                              "temperature": 0, "enable_thinking": False, "ignore_eos": False})
    generation = {**generation, **protocol.get("generation_by_dataset", {}).get(dataset, {})}
    return (all(header.get(key) == generation[key] for key in ("max_tokens", "max_model_len", "temperature", "enable_thinking", "ignore_eos"))
            and header.get("model") == "qwen3-full" and header.get("max_inflight") == 64
            and header.get("http_timeout_s") == protocol.get("execution", {}).get("http_timeout_s", 1800)
            and header.get("logprobs") is diagnostics)


def output_length_issues(header, records, item, protocol, root):
    if not protocol.get("equal_output_length", False):
        return []
    targets = json.loads((root / item["output_lengths_file"]).read_bytes())
    if (header.get("ignore_eos") is not True or header.get("output_length_mode") != "fixed_previous_off_reference"
            or header.get("output_lengths_sha256") != item["output_lengths_sha256"]
            or {r["request_id"] for r in records} != set(targets)
            or any(r.get("requested_output_tokens") != targets.get(r["request_id"])
                   or r.get("usage", {}).get("completion_tokens") != targets.get(r["request_id"])
                   or r.get("finish_reason") != "length" for r in records)):
        return ["fixed per-request output length, target identity or length termination differs"]
    return []


def schedule_evidence(root, protocol, status):
    expected = compare.schedule(protocol)
    compare.require(json.loads((root / "schedule.json").read_bytes()) == expected, "archived schedule differs from frozen counterbalance")
    ids = [job["session_id"] for job in expected]
    completed = status.get("completed_sessions", [])
    compare.require(isinstance(completed, list) and completed == ids[:len(completed)], "completed sessions do not follow the frozen prefix")
    if status.get("state") == "comparison_completed":
        compare.require(completed == ids, "controller claims completion without all60 sessions")
    issues, steps, previous_step, previous_start = {}, set(), -1, None
    for job in expected:
        directory = root / job["session_id"]
        failures = []
        try:
            launch = json.loads((directory / "launch.json").read_bytes())
            if any(launch.get(k) != v for k, v in job.items()):
                raise ValueError
            command = launch["command"]
            if (command[0] != "srun" or command[-4] != job["arm"]
                    or not str(command[-5]).endswith("/full_serving/custom_compare_node.sh")
                    or not {"--jobid=" + compare.native.JOB, "--nodes=2", "--ntasks=2", "--gpus-per-node=4"} <= set(command)):
                raise ValueError
            started = datetime.fromisoformat(launch["started_utc"])
            if started.tzinfo is None or (previous_start is not None and started < previous_start):
                raise ValueError
            previous_start = started
            recorded_steps = [(directory / f"server-step-{role}.txt").read_text().strip() for role in ("primary", "worker")]
            if recorded_steps[0] != recorded_steps[1] or not re.fullmatch(compare.native.JOB + r"\.\d+", recorded_steps[0]):
                raise ValueError
            step = recorded_steps[0]
            step_number = int(step.split(".")[1])
            if step in steps or step_number <= previous_step:
                raise ValueError
            steps.add(step)
            previous_step = step_number
            if job["session_id"] in completed:
                closed = json.loads((directory / "status.json").read_bytes())
                if closed.get("state") != "session_completed" or any(closed.get(k) != v for k, v in job.items()):
                    raise ValueError
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            failures.append("launch order, fresh Slurm step or session closure evidence missing/mismatched")
        issues[job["session_id"]] = failures
    return issues


def report_block(root, protocol, dataset, block, expected_client_hash, request_admission, session_issues):
    base = root / dataset["dataset"] / block["block_id"]
    result = {"block_id": block["block_id"], "arm_order": block["arm_order"], "valid": False,
              "issues": [], "arms": {}, "contrasts": {}}
    rows = {}
    try:
        calendar_path = root / "calendars" / dataset["dataset"] / (block["block_id"] + ".json")
        if compare.native.sha(calendar_path) != block["calendar_sha256"]:
            raise ValueError
        calendar = json.loads(calendar_path.read_bytes())
    except (OSError, ValueError):
        calendar = None
        result["issues"].append("archived frozen calendar missing/changed")
    result["runtime_admission"] = runtime_triplet(base, protocol)
    if result["runtime_admission"].get("status") != "passed":
        result["issues"].append("native/custom runtime identity and source/hardware admission incomplete or failed")
    for arm in compare.ARMS:
        directory = base / arm
        item, records = read_phase(directory / "measured.jsonl")
        item["issues"].extend(session_issues.get(f'{dataset["dataset"]}/{block["block_id"]}/{arm}', ["session schedule evidence missing"]))
        rows[arm] = records
        header = item.get("header", {})
        if (header.get("arm") != arm or header.get("requests_sha256") != block["requests_sha256"]
                or header.get("calendar_sha256") != block["calendar_sha256"]
                or header.get("client_source_sha256") != expected_client_hash
                or [r["request_id"] for r in records] != block["request_ids"]
                or len(records) != 16 or header.get("logprobs") is not False):
            item["issues"].append("frozen request/calendar/client/arm identity or count differs")
        if not expected_generation(header, protocol, False, dataset["dataset"]):
            item["issues"].append("request generation configuration differs from frozen protocol")
        item["issues"].extend(output_length_issues(header, records, block, protocol, root))
        item["replay_evidence"] = (replay_evidence(directory / "measured.jsonl", records, calendar, block["calendar_sha256"])
                                   if calendar is not None else {"valid": False, "issues": ["frozen calendar unavailable"]})
        item["issues"].extend(item["replay_evidence"]["issues"])
        if any(r.get("input_sha256") != request_admission.get(r["request_id"], {}).get("client_input_sha256") for r in records):
            item["issues"].append("client input differs from source-bound private admission")
        item["warmup"] = []
        for phase in ("warmup-00", "warmup-01"):
            warm, warm_rows = read_phase(directory / f"{phase}.jsonl")
            warm_header = warm.get("header", {})
            warm_calendar = {"kind": "synthetic_pilot_burst", "rate_requests_per_s": 0.0,
                             "arrival_offsets_ns": [0] * 8, "request_ids": dataset["warmup"]["request_ids"]}
            warm_hash = hashlib.sha256(json.dumps(warm_calendar, sort_keys=True).encode()).hexdigest()
            warm_timing = replay_evidence(directory / f"{phase}.jsonl", warm_rows, warm_calendar, warm_hash)
            length_issues = output_length_issues(warm_header, warm_rows, dataset["warmup"], protocol, root)
            warm_valid = not (not warm["valid"] or len(warm_rows) != 8 or warm_header.get("arm") != arm
                    or length_issues
                    or warm_header.get("requests_sha256") != dataset["warmup"]["requests_sha256"]
                    or warm_header.get("client_source_sha256") != expected_client_hash
                    or warm_header.get("logprobs") is not (phase == "warmup-00")
                    or [r["request_id"] for r in warm_rows] != dataset["warmup"]["request_ids"]
                    or not expected_generation(warm_header, protocol, phase == "warmup-00", dataset["dataset"])
                    or warm_header.get("arrival_kind") != "synthetic_pilot_burst"
                    or warm_header.get("synthetic_rate_requests_per_s") != 0.0 or not warm_timing["valid"]
                    or any(r.get("input_sha256") != request_admission.get(r["request_id"], {}).get("client_input_sha256") for r in warm_rows))
            if not warm_valid:
                item["issues"].append(phase + " missing/invalid or differs from frozen pilot warmup")
            item["warmup"].append({"phase": phase, "valid": warm_valid, "summary": warm.get("summary"), "replay_evidence": warm_timing,
                "cache_preemption_deltas": cache_counter_deltas(directory / f"{phase}-metrics-before.txt", directory / f"{phase}-metrics-after.txt")})
        item["native_metrics"] = metric_deltas(directory / "measured-metrics-before.txt", directory / "measured-metrics-after.txt")
        item["phase_balance"] = phase_balance(item["native_metrics"])
        item["http_timing_valid"] = item["valid"] and not item["issues"]
        item["architecture_metrics_available"] = item["native_metrics"].get("available", False)
        item["native_completion_counters"] = audit_counters(directory / "measured-metrics-before.txt", directory / "measured-metrics-after.txt",
                                                             16, item.get("summary", {}).get("finish_reason_counts", {}))
        if item["native_completion_counters"]["status"] != "matched":
            item["issues"].append("native completion/drained-engine counters unavailable or unreconciled")
        item["cache_preemption_deltas"] = cache_counter_deltas(directory / "measured-metrics-before.txt", directory / "measured-metrics-after.txt")
        item["native_eplb"] = logs = eplb_log_evidence(directory)
        if logs.get("async_error_log_entries") or logs.get("coordinated_stop_log_entries"):
            item["issues"].append("native asynchronous worker issue")
        if arm in ("off", "custom") and any(logs.get(key) for key in (
                "ordinary_async_launch_log_entries", "ordinary_sync_launch_log_entries", "profile_launch_log_entries")):
            item["issues"].append("unexpected native EPLB launch in control-disabled arm")
        if arm == "custom":
            try:
                view = copy.deepcopy(protocol)
                view["custom"]["config_file"] = "implementation/config.json"
                if compare.native.sha(root / "implementation/config.json") != protocol["custom"]["config_sha256"]:
                    raise ValueError("archived custom config mismatch")
                if json.loads((root / "implementation/config.json").read_bytes()).get("startup_diagnostic", False) is not False:
                    raise ValueError("correctness-only startup instrumentation enabled in timing comparison")
                item["custom_bindings"] = compare.verify_custom_bindings(directory, view, root / "frozen-protocol.json")
            except (OSError, ValueError, TypeError, KeyError):
                item["issues"].append("custom eight-rank/layer/source/map binding audit failed")
        item["valid"] = item["valid"] and not item["issues"]
        result["arms"][arm] = item
    if not all(item["valid"] for item in result["arms"].values()):
        result["issues"].append("complete valid three-arm block required; no successful-subset contrast")
        return result
    reference = result["arms"]["off"]["header"]
    for arm in ("on", "custom"):
        header = result["arms"][arm]["header"]
        if any(reference.get(key) != header.get(key) for key in IDENTITY_KEYS):
            result["issues"].append("paired header configuration/input/calendar mismatch")
        for off, other in zip(rows["off"], rows[arm]):
            if (any(off.get(key) != other.get(key) for key in ("request_id", "input_sha256", "payload_sha256", "arrival_offset_ns", "expected_input_tokens"))
                    or off["usage"]["prompt_tokens"] != other["usage"]["prompt_tokens"]):
                result["issues"].append("paired raw input/payload/schedule identity mismatch")
                break
    if result["issues"]:
        return result
    result["valid"] = True
    for baseline in ("off", "on"):
        b, t = result["arms"][baseline]["summary"], result["arms"]["custom"]["summary"]
        pairs = list(zip(rows[baseline], rows["custom"]))
        result["contrasts"]["custom/" + baseline] = {
            "ratios": {name: ratio(at(t, keys), at(b, keys)) for name, keys in METRICS.items()},
            "paired_requests": 16, "equal_generated_texts": sum(a["output_sha256"] == z["output_sha256"] for a, z in pairs),
            "equal_output_token_counts": sum(a["usage"]["completion_tokens"] == z["usage"]["completion_tokens"] for a, z in pairs),
            "baseline_output_tokens": b["completed_output_tokens"], "custom_output_tokens": t["completed_output_tokens"],
            "generation_limit": "Identical prompts and arrival calendars do not imply identical generated work; main runs do not return generated token IDs."}
    return result


def build_report(root):
    root = Path(root)
    protocol = compare.validate_protocol(json.loads((root / "frozen-protocol.json").read_bytes()))
    compact = protocol.get("schema") == compare.COMPACT_SCHEMA
    source = json.loads((root / "controller-source.json").read_bytes())
    expected_client_hash = source["files"]["full_serving/client.py"]
    status = json.loads((root / "status.json").read_bytes())
    compare.require(status.get("protocol_sha256") == compare.native.sha(root / "frozen-protocol.json"), "frozen protocol/status identity mismatch")
    sessions = schedule_evidence(root, protocol, status)
    view = copy.deepcopy(protocol)
    view["custom"].update(config_file="implementation/config.json", source_manifest_file="implementation/source-manifest.json")
    view["pilot_admission"]["file"] = "implementation/pilot-admission.json"
    for key in view.get("source_admission", {}):
        if key.endswith("_file"):
            view["source_admission"][key] = "source-admission/" + key.removesuffix("_file") + ".json"
    compare.validate_admission_artifacts(view, root / "frozen-protocol.json")
    source_metadata = compare.load_source_metadata(view, root / "frozen-protocol.json")
    request_path = root / "request-admission.json"
    compare.require(compare.native.sha(request_path) == status.get("request_admission_sha256"), "request admission manifest changed")
    request_manifest = json.loads(request_path.read_bytes())
    compare.require(request_manifest.get("status") == "passed", "private source admission missing")
    requests = request_manifest["requests"]
    expected_ids = {rid for dataset in protocol["datasets"] for item in [dataset["warmup"], *dataset["blocks"]] for rid in item["request_ids"]}
    compare.require(set(requests) == expected_ids, "source admission identities differ")
    for rid, admission in requests.items():
        compare.require(admission.get("status") == "passed" and admission.get("source_metadata_sha256") == compare.canonical_hash(source_metadata["by_id"][rid]),
                        "request admission differs from sealed public metadata")
    result = {"schema": "resident_compare_report_v1", "complete": False, "datasets": [],
              "protocol_sha256": compare.native.sha(root / "frozen-protocol.json"), "controller_state": status.get("state"),
              "primary_contrast": "custom/off", "secondary_contrast": "custom/on", "final_speedup_claim": False,
              "scope": "Whole resident method versus native configurations; placement, dispatch and recurring control differ together.",
              "uncertainty": "Four content/arrival blocks per source; pointwise exploratory t intervals on block log ratios. No cross-source pooling or request-level resampling.",
              "warmup_state": "Each server runs the same8 pilot requests twice. Native ON may change maps/history during warmup; HTTP drain does not attest an EPLB commit.",
              "commit_and_numerical_limit": "Native async normal commits remain unobservable in ordinary logs. Binding manifests and HTTP success do not establish all-layer numerical equivalence.",
              "source_dependence": "Arrival dates are reused across content sources; do not treat20blocks as independent replicates of one universal effect."}
    result["design"] = "two_workloads_one_group" if compact else "five_datasets_four_groups"
    if compact:
        result["uncertainty"] = "One paired group per workload; descriptive ratios only, no confidence intervals or significance claims. Six sessions cannot estimate between-group or repeated-run variability."
        result["source_dependence"] = "The two workloads share an Azure arrival block shape scaled to their previously frozen pilot rates; do not pool them as independent repetitions."
        result["workload_settings"] = protocol["generation_by_dataset"]
        result["phase_scope"] = "LongBench retains full prompts with a 16-token generation cap; NuminaMath retains a 2048-token cap. Check native measured phase durations before calling a workload prefill/decode dominated. Capped answers are not a quality evaluation."
    if protocol.get("equal_output_length", False):
        result["equal_output_length"] = True
        result["phase_scope"] = "Each prompt generates exactly its previous natural-EOS OFF output-token count, with min_tokens=max_tokens=N and early stopping suppressed for all arms. Full prompts and absolute calendars are reused. Equal counts do not imply identical token identities, expert routing, batch shapes or answer quality. Native phase dominance is measured, not assumed."
    runtime_reference = None
    for dataset in protocol["datasets"]:
        item = {"dataset": dataset["dataset"], "blocks": [report_block(root, protocol, dataset, block, expected_client_hash, requests, sessions)
                                                               for block in dataset["blocks"]], "summary": {}}
        item["equal_output_length"] = protocol.get("equal_output_length", False)
        for block in item["blocks"]:
            runtime = block["runtime_admission"]
            if runtime.get("status") == "passed":
                fingerprint = (runtime.get("common_runtime_fingerprint_sha256"), runtime.get("native_engine_configuration_sha256"))
                if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprint):
                    block["issues"].append("runtime admission omitted its source/hardware fingerprints")
                elif runtime_reference is None:
                    runtime_reference = fingerprint
                elif fingerprint != runtime_reference:
                    block["issues"].append("hardware/source/engine identity changed across experiment blocks")
                if block["issues"]:
                    block["valid"] = False
                    block["contrasts"] = {}
        item["complete"] = all(block["valid"] for block in item["blocks"])
        for contrast in ("custom/off", "custom/on"):
            estimates = {name: (descriptive_ratio if compact else paired_log_interval)([block.get("contrasts", {}).get(contrast, {}).get("ratios", {}).get(name)
                                                  for block in item["blocks"]]) for name in METRICS}
            primary = estimates["mean_response_ms"]
            if primary["available"]:
                point = primary["equal_block_geometric_mean_ratio"]
                primary["useful_gain_margin_met_by_point_only"] = point <= 1 - protocol["acceptance"].get("useful_gain_percent", 5) / 100
                primary["regression_margin_exceeded_by_point_only"] = point >= 1 + protocol["acceptance"].get("acceptable_regression_percent", 2) / 100
                primary["confirmatory_acceptance"] = False
            item["summary"][contrast] = estimates
        result["datasets"].append(item)
    result["complete"] = status.get("state") == "comparison_completed" and all(d["complete"] for d in result["datasets"])
    result["cross_block_runtime_reference"] = runtime_reference
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": report["complete"], "output": str(args.output), "final_speedup_claim": False}))
    return 0 if report["complete"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
