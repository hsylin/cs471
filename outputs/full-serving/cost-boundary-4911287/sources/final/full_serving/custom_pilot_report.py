"""Read-only descriptive comparison of three explicitly named pilot servers.

Pass0 requests diagnostic token IDs/logprobs; pass1 uses ordinary payloads.
Neither is a held-out or counterbalanced performance experiment. This report
does not replace the independent runtime/correctness admission audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full_serving.custom_report import (IDENTITY_KEYS, METRICS, at, cache_counter_deltas,
                                        expected_generation, ratio, replay_evidence)
from full_serving.report_pilots import metric_deltas, read_phase

ARMS = ("off", "resident", "placement_only")
DATASETS = ("wildchat", "sharegpt", "instructcoder", "numinamath", "longbench")
CONTRASTS = (("resident", "off"), ("resident", "placement_only"), ("placement_only", "off"))
LIMITS = [
    "Pilot inputs were previously used; these are not held-out generalization results.",
    "One fresh server per arm, fixed arm/source/pass order; no independent repetitions or inferential intervals.",
    "Pass0 includes diagnostic IDs/logprobs. Pass1 repeats the same inputs after pass0 and can benefit from prefix caching. Do not pool the passes.",
    "Resident versus placement_only keeps the intended placement and control suppression fixed, but fixed order, prefix-cache state, and generated-work differences prevent an inferential causal claim.",
    "Output tokens per second alone is not response-time speedup when generated lengths differ.",
    "HTTP success and diagnostic IDs/logprobs do not establish all-layer numerical equivalence; this report is not runtime or correctness admission.",
    "Completed elapsed covers the offered episode and drain, excluding startup and time between phases. Request RUNNING metrics are not isolated GPU service.",
]


def read_json(path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def phase(directory, dataset, arm, number):
    stem = f"{dataset}-pass{number}"
    path = directory / (stem + ".jsonl")
    item, records = read_phase(path)
    item.get("summary", {}).pop("speed_claim_allowed", None)
    item["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    item["native_phases"] = metric_deltas(directory / (stem + "-metrics-before.txt"), directory / (stem + "-metrics-after.txt"))
    item["cache"] = cache_counter_deltas(directory / (stem + "-metrics-before.txt"), directory / (stem + "-metrics-after.txt"))
    if not item["valid"]:
        item["metrics"] = None
        return item, records
    header = item["header"]
    if (header.get("arm") != arm or header.get("request_count") != 8
            or header.get("load_id") != f"pass{number}"
            or header.get("block_id") != dataset + "-resident-pilot"
            or header.get("arrival_kind") != "synthetic_regular_pilot"
            or header.get("synthetic_rate_requests_per_s") != 2
            or not expected_generation(header, {}, number == 0)):
        item["issues"].append("pilot phase, arm, rate, request count or generation contract mismatch")
    calendar = {"kind": "synthetic_regular_pilot", "rate_requests_per_s": 2.0,
                "arrival_offsets_ns": [i * 500_000_000 for i in range(8)],
                "request_ids": [r["request_id"] for r in records]}
    calendar_hash = hashlib.sha256(json.dumps(calendar, sort_keys=True).encode()).hexdigest()
    item["raw_timing_audit"] = replay_evidence(path, records, calendar, calendar_hash)
    item["issues"].extend(item["raw_timing_audit"]["issues"])
    item["valid"] = not item["issues"]
    summary = item["summary"]
    item["metrics"] = ({**{name: at(summary, keys) for name, keys in METRICS.items()},
                        **{name: summary[name] for name in ("completed_requests", "failed_requests", "completed_input_tokens",
                            "completed_output_tokens", "output_cap_completions", "finish_reason_counts")}}
                       if item["valid"] else None)
    return item, records


def paired(treatment, baseline, treatment_rows, baseline_rows):
    result = {"valid": False, "issues": [], "ratios": {}, "requests": [], "inferential_claim": False}
    if not treatment["valid"] or not baseline["valid"]:
        result["issues"].append("both complete valid phases required; successful subsets are not compared")
        return result
    if any(treatment["header"].get(key) != baseline["header"].get(key) for key in IDENTITY_KEYS):
        result["issues"].append("paired header input/calendar/configuration mismatch")
    for a, b in zip(treatment_rows, baseline_rows):
        if (any(a.get(key) != b.get(key) for key in ("request_id", "input_sha256", "payload_sha256", "arrival_offset_ns", "expected_input_tokens"))
                or a["usage"]["prompt_tokens"] != b["usage"]["prompt_tokens"]):
            result["issues"].append("paired actual input/payload/prompt/calendar mismatch")
            break
    if result["issues"]:
        return result
    result["valid"] = True
    result["ratios"] = {name: ratio(treatment["metrics"][name], baseline["metrics"][name]) for name in METRICS}
    result["percent_changes"] = {name: (value - 1) * 100 if value is not None else None for name, value in result["ratios"].items()}
    for a, b in zip(treatment_rows, baseline_rows):
        token_hashes = (a.get("completion_token_ids_sha256"), b.get("completion_token_ids_sha256"))
        result["requests"].append({"request_id": a["request_id"], "prompt_tokens": a["usage"]["prompt_tokens"],
            "treatment_output_tokens": a["usage"]["completion_tokens"], "baseline_output_tokens": b["usage"]["completion_tokens"],
            "output_token_difference": a["usage"]["completion_tokens"] - b["usage"]["completion_tokens"],
            "generated_text_hash_equal": a["output_sha256"] == b["output_sha256"],
            "generated_token_hash_equal": token_hashes[0] == token_hashes[1] if all(token_hashes) else None,
            "treatment_finish_reason": a["finish_reason"], "baseline_finish_reason": b["finish_reason"]})
    result["equal_generated_texts"] = sum(r["generated_text_hash_equal"] for r in result["requests"])
    result["equal_output_counts"] = sum(r["output_token_difference"] == 0 for r in result["requests"])
    return result


def build_report(directories):
    if set(directories) != set(ARMS):
        raise ValueError("exactly three explicit pilot directories are required")
    directories = {k: Path(v) for k, v in directories.items()}
    if len({p.resolve() for p in directories.values()}) != 3:
        raise ValueError("pilot directories must be distinct")
    result = {"schema": "resident_pilot_comparison_v1", "complete": False, "pilot_only": True,
              "final_speedup_claim": False, "inferential_intervals": None, "limits": LIMITS,
              "sources": {}, "datasets": []}
    for arm, directory in directories.items():
        status = read_json(directory / "status.json")
        valid = (isinstance(status, dict) and status.get("pilot_only") is True
                 and status.get("final_inputs_opened") is False and status.get("arm") == arm)
        phases = ([(p.get("dataset"), p.get("phase")) for p in status.get("completed_phases", [])]
                  if valid else [])
        complete = (valid and status.get("state") == "pilot_completed"
                    and phases == [(name, n) for name in DATASETS for n in (0, 1)])
        public_status = ({k: v for k, v in status.items() if k != "completed_phases"} if isinstance(status, dict) else status)
        if isinstance(public_status, dict):
            public_status["completed_phases"] = [{"dataset": d, "phase": n} for d, n in phases]
        result["sources"][arm] = {"directory": str(directory), "status": public_status, "controller_identity_valid": valid,
                                  "controller_complete": complete, "startup": read_json(directory / "startup.json"),
                                  "runtime_admission_verified_by_this_report": False}
    for dataset in DATASETS:
        item = {"dataset": dataset, "passes": []}
        for number in (0, 1):
            p = {"pass": number, "diagnostic_payload": number == 0, "arms": {}, "contrasts": {}}
            records = {}
            for arm, directory in directories.items():
                view, records[arm] = phase(directory, dataset, arm, number)
                if not result["sources"][arm]["controller_identity_valid"]:
                    view["issues"].append("controller identity missing or mismatched")
                    view["valid"] = False
                    view["metrics"] = None
                p["arms"][arm] = view
            for treatment, baseline in CONTRASTS:
                p["contrasts"][treatment + "/" + baseline] = paired(p["arms"][treatment], p["arms"][baseline], records[treatment], records[baseline])
            item["passes"].append(p)
        result["datasets"].append(item)
    result["complete"] = (all(s["controller_complete"] for s in result["sources"].values())
        and all(p["arms"][arm]["valid"] for d in result["datasets"] for p in d["passes"] for arm in ARMS)
        and all(c["valid"] for d in result["datasets"] for p in d["passes"] for c in p["contrasts"].values()))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    dirs = {arm: getattr(args, arm) for arm in ARMS}
    if any(args.output.resolve().is_relative_to(d.resolve()) for d in dirs.values()):
        raise ValueError("write the report outside the read-only evidence directories")
    report = build_report(dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": report["complete"], "final_speedup_claim": False, "output": str(args.output)}))
    return 0 if report["complete"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
