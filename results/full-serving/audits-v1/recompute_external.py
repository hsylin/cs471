"""Recompute descriptive pilot metrics directly from audited public timestamps.

Run from the repository root, for example:
python results/full-serving/audits-v1/recompute_external.py \
  --root results/full-serving/pilots-v1 \
  --audit results/full-serving/audits-v1/pilot-audit-06.json \
  --output results/full-serving/audits-v1/pilot-external-recompute-06.json

This deliberately does not import the serving summary/report implementation.
It does not read private text or publish any generated text/token arrays.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(value):
    body = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p95": None}
    p = (len(values)-1)*.95
    lo, hi = math.floor(p), math.ceil(p)
    return {"n": len(values), "mean": statistics.fmean(values),
            "p95": values[lo]+(values[hi]-values[lo])*(p-lo)}


def recompute(root, audit):
    if audit["report_sha256"] != sha({k: v for k, v in audit.items() if k != "report_sha256"}):
        raise ValueError("Audit seal differs")
    report = {"schema": "independent_pilot_external_timestamp_recompute_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "audit_report_sha256": audit["report_sha256"], "recompute_source_sha256": sha(Path(__file__).read_bytes()),
              "comparison": "100 * (EPLB_on / EPLB_off - 1); negative latency change is lower",
              "scope": "One 24-request external-arrival block per source; descriptive only, fixed OFF-then-ON order; no request-level confidence interval or superiority claim.",
              "timing": "Response=end_ns-scheduled_ns; TTFT=first_content_ns-scheduled_ns; elapsed=max(last_end,epoch+horizon)-epoch. Initialization excluded; all outputs generated with EOS respected and max_tokens=2048.",
              "cohorts": [], "uncompleted_or_unaudited_cohorts": []}
    for evidence in audit["cohorts"]:
        dataset = evidence["dataset"]
        if evidence["status"] != "passed":
            report["uncompleted_or_unaudited_cohorts"].append({"dataset": dataset, "status": evidence["status"]})
            continue
        if evidence["native_counter_reconciliation"] != "matched" or evidence["admitted_private_binding"] != "passed":
            raise ValueError(f"{dataset}: incomplete native/private audit coverage")
        arms, raw = {}, {}
        for arm in ("off", "on"):
            body = (root/dataset/arm/"external.jsonl").read_bytes()
            if sha(body) != evidence["stages"]["external"]["arms"][arm]["record_file_sha256"]:
                raise ValueError(f"{dataset}/{arm}: raw records changed since audit")
            rows = [json.loads(line) for line in body.split(b"\n") if line]
            requests = sorted((r for r in rows if r["type"] == "request"), key=lambda r: r["request_index"])
            timing = next(r for r in rows if r["type"] == "run_summary")
            if len(requests) != 24 or not all(r["success"] for r in requests):
                raise ValueError(f"{dataset}/{arm}: expected 24 completed requests")
            raw[arm] = requests
            epoch, horizon = timing["epoch_monotonic_ns"], timing["offered_horizon_ns"]
            last_end = max(r["end_ns"] for r in requests)
            elapsed = (max(last_end, epoch+horizon)-epoch)/1e9
            output = sum(r["usage"]["completion_tokens"] for r in requests)
            arms[arm] = {"raw_file_sha256": sha(body), "requests": 24,
                         "response_from_schedule_s": describe([(r["end_ns"]-r["scheduled_ns"])/1e9 for r in requests]),
                         "ttft_from_schedule_s": describe([(r["first_content_ns"]-r["scheduled_ns"])/1e9 for r in requests if r["first_content_ns"] is not None]),
                         "completed_elapsed_s": elapsed, "drain_s": max(0, last_end-epoch-horizon)/1e9,
                         "completed_requests_per_s": 24/elapsed, "completed_output_tokens_per_s": output/elapsed,
                         "input_tokens": sum(r["usage"]["prompt_tokens"] for r in requests), "output_tokens": output,
                         "finish_reasons": dict(Counter(r["finish_reason"] for r in requests))}
        pairs = {f"{metric}.{stat}": (arms["off"][metric][stat], arms["on"][metric][stat])
                 for metric in ("response_from_schedule_s", "ttft_from_schedule_s") for stat in ("mean", "p95")}
        pairs.update({metric: (arms["off"][metric], arms["on"][metric])
                      for metric in ("completed_elapsed_s", "drain_s", "completed_requests_per_s", "completed_output_tokens_per_s", "output_tokens")})
        report["cohorts"].append({"dataset": dataset, "arms": arms,
            "on_vs_off_change_percent": {k: 100*(on/off-1) if off and on is not None else None for k, (off, on) in pairs.items()},
            "same_generated_text_sha256_requests": sum(b["output_sha256"] == t["output_sha256"] for b, t in zip(raw["off"], raw["on"])),
            "same_output_token_count_requests": sum(b["usage"]["completion_tokens"] == t["usage"]["completion_tokens"] for b, t in zip(raw["off"], raw["on"])),
            "input_identity_and_raw_timing_audit": "passed",
            "compiler_cache_audit": evidence["native_configuration_pair"]["native_build_evidence_pair"]["flashinfer_prebuild"]["status"]})
    report["report_sha256"] = sha(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output path; prior artifacts are immutable")
    result = recompute(args.root, json.loads(args.audit.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "report_sha256": result["report_sha256"],
                      "cohorts": {c["dataset"]: c["on_vs_off_change_percent"] for c in result["cohorts"]},
                      "pending": result["uncompleted_or_unaudited_cohorts"]}))
