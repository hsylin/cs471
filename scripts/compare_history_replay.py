#!/usr/bin/env python3
"""Compare bounded-history B4 replay with the valid continuous-history pilot.

This checks semantic identity and capture accounting only. Elapsed timings are
deliberately neither compared nor reported; this is not final performance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

REFERENCE_SHA = "036182c3a74868375d07e3114b75266d22e12975f04880b17fe54c63f6820754"
CANDIDATE_SHA = "e13b464cad82c6990d0459c3aa1a523703dce4241d4caa28440d0f38dffa1498"
KEY = ("regime", "trace_variant", "batch_tokens", "seed", "repetition")
COMMON = ("phase_steps", "trace_sha256", "trace_config", "weights_source", "policy_config",
    "native_vllm_commit", "native_source_attestation", "eplb_mode", "dispatch_backend",
    "output_collective", "weight_communicator", "reject_execution", "world_size", "gpu_name",
    "torch_version", "vllm_version", "cuda_version", "torch_num_threads", "dtype",
    "hidden_input_source", "logical_routing_source", "startup_contract", "native_threshold_active",
    "steps_completed", "measured_horizon_steps", "raw_trace_steps", "observation_warmup_steps",
    "final_map", "accepted", "rejected", "migration_bytes", "migration_sent_messages",
    "replica_budget_actual", "terminal_proposals_suppressed", "observation_count_multiplicity",
    "observation_window_assignment_total_expected", "observation_conservation_checks")
DECISION = ("step", "allowed", "reason", "tv", "predicted_gain", "current_counts", "previous_counts",
    "current_map", "proposed_map", "committed_map", "committed_map_sha256", "candidate_changed_slots",
    "physical_slot_assignments_window", "physical_slot_assignments_sha256", "physical_window_next_write",
    "physical_counter_scope", "physical_counter_input_multiplicity", "physical_owner_assignments_per_step",
    "physical_window_measured_step_range", "transfer_sent_bytes_by_rank", "transfer_sent_bytes", "commit_completed")
CAPTURES = {"logical_histogram_collection_enabled": True,
    "logical_histogram_measured_capture_steps": 128, "logical_histogram_warmup_capture_steps": 32,
    "previous_window_timestamp_checks": 7, "observation_conservation_checks": 14,
    "logical_histogram_count_method": "preallocated_int64_scatter_add"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    indexed = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") != "B4":
            continue
        for field, expected in (("measurement_kind", "gpu"), ("split", "pilot"), ("status", "complete"),
                                ("correctness_pass", True), ("evidence_eligible", True)):
            if row.get(field) != expected:
                raise ValueError(f"{path}: B4 record has invalid {field}")
        key = tuple(row[k] for k in KEY)
        if key in indexed:
            raise ValueError(f"{path}: duplicate B4 state {key}")
        indexed[key] = row
    if len(indexed) != 28:
        raise ValueError(f"{path}: expected 28 B4 states, found {len(indexed)}")
    return indexed


def compare(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("Reference and candidate workload/seed/repetition sets differ")
    differences, checked = [], []
    for key in sorted(reference):
        old, new = reference[key], candidate[key]
        start = len(differences)
        def mismatch(field):
            differences.append({"state": key, "field": field})
        for label, row, expected in (("reference", old, REFERENCE_SHA), ("candidate", new, CANDIDATE_SHA)):
            if row.get("harness_sha256") != expected or digest(row.get("harness_file_sha256")) != expected:
                mismatch(f"{label}.harness_sha256")
        for field in COMMON:
            if field not in old or field not in new or old[field] != new[field]:
                mismatch(field)
        for field, expected in CAPTURES.items():
            if new.get(field) != expected:
                mismatch(f"candidate.{field}")
        a, b = old["decision_log"], new["decision_log"]
        if len(a) != 7 or len(b) != 7:
            mismatch("decision_log.length")
        for index, (before, after) in enumerate(zip(a, b)):
            for field in DECISION:
                if field not in before or field not in after or before[field] != after[field]:
                    mismatch(f"decision_log[{index}].{field}")
        checked.append({"state": key, "matched": len(differences) == start,
            "decisions_compared": min(len(a), len(b)), "candidate_capture_accounting": {k: new.get(k) for k in CAPTURES}})
    return {"status": "identical_replay" if not differences else "mismatch", "expected_pairs": 28,
        "matched_pairs": sum(row["matched"] for row in checked), "reference_harness_sha256": REFERENCE_SHA,
        "candidate_harness_sha256": CANDIDATE_SHA, "differences": differences, "rows": checked,
        "comparison_scope": "Same logical traces, checkpoint weights, policy/native configuration, exact decisions, maps, physical dispatch windows and payloads. Bounded capture counts and all seven previous-window timestamp checks are required.",
        "timings_compared": False, "interpretation": "Semantic replay validation only, not a speedup comparison or final evidence. Manifest hashes and record IDs intentionally differ between implementation cohorts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = compare(load(args.reference), load(args.candidate))
    for name in ("reference", "candidate"):
        path = getattr(args, name)
        report[f"{name}_path"] = str(path.resolve())
        report[f"{name}_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "matched_pairs", "timings_compared")}))
    raise SystemExit(0 if report["status"] == "identical_replay" else 1)


if __name__ == "__main__":
    main()
