#!/usr/bin/env python3
"""Describe completed three-arm records without changing the frozen analysis.

No confidence intervals, hypothesis tests, adaptive rules or policy selection are
computed. Technical repetitions are exposed separately from unique source states.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ("KEEP", "PREINSTALLED", "CHARGED")
STATE_FIELDS = ("split", "block_id", "regime", "batch_tokens")
PAIR_FIELDS = ("config_hash", *STATE_FIELDS, "candidate_id", "horizon_steps", "repetition")
METRICS = (
    "charged_payload_bytes", "changed_slots", "native_call_ms",
    "net_payback_ms", "headroom_ms", "transition_ms",
    "future_peak_assignments_keep", "future_peak_assignments_candidate",
    "future_peak_assignment_reduction", "future_peak_assignment_reduction_fraction",
    "mean_active_experts_per_rank_keep", "mean_active_experts_per_rank_candidate",
    "mean_active_experts_per_rank_change",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_of(row, fields):
    return tuple(row[field] for field in fields)


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected finite number: {label}")
    return value


def describe(values):
    values = list(values)
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    for value in values:
        finite_number(value, "descriptive sample")
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": max(values)}


def common_source_signature(row):
    prefix = [{k: d.get(k) for k in ("step", "allowed", "would_allow", "current_map",
               "proposed_map", "committed_map", "current_counts", "transfer_sent_bytes")}
              for d in row["prefix_decisions"]]
    return {"source_map_sha256": row["source_map_sha256"],
            "source_weights_sha256": row["source_weights_sha256"],
            "capture_sha256": row["capture_sha256"], "replay_sha256": row["replay_sha256"],
            "current_counts": row["current_counts"], "prefix": prefix,
            "target_b3_would_allow_native": row["b3_would_allow_native"],
            "source_prompt_ids": row["source_prompt_ids"], "decision_step": row["decision_step"]}


def dispatch_summary(row):
    diagnostic = row["physical_dispatch_diagnostic"]
    counts = diagnostic["owner_assignments_per_step"]
    active = diagnostic["active_physical_experts_per_rank_per_step"]
    horizon, ranks = row["horizon_steps"], row["world_size"]
    if len(counts) != horizon or len(active) != horizon:
        raise ValueError("Physical diagnostic does not cover the declared horizon")
    slots = len(row["final_map"]) // ranks
    expected = row["batch_tokens"] * row["weights_source"]["top_k"]
    for assigned, activated in zip(counts, active):
        if len(assigned) != ranks or len(activated) != ranks:
            raise ValueError("Physical diagnostic rank count differs")
        if sum(assigned) != expected:
            raise ValueError("Physical assignments fail per-input token conservation")
        for count, count_active in zip(assigned, activated):
            if (int(count) != count or int(count_active) != count_active or count < 0
                    or not 0 <= count_active <= min(slots, count)):
                raise ValueError("Invalid physical count or active-expert count")
    per_rank = [statistics.fmean(step[rank] for step in active) for rank in range(ranks)]
    return {"mean_per_input_peak_assignments": statistics.fmean(max(step) for step in counts),
            "mean_active_experts_by_rank": per_rank,
            "mean_active_experts_per_rank": statistics.fmean(per_rank)}


def paired_measurement(arms):
    keep, preinstalled, charged = (arms[a] for a in ARMS)
    for field in ("source_map_sha256", "source_weights_sha256", "candidate_map_sha256",
                  "capture_sha256", "replay_sha256", "candidate_changed_slots", "predicted_gain"):
        if any(row[field] != keep[field] for row in arms.values()):
            raise ValueError(f"Paired arm mismatch: {field}")
    if preinstalled["physical_dispatch_diagnostic"] != charged["physical_dispatch_diagnostic"]:
        raise ValueError("Preinstalled/charged candidate dispatch differs")
    for row in arms.values():
        elapsed = finite_number(row["elapsed_seconds"], "elapsed_seconds")
        if elapsed <= 0 or elapsed != max(row["per_rank_elapsed_seconds"]):
            raise ValueError("Elapsed time is not positive slowest-rank time")
        if not row["commit_completed"] or row["steps_completed"] != row["horizon_steps"]:
            raise ValueError("Arm has incomplete commit or future")
        if row["future_rebalancing_enabled"]:
            raise ValueError("Future rebalancing violates fixed-action diagnostic")
        if sum(row["migration_sent_bytes_by_rank"]) != row["migration_bytes"]:
            raise ValueError("Payload rank totals differ")
    if keep["migration_bytes"] != 0 or preinstalled["migration_bytes"] != 0:
        raise ValueError("Non-charged arm migrated inside timer")
    if charged["migration_sent_bytes_by_rank"] != charged["candidate_preflight_payload_by_rank"]:
        raise ValueError("Charged payload differs from candidate preflight")
    k, p = dispatch_summary(keep), dispatch_summary(preinstalled)
    reduction = k["mean_per_input_peak_assignments"] - p["mean_per_input_peak_assignments"]
    output = {field: keep[field] for field in PAIR_FIELDS}
    output.update(
        state_id=keep["state_id"], candidate_changed_slots=keep["candidate_changed_slots"],
        current_window_proxy_gain=keep["predicted_gain"],
        charged_payload_bytes=charged["migration_bytes"], changed_slots=keep["candidate_changed_slots"],
        native_call_ms=1000 * finite_number(charged["migration_seconds"], "migration_seconds"),
        net_payback_ms=1000 * (keep["elapsed_seconds"] - charged["elapsed_seconds"]),
        headroom_ms=1000 * (keep["elapsed_seconds"] - preinstalled["elapsed_seconds"]),
        transition_ms=1000 * (charged["elapsed_seconds"] - preinstalled["elapsed_seconds"]),
        future_peak_assignments_keep=k["mean_per_input_peak_assignments"],
        future_peak_assignments_candidate=p["mean_per_input_peak_assignments"],
        future_peak_assignment_reduction=reduction,
        future_peak_assignment_reduction_fraction=reduction / k["mean_per_input_peak_assignments"],
        mean_active_experts_per_rank_keep=k["mean_active_experts_per_rank"],
        mean_active_experts_per_rank_candidate=p["mean_active_experts_per_rank"],
        mean_active_experts_per_rank_change=(p["mean_active_experts_per_rank"] - k["mean_active_experts_per_rank"]),
        active_experts_mean_by_rank_keep=k["mean_active_experts_by_rank"],
        active_experts_mean_by_rank_candidate=p["mean_active_experts_by_rank"],
        allocation_ids=sorted({str(r["allocation_id"]) for r in arms.values()}),
        process_run_ids=sorted({r["process_run_id"] for r in arms.values()}),
    )
    return output


def summarize(records_paths):
    inputs, excluded, rows = [], Counter(), []
    record_ids = set()
    for path in records_paths:
        path = Path(path)
        inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_hash(path)})
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("status") != "complete" or row.get("correctness_pass") is not True
                        or row.get("evidence_eligible") is not True
                        or row.get("measurement_kind") != "gpu_payback"):
                    excluded["incomplete_failed_incorrect_or_ineligible"] += 1
                    continue
                if row["record_id"] in record_ids:
                    raise ValueError(f"Duplicate record ID at {path}:{line_number}")
                record_ids.add(row["record_id"])
                rows.append(row)
    if not rows:
        raise ValueError("No complete eligible GPU payback records")
    source_signatures, source_rows, candidate_signatures, pairs = {}, {}, {}, defaultdict(dict)
    for row in rows:
        state = key_of(row, STATE_FIELDS)
        signature = common_source_signature(row)
        if state in source_signatures and source_signatures[state] != signature:
            raise ValueError(f"Source state differs across candidates, horizons or repetitions: {state}")
        source_signatures[state], source_rows[state] = signature, row
        candidate_key = (*state, row["candidate_id"])
        candidate_signature = {field: row[field] for field in (
            "candidate_map_sha256", "candidate_changed_slots", "predicted_gain",
            "candidate_preflight_payload_by_rank")}
        if candidate_key in candidate_signatures and candidate_signatures[candidate_key] != candidate_signature:
            raise ValueError(f"Candidate differs across horizons or repetitions: {candidate_key}")
        candidate_signatures[candidate_key] = candidate_signature
        pair = key_of(row, PAIR_FIELDS)
        if row["arm"] not in ARMS or row["arm"] in pairs[pair]:
            raise ValueError(f"Invalid or duplicate arm in pair: {pair}")
        pairs[pair][row["arm"]] = row
    incomplete, measurements = [], []
    for pair, arms in sorted(pairs.items()):
        if set(arms) != set(ARMS):
            incomplete.append({**dict(zip(PAIR_FIELDS, pair)), "missing_arms": sorted(set(ARMS) - set(arms))})
        else:
            measurements.append(paired_measurement(arms))
    states = []
    for state, row in sorted(source_rows.items()):
        earlier = [d for d in row["prefix_decisions"] if d["step"] < row["decision_step"]]
        accepted = [d for d in earlier if d["allowed"]]
        states.append({**dict(zip(STATE_FIELDS, state)), "state_id": row["state_id"],
                       "source_prompt_ids": row["source_prompt_ids"],
                       "earlier_prefix_opportunities": len(earlier),
                       "earlier_prefix_accepted_decisions": len(accepted),
                       "earlier_prefix_actual_move_decisions": sum(d["transfer_sent_bytes"] > 0 for d in accepted),
                       "earlier_prefix_accepted_any": bool(accepted),
                       "earlier_prefix_actual_move_any": any(d["transfer_sent_bytes"] > 0 for d in accepted),
                       "target_native_b3_would_allow": row["b3_would_allow_native"],
                       "source_map_sha256": row["source_map_sha256"],
                       "source_weights_sha256": row["source_weights_sha256"]})
    by_group = defaultdict(list)
    for measurement in measurements:
        by_group[key_of(measurement, ("split", "candidate_id", "batch_tokens", "horizon_steps"))].append(measurement)
    summaries = []
    for group, values in sorted(by_group.items()):
        by_state = defaultdict(list)
        for value in values:
            by_state[key_of(value, STATE_FIELDS)].append(value)
        # These are descriptive means of technical repeats, not independent draws.
        state_means = [{metric: statistics.fmean(x[metric] for x in repeats) for metric in METRICS}
                       for repeats in by_state.values()]
        positive = lambda xs, field: sum(x[field] > 0 for x in xs)
        summaries.append({**dict(zip(("split", "candidate_id", "batch_tokens", "horizon_steps"), group)),
                          "technical_paired_groups": len(values), "unique_source_states": len(by_state),
                          "source_blocks": sorted({v["block_id"] for v in values}),
                          "repetitions": sorted({v["repetition"] for v in values}),
                          "technical_repeat_metrics": {metric: describe(v[metric] for v in values) for metric in METRICS},
                          "source_state_mean_metrics": {metric: describe(v[metric] for v in state_means) for metric in METRICS},
                          "positive_counts": {
                              "technical_pairs_net_payback": positive(values, "net_payback_ms"),
                              "technical_pairs_headroom": positive(values, "headroom_ms"),
                              "technical_pairs_transition_penalty": positive(values, "transition_ms"),
                              "technical_pairs_actual_peak_reduction": positive(values, "future_peak_assignment_reduction"),
                              "source_state_mean_net_payback": positive(state_means, "net_payback_ms"),
                              "source_state_mean_headroom": positive(state_means, "headroom_ms"),
                              "source_state_mean_actual_peak_reduction": positive(state_means, "future_peak_assignment_reduction"),
                          }})
    return {"schema_version": 1, "descriptive_only": True,
            "inputs": inputs, "complete_eligible_arm_records": len(rows),
            "excluded_records": dict(excluded), "complete_paired_groups": len(measurements),
            "incomplete_pairs": incomplete, "unique_source_states": len(states),
            "independent_source_blocks": sorted({(s["split"], s["block_id"]) for s in states}),
            "source_state_counts": {
                "prefix_accepted_any": sum(s["earlier_prefix_accepted_any"] for s in states),
                "prefix_actual_move_any": sum(s["earlier_prefix_actual_move_any"] for s in states),
                "target_native_b3_would_allow": sum(s["target_native_b3_would_allow"] for s in states)},
            "definitions": {
                "net_payback_ms": "1000*(KEEP-CHARGED); positive means observed charged benefit",
                "headroom_ms": "1000*(KEEP-PREINSTALLED); positive means observed free-install benefit",
                "transition_ms": "1000*(CHARGED-PREINSTALLED); includes native transition and cache-conditioning differences",
                "native_call_ms": "Full synchronized native transfer/commit call, including CPU preparation, copies and map update; not network time",
                "future_peak_assignment_reduction": "Mean per-input maximum rank assignment count under KEEP minus PREINSTALLED, from exact native mapper diagnostics",
                "active_experts": "Physical slots with positive assignments in each input, averaged over inputs and ranks; rankwise means also retained",
                "state": "One split/block/regime/batch source snapshot, verified equal across candidate families, horizons and technical repetitions",
                "independence": "Source-document blocks are units; regimes, batch sizes, horizons, arms and technical repetitions are correlated",
                "limits": "No confidence intervals, hypothesis tests, policy decisions, causal component attribution or final-data tuning"},
            "summaries": summaries, "source_states": states, "paired_measurements": measurements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, nargs="+", help="JSONL or JSONL.gz input files")
    parser.add_argument("--output", required=True, help="Fresh descriptive JSON; refuses overwrite")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    result = summarize(args.records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "arm_records": result["complete_eligible_arm_records"],
                      "paired_groups": result["complete_paired_groups"],
                      "source_states": result["unique_source_states"],
                      "incomplete_pairs": len(result["incomplete_pairs"])}))


if __name__ == "__main__":
    main()
