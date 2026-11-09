"""Paired resident-dispatch analysis. Test fixtures are never GPU evidence."""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path
import random
import statistics

REGIMES = ("stable", "transient", "persistent", "switching")
GROUP_FIELDS = ("block_id", "regime", "batch_tokens", "repetition")
MATCH_FIELDS = ("config_hash", "capture_sha256", "replay_sha256", "source_map_sha256",
                "source_weights_sha256", "logical_routing_sha256",
                "decision_trajectory_sha256", "logical_trajectory_sha256", "timing_scope",
                "resolved_kernel_config", "source_prompt_ids", "steps_completed", "policy")


def criterion(interval, limit):
    """A reduction/guard passes only when its entire interval meets the limit."""
    if interval is None:
        return "inconclusive"
    if interval[1] <= limit:
        return "pass"
    if interval[0] > limit:
        return "fail"
    return "inconclusive"


def _quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def _estimate(logs, samples, seed, family=1, inferential=False):
    values = [logs[key] for key in sorted(logs)]
    result = {"source_blocks": len(values), "ratio": None, "ci95": None,
              "adjusted_ci": None, "family_size": family}
    if not values:
        return result
    result["ratio"] = math.exp(statistics.mean(values))
    if not inferential or len(values) < 8:
        return result
    rng = random.Random(seed)
    boot = [math.exp(statistics.mean(rng.choices(values, k=len(values)))) for _ in range(samples)]
    result["ci95"] = [_quantile(boot, .025), _quantile(boot, .975)]
    result["adjusted_ci"] = [_quantile(boot, .025 / family), _quantile(boot, 1 - .025 / family)]
    return result


def _unique(values, name):
    values = tuple(values)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty and unique")
    return values


def _collect(records, blocks, regimes, batches, repetitions, arms,
             allocation_by_repetition=None, require_two_allocations=False):
    groups = list(itertools.product(blocks, regimes, batches, repetitions))
    expected = set(itertools.product(groups, arms))
    index = {}
    for row in records:
        group = tuple(row.get(field) for field in GROUP_FIELDS)
        key = (group, row.get("arm"))
        if key not in expected:
            raise ValueError(f"unexpected row key: {key}")
        if key in index:
            raise ValueError(f"duplicate row key: {key}")
        index[key] = row
    allocation_sets = {rep: {str(row["allocation_id"]) for (group, _), row in index.items()
                             if group[-1] == rep and row.get("allocation_id") is not None
                             and str(row["allocation_id"]).strip()} for rep in repetitions}
    allocation_errors = []
    if require_two_allocations:
        if len(repetitions) != 2 or any(len(values) != 1 for values in allocation_sets.values()):
            allocation_errors.append("each of two repetitions must use one allocation")
        elif len(set(next(iter(values)) for values in allocation_sets.values())) != 2:
            allocation_errors.append("two distinct allocation windows required")
    if allocation_by_repetition is not None:
        expected_allocations = {int(rep): str(value) for rep, value in allocation_by_repetition.items()}
        if set(expected_allocations) != set(repetitions):
            raise ValueError("allocation_by_repetition must cover every repetition")
        for rep, values in allocation_sets.items():
            if values != {expected_allocations[rep]}:
                allocation_errors.append(f"repetition {rep} allocation differs from plan")
    missing, invalid, valid = [], [], []
    for group in groups:
        absent = [arm for arm in arms if (group, arm) not in index]
        identity = dict(zip(GROUP_FIELDS, group))
        if absent:
            missing.append({**identity, "arms": absent})
            continue
        rows = {arm: index[group, arm] for arm in arms}
        errors = []
        for arm, row in rows.items():
            if row.get("status") != "complete":
                errors.append(f"{arm}:status")
            for field in ("correctness_pass", "evidence_eligible"):
                if row.get(field) is not True:
                    errors.append(f"{arm}:{field}")
            value = row.get("elapsed_seconds")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"{arm}:elapsed_seconds")
            if row.get("allocation_id") is None or not str(row["allocation_id"]).strip():
                errors.append(f"{arm}:allocation_id")
            for field in ("source_restore_verified", "expected_arm_weights_verified",
                          "commit_completed", "logical_routing_preserved", "rank_assignment_agreement"):
                if field in row and row[field] is not True:
                    errors.append(f"{arm}:{field}")
            for field in ("migration_bytes", "migration_sent_messages"):
                if field in row and row[field] != 0:
                    errors.append(f"{arm}:fixed-map {field}")
            if "source_map" in row and "final_map" in row and row["source_map"] != row["final_map"]:
                errors.append(f"{arm}:fixed map changed")
        if len({str(row.get("allocation_id")) for row in rows.values()}) != 1:
            errors.append("pair crosses allocations")
        for field in MATCH_FIELDS:
            if any(field in row for row in rows.values()):
                values = [json.dumps(row.get(field), sort_keys=True) for row in rows.values()]
                if any(field not in row for row in rows.values()) or len(set(values)) != 1:
                    errors.append(f"paired {field} differs")
        if errors:
            invalid.append({**identity, "issues": errors})
        else:
            valid.append((identity, rows))
    return {"expected_records": len(expected), "observed_records": len(index),
            "missing": missing, "invalid_groups": invalid,
            "allocation_errors": allocation_errors,
            "allocations_by_repetition": {str(rep): sorted(values) for rep, values in allocation_sets.items()}}, valid


def analyze(records, *, phase, expected_blocks, expected_regimes=REGIMES,
            expected_batches=(128, 1024), expected_repetitions=(0, 1),
            allocation_by_repetition=None, native_arm="native", treatment_arm="online",
            bootstrap_samples=20000, bootstrap_seed=10471,
            useful_gain_ratio=.95, stable_ratio_limit=1.02):
    """Analyze a frozen two-arm matrix; only eight-block final data can pass.

    Allocation IDs are retained, not resampled as if two windows estimated a
    hardware population. Source blocks are the resampling units. Extra fields
    listed in MATCH_FIELDS are checked when supplied; GPU provenance, native
    counters and numerical correctness still need the separate raw-data audit.
    """
    if phase not in {"pilot", "validation", "final"}:
        raise ValueError("unsupported phase")
    blocks = _unique(expected_blocks, "blocks")
    regimes = _unique(expected_regimes, "regimes")
    batches = _unique(expected_batches, "batches")
    repetitions = _unique(expected_repetitions, "repetitions")
    if native_arm == treatment_arm or "stable" not in regimes or len(regimes) < 2:
        raise ValueError("distinct arms and stable/changing regimes required")
    if bootstrap_samples < 100 or not 0 < useful_gain_ratio < 1 or stable_ratio_limit < 1:
        raise ValueError("invalid analysis plan")
    if phase == "final" and (len(blocks) != 8 or set(regimes) != set(REGIMES)
                             or set(batches) != {128, 1024} or len(repetitions) != 2):
        raise ValueError("final requires eight blocks, four regimes, two batches and two allocation repetitions")
    summary, valid = _collect(records, blocks, regimes, batches, repetitions,
                              (native_arm, treatment_arm), allocation_by_repetition,
                              require_two_allocations=phase == "final")
    complete = not (summary["missing"] or summary["invalid_groups"] or summary["allocation_errors"])
    inference = complete and phase == "final"
    pairs = []
    for identity, rows in valid:
        native, treatment = rows[native_arm], rows[treatment_arm]
        pairs.append({**identity, "allocation_id": str(native["allocation_id"]),
                      "native_seconds": native["elapsed_seconds"],
                      "treatment_seconds": treatment["elapsed_seconds"],
                      "log_ratio": math.log(treatment["elapsed_seconds"]) - math.log(native["elapsed_seconds"])})
    cell_logs, cells = {}, []
    for regime, batch in itertools.product(regimes, batches):
        logs = {}
        for block in blocks:
            selected = [p["log_ratio"] for p in pairs if (p["block_id"], p["regime"], p["batch_tokens"]) == (block, regime, batch)]
            if len(selected) == len(repetitions):
                logs[block] = statistics.mean(selected)
        cell_logs[regime, batch] = logs
        family = len(batches) if regime == "stable" else (len(regimes) - 1) * len(batches)
        result = _estimate(logs, bootstrap_samples, bootstrap_seed, family, inference)
        limit = stable_ratio_limit if regime == "stable" else useful_gain_ratio
        cells.append({"regime": regime, "batch_tokens": batch, **result,
                      "limit": limit, "criterion": criterion(result["adjusted_ci"], limit)})
    aggregates = []
    for group in ("changing", "stable"):
        keys = [(r, b) for r, b in cell_logs if (r == "stable") == (group == "stable")]
        logs = {block: statistics.mean(cell_logs[key][block] for key in keys)
                for block in blocks if all(block in cell_logs[key] for key in keys)}
        result = _estimate(logs, bootstrap_samples, bootstrap_seed, inferential=inference)
        limit = stable_ratio_limit if group == "stable" else useful_gain_ratio
        aggregates.append({"group": group, **result, "limit": limit,
                           "criterion": criterion(result["ci95"], limit)})
    by_allocation = []
    for allocation in sorted({pair["allocation_id"] for pair in pairs}):
        for group in ("changing", "stable"):
            selected = [p for p in pairs if p["allocation_id"] == allocation
                        and (p["regime"] == "stable") == (group == "stable")]
            by_allocation.append({"allocation_id": allocation, "group": group,
                                  "pairs": len(selected),
                                  "ratio": math.exp(statistics.mean(p["log_ratio"] for p in selected)) if selected else None,
                                  "interpretation": "descriptive allocation-specific paired ratio; no hardware-population interval"})
    useful = next(row["criterion"] for row in aggregates if row["group"] == "changing")
    guards = [row["criterion"] for row in cells if row["regime"] == "stable"]
    guards += [row["criterion"] for row in aggregates if row["group"] == "stable"]
    stable = "fail" if "fail" in guards else "pass" if all(x == "pass" for x in guards) else "inconclusive"
    acceptance = "fail" if "fail" in (useful, stable) else "pass" if useful == stable == "pass" else "inconclusive"
    return {**summary, "phase": phase, "status": "complete" if complete else "incomplete",
            "claim_eligible": inference, "native_arm": native_arm, "treatment_arm": treatment_arm,
            "eligible_matched_pairs": len(pairs), "pairs": pairs, "cells": cells,
            "aggregates": aggregates, "by_allocation": by_allocation,
            "primary_useful_gain": useful, "stable_guard": stable, "acceptance": acceptance,
            "interval_scope": "Eight selected source-document blocks, conditional on two allocation windows and the replay construction; not independent steps, production demand or a hardware population."}


def analyze_pilot(records, *, expected_blocks, expected_regimes=("domain0", "domain1"),
                  expected_batches=(128, 1024), expected_repetitions=(0, 1),
                  native_arm="native", noop_arm="native_aa", precomputed_native_arm="pre_native",
                  candidate_arms=("one_copy_home", "token_balanced"),
                  null_band=.01, noise_floor=.01, tax_reserve=.01,
                  useful_gain_ratio=.95, stable_ratio_limit=1.02):
    """Descriptive headroom screen, never confirmatory inference from two blocks."""
    blocks = _unique(expected_blocks, "blocks")
    regimes = _unique(expected_regimes, "regimes")
    batches = _unique(expected_batches, "batches")
    repetitions = _unique(expected_repetitions, "repetitions")
    arms = _unique((native_arm, noop_arm, precomputed_native_arm, *candidate_arms), "arms")
    if len(blocks) != 2 or len(regimes) != 2 or set(batches) != {128, 1024} or len(repetitions) != 2:
        raise ValueError("pilot requires two blocks, two domains, both batches and two repetitions")
    if any(not math.isfinite(x) or x < 0 for x in (null_band, noise_floor, tax_reserve)):
        raise ValueError("invalid pilot reserves")
    summary, valid = _collect(records, blocks, regimes, batches, repetitions, arms)
    complete = not (summary["missing"] or summary["invalid_groups"])
    cells = []
    for block, regime, batch, arm in itertools.product(blocks, regimes, batches, arms[1:]):
        logs = [math.log(rows[arm]["elapsed_seconds"]) - math.log(rows[native_arm]["elapsed_seconds"])
                for identity, rows in valid if (identity["block_id"], identity["regime"], identity["batch_tokens"]) == (block, regime, batch)]
        if len(logs) == len(repetitions):
            cells.append({"block_id": block, "regime": regime, "batch_tokens": batch,
                          "arm": arm, "ratio": math.exp(statistics.mean(logs))})
    null_ratios = [row["ratio"] for row in cells if row["arm"] == noop_arm]
    null_aggregate = math.exp(statistics.mean(math.log(r) for r in null_ratios)) if null_ratios else None
    null_deviation = max((abs(r - 1) for r in null_ratios), default=None)
    calibrated = (complete and null_aggregate is not None
                  and all(1 - null_band <= ratio <= 1 + null_band for ratio in null_ratios)
                  and 1 - null_band <= null_aggregate <= 1 + null_band)
    reserve = max(noise_floor, null_deviation) if null_deviation is not None else None
    candidates = []
    for arm in candidate_arms:
        by_block = []
        for block in blocks:
            selected = [row["ratio"] for row in cells if row["arm"] == arm and row["block_id"] == block]
            if len(selected) == len(regimes) * len(batches):
                by_block.append({"block_id": block, "ratio": math.exp(statistics.mean(math.log(r) for r in selected))})
        candidate_cells = [row for row in cells if row["arm"] == arm]
        assignment_cells = []
        for row in candidate_cells:
            reference = next(c for c in cells if c["arm"] == precomputed_native_arm
                             and all(c[key] == row[key] for key in ("block_id", "regime", "batch_tokens")))
            assignment_cells.append({**{key: row[key] for key in ("block_id", "regime", "batch_tokens")},
                                     "ratio": row["ratio"] / reference["ratio"]})
        enough = len(by_block) == len(blocks) and len(candidate_cells) == len(blocks) * len(regimes) * len(batches)
        passes = calibrated and enough and all(row["ratio"] + reserve + tax_reserve <= useful_gain_ratio for row in by_block) and all(row["ratio"] + reserve + tax_reserve <= stable_ratio_limit for row in candidate_cells)
        candidates.append({"arm": arm, "candidate_over_native_by_block": by_block,
                           "candidate_over_pre_native_cells": assignment_cells,
                           "advances_to_online_implementation": bool(passes),
                           "screen": "pass" if passes else "inconclusive" if not complete else "stop"})
    passing = [row for row in candidates if row["advances_to_online_implementation"]]
    selected = min(passing, key=lambda row: (max(b["ratio"] for b in row["candidate_over_native_by_block"]),
                                             row["arm"] != "one_copy_home", row["arm"]))["arm"] if passing else None
    return {**summary, "phase": "pilot", "status": "complete" if complete else "incomplete",
            "claim_eligible": False, "calibration": "pass" if calibrated else "stop" if complete else "inconclusive",
            "max_null_cell_deviation": null_deviation, "null_band": null_band,
            "null_aggregate_ratio": null_aggregate,
            "noise_reserve": reserve, "selector_tax_reserve": tax_reserve,
            "cells": cells, "candidates": candidates, "selected_candidate": selected,
            "interpretation": "Cyclic pilot: 16 conditioning inputs, then 32 inputs repeated eight times for 256 timed forwards with native SKIP observation/planning. Precomputed assignments omit selection cost; pre_native/native is an adapter contrast, not A/A. This is a descriptive development screen, not statistical equivalence, an online-policy result or a runtime bound."}


def describe_validation(records, *, expected_blocks,
                        expected_regimes=("domain0", "domain1"),
                        expected_batches=(128, 1024), expected_repetitions=(0, 1),
                        native_arm="native", treatment_arm="online"):
    """Describe the fixed online implementation on 32 development rows.

    Two development blocks support paired descriptive ratios, not confirmation.
    No timing threshold selects, retunes or disqualifies the fixed algorithm;
    fresh evaluation is conditional on correctness under the explicit amendment.
    """
    blocks = _unique(expected_blocks, "blocks")
    regimes = _unique(expected_regimes, "regimes")
    batches = _unique(expected_batches, "batches")
    repetitions = _unique(expected_repetitions, "repetitions")
    if (len(blocks) != 2 or len(regimes) != 2 or set(batches) != {128, 1024}
            or len(repetitions) != 2 or native_arm == treatment_arm):
        raise ValueError("online validation requires two blocks/domains/batches/repetitions and distinct arms")
    summary, valid = _collect(records, blocks, regimes, batches, repetitions,
                              (native_arm, treatment_arm))
    complete = not (summary["missing"] or summary["invalid_groups"] or summary["allocation_errors"])
    pairs = []
    for identity, rows in valid:
        native, treatment = rows[native_arm], rows[treatment_arm]
        pairs.append({**identity, "allocation_id": str(native["allocation_id"]),
                      "native_seconds": native["elapsed_seconds"],
                      "treatment_seconds": treatment["elapsed_seconds"],
                      "log_ratio": math.log(treatment["elapsed_seconds"]) - math.log(native["elapsed_seconds"])})
    cells = []
    for block, regime, batch in itertools.product(blocks, regimes, batches):
        selected = [pair["log_ratio"] for pair in pairs
                    if (pair["block_id"], pair["regime"], pair["batch_tokens"]) == (block, regime, batch)]
        if len(selected) == len(repetitions):
            cells.append({"block_id": block, "regime": regime, "batch_tokens": batch,
                          "paired_repetitions": len(selected),
                          "log_ratio": statistics.mean(selected),
                          "ratio": math.exp(statistics.mean(selected))})
    by_block = []
    for block in blocks:
        selected = [cell["log_ratio"] for cell in cells if cell["block_id"] == block]
        by_block.append({"block_id": block, "complete_cells": len(selected),
                         "ratio": math.exp(statistics.mean(selected))
                         if len(selected) == len(regimes) * len(batches) else None})
    by_batch = []
    for batch in batches:
        selected = [cell["log_ratio"] for cell in cells if cell["batch_tokens"] == batch]
        by_batch.append({"batch_tokens": batch, "complete_cells": len(selected),
                         "ratio": math.exp(statistics.mean(selected))
                         if len(selected) == len(blocks) * len(regimes) else None})
    return {**summary, "phase": "paired_descriptive", "status": "complete" if complete else "incomplete",
            "claim_eligible": False, "native_arm": native_arm, "treatment_arm": treatment_arm,
            "eligible_matched_pairs": len(pairs), "pairs": pairs, "cells": cells,
            "by_block": by_block, "by_batch": by_batch,
            "overall_ratio": math.exp(statistics.mean(cell["log_ratio"] for cell in cells))
            if complete and len(cells) == len(blocks) * len(regimes) * len(batches) else None,
            "performance_gate": "not_applied_by_predeclared_user_directed_fresh_test_amendment",
            "interpretation": "Descriptive fully charged online/native comparison on two development blocks; no confidence interval, parameter selection or confirmatory acceptance. Numerical correctness, native accounting and source/input audits are separate prerequisites for fresh evaluation. Selector cost is not inferred by subtracting these whole-replay times."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--plan", required=True, help="JSON kwargs; phase=pilot_screen, paired_descriptive, or pilot/validation/final")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    opener = gzip.open if args.records.endswith(".gz") else open
    with opener(args.records, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("phase") == "pilot_screen":
        plan.pop("phase")
        result = analyze_pilot(records, **plan)
    elif plan.get("phase") == "paired_descriptive":
        plan.pop("phase")
        result = describe_validation(records, **plan)
    else:
        result = analyze(records, **plan)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "claim_eligible", "acceptance", "calibration") if key in result}))


if __name__ == "__main__":
    main()
