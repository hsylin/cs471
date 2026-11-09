#!/usr/bin/env python3
"""Independent CPU audit of the frozen resident-dispatch pilot and its summary.

Does not import resident_dispatch.analysis. Exact input/physical-counter replay
and numerical outputs require the separate raw-data audit; flags alone are not
independent correctness evidence. No GPU execution or measurements are created.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ARMS = ("native", "native_aa", "pre_native", "one_copy_home", "token_balanced")
CANDIDATES = ARMS[3:]
IDENTITY = ("block_id", "regime", "batch_tokens", "repetition", "arm")
COMMON = ("config_hash", "capture_sha256", "source_map_sha256", "source_weights_sha256",
          "logical_trajectory_sha256", "resolved_kernel_config", "source_prompt_ids",
          "timing_scope", "policy", "steps_completed")
TOLERANCE = 1e-12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def geometric(values):
    """Independent product/root calculation, rather than analyzer log averaging."""
    values = list(values)
    return math.prod(values) ** (1.0 / len(values))


def advance(block_ratios, cell_ratios, calibrated, noise, tax, gain=.95, guard=1.02):
    return calibrated and all(r + noise + tax <= gain for r in block_ratios) and all(r + noise + tax <= guard for r in cell_ratios)


def verify(manifest, records, summary, *, expected_allocation_id=None):
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def same(actual, expected, label):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                  and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE), label)
        else:
            check(actual == expected, label)

    config_hash = digest({key: value for key, value in manifest.items() if key != "freeze"})
    check(manifest.get("freeze", {}).get("sha256") == config_hash, "manifest freeze differs")
    matrix, plan = manifest["matrix"], manifest["analysis"]
    blocks, domains, batches = matrix["block_ids"], matrix["domains"], matrix["batch_tokens"]
    regimes, repetitions = [f"domain{d}" for d in domains], tuple(range(matrix["repetitions"]))
    check(len(blocks) == len(set(blocks)) == 2 and len(domains) == len(set(domains)) == 2
          and set(batches) == {128, 1024} and repetitions == (0, 1) and tuple(matrix["arms"]) == ARMS,
          "pilot is not the frozen eight-state/five-arm/two-repeat matrix")
    for field, expected in (("phase", "pilot_screen"), ("expected_blocks", blocks),
                            ("expected_regimes", regimes), ("expected_batches", batches),
                            ("expected_repetitions", list(repetitions))):
        same(plan.get(field), expected, f"analysis plan {field}")
    expected_keys = set(itertools.product(blocks, regimes, batches, repetitions, ARMS))
    check(len(expected_keys) == 80, "expected matrix is not 80 rows")
    indexed = {}
    for row in records:
        key = tuple(row.get(field) for field in IDENTITY)
        check(key in expected_keys, f"unexpected row {key}")
        check(key not in indexed, f"duplicate row {key}")
        indexed[key] = row
    check(set(indexed) == expected_keys and len(records) == 80, "missing/extra pilot rows")
    allocation_ids = {str(row.get("allocation_id")) for row in records}
    check(len(allocation_ids) == 1 and None not in [row.get("allocation_id") for row in records],
          "complete pilot must come from one allocation; attempts cannot be mixed")
    if expected_allocation_id is not None:
        check(allocation_ids == {str(expected_allocation_id)}, "pilot allocation differs from the authorized clean attempt")
    # A malformed matrix must not become a partial, apparently valid selection.
    if errors:
        return {"pass": False, "errors": errors, "observed_rows": len(records), "selection_recomputed": False}
    banks = {row["block_id"]: row["capture_sha256"] for row in manifest["banks"]}
    docs = {row["block_id"]: row["source_document_ids"] for row in manifest["blocks"]}
    for key, row in indexed.items():
        label = "/".join(map(str, key))
        check(row.get("status") == "complete", f"{label}: status")
        for flag in ("correctness_pass", "evidence_eligible", "source_restore_verified",
                     "expected_arm_weights_verified", "commit_completed", "logical_routing_preserved",
                     "rank_assignment_agreement"):
            check(row.get(flag) is True, f"{label}: {flag}")
        same(row.get("config_hash"), config_hash, f"{label}: config hash")
        same(row.get("resident_dispatch_sha256"), manifest["environment"]["resident_dispatch_sha256"], f"{label}: execution hash")
        same(row.get("resident_dispatch_file_sha256"), manifest["execution_source_sha256"], f"{label}: execution file hashes")
        same(row.get("capture_sha256"), banks[key[0]], f"{label}: bank hash")
        same(row.get("source_prompt_ids"), docs[key[0]], f"{label}: source documents")
        same(row.get("resolved_kernel_config"), manifest["default_configs"][str(key[2])], f"{label}: kernel configuration")
        same(row.get("timing_scope"), manifest["execution"]["timing_scope"], f"{label}: timing scope")
        same(row.get("steps_completed"), 256, f"{label}: measured steps")
        same(row.get("terminal_proposals_suppressed"), 1, f"{label}: terminal suppression")
        same(row.get("migration_bytes"), 0, f"{label}: migration bytes")
        same(row.get("migration_sent_messages"), 0, f"{label}: migration messages")
        check(row.get("policy") == "SKIP", f"{label}: policy")
        check(bool(str(row.get("allocation_id", "")).strip()), f"{label}: allocation ID")
        elapsed = row.get("elapsed_seconds")
        valid_time = isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed > 0
        check(valid_time, f"{label}: invalid elapsed time")
        ranks = row.get("per_rank_elapsed_seconds", [])
        valid_ranks = len(ranks) == 2 and all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in ranks)
        check(valid_ranks, f"{label}: rank times")
        if valid_ranks:
            same(elapsed, max(ranks), f"{label}: elapsed is not slowest rank")
        mapping = row.get("source_map", [])
        legal = len(mapping) == 160 and all(type(x) is int and 0 <= x < 128 for x in mapping) and set(mapping) == set(range(128))
        check(legal and row.get("final_map") == mapping, f"{label}: fixed map legality")
        decisions = row.get("decision_log", [])
        check([d.get("step") for d in decisions] == [63, 127, 191], f"{label}: decision cadence")
        for decision in decisions:
            counts = decision.get("current_counts", [])
            check(len(counts) == 128 and all(type(x) is int and x >= 0 for x in counts)
                  and sum(counts) == 16 * key[2] * 8 * 2, f"{label}: logical count conservation")
            check(decision.get("allowed") is False and decision.get("current_map") == mapping
                  and decision.get("committed_map") == mapping, f"{label}: rejected fixed-map decision")
        try:
            trajectory = [{name: d[name] for name in ("step", "current_counts", "current_map", "proposed_map", "committed_map")} for d in decisions]
            same(row.get("logical_trajectory_sha256"), digest(trajectory), f"{label}: logical trajectory hash")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: incomplete decision trajectory")
    # Five arms and both reversed repetitions must describe the same input/state.
    for state in itertools.product(blocks, regimes, batches):
        selected = [indexed[(*state, rep, arm)] for rep in repetitions for arm in ARMS]
        for field in COMMON:
            check(all(field in row for row in selected) and len({json.dumps(row.get(field), sort_keys=True) for row in selected}) == 1,
                  f"{state}: paired {field} differs")
        for rep in repetitions:
            rows = [indexed[(*state, rep, arm)] for arm in ARMS]
            check(len({row.get("allocation_id") for row in rows}) == 1, f"{state}/{rep}: pair crosses allocations")
            hashes = [indexed[(*state, rep, arm)].get("assignment_sha256") for arm in ARMS[:3]]
            check(hashes[0] is not None and len(set(hashes)) == 1, f"{state}/{rep}: native/A-A/pre-native assignments differ")
    ordered_cells = list(itertools.product(blocks, domains, batches))
    random.Random(manifest["execution"]["order_seed"]).shuffle(ordered_cells)
    expected_order = []
    for position, (block, domain, batch) in enumerate(ordered_cells):
        shift = position % len(ARMS)
        first = ARMS[shift:] + ARMS[:shift]
        for rep, order in enumerate((first, first[::-1])):
            for arm_position, arm in enumerate(order):
                key = (block, f"domain{domain}", batch, rep, arm)
                expected_order.append(key)
                same(indexed[key].get("arm_order"), list(order), f"{key}: execution order")
                same(indexed[key].get("arm_position"), arm_position, f"{key}: arm position")
    check([tuple(row[field] for field in IDENTITY) for row in records] == expected_order, "raw file execution sequence differs")
    if errors:
        return {"pass": False, "errors": errors, "observed_rows": len(records), "selection_recomputed": False}
    ratios = {}
    for state in itertools.product(blocks, regimes, batches):
        for arm in ARMS[1:]:
            ratios[(*state, arm)] = math.sqrt(math.prod(indexed[(*state, rep, arm)]["elapsed_seconds"] /
                                                        indexed[(*state, rep, "native")]["elapsed_seconds"] for rep in repetitions))
    nulls = [value for key, value in ratios.items() if key[-1] == "native_aa"]
    null_band, floor, tax = (plan.get(key, default) for key, default in (("null_band", .01), ("noise_floor", .01), ("tax_reserve", .01)))
    gain, guard = plan.get("useful_gain_ratio", .95), plan.get("stable_ratio_limit", 1.02)
    null_aggregate, deviation = geometric(nulls), max(abs(value - 1) for value in nulls)
    calibrated = all(1 - null_band <= value <= 1 + null_band for value in [*nulls, null_aggregate])
    noise = max(floor, deviation)
    candidates = []
    for arm in CANDIDATES:
        block_ratios = {block: geometric(ratios[block, regime, batch, arm] for regime, batch in itertools.product(regimes, batches)) for block in blocks}
        cell_ratios = [ratios[block, regime, batch, arm] for block, regime, batch in itertools.product(blocks, regimes, batches)]
        passing = advance(block_ratios.values(), cell_ratios, calibrated, noise, tax, gain, guard)
        candidates.append({"arm": arm, "block_ratios": block_ratios, "advances": passing,
                           "worst_block_with_reserves": max(block_ratios.values()) + noise + tax,
                           "worst_cell_with_reserves": max(cell_ratios) + noise + tax})
    passing = [candidate for candidate in candidates if candidate["advances"]]
    chosen = min(passing, key=lambda candidate: (max(candidate["block_ratios"].values()), candidate["arm"] != "one_copy_home", candidate["arm"]))["arm"] if passing else None
    for field, expected in (("status", "complete"), ("expected_records", 80), ("observed_records", 80),
                            ("claim_eligible", False), ("calibration", "pass" if calibrated else "stop"),
                            ("null_aggregate_ratio", null_aggregate), ("max_null_cell_deviation", deviation),
                            ("noise_reserve", noise), ("selector_tax_reserve", tax), ("selected_candidate", chosen)):
        same(summary.get(field), expected, f"summary {field}")
    for field in ("missing", "invalid_groups", "allocation_errors"):
        same(summary.get(field), [], f"summary {field}")
    reported_cells = {tuple(row[field] for field in ("block_id", "regime", "batch_tokens", "arm")): row for row in summary.get("cells", [])}
    check(len(summary.get("cells", [])) == len(ratios) and set(reported_cells) == set(ratios), "summary cell keys")
    maximum_difference = 0.
    for key, value in ratios.items():
        actual = reported_cells.get(key, {}).get("ratio")
        same(actual, value, f"summary cell {key}")
        if isinstance(actual, (int, float)):
            maximum_difference = max(maximum_difference, abs(actual - value))
    reported_candidates = {row["arm"]: row for row in summary.get("candidates", [])}
    check(len(summary.get("candidates", [])) == 2 and set(reported_candidates) == set(CANDIDATES), "summary candidate keys")
    for candidate in candidates:
        arm, row = candidate["arm"], reported_candidates.get(candidate["arm"], {})
        same(row.get("advances_to_online_implementation"), candidate["advances"], f"summary {arm} advance")
        same(row.get("screen"), "pass" if candidate["advances"] else "stop", f"summary {arm} screen")
        reported_blocks = {b["block_id"]: b["ratio"] for b in row.get("candidate_over_native_by_block", [])}
        check(len(row.get("candidate_over_native_by_block", [])) == len(blocks)
              and set(reported_blocks) == set(blocks), f"summary {arm} block keys")
        for block, value in candidate["block_ratios"].items():
            same(reported_blocks.get(block), value, f"summary {arm}/{block} aggregate")
        assignment = {tuple(c[field] for field in ("block_id", "regime", "batch_tokens")): c["ratio"] for c in row.get("candidate_over_pre_native_cells", [])}
        check(len(row.get("candidate_over_pre_native_cells", [])) == 8
              and set(assignment) == set(itertools.product(blocks, regimes, batches)), f"summary {arm} assignment keys")
        for state in itertools.product(blocks, regimes, batches):
            same(assignment.get(state), ratios[(*state, arm)] / ratios[(*state, "pre_native")], f"summary {arm}/{state} assignment contrast")
    return {"pass": not errors, "errors": errors, "observed_rows": len(records), "expected_rows": 80,
            "matched_five_arm_repetitions": 16, "source_blocks": 2, "allocation_ids": sorted(allocation_ids), "selection_recomputed": True,
            "calibration": "pass" if calibrated else "stop", "null_aggregate_ratio": null_aggregate,
            "max_null_cell_deviation": deviation, "noise_reserve": noise, "selector_tax_reserve": tax,
            "candidates": candidates, "selected_candidate": chosen,
            "maximum_cell_ratio_difference": maximum_difference,
            "cells": [{"block_id": key[0], "regime": key[1], "batch_tokens": key[2], "arm": key[3], "ratio": value} for key, value in ratios.items()],
            "limits": ["No pilot confidence interval or statistical equivalence claim.",
                       "Source/flag/map checks do not replace direct NPZ counter reconstruction or output numerical validation.",
                       "Candidate/native may reflect adapter savings; candidate/pre_native is the assignment contrast.",
                       "Precomputed selection is excluded; a pass is not an online-policy performance result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("manifest", "records", "summary", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--allocation-id", help="Require the clean authorized allocation; reject invalidated attempts")
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == ".gz" else open
    with opener(args.records, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    result = verify(json.loads(args.manifest.read_text()), records, json.loads(args.summary.read_text()),
                    expected_allocation_id=args.allocation_id)
    result["files"] = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in (("manifest", args.manifest), ("records", args.records), ("summary", args.summary), ("verifier", Path(__file__)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pass", "observed_rows", "selection_recomputed", "calibration", "selected_candidate") if key in result}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
