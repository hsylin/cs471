#!/usr/bin/env python3
"""Independent final resident-dispatch statistics audit, without production imports.

Recomputes paired log ratios and the frozen joint eight-source-block bootstrap.
Raw NPZ routing and actual numerical outputs need the separate correctness audit.
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

import numpy as np

ARMS = ("native", "online")
REGIMES = ("stable", "transient", "persistent", "switching")
IDENTITY = ("block_id", "regime", "batch_tokens", "repetition", "arm")
PAIRED = ("capture_sha256", "replay_sha256", "source_map_sha256", "source_weights_sha256",
          "logical_routing_sha256", "decision_trajectory_sha256", "logical_trajectory_sha256",
          "resolved_kernel_config", "source_prompt_ids", "timing_scope")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def decision(interval, limit):
    return "pass" if interval[1] <= limit else "fail" if interval[0] > limit else "inconclusive"


def draw_blocks(samples=20000, seed=10471):
    # random.choices uses floor(random()*n). Implement the draw explicitly;
    # vectorized NumPy estimation is independent of the production reducer.
    generator = random.Random(seed)
    return np.fromiter((int(generator.random() * 8) for _ in range(samples * 8)),
                       dtype=np.int64, count=samples * 8).reshape(samples, 8)


def estimate(block_log_ratios, draws, family=1):
    values = np.asarray(block_log_ratios, dtype=np.float64)
    if values.shape != (8,):
        raise ValueError("Exactly eight workload blocks are the resampling units")
    bootstrap = np.exp(values[draws].mean(axis=1))
    return {"source_blocks": 8, "ratio": float(np.exp(values.mean())),
            "ci95": np.quantile(bootstrap, [.025, .975], method="linear").tolist(),
            "adjusted_ci": np.quantile(bootstrap, [.025 / family, 1 - .025 / family], method="linear").tolist(),
            "family_size": family}


def verify(manifest, records, summary, *, allocation_by_repetition=None):
    errors, maximum_difference = [], 0.

    def check(condition, label):
        if not condition:
            errors.append(label)

    def compare(actual, expected, label):
        nonlocal maximum_difference
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            numeric = isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
            check(numeric and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), label)
            if numeric:
                maximum_difference = max(maximum_difference, abs(actual - expected))
        elif isinstance(expected, list):
            check(isinstance(actual, list) and len(actual) == len(expected), label + ": length")
            if isinstance(actual, list) and len(actual) == len(expected):
                for index, (a, e) in enumerate(zip(actual, expected)):
                    compare(a, e, f"{label}[{index}]")
        else:
            check(actual == expected, label)

    config_hash = digest({key: value for key, value in manifest.items() if key != "freeze"})
    check(config_hash == manifest.get("freeze", {}).get("sha256"), "manifest freeze differs")
    matrix, plan = manifest["matrix"], manifest["analysis"]
    blocks, regimes, batches = matrix["block_ids"], matrix["regimes"], matrix["batch_tokens"]
    check(manifest.get("phase") == manifest.get("split") == "final", "not final data")
    check(len(blocks) == len(set(blocks)) == 8 and len(regimes) == 4 and set(regimes) == set(REGIMES)
          and len(batches) == 2 and set(batches) == {128, 1024}
          and matrix["repetitions"] == 2 and tuple(matrix["arms"]) == ARMS, "wrong final matrix")
    expected_plan = dict(phase="final", expected_blocks=blocks, expected_regimes=regimes,
                         expected_batches=batches, expected_repetitions=[0, 1],
                         native_arm="native", treatment_arm="online", bootstrap_samples=20000,
                         bootstrap_seed=10471, useful_gain_ratio=.95, stable_ratio_limit=1.02)
    check(plan == expected_plan, "analysis plan differs from frozen 5%/2% contract")
    for field, expected in (("raw_steps", 544), ("warmup_steps", 32), ("measured_steps", 512), ("window_steps", 16)):
        compare(manifest["trace"].get(field), expected, "manifest trace " + field)
    expected_keys = set(itertools.product(blocks, regimes, batches, (0, 1), ARMS))
    index = {}
    for row in records:
        key = tuple(row.get(field) for field in IDENTITY)
        check(key in expected_keys, f"unexpected record {key}")
        check(key not in index, f"duplicate record {key}")
        index[key] = row
    check(len(records) == len(index) == 256 and set(index) == expected_keys, "missing/extra final records")
    if errors:
        return {"pass": False, "errors": errors, "observed_rows": len(records), "statistics_recomputed": False}
    allocations = {rep: {str(row.get("allocation_id")) for key, row in index.items() if key[3] == rep} for rep in (0, 1)}
    check(all(len(values) == 1 and next(iter(values)) not in ("None", "", "unallocated") for values in allocations.values()),
          "each repetition must occupy one identified allocation")
    check(allocations[0].isdisjoint(allocations[1]), "repetitions share an allocation")
    if allocation_by_repetition is not None:
        check(set(map(int, allocation_by_repetition)) == {0, 1}, "authorized allocation map missing repetition")
        for rep, allocation in allocation_by_repetition.items():
            check(allocations[int(rep)] == {str(allocation)}, f"repetition {rep}: wrong authorized allocation")
    tape_rows = manifest["tapes"]
    tapes = {(row["block_id"], row["regime"], row["batch_tokens"]): row for row in tape_rows}
    check(len(tape_rows) == len(tapes) == 64 and set(tapes) == set(itertools.product(blocks, regimes, batches)), "incomplete/duplicate tape inventory")
    documents = {row["block_id"]: row["source_document_ids"] for row in manifest["blocks"]}
    all_documents = [doc for source_ids in documents.values() for doc in source_ids]
    check(set(documents) == set(blocks) and len(all_documents) == len(set(all_documents))
          and not set(all_documents).intersection(manifest.get("excluded_source_document_ids", [])), "source documents overlap or lack coverage")
    fixed_map = list(range(128)) + list(range(32))
    for key, row in index.items():
        label = "/".join(map(str, key))
        compare(row.get("config_hash"), config_hash, label + ": config")
        compare(row.get("resident_dispatch_sha256"), manifest["environment"]["resident_dispatch_sha256"], label + ": execution source")
        compare(row.get("resident_dispatch_file_sha256"), manifest["execution_source_sha256"], label + ": execution source files")
        for field, value in (("phase", "final"), ("split", "final"), ("status", "complete"), ("policy", "SKIP"),
                             ("steps_completed", 512), ("migration_bytes", 0), ("migration_sent_messages", 0),
                             ("terminal_proposals_suppressed", 1), ("measurement_setup_revision", 2),
                             ("full_policy_conditioning_steps", 64), ("full_policy_conditioning_calls", 1)):
            compare(row.get(field), value, label + ": " + field)
        for field in ("correctness_pass", "evidence_eligible", "assignment_selection_cost_included", "source_restore_verified",
                      "expected_arm_weights_verified", "commit_completed", "logical_routing_preserved", "rank_assignment_agreement",
                      "timing_events_initialized_before_timer", "gc_enabled_during_measurement"):
            check(row.get(field) is True, label + ": " + field)
        compare(row.get("online_algorithm"), manifest["online_algorithm"] if key[-1] == "online" else "native_knuth_hash", label + ": algorithm")
        compare(row.get("source_map"), fixed_map, label + ": source map")
        compare(row.get("final_map"), fixed_map, label + ": final map")
        compare(row.get("resolved_kernel_config"), manifest["default_configs"][str(key[2])], label + ": kernel")
        compare(row.get("source_prompt_ids"), documents.get(key[0]), label + ": documents")
        compare(row.get("timing_scope"), manifest["execution"]["timing_scope"], label + ": timer")
        tape = tapes.get(key[:3], {})
        for field in ("capture_sha256", "replay_sha256"):
            compare(row.get(field), tape.get(field), label + ": " + field)
        elapsed, rank_times = row.get("elapsed_seconds"), row.get("per_rank_elapsed_seconds", [])
        valid_elapsed = type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
        valid_ranks = len(rank_times) == 2 and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in rank_times)
        check(valid_elapsed and valid_ranks, label + ": finite positive completed time")
        if valid_ranks:
            compare(elapsed, max(rank_times), label + ": slowest-rank time")
        log = row.get("decision_log", [])
        check([d.get("step") for d in log] == list(range(63, 511, 64)), label + ": decision cadence")
        check(all(d.get("allowed") is False for d in log), label + ": migration admitted")
        try:
            trajectory = digest([{field: d[field] for field in ("step", "current_counts", "current_map", "proposed_map", "committed_map")} for d in log])
            compare(row.get("decision_trajectory_sha256"), trajectory, label + ": logical decision hash")
            compare(row.get("logical_trajectory_sha256"), trajectory, label + ": logical trajectory hash")
        except (KeyError, TypeError, ValueError):
            errors.append(label + ": malformed decision trajectory")
    for state in itertools.product(blocks, regimes, batches):
        rows = [index[(*state, rep, arm)] for rep, arm in itertools.product((0, 1), ARMS)]
        for field in PAIRED:
            check(all(field in row for row in rows) and len({json.dumps(row.get(field), sort_keys=True) for row in rows}) == 1,
                  f"{state}: paired {field} differs")
    # Verify each allocation's sequence independently, so concatenation order is irrelevant.
    cells = list(itertools.product(blocks, regimes, batches))
    random.Random(manifest["execution"]["order_seed"]).shuffle(cells)
    for rep in (0, 1):
        expected_order = []
        for position, state in enumerate(cells):
            order = list(ARMS if position % 2 == 0 else ARMS[::-1])
            if rep == 1:
                order.reverse()
            for arm_position, arm in enumerate(order):
                key = (*state, rep, arm)
                expected_order.append(key)
                compare(index[key].get("arm_order"), order, f"{key}: arm order")
                compare(index[key].get("arm_position"), arm_position, f"{key}: arm position")
        observed_order = [tuple(row[field] for field in IDENTITY) for row in records if row["repetition"] == rep]
        check(observed_order == expected_order, f"repetition {rep}: raw execution sequence")
    if errors:
        return {"pass": False, "errors": errors, "observed_rows": len(records), "statistics_recomputed": False}
    sorted_blocks = sorted(blocks)
    logs = np.empty((8, 4, 2, 2), dtype=np.float64)
    pairs = {}
    for bi, block in enumerate(sorted_blocks):
        for ri, regime in enumerate(regimes):
            for ti, batch in enumerate(batches):
                for rep in (0, 1):
                    state = (block, regime, batch, rep)
                    native, online = index[(*state, "native")], index[(*state, "online")]
                    value = math.log(online["elapsed_seconds"] / native["elapsed_seconds"])
                    logs[bi, ri, ti, rep] = value
                    pairs[state] = {"native_seconds": native["elapsed_seconds"], "treatment_seconds": online["elapsed_seconds"],
                                    "log_ratio": value, "allocation_id": str(native["allocation_id"])}
    by_cell = logs.mean(axis=3)
    draws = draw_blocks(plan["bootstrap_samples"], plan["bootstrap_seed"])
    estimates = {}
    for ri, regime in enumerate(regimes):
        for ti, batch in enumerate(batches):
            limit, family = (1.02, 2) if regime == "stable" else (.95, 6)
            result = estimate(by_cell[:, ri, ti], draws, family)
            result.update(limit=limit, criterion=decision(result["adjusted_ci"], limit))
            estimates[regime, batch] = result
    aggregates = {}
    for group in ("changing", "stable"):
        selected = [i for i, regime in enumerate(regimes) if (regime == "stable") == (group == "stable")]
        limit = 1.02 if group == "stable" else .95
        result = estimate(by_cell[:, selected, :].mean(axis=(1, 2)), draws)
        result.update(limit=limit, criterion=decision(result["ci95"], limit))
        aggregates[group] = result
    guard_statuses = [estimates["stable", batch]["criterion"] for batch in batches] + [aggregates["stable"]["criterion"]]
    guard = "fail" if "fail" in guard_statuses else "pass" if set(guard_statuses) == {"pass"} else "inconclusive"
    useful = aggregates["changing"]["criterion"]
    acceptance = "fail" if "fail" in (useful, guard) else "pass" if useful == guard == "pass" else "inconclusive"
    for field, expected in (("phase", "final"), ("status", "complete"), ("expected_records", 256), ("observed_records", 256),
                            ("eligible_matched_pairs", 128), ("claim_eligible", True), ("native_arm", "native"), ("treatment_arm", "online"),
                            ("primary_useful_gain", useful), ("stable_guard", guard), ("acceptance", acceptance)):
        compare(summary.get(field), expected, "summary " + field)
    for field in ("missing", "invalid_groups", "allocation_errors"):
        compare(summary.get(field), [], "summary " + field)
    compare(summary.get("allocations_by_repetition"), {str(rep): sorted(ids) for rep, ids in allocations.items()}, "summary allocations")
    reported_pairs = {tuple(row[field] for field in IDENTITY[:-1]): row for row in summary.get("pairs", [])}
    check(len(summary.get("pairs", [])) == 128 and set(reported_pairs) == set(pairs), "summary pair keys")
    for key, expected in pairs.items():
        for field, value in expected.items():
            compare(reported_pairs.get(key, {}).get(field), value, f"summary pair {key}/{field}")
    reported_cells = {(row["regime"], row["batch_tokens"]): row for row in summary.get("cells", [])}
    check(len(summary.get("cells", [])) == 8 and set(reported_cells) == set(estimates), "summary cell keys")
    for key, expected in estimates.items():
        for field, value in expected.items():
            compare(reported_cells.get(key, {}).get(field), value, f"summary cell {key}/{field}")
    reported_aggregates = {row["group"]: row for row in summary.get("aggregates", [])}
    check(len(summary.get("aggregates", [])) == 2 and set(reported_aggregates) == set(aggregates), "summary aggregate keys")
    for key, expected in aggregates.items():
        for field, value in expected.items():
            compare(reported_aggregates.get(key, {}).get(field), value, f"summary aggregate {key}/{field}")
    by_allocation = []
    for rep in (0, 1):
        allocation = next(iter(allocations[rep]))
        for group in ("changing", "stable"):
            selected = [i for i, regime in enumerate(regimes) if (regime == "stable") == (group == "stable")]
            by_allocation.append({"allocation_id": allocation, "group": group,
                                  "pairs": 8 * len(selected) * 2,
                                  "ratio": float(np.exp(logs[:, selected, :, rep].mean()))})
    reported_allocations = {(row["allocation_id"], row["group"]): row for row in summary.get("by_allocation", [])}
    check(len(summary.get("by_allocation", [])) == 4 and set(reported_allocations) == {(row["allocation_id"], row["group"]) for row in by_allocation}, "summary allocation keys")
    for expected in by_allocation:
        key = (expected["allocation_id"], expected["group"])
        for field in ("pairs", "ratio"):
            compare(reported_allocations.get(key, {}).get(field), expected[field], f"summary allocation {key}/{field}")
    return {"pass": not errors, "errors": errors, "statistics_recomputed": True,
            "observed_rows": len(records), "matched_pairs": 128, "source_blocks": 8,
            "allocations_by_repetition": {str(rep): sorted(ids) for rep, ids in allocations.items()},
            "primary_useful_gain": useful, "stable_guard": guard, "acceptance": acceptance,
            "maximum_numeric_difference": maximum_difference,
            "cells": [{"regime": key[0], "batch_tokens": key[1], **value} for key, value in estimates.items()],
            "aggregates": [{"group": key, **value} for key, value in aggregates.items()],
            "by_allocation": by_allocation,
            "limits": ["Intervals resample eight chosen source blocks, conditional on two allocation windows.",
                       "Technical repetitions, ranks and steps are not independent workload units.",
                       "Record checks do not replace actual tape/physical-counter reconstruction or numerical-output validation.",
                       "An independent same-team implementation is not external reproduction."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "records", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allocation-map", type=Path, help='Optional JSON {"0":"job0","1":"job1"} of authorized allocations')
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == ".gz" else open
    with opener(args.records, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    result = verify(json.loads(args.manifest.read_text()), records, json.loads(args.summary.read_text()),
                    allocation_by_repetition=json.loads(args.allocation_map.read_text()) if args.allocation_map else None)
    result["files"] = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in (("manifest", args.manifest), ("records", args.records), ("summary", args.summary), ("verifier", Path(__file__)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pass", "observed_rows", "statistics_recomputed", "primary_useful_gain", "stable_guard", "acceptance") if key in result}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
