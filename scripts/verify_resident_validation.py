#!/usr/bin/env python3
"""Independent descriptive audit of 32 fully charged resident development runs."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def geometric(values):
    values = list(values)
    return math.prod(values) ** (1 / len(values))


def verify(manifest, records, summary, allocation_id=None):
    errors, difference = [], 0.

    def check(condition, label):
        if not condition:
            errors.append(label)

    def equal(actual, expected, label):
        nonlocal difference
        if type(expected) in (int, float):
            numeric = type(actual) in (int, float) and math.isfinite(actual)
            check(numeric and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), label)
            if numeric:
                difference = max(difference, abs(actual - expected))
        else:
            check(actual == expected, label)

    config_hash = digest({key: value for key, value in manifest.items() if key != "freeze"})
    check(config_hash == manifest["freeze"]["sha256"], "manifest hash")
    check(manifest["phase"] == "validation" and manifest["split"] == "pilot", "wrong phase/split")
    matrix, plan = manifest["matrix"], manifest["analysis"]
    blocks, regimes, batches = matrix["block_ids"], matrix["regimes"], matrix["batch_tokens"]
    expected_plan = dict(phase="paired_descriptive", expected_blocks=blocks, expected_regimes=regimes,
                         expected_batches=batches, expected_repetitions=[0, 1], native_arm="native", treatment_arm="online")
    check(plan == expected_plan, "descriptive plan differs")
    check(len(set(blocks)) == len(blocks) == 2 and len(set(regimes)) == len(regimes) == 2
          and set(batches) == {128, 1024} and matrix["repetitions"] == 2 and matrix["arms"] == ["native", "online"], "wrong matrix")
    fields = ("block_id", "regime", "batch_tokens", "repetition", "arm")
    index = {}
    expected = set(itertools.product(blocks, regimes, batches, (0, 1), ("native", "online")))
    for row in records:
        key = tuple(row.get(field) for field in fields)
        check(key not in index and key in expected, f"duplicate/unexpected {key}")
        index[key] = row
    check(len(records) == 32 and set(index) == expected, "missing/extra records")
    allocations = {str(row.get("allocation_id")) for row in records}
    check(len(allocations) == 1 and not allocations.intersection({"None", "", "unallocated"}), "mixed or absent allocation")
    if allocation_id is not None:
        check(allocations == {str(allocation_id)}, "wrong authorized allocation")
    if errors:
        return {"pass": False, "errors": errors, "statistics_recomputed": False}
    for key, row in index.items():
        label = str(key)
        equal(row.get("config_hash"), config_hash, label + ": config hash")
        equal(row.get("resident_dispatch_sha256"), manifest["environment"]["resident_dispatch_sha256"], label + ": source fingerprint")
        equal(row.get("resident_dispatch_file_sha256"), manifest["execution_source_sha256"], label + ": source file hashes")
        for field in ("correctness_pass", "evidence_eligible", "assignment_selection_cost_included", "source_restore_verified",
                      "expected_arm_weights_verified", "commit_completed", "logical_routing_preserved", "rank_assignment_agreement",
                      "timing_events_initialized_before_timer", "gc_enabled_during_measurement"):
            check(row.get(field) is True, label + ": " + field)
        for field, value in (("phase", "validation"), ("split", "pilot"), ("status", "complete"), ("steps_completed", 256),
                             ("policy", "SKIP"), ("migration_bytes", 0), ("migration_sent_messages", 0),
                             ("measurement_setup_revision", 2), ("full_policy_conditioning_steps", 64),
                             ("full_policy_conditioning_calls", 1), ("terminal_proposals_suppressed", 1)):
            equal(row.get(field), value, label + ": " + field)
        equal(row.get("final_map"), row.get("source_map"), label + ": fixed map")
        times = row.get("per_rank_elapsed_seconds", [])
        elapsed = row.get("elapsed_seconds")
        valid = len(times) == 2 and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in times)
        check(valid, label + ": rank durations")
        if valid:
            equal(elapsed, max(times), label + ": slowest-rank time")
        equal([d["step"] for d in row.get("decision_log", [])], [63, 127, 191], label + ": decision cadence")
    for state in itertools.product(blocks, regimes, batches):
        matched = [index[(*state, rep, arm)] for rep, arm in itertools.product((0, 1), ("native", "online"))]
        for field in ("capture_sha256", "replay_sha256", "source_map_sha256", "source_weights_sha256",
                      "logical_routing_sha256", "decision_trajectory_sha256", "resolved_kernel_config", "source_prompt_ids", "timing_scope"):
            check(all(field in row for row in matched) and len({json.dumps(row.get(field), sort_keys=True) for row in matched}) == 1,
                  f"{state}: paired {field}")
    cells_order = list(itertools.product(blocks, regimes, batches))
    random.Random(manifest["execution"]["order_seed"]).shuffle(cells_order)
    expected_order = []
    for position, state in enumerate(cells_order):
        first = ["native", "online"] if position % 2 == 0 else ["online", "native"]
        for rep in (0, 1):
            order = first if rep == 0 else list(reversed(first))
            for arm in order:
                key = (*state, rep, arm)
                expected_order.append(key)
                equal(index[key].get("arm_order"), order, f"{key}: order")
                equal(index[key].get("arm_position"), order.index(arm), f"{key}: position")
    equal([tuple(row[field] for field in fields) for row in records], expected_order, "execution sequence")
    if errors:
        return {"pass": False, "errors": errors, "statistics_recomputed": False}
    ratios = {state: math.sqrt(math.prod(index[(*state, rep, "online")]["elapsed_seconds"] /
                                        index[(*state, rep, "native")]["elapsed_seconds"] for rep in (0, 1)))
              for state in itertools.product(blocks, regimes, batches)}
    overall = geometric(ratios.values())
    by_block = {block: geometric(value for state, value in ratios.items() if state[0] == block) for block in blocks}
    by_batch = {batch: geometric(value for state, value in ratios.items() if state[2] == batch) for batch in batches}
    for field, value in (("status", "complete"), ("phase", "paired_descriptive"), ("expected_records", 32), ("observed_records", 32),
                         ("eligible_matched_pairs", 16), ("claim_eligible", False), ("overall_ratio", overall),
                         ("performance_gate", "not_applied_by_predeclared_user_directed_fresh_test_amendment")):
        equal(summary.get(field), value, "summary " + field)
    check("acceptance" not in summary, "descriptive result has acceptance")
    for field in ("missing", "invalid_groups", "allocation_errors"):
        equal(summary.get(field), [], "summary " + field)
    reported = {(r["block_id"], r["regime"], r["batch_tokens"]): r for r in summary.get("cells", [])}
    check(len(summary.get("cells", [])) == 8 and set(reported) == set(ratios), "summary cell coverage")
    for state, value in ratios.items():
        equal(reported.get(state, {}).get("ratio"), value, f"summary {state} ratio")
        equal(reported.get(state, {}).get("paired_repetitions"), 2, f"summary {state} repetitions")
        check("ci95" not in reported.get(state, {}), f"summary {state}: no development CI")
    for field, key, expected in (("by_block", "block_id", by_block), ("by_batch", "batch_tokens", by_batch)):
        reported = {r[key]: r for r in summary.get(field, [])}
        check(len(summary.get(field, [])) == 2 and set(reported) == set(expected), "summary " + field + " coverage")
        for item, value in expected.items():
            equal(reported.get(item, {}).get("ratio"), value, f"summary {field}/{item}")
            equal(reported.get(item, {}).get("complete_cells"), 4, f"summary {field}/{item} cells")
    return {"pass": not errors, "errors": errors, "statistics_recomputed": True,
            "rows": 32, "matched_pairs": 16, "source_blocks": 2, "allocation_ids": sorted(allocations),
            "overall_ratio": overall, "by_block": by_block, "by_batch": {str(k): v for k, v in by_batch.items()},
            "cells": [{"block_id": key[0], "regime": key[1], "batch_tokens": key[2], "ratio": value} for key, value in ratios.items()],
            "maximum_numeric_difference": difference,
            "limits": "Descriptive two-block development result; no confidence interval, selection or confirmatory acceptance. Source/flag checks do not replace raw route reconstruction or numerical output validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "records", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allocation-id")
    args = parser.parse_args()
    opener = gzip.open if args.records.suffix == ".gz" else open
    with opener(args.records, "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    result = verify(json.loads(args.manifest.read_text()), records, json.loads(args.summary.read_text()), args.allocation_id)
    result["files"] = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in (("manifest", args.manifest), ("records", args.records), ("summary", args.summary), ("verifier", Path(__file__)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pass", "statistics_recomputed", "rows", "overall_ratio", "maximum_numeric_difference") if key in result}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
