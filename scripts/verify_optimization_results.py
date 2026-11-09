"""Independent stdlib audit of complete kernel-optimization raw results.

No optimization/, followup/, or skewbench/ helpers are imported. This rebuilds
paired log ratios, source-block bootstrap intervals and frozen criteria, and
checks recorded identities/trajectories. It is a separate implementation by the
same team, not external reproduction or verification of GPU execution itself.
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
import struct

FIELDS = ("block_id", "regime", "batch_tokens", "repetition")
ARMS = (("DEFAULT", "B3"), ("DEFAULT", "SKIP"), ("TUNED", "B3"), ("TUNED", "SKIP"))
CONTRASTS = {
    "kernel_B3": (("TUNED", "B3"), ("DEFAULT", "B3")),
    "kernel_SKIP": (("TUNED", "SKIP"), ("DEFAULT", "SKIP")),
    "policy_DEFAULT": (("DEFAULT", "B3"), ("DEFAULT", "SKIP")),
    "policy_TUNED": (("TUNED", "B3"), ("TUNED", "SKIP")),
    "combined_vs_default_SKIP": (("TUNED", "B3"), ("DEFAULT", "SKIP")),
}
IDENTITIES = ("capture_sha256", "replay_sha256", "source_map_sha256", "source_weights_sha256")
TRAJECTORY_FIELDS = ("step", "allowed", "reason", "current_counts", "current_map", "proposed_map",
                     "committed_map", "transfer_sent_bytes", "physical_slot_assignments_window")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def frozen_hash(manifest):
    digest = canonical({k: v for k, v in manifest.items() if k != "freeze"})
    require(manifest.get("freeze", {}).get("sha256") == digest, "Manifest freeze mismatch")
    return digest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(values):
    return math.fsum(values) / len(values)


def percentile(values, probability):
    position = (len(values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    weight = position - left
    return (1 - weight) * values[left] + weight * values[right]


def criterion(interval, limit):
    if interval is None:
        return "inconclusive"
    if interval[1] <= limit:
        return "pass"
    return "fail" if interval[0] > limit else "inconclusive"


def index_rows(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"Duplicate {label}: {key}")
        result[key] = row
    return result


def map_hash(values):
    # Native records use contiguous little-endian int64 maps on the target host.
    def flatten(items):
        for value in items:
            if isinstance(value, list):
                yield from flatten(value)
            else:
                require(type(value) is int, "Noninteger map entry")
                yield value
    return hashlib.sha256(b"".join(struct.pack("<q", v) for v in flatten(values))).hexdigest()


def finite_positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def audit_provenance(manifest, selection_path=None, pilot_paths=()):
    if selection_path is None and not pilot_paths:
        return {"status": "not_requested", "scope": "Provide --selection and --pilots to check file-level pilot linkage."}
    require(selection_path is not None, "Pilot provenance audit requires --selection")
    selection = json.loads(Path(selection_path).read_text())
    require(selection.get("status") == "selected", "Incomplete pilot selection")
    require(file_hash(selection_path) == manifest["selection_sha256"], "Selection file SHA mismatch")
    require(selection["config_hash"] == manifest["tuning_manifest_sha256"], "Selection tuning digest mismatch")
    require(selection["selected_configs"] == manifest["selected_configs"], "Selected configs mismatch")
    hashes, phases = [], set()
    final_docs = {d for block in manifest["blocks"] for d in block["source_document_ids"]}
    for path in pilot_paths:
        prior = json.loads(Path(path).read_text())
        digest = frozen_hash(prior)
        hashes.append(digest); phases.add(prior["phase"])
        require(prior["split"] == "pilot", "Reference is not a pilot")
        if manifest["phase"] == "final":
            require(not final_docs.intersection(d for b in prior["blocks"] for d in b["source_document_ids"]),
                    "Pilot/final document overlap")
        if prior["phase"] == "tuning":
            require(digest == selection["config_hash"], "Referenced tuning differs from selection")
        if prior["phase"] == "validation":
            require(prior["selection_sha256"] == manifest["selection_sha256"]
                    and prior["tuning_manifest_sha256"] == selection["config_hash"]
                    and prior["selected_configs"] == selection["selected_configs"], "Validation selection differs")
    if manifest["phase"] == "final":
        require(phases == {"tuning", "validation"}, "Final provenance requires tuning and validation")
        require(sorted(hashes) == sorted(manifest["pilot_manifest_sha256"]), "Pilot reference inventory mismatch")
    return {"status": "verified", "selection_sha256": file_hash(selection_path), "pilot_manifest_sha256": hashes}


def verify(manifest, paths, report, *, tolerance=1e-12, selection_path=None, pilot_paths=()):
    digest = frozen_hash(manifest)
    require(digest == report["config_hash"], "Report config differs")
    require(manifest["optimization_schema_version"] == report["optimization_schema_version"] == 1, "Unknown schema")
    require(manifest["phase"] in ("validation", "final"), "Only whole-policy matrices are supported")
    require(report["phase"] == manifest["phase"] and report["split"] == manifest["split"], "Report phase/split differs")
    require(report["status"] == "complete" and not report["missing"] and not report["invalid_groups"],
            "Only complete valid matrices can be certified")
    m, plan = manifest["matrix"], manifest["analysis"]
    for field in ("block_ids", "regimes", "batch_tokens", "kernel_variants", "policies"):
        require(m[field] and len(m[field]) == len(set(m[field])), f"Invalid matrix {field}")
    require(set(m["kernel_variants"]) == {"DEFAULT", "TUNED"} and set(m["policies"]) == {"B3", "SKIP"}, "Expected four arms")
    require(type(m["repetitions"]) is int and m["repetitions"] > 0, "Invalid repetitions")
    require(set(m["regimes"]) == {"stable", "transient", "persistent", "switching"}, "Unexpected regime family")
    require(plan["cell_families"] == {"changing": 3*len(m["batch_tokens"]), "stable": len(m["batch_tokens"])}, "Incorrect multiplicity family")
    documents = {b["block_id"]: b["source_document_ids"] for b in manifest["blocks"]}
    flat_docs = [d for ids in documents.values() for d in ids]
    require(len(documents) == len(manifest["blocks"]) and set(documents) == set(m["block_ids"]), "Block inventory differs")
    require(len(flat_docs) == len(set(flat_docs)) and not set(flat_docs).intersection(manifest.get("excluded_source_document_ids", [])), "Source document reuse")
    tapes = index_rows(manifest["tapes"], FIELDS[:3], "tape")
    require(set(tapes) == set(itertools.product(m["block_ids"], m["regimes"], m["batch_tokens"])), "Tape inventory differs")
    groups = set(itertools.product(m["block_ids"], m["regimes"], m["batch_tokens"], range(m["repetitions"])))
    expected = {(g, a) for g in groups for a in ARMS}
    records = {}
    differences = []

    def equal(actual, reported, label):
        require(type(reported) in (float, int) and math.isfinite(reported), f"Invalid number: {label}")
        difference = abs(actual - reported)
        require(difference <= tolerance, f"Mismatch {label}: reconstructed={actual!r}, reported={reported!r}")
        differences.append(difference)

    for path in map(Path, paths):
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line); location = f"{path}:{line_number}"
                group = tuple(row[f] for f in FIELDS); arm = (row["kernel_variant"], row["policy"])
                key = (group, arm)
                require(key in expected and key not in records, f"Unexpected/duplicate record: {location}")
                required = {"config_hash": digest, "optimization_schema_version": 1, "phase": manifest["phase"],
                            "split": manifest["split"], "measurement_kind": "gpu_optimization", "status": "complete",
                            "correctness_pass": True, "evidence_eligible": True, "kernel_config_rank_agreement": True,
                            "steps_completed": manifest["trace"]["measured_steps"],
                            "observation_warmup_steps": manifest["trace"]["warmup_steps"],
                            "raw_trace_steps": manifest["trace"]["raw_steps"],
                            "source_restore_verified": True, "expected_arm_weights_verified": True,
                            "commit_completed": True, "logical_routing_preserved": True,
                            **manifest.get("required_record_metadata", {})}
                for field, value in required.items():
                    require(row.get(field) == value, f"Metadata mismatch {field}: {location}")
                require(finite_positive(row["elapsed_seconds"]), f"Invalid elapsed time: {location}")
                rank_times = row["per_rank_elapsed_seconds"]
                require(len(rank_times) == row["world_size"] and all(finite_positive(v) for v in rank_times), "Invalid rank elapsed times")
                equal(max(rank_times), row["elapsed_seconds"], "elapsed/max-rank")
                selected = {"candidate_id": "DEFAULT", "config": None} if arm[0] == "DEFAULT" else manifest["selected_configs"][str(group[2])]
                config = selected["config"]
                require(row["kernel_config_id"] == selected["candidate_id"] and row["kernel_override_config"] == config
                        and row["kernel_config_sha256"] == canonical(config), "Kernel selection/config differs")
                resolved = config if config is not None else manifest["default_configs"][str(group[2])]
                require(row["resolved_kernel_config"] == resolved, "Resolved kernel differs from frozen setting")
                for map_kind in ("source", "final"):
                    require(map_hash(row[map_kind+"_map"]) == row[map_kind+"_map_sha256"], "Map content/hash differs")
                log = row["decision_log"]
                require(canonical([{f: d.get(f) for f in TRAJECTORY_FIELDS} for d in log]) == row["decision_trajectory_sha256"], "Trajectory content/hash differs")
                interval = manifest["policy"]["rebalance_interval"]
                require([d["step"] for d in log] == list(range(interval-1, row["steps_completed"]-1, interval)), "Policy decision schedule differs")
                require(row["accepted"] == sum(d["allowed"] for d in log), "Acceptance count differs")
                require(row["migration_bytes"] == sum(d["transfer_sent_bytes"] for d in log), "Payload total differs")
                current = row["source_map"]
                for decision in log:
                    require(decision["current_map"] == current, "Decision map chain differs")
                    current = decision["proposed_map"] if decision["allowed"] else current
                    require(decision["committed_map"] == current and decision.get("commit_completed") is True, "Committed map differs")
                require(current == row["final_map"], "Final map differs from decision chain")
                if arm[1] == "SKIP":
                    require(row["migration_bytes"] == 0 and row["accepted"] == 0 and row["source_map"] == row["final_map"], "SKIP changed placement")
                records[key] = row
    require(set(records) == expected, "Missing raw arm records")
    require(report["expected_records"] == len(records) and report["eligible_matched_groups"] == len(groups), "Report matrix counts differ")
    reported_pairs = index_rows(report["pairs"], FIELDS, "reported group")
    require(set(reported_pairs) == groups, "Reported paired group inventory differs")
    trajectory_by_state, identity_by_state, logs = {}, {}, {}
    allocations, hosts, processes = set(), set(), set()
    for group in sorted(groups):
        rows = {arm: records[group, arm] for arm in ARMS}
        reference = rows[ARMS[0]]
        identities = tuple(reference[f] for f in IDENTITIES)
        require(all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in identities), "Invalid source hashes")
        for arm, row in rows.items():
            require(tuple(row[f] for f in IDENTITIES) == identities, "Four-arm source identity differs")
            require(row["source_prompt_ids"] == documents[group[0]], "Source documents differ")
            for field in ("allocation_id", "hostname", "process_run_id"):
                require(row.get(field) and row[field] == reference.get(field), f"Unmatched execution block: {field}")
            for field in ("capture_sha256", "replay_sha256"):
                require(row[field] == tapes[group[:3]][field], "Frozen input hash differs")
            trajectory = (row["decision_trajectory_sha256"], row["final_map_sha256"], row["migration_bytes"])
            state = (*group[:3], arm[1])
            require(state not in trajectory_by_state or trajectory_by_state[state] == trajectory, "Kernel/repetition changed policy trajectory")
            trajectory_by_state[state] = trajectory
            equal(row["elapsed_seconds"], reported_pairs[group]["_".join(arm)+"_seconds"], "paired runtime")
        require(group[:3] not in identity_by_state or identity_by_state[group[:3]] == identities, "Repetition changed inputs/source")
        identity_by_state[group[:3]] = identities
        allocations.add(reference["allocation_id"]); hosts.add(reference["hostname"]); processes.add(reference["process_run_id"])
        for contrast, (numerator, denominator) in CONTRASTS.items():
            value = math.log(rows[numerator]["elapsed_seconds"]) - math.log(rows[denominator]["elapsed_seconds"])
            logs[group, contrast] = value
            equal(value, reported_pairs[group][contrast+"_log_ratio"], contrast)
        value = logs[group, "kernel_B3"] - logs[group, "kernel_SKIP"]
        logs[group, "interaction"] = value
        equal(value, reported_pairs[group]["interaction_log_ratio"], "interaction")

    blocks = sorted(m["block_ids"]); count = len(blocks)
    rng = random.Random(plan["bootstrap_seed"])
    # Independently reconstruct random.choices' equal-weight index stream.
    draws = [[math.floor(rng.random()*count) for _ in range(count)]
             for _ in range(plan["bootstrap_samples"])] if count >= plan["minimum_source_blocks"] else []

    def estimate(values, family, reported, label):
        ratio = math.exp(average(values)); equal(ratio, reported["ratio"], label+":ratio")
        require(reported["source_blocks"] == count and reported["family_size"] == family, "Block/family count differs")
        intervals = {}
        if draws:
            samples = sorted(math.exp(average([values[i] for i in draw])) for draw in draws)
            for field, tail in (("ci95", .025), ("adjusted_ci", .025/family)):
                actual = [percentile(samples, tail), percentile(samples, 1-tail)]
                require(isinstance(reported[field], list) and len(reported[field]) == 2, "Missing interval")
                for computed, observed in zip(actual, reported[field]): equal(computed, observed, label+":"+field)
                intervals[field] = actual
            require(reported["status"] == "estimated", "Estimate status differs")
        else:
            require(reported["ci95"] is None and reported["adjusted_ci"] is None
                    and reported["status"] == "insufficient_blocks", "Intervals with insufficient blocks")
            intervals = {"ci95": None, "adjusted_ci": None}
        return {"ratio": ratio, **intervals}

    contrasts = [*CONTRASTS, "interaction"]
    cells = index_rows(report["cells"], ("regime", "batch_tokens", "contrast"), "reported cell")
    expected_cells = set(itertools.product(m["regimes"], m["batch_tokens"], contrasts))
    require(set(cells) == expected_cells, "Cell inventory differs")
    cell_values, audited_cells = {}, []
    for regime, batch, contrast in sorted(expected_cells):
        values = [average([logs[(block, regime, batch, rep), contrast] for rep in range(m["repetitions"])]) for block in blocks]
        key = regime, batch, contrast; cell_values[key] = values; row = cells[key]
        inferential = contrast in ("kernel_B3", "kernel_SKIP")
        family = plan["cell_families"]["stable" if regime == "stable" else "changing"] if inferential else 1
        computed = estimate(values, family, row, str(key))
        limit = plan["stable_ratio_limit"] if regime == "stable" else plan["useful_gain_ratio"]
        expected_status = criterion(computed["adjusted_ci"], limit) if inferential else "descriptive"
        require(row["limit"] == limit and row["criterion"] == expected_status, "Cell criterion differs")
        require(row["confirmatory_eligible"] == (manifest["phase"] == "final" and contrast == "kernel_B3"), "Cell eligibility differs")
        audited_cells.append({"regime": regime, "batch_tokens": batch, "contrast": contrast, **computed,
                              "criterion": expected_status, "family_size": family,
                              "block_log_ratios": dict(zip(blocks, values))})
    aggregates = index_rows(report["aggregates"], ("group", "contrast"), "reported aggregate")
    require(set(aggregates) == set(itertools.product(("changing", "stable", "all"), contrasts)), "Aggregate inventory differs")
    audited_aggregates = []
    statuses = {}
    for group, contrast in sorted(aggregates):
        regimes = [r for r in m["regimes"] if group == "all" or (r == "stable") == (group == "stable")]
        values = [average([cell_values[r, b, contrast][i] for r in regimes for b in m["batch_tokens"]]) for i in range(count)]
        row = aggregates[group, contrast]
        computed = estimate(values, 1, row, f"{group}/{contrast}")
        limit = plan["stable_ratio_limit"] if group == "stable" else plan["useful_gain_ratio"]
        inferential = contrast in ("kernel_B3", "kernel_SKIP") and group != "all"
        status = criterion(computed["ci95"], limit) if inferential else "descriptive"
        require(row["criterion"] == status and row["limit"] == limit, "Aggregate criterion differs")
        require(row["confirmatory_eligible"] == (manifest["phase"] == "final" and contrast == "kernel_B3" and group != "all"), "Aggregate eligibility differs")
        statuses[group, contrast] = status
        audited_aggregates.append({"group": group, "contrast": contrast, **computed, "criterion": status,
                                   "block_log_ratios": dict(zip(blocks, values))})
    guards = [c["criterion"] for c in audited_cells if c["regime"] == "stable" and c["contrast"] == "kernel_B3"]
    guards.append(statuses["stable", "kernel_B3"])
    stable = "fail" if "fail" in guards else "pass" if all(s == "pass" for s in guards) else "inconclusive"
    primary = statuses["changing", "kernel_B3"]
    acceptance = "fail" if "fail" in (primary, stable) else "pass" if primary == stable == "pass" else "inconclusive"
    for field, value in (("primary_useful_gain", primary), ("stable_guard", stable), ("acceptance", acceptance),
                         ("claim_eligible", manifest["phase"] == "final")):
        require(report[field] == value, f"Report {field} differs")
    return {"status": "verified", "config_hash": digest, "records": len(records), "matched_four_arm_groups": len(groups),
            "source_blocks": count, "technical_repetitions": m["repetitions"], "verified_cells": len(cells),
            "verified_aggregates": len(aggregates), "bootstrap_samples": len(draws), "bootstrap_seed": plan["bootstrap_seed"],
            "maximum_absolute_numerical_difference": max(differences, default=0.), "numerical_tolerance": tolerance,
            "allocation_ids": sorted(allocations), "hostnames": sorted(hosts), "process_run_ids": sorted(processes),
            "primary_useful_gain": primary, "stable_guard": stable, "acceptance": acceptance,
            "cells": audited_cells, "aggregates": audited_aggregates,
            "provenance": audit_provenance(manifest, selection_path, pilot_paths),
            "scope": "Independent standard-library reimplementation by the same team; not external human reproduction. Checks raw-record consistency, not actual GPU execution or saved tensor bytes. Block intervals describe the fixed source corpus under resampling assumptions; technical repeats are not new blocks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--pilots", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an audit")
    result = verify(json.loads(args.manifest.read_text()), args.records, json.loads(args.report.read_text()),
                    selection_path=args.selection, pilot_paths=args.pilots)
    result["input_files_sha256"] = {str(p): file_hash(p) for p in [args.manifest, *args.records, args.report]}
    result["verifier_sha256"] = file_hash(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "records", "matched_four_arm_groups", "source_blocks",
                                           "maximum_absolute_numerical_difference", "primary_useful_gain", "stable_guard", "acceptance")}))


if __name__ == "__main__":
    main()
