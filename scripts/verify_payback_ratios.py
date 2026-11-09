"""Independent standard-library reconstruction of the frozen payback statistics.

This script never imports followup/, skewbench/, or their analysis helpers.
It checks complete raw matrices, matched identities, paired geometric means,
and the declared block bootstrap against an existing summary.json. It is an
independent implementation, not external human reproduction or a new experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ARM_NAMES = ("KEEP", "PREINSTALLED", "CHARGED")
PAIR_FIELDS = ("block_id", "regime", "batch_tokens", "horizon_steps", "candidate_id", "repetition")
CONTRAST_ARMS = {"headroom": ("PREINSTALLED", "KEEP"),
                 "payback": ("CHARGED", "KEEP"),
                 "transition": ("CHARGED", "PREINSTALLED")}
IDENTITIES = ("source_map_sha256", "source_weights_sha256", "candidate_map_sha256",
              "capture_sha256", "replay_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def average(values):
    return math.fsum(values) / len(values)


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    if left == right:
        return sorted_values[left]
    fraction = position - left
    return (1 - fraction) * sorted_values[left] + fraction * sorted_values[right]


def limit_status(interval, limit):
    if interval is None:
        return "inconclusive"
    if interval[1] <= limit:
        return "pass"
    if interval[0] > limit:
        return "fail"
    return "inconclusive"


def unique_index(rows, fields, description):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"Duplicate {description}: {key}")
        result[key] = row
    return result


def verify(manifest, raw_paths, report, tolerance=1e-12):
    digest = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "freeze"},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(digest == manifest["freeze"]["sha256"] == report["config_hash"], "Manifest/report SHA mismatch")
    require(manifest["payback_schema_version"] == 1 and report["payback_schema_version"] == 1, "Unsupported schema")
    require(report["split"] == manifest["split"], "Report split mismatch")
    require(report["status"] == "complete" and not report["missing"] and not report["invalid_pairs"],
            "Only complete, valid matrices can be verified")
    m, plan = manifest["matrix"], manifest["analysis"]
    require(set(m["arms"]) == set(ARM_NAMES) and len(m["arms"]) == 3, "Three unique arms required")
    for field in ("block_ids", "regimes", "batch_tokens", "horizons", "candidate_ids"):
        require(m[field] and len(set(m[field])) == len(m[field]), f"Invalid matrix {field}")
    require(isinstance(m["repetitions"], int) and m["repetitions"] > 0, "Invalid repetition count")
    require(manifest["split"] in ("pilot", "final"), "Unknown split")
    stage = report["stage"]
    require(stage in ("screening", "confirmation"), "Unknown report stage")
    documents = {}
    all_documents = set()
    for block in manifest["blocks"]:
        identity = block["block_id"]
        ids = block["source_document_ids"]
        require(identity not in documents and ids and len(ids) == len(set(ids)), "Invalid block provenance")
        require(not (all_documents & set(ids)), "Source documents overlap across blocks")
        documents[identity] = ids
        all_documents.update(ids)
    require(set(documents) == set(m["block_ids"]), "Block inventory mismatch")
    tapes = unique_index(manifest["tapes"], ("block_id", "regime", "batch_tokens"), "tape")
    require(set(tapes) == set(itertools.product(m["block_ids"], m["regimes"], m["batch_tokens"])), "Tape inventory mismatch")

    expected = set()
    for key in itertools.product(m["block_ids"], m["regimes"], m["batch_tokens"], m["horizons"],
                                 m["candidate_ids"], range(m["repetitions"])):
        block, regime, batch, horizon, candidate, repetition = key
        if manifest["split"] == "pilot" and repetition > 0:
            if stage == "screening" or regime == "stable" or horizon != plan["primary_horizon"]:
                continue
        expected.add(key)
    raw = {}
    ignored = 0
    for path in map(Path, raw_paths):
        reader = gzip.open if path.suffix == ".gz" else open
        with reader(path, "rt") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                location = f"{path}:{number}"
                require(row["config_hash"] == digest, f"Raw config mismatch: {location}")
                key = tuple(row[field] for field in PAIR_FIELDS)
                if key not in expected and manifest["split"] == "pilot" and stage == "screening" and row["repetition"] > 0:
                    ignored += 1
                    continue
                require(key in expected and row["arm"] in ARM_NAMES, f"Unexpected raw matrix key: {location}")
                arm_key = (*key, row["arm"])
                require(arm_key not in raw, f"Duplicate raw record: {location}")
                require(row.get("payback_schema_version") == 1 and row.get("split") == manifest["split"], f"Raw schema/split mismatch: {location}")
                require(row.get("status") == "complete" and row.get("measurement_kind") == "gpu_payback"
                        and row.get("correctness_pass") is True and row.get("evidence_eligible") is True,
                        f"Ineligible raw measurement: {location}")
                seconds = row["elapsed_seconds"]
                require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                        and math.isfinite(seconds) and seconds > 0, f"Invalid elapsed time: {location}")
                require(row["steps_completed"] == row["horizon_steps"], f"Incomplete horizon: {location}")
                for field, value in manifest.get("required_record_metadata", {}).items():
                    require(row.get(field) == value, f"Execution metadata {field} mismatch: {location}")
                raw[arm_key] = row
    require(len(raw) == 3 * len(expected), "Raw matrix is incomplete")
    require(report["expected_arm_records"] == len(raw) and report["eligible_pairs"] == len(expected), "Report matrix counts mismatch")
    require(len(report.get("ignored", [])) == ignored, "Report ignored-row count mismatch")

    differences = []
    def equal_number(actual, reported, label):
        require(isinstance(reported, (float, int)) and not isinstance(reported, bool) and math.isfinite(reported), f"Invalid reported number: {label}")
        difference = abs(actual - reported)
        require(difference <= tolerance, f"Numerical mismatch {label}: reconstructed={actual!r}, reported={reported!r}")
        differences.append(difference)

    reported_pairs = unique_index(report["pairs"], PAIR_FIELDS, "reported pair")
    require(set(reported_pairs) == expected, "Reported pair inventory differs")
    logs = {}
    payback_seconds = {}
    state_identities, prefix_identities = {}, {}
    for key in sorted(expected):
        arms = {arm: raw[(*key, arm)] for arm in ARM_NAMES}
        reference = arms["KEEP"]
        require(reference["source_prompt_ids"] == documents[key[0]], "Source prompt IDs differ from declared documents")
        require(reference.get("state_id"), "State identity missing")
        identity = tuple(reference.get(field) for field in IDENTITIES)
        for field, value in zip(IDENTITIES, identity):
            require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"Invalid SHA for {field}")
        for arm, row in arms.items():
            require(tuple(row.get(field) for field in IDENTITIES) == identity, "Matched arm identity mismatch")
            require(row["state_id"] == reference["state_id"] and row["source_prompt_ids"] == reference["source_prompt_ids"], "Matched state mismatch")
            for field in ("allocation_id", "hostname", "process_run_id"):
                require(row.get(field) and row.get(field) == reference.get(field), f"Matched run-block mismatch: {field}")
            for field in ("capture_sha256", "replay_sha256"):
                require(row[field] == tapes[key[:3]][field], f"Frozen tape hash mismatch: {field}")
            if arm != "CHARGED":
                require(row["migration_bytes"] == 0, "Uncharged arm records timed migration")
        state = (*key[:3], key[4])
        require(state not in state_identities or state_identities[state] == identity,
                "Source/candidate/replay differs across repeats or horizons")
        state_identities[state] = identity
        prefix = tuple(reference[field] for field in IDENTITIES if field != "candidate_map_sha256")
        require(key[:3] not in prefix_identities or prefix_identities[key[:3]] == prefix,
                "Candidate families do not share source prefix/replay")
        prefix_identities[key[:3]] = prefix
        pair = reported_pairs[key]
        times = {arm: row["elapsed_seconds"] for arm, row in arms.items()}
        for arm, seconds in times.items():
            equal_number(seconds, pair[arm.lower() + "_seconds"], f"pair {key} {arm}")
        for name, (top, bottom) in CONTRAST_ARMS.items():
            value = math.log(times[top]) - math.log(times[bottom])
            logs[(*key, name)] = value
            equal_number(value, pair[name + "_log_ratio"], f"pair {key} {name}")
        for name, value in (("headroom_seconds", times["KEEP"]-times["PREINSTALLED"]),
                            ("payback_seconds", times["KEEP"]-times["CHARGED"]),
                            ("transition_seconds", times["CHARGED"]-times["PREINSTALLED"])):
            equal_number(value, pair[name], f"pair {key} {name}")
        payback_seconds[key] = times["KEEP"]-times["CHARGED"]

    block_ids = sorted(m["block_ids"])
    count = len(block_ids)
    # Reconstruct random.choices' index stream explicitly, without calling the
    # estimator's resampling function or importing its implementation.
    indexes = []
    if count >= plan["minimum_independent_blocks"]:
        rng = random.Random(plan["bootstrap_seed"])
        indexes = [[math.floor(rng.random() * count) for _ in range(count)]
                   for _ in range(plan["bootstrap_samples"])]

    def check_estimate(values, family, reported, label):
        ratio = math.exp(average(values))
        equal_number(ratio, reported["ratio"], label + ":ratio")
        require(reported["independent_blocks"] == count and reported["family_size"] == family, label + ":sample/family size mismatch")
        if not indexes:
            require(reported["ci95"] is None and reported["family_adjusted_ci"] is None
                    and reported["status"] == "insufficient_blocks", label + ":interval with too few blocks")
            adjusted = None
        else:
            draws = sorted(math.exp(average([values[i] for i in sample])) for sample in indexes)
            for field, tail in (("ci95", .025), ("family_adjusted_ci", .025/family)):
                actual = [percentile(draws, tail), percentile(draws, 1-tail)]
                require(isinstance(reported[field], list) and len(reported[field]) == 2, label + ":interval missing")
                for index, number in enumerate(actual):
                    equal_number(number, reported[field][index], label + ":" + field)
                if field == "family_adjusted_ci":
                    adjusted = actual
            require(reported["status"] == "estimated", label + ":estimate status mismatch")
        return ratio, adjusted

    changing = [regime for regime in m["regimes"] if regime != "stable"]
    cell_fields = ("candidate_id", "horizon_steps", "regime", "batch_tokens", "contrast")
    reported_cells = unique_index(report["cells"], cell_fields, "reported cell")
    cells = list(itertools.product(m["candidate_ids"], m["horizons"], m["regimes"], m["batch_tokens"], CONTRAST_ARMS))
    require(set(reported_cells) == set(cells), "Reported cell inventory mismatch")
    block_cell_logs = {}
    for candidate, horizon, regime, batch, contrast in cells:
        values = []
        for block in block_ids:
            reps = [logs[(*key, contrast)] for key in sorted(expected)
                    if key[:5] == (block, regime, batch, horizon, candidate)]
            require(reps, "Missing repetitions in cell")
            values.append(average(reps))
        cell_key = (candidate, horizon, regime, batch, contrast)
        block_cell_logs[cell_key] = values
        row = reported_cells[cell_key]
        family = len(m["candidate_ids"]) * len(m["batch_tokens"]) * (len(changing) if regime != "stable" else 1)
        _, interval = check_estimate(values, family, row, str(cell_key))
        limit = plan["stable_ratio_limit"] if regime == "stable" else plan["useful_gain_ratio"]
        require(row["limit"] == limit, "Cell decision margin mismatch")
        expected_status = "descriptive" if contrast == "transition" else limit_status(interval, limit)
        require(row["criterion"] == expected_status, "Cell criterion mismatch")
        require(row["confirmatory_eligible"] == (manifest["split"] == "final" and horizon == plan["primary_horizon"]), "Horizon/split eligibility mismatch")

    aggregate_fields = ("candidate_id", "horizon_steps", "group", "contrast")
    reported_aggregates = unique_index(report["aggregates"], aggregate_fields, "reported aggregate")
    aggregates = list(itertools.product(m["candidate_ids"], m["horizons"], ("changing", "stable"), CONTRAST_ARMS))
    require(set(reported_aggregates) == set(aggregates), "Reported aggregate inventory mismatch")
    checked_aggregates = []
    for candidate, horizon, group, contrast in aggregates:
        regimes = changing if group == "changing" else ["stable"]
        constituents = [block_cell_logs[candidate, horizon, regime, batch, contrast]
                        for regime in regimes for batch in m["batch_tokens"]]
        values = [average([cell[i] for cell in constituents]) for i in range(count)]
        key = (candidate, horizon, group, contrast)
        row = reported_aggregates[key]
        ratio, interval = check_estimate(values, len(m["candidate_ids"]), row, str(key))
        limit = plan["stable_ratio_limit"] if group == "stable" else plan["useful_gain_ratio"]
        require(row["limit"] == limit and row["criterion"] == ("descriptive" if contrast == "transition" else limit_status(interval, limit)), "Aggregate criterion mismatch")
        require(row["confirmatory_eligible"] == (manifest["split"] == "final" and horizon == plan["primary_horizon"]), "Aggregate eligibility mismatch")
        checked_aggregates.append(dict(zip(aggregate_fields, key)) | {"ratio":ratio,"family_adjusted_ci":interval})

    if "pilot_gate" in report:
        require(manifest["split"] == "pilot", "Final report must not make a pilot decision")
        confirming = stage == "confirmation"
        qualifying = Counter()
        for candidate, block, regime, batch in itertools.product(m["candidate_ids"], block_ids, changing, m["batch_tokens"]):
            values = [payback_seconds[key] for key in sorted(expected)
                      if key[:5] == (block, regime, batch, 512, candidate)]
            if len(values) == (3 if confirming else 1) and average(values) >= .001 and sum(v > 0 for v in values) >= (2 if confirming else 1):
                qualifying[(candidate, block)] += 1
        for row in report["pilot_gate"]["candidates"]:
            expected_continue = sum(qualifying[row["candidate_id"], block] > 0 for block in block_ids) >= 2
            require(row["continue"] == expected_continue, "Pilot continuation mismatch")

    return {"status":"verified","config_hash":digest,"split":manifest["split"],"stage":stage,
            "complete_arm_records":len(raw),"complete_matched_triples":len(expected),
            "source_blocks":count,"cell_estimates_checked":len(cells),"aggregate_estimates_checked":len(aggregates),
            "bootstrap_draws_per_estimate":len(indexes),"maximum_absolute_numerical_difference":max(differences, default=0.),
            "absolute_tolerance":tolerance,"repetitions_per_pair_cell":dict(Counter(str(sum(key[:5] == group for key in expected)) for group in {key[:5] for key in expected})),
            "aggregates":checked_aggregates,
            "scope":"Independent standard-library arithmetic/schema reimplementation by the same agent team; not external human reproduction, an independent experiment, or proof of interval coverage.",
            "interval_scope":"Fixed hand-selected corpus, block-resampling assumption; not production-workload population confidence intervals."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.manifest.read_text()), args.records, json.loads(args.report.read_text()))
    result["input_sha256"] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [args.manifest, *args.records, args.report]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("aggregates","input_sha256")}))


if __name__ == "__main__":
    main()
