#!/usr/bin/env python3
"""Audit unchanged pilot branch records against the declared execution selectors.

This supplements, and does not rewrite, the frozen analysis selector or data.
Numerical passes and snapshot hashes are runtime attestations, not tensor reruns.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import random
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def plans(manifest):
    d, matrix = manifest["diagnostics"], manifest["matrix"]
    reps = d.get("repetitions", matrix["repetitions"])
    reps = range(reps) if isinstance(reps, int) else reps
    execution = {(c["regime"], c["trace_variant"], b, s, r, t): c["phase_steps"]
                 for c, b, s, r, t in itertools.product(d.get("cases", matrix["cases"]),
                     d.get("batch_tokens", matrix["batch_tokens"]), d.get("seeds", matrix["seeds"]), reps, d["decision_steps"])}
    analysis = set(itertools.product(d["analysis_regimes"], d["analysis_trace_variants"],
        matrix["batch_tokens"], d["analysis_seeds"], d["analysis_repetitions"], d["decision_steps"]))
    return execution, analysis


def balanced_orders(manifest):
    d, matrix = manifest["diagnostics"], manifest["matrix"]
    reps = d.get("repetitions", matrix["repetitions"])
    reps = range(reps) if isinstance(reps, int) else reps
    keys = [(c["regime"], c["trace_variant"], b, s, r, t) for s, r, c, b, t in itertools.product(
        d.get("seeds", matrix["seeds"]), reps, d.get("cases", matrix["cases"]),
        d.get("batch_tokens", matrix["batch_tokens"]), d["decision_steps"])]
    orders = [["allow", "skip"] if index % 2 == 0 else ["skip", "allow"] for index in range(len(keys))]
    random.Random(d.get("order_seed", 471)).shuffle(orders)
    return dict(zip(keys, orders))


def audit(manifest, records, final_template):
    digest = sha(canonical({k: v for k, v in manifest.items() if k != "freeze"}))
    require(manifest["split"] == "pilot" and digest == manifest["freeze"]["sha256"], "Pilot manifest integrity/split mismatch")
    expected, original_analysis = plans(manifest)
    final_execution, final_analysis = plans(final_template)
    require(final_template["split"] == "final" and set(final_execution) == final_analysis, "Final execution/analysis selectors disagree")
    require(len(expected) == 8 and len(records) == 8, "Expected exactly eight declared execution states and records")
    env, horizon = manifest["environment"], manifest["diagnostics"]["horizon_steps"]
    require(horizon == manifest["diagnostics"]["future_steps"], "Conflicting horizon declarations")
    declared_orders = balanced_orders(manifest)
    explicit_order = sum("branch_order" in row for row in records)
    require(explicit_order in (0, 8), "Mixed explicit and legacy branch-order records")
    seen, rows, actual_orders = set(), [], []
    for record in records:
        key = tuple(record[k] for k in ("regime", "trace_variant", "batch_tokens", "seed", "repetition", "decision_step"))
        require(key in expected and key not in seen, f"Unexpected or duplicate execution state: {key}")
        seen.add(key)
        for field, value in (("config_hash", digest), ("split", "pilot"), ("measurement_kind", "gpu_branch"),
                ("status", "complete"), ("correctness_pass", True), ("evidence_eligible", True),
                ("future_steps", horizon), ("future_rebalancing_enabled", False), ("phase_steps", expected[key]),
                ("harness_sha256", env["harness_sha256"]), ("native_vllm_commit", env["vllm_commit"]),
                ("world_size", env["world_size"]), ("dtype", env["dtype"]),
                ("dispatch_backend", env["dispatch_backend"]), ("eplb_mode", env["eplb_mode"])):
            require(record.get(field) == value, f"{key}: {field} mismatch")
        require(sha(canonical(record["harness_file_sha256"])) == env["harness_sha256"], "Harness file-hash aggregate mismatch")
        require(record["policy_config"] == manifest["policy"], "Policy configuration differs from frozen pilot")
        trace_expected = {k: manifest["trace"][k] for k in ("num_experts", "top_k", "window_steps")}
        trace_expected.update(seed=key[3], regime=key[0], tokens_per_step=key[2], phase_steps=expected[key],
                              num_steps=manifest["trace"]["num_steps"] + 2 * manifest["trace"]["window_steps"])
        require(all(record["trace_config"].get(k) == v for k, v in trace_expected.items()), "Replay trace configuration mismatch")
        job = {k: record[k] for k in ("regime", "trace_variant", "phase_steps", "method", "seed", "repetition", "batch_tokens", "decision_step")}
        if "branch_order" in record:
            job["branch_order"] = record["branch_order"]
            require(record["branch_order"] == declared_orders[key], "Declared branch order differs from seeded balanced assignment")
        require(record["method"] == "BRANCH" and record["record_id"] == sha(json.dumps([digest, job], sort_keys=True).encode()), "Record identity mismatch")
        source = record["weights_source"]
        require((source["source_model"], source["source_revision"], source["layer"], source["weights_kind"]) ==
                (env["model_id"], env["model_revision"], env["layer_id"], "pretrained"), "Checkpoint provenance mismatch")
        for map_field, hash_field in (("current_map", "source_map_sha256"), ("proposed_map", "candidate_map_sha256")):
            require(sha(struct.pack(f"<{len(record[map_field])}q", *record[map_field])) == record[hash_field], "Map bytes/hash mismatch")
        require(sha(json.dumps(record["source_weight_rank_sha256"]).encode()) == record["source_weights_sha256"], "Rank weight-hash aggregate mismatch")
        require(record["trace_artifact"]["trace_sha256"] == record["trace_sha256"], "Route artifact identity mismatch")
        arms = {arm["branch"]: arm for arm in record["branches"]}
        require(len(record["branches"]) == 2 and set(arms) == {"allow", "skip"}, "Need exactly one allow/skip pair")
        actual_order = [arm["branch"] for arm in record["branches"]]
        actual_orders.append(actual_order)
        if "branch_order" in record:
            require(actual_order == record["branch_order"], "Actual arm execution order differs from declared job")
        for arm in arms.values():
            require(arm["correctness_pass"] is True and arm["future_steps"] == horizon, "Arm correctness/horizon mismatch")
            for field in ("source_map_sha256", "candidate_map_sha256", "source_weights_sha256", "trace_sha256"):
                require(re.fullmatch(r"[0-9a-f]{64}", record[field]) and arm[field] == record[field], f"Arm {field} mismatch")
            for field in ("elapsed_seconds", "migration_bytes", "correctness_max_abs_error", "correctness_atol", "correctness_rtol"):
                value = arm[field]
                require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, f"Invalid numerical field {field}")
            require(arm["elapsed_seconds"] > 0 and arm["correctness_atol"] > 0, "Invalid duration/tolerance")
        rows.append({"state": key, "status": "validated", "tv": record["gate_decision"]["tv"],
            "payback_ms": 1000 * (arms["skip"]["elapsed_seconds"] - arms["allow"]["elapsed_seconds"]),
            "numerical_checks": {name: {k: arm[k] for k in ("correctness_pass", "correctness_max_abs_error", "correctness_atol", "correctness_rtol")} for name, arm in arms.items()}})
    require(seen == set(expected), "Missing declared execution states")
    allow_first = sum(order[0] == "allow" for order in actual_orders)
    require(not explicit_order or allow_first == 4, "Explicit branch plan must execute four allow-first and four skip-first pairs")
    return {"status": "validated_with_original_analysis_selector_mismatch" if set(expected) != original_analysis else "validated",
        "manifest_sha256": digest, "harness_sha256": env["harness_sha256"], "split": "pilot",
        "expected_execution_pairs": len(expected), "validated_pairs": len(rows), "fixed_future_steps": horizon,
        "original_analysis_selector_mismatch": set(expected) != original_analysis,
        "execution_states_excluded_by_original_analysis": sorted(set(expected) - original_analysis),
        "original_analysis_states_without_execution": sorted(original_analysis - set(expected)),
        "final_execution_analysis_selectors_agree": True, "final_expected_pairs": len(final_execution), "rows": rows,
        "branch_order_audit": {"explicit_seeded_plan": bool(explicit_order), "order_seed": manifest["diagnostics"].get("order_seed", 471),
            "allow_first_pairs": allow_first, "skip_first_pairs": 8 - allow_first,
            "balanced_four_each_required": bool(explicit_order), "legacy_order_preserved": not explicit_order},
        "interpretation": "Supplemental audit of the original declared pilot execution. No frozen manifest, record, seed, or split was changed. These are descriptive pilot diagnostics, not confirmatory evidence. Numerical and tensor-equality claims remain runtime attestations; this audit checks their fields and hashes, not saved output tensors."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "records", "final-template", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.manifest.read_text()),
        [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()], json.loads(args.final_template.read_text()))
    report["records_file_sha256"] = sha(args.records.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "validated_pairs", "original_analysis_selector_mismatch", "final_execution_analysis_selectors_agree")}))


if __name__ == "__main__":
    main()
