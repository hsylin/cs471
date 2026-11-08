"""Predeclared paired analysis; no fabricated values for unexecuted runs.

The independent sampling unit is the trace seed. Repetitions are averaged on
the log runtime-ratio scale before bootstrap resampling. A seed is resampled
as one block across the predeclared cells, preserving cross-cell dependence.
Only measured GPU run records are eligible for performance conclusions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAPS = 10000
PAIR_FIELDS = ("config_hash", "split", "regime", "trace_variant", "batch_tokens", "seed", "repetition")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def manifest_hash(manifest: dict) -> str:
    """Hash the payload, excluding the self-referential frozen integrity field."""
    return hashlib.sha256(canonical_bytes({k: v for k, v in manifest.items() if k != "freeze"})).hexdigest()


def _matrix(manifest: dict) -> dict:
    matrix = manifest["matrix"]
    for field in ("methods", "batch_tokens", "seeds"):
        values = matrix[field]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"matrix.{field} must be nonempty and unique")
    if not isinstance(matrix["repetitions"], int) or matrix["repetitions"] < 1:
        raise ValueError("matrix.repetitions must be a positive integer")
    cases = _cases(matrix)
    keys = [(c["regime"], c["trace_variant"]) for c in cases]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("matrix cases must be nonempty and unique by regime/trace_variant")
    return matrix


def _cases(matrix: dict) -> list[dict]:
    if "cases" in matrix:
        return matrix["cases"]
    return [{"regime": r, "trace_variant": "core", "phase_steps": 0} for r in matrix["regimes"]]


def verify_frozen_manifest(manifest: dict) -> str:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    digest = manifest_hash(manifest)
    frozen = manifest.get("freeze", {})
    if frozen.get("sha256") != digest:
        raise ValueError("manifest is not frozen or its content changed after freezing")
    _matrix(manifest)
    return digest


def freeze_manifest(source: Path, destination: Path, pilot: Path | None = None) -> dict:
    """Write once (O_EXCL); changed files fail integrity checks on later reads.

    This detects accidental post-freeze edits, not tampering by a malicious user
    who can recompute SHA-256. Keep the frozen manifest in version control.
    """
    manifest = json.loads(source.read_text())
    if "freeze" in manifest:
        raise ValueError("freeze expects an unfrozen template")
    matrix = _matrix(manifest)
    if manifest.get("split") not in {"pilot", "final"}:
        raise ValueError("split must be pilot or final")
    if manifest["split"] == "final":
        if pilot is None:
            raise ValueError("final freezing requires the completed pilot's frozen manifest")
        prior = json.loads(pilot.read_text())
        pilot_hash = verify_frozen_manifest(prior)
        if prior.get("split") != "pilot":
            raise ValueError("pilot reference must have split=pilot")
        if set(matrix["seeds"]) & set(prior["matrix"]["seeds"]):
            raise ValueError("pilot and final trace seeds overlap")
        manifest["pilot_manifest_sha256"] = pilot_hash
        unresolved = _unresolved(manifest)
        if unresolved:
            raise ValueError("final template has unresolved fields: " + ", ".join(unresolved))
        for key in ("model_id", "model_revision", "hardware", "vllm_commit"):
            if not manifest.get("environment", {}).get(key):
                raise ValueError(f"final environment.{key} is required")
    digest = manifest_hash(manifest)
    manifest["freeze"] = {"algorithm": "sha256", "sha256": digest, "schema_version": 1}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    destination.chmod(0o444)
    return manifest


def _unresolved(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _unresolved(v, f"{prefix}.{k}".strip("."))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _unresolved(v, f"{prefix}[{i}]")]
    return [prefix] if value is None or (isinstance(value, str) and value.startswith("UNSET")) else []


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: record must be an object")
            record["_source"] = f"{path}:{number}"
            records.append(record)
    return records


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = int(position)
    fraction = position - low
    return values[low] * (1 - fraction) + values[min(low + 1, len(values) - 1)] * fraction


def paired_cluster_estimate(seed_logs: dict[int, float], *, bootstrap_samples: int = DEFAULT_BOOTSTRAPS,
                            bootstrap_seed: int = 471, minimum_seeds: int = 5) -> dict:
    """Geometric mean candidate/reference ratio and seed-block percentile CI."""
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if minimum_seeds < 2:
        raise ValueError("at least two independent seeds are required for an interval")
    values = [seed_logs[k] for k in sorted(seed_logs)]
    result = {"independent_seeds": len(values), "ratio": None, "ci95": None,
              "status": "missing" if not values else "insufficient_seeds"}
    if not values:
        return result
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite paired log ratio")
    result["ratio"] = math.exp(statistics.mean(values))
    if len(values) < minimum_seeds:
        return result
    rng = random.Random(bootstrap_seed)
    boot = [math.exp(statistics.mean(rng.choices(values, k=len(values)))) for _ in range(bootstrap_samples)]
    result.update(ci95=[_quantile(boot, .025), _quantile(boot, .975)], status="estimated")
    return result


def _eligible(record: dict) -> bool:
    seconds = record.get("elapsed_seconds")
    return (record.get("schema_version") == SCHEMA_VERSION
            and record.get("status") == "complete" and record.get("correctness_pass") is True
            and record.get("evidence_eligible") is True
            and not record.get("_contract_issues")
            and isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
            and math.isfinite(seconds) and seconds > 0
            and record.get("measurement_kind") == "gpu")


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _record_contract_issues(record: dict, manifest: dict) -> list[str]:
    """Check what ran, not merely whether its record quotes the manifest hash."""
    if not manifest["analysis"].get("require_execution_metadata") or record.get("status") != "complete":
        return []
    env, trace = manifest["environment"], manifest["trace"]
    issues = []
    fields = {"native_vllm_commit": env["vllm_commit"],
              **{k: env[k] for k in ("world_size", "dtype", "dispatch_backend", "eplb_mode")},
              "steps_completed": trace["num_steps"], "measured_horizon_steps": trace["num_steps"],
              "observation_warmup_steps": 2 * trace["window_steps"],
              "raw_trace_steps": trace["num_steps"] + 2 * trace["window_steps"],
              "policy_config": manifest["policy"]}
    for key, expected in fields.items():
        if record.get(key) != expected:
            issues.append(f"{key}_mismatch")
    if env.get("harness_sha256") and record.get("harness_sha256") != env["harness_sha256"]:
        issues.append("harness_sha256_mismatch")
    if env.get("gpu_name_contains") and env["gpu_name_contains"] not in record.get("gpu_name", ""):
        issues.append("gpu_name_mismatch")
    source = record.get("weights_source")
    if not isinstance(source, dict):
        issues.append("missing_weights_metadata")
    else:
        expected = {"source_model": env["model_id"], "source_revision": env["model_revision"],
                    "layer": env["layer_id"], "weights_kind": "pretrained", "dtype": env["dtype"],
                    "num_experts": trace["num_experts"], "top_k": trace["top_k"]}
        expected.update({k: env[k] for k in ("hidden_size", "intermediate_size") if k in env})
        issues.extend(f"weights_{k}_mismatch" for k, v in expected.items() if source.get(k) != v)
        if not _sha256(source.get("weights_sha256")):
            issues.append("missing_weights_sha256")
    if not _sha256(record.get("trace_sha256")):
        issues.append("missing_trace_sha256")
    actual_trace = record.get("trace_config", {})
    case = next(c for c in _cases(manifest["matrix"])
                if c["regime"] == record["regime"] and c["trace_variant"] == record.get("trace_variant", "core"))
    expected_trace = {**trace, "num_steps": trace["num_steps"] + 2 * trace["window_steps"],
                      "warmup_steps": 2 * trace["window_steps"], "phase_steps": case["phase_steps"],
                      "seed": record["seed"], "regime": record["regime"], "tokens_per_step": record["batch_tokens"]}
    issues.extend(f"trace_{k}_mismatch" for k, v in expected_trace.items() if actual_trace.get(k) != v)
    cadence = manifest["policy"]["rebalance_interval"]
    expected_decisions = [] if record["method"] == "B0" else list(range(cadence - 1, trace["num_steps"] - 1, cadence))
    decisions = record.get("decision_log", [])
    if not isinstance(decisions, list) or [d.get("step") for d in decisions] != expected_decisions:
        issues.append("scheduled_decision_steps_mismatch")
    if record.get("accepted", 0) + record.get("rejected", 0) != len(expected_decisions):
        issues.append("decision_count_mismatch")
    return issues


def analyze(records: list[dict], manifest: dict, *, bootstrap_samples: int | None = None) -> dict:
    digest = verify_frozen_manifest(manifest)
    matrix = _matrix(manifest)
    plan = manifest["analysis"]
    samples = plan.get("bootstrap_samples", DEFAULT_BOOTSTRAPS) if bootstrap_samples is None else bootstrap_samples
    minimum = plan.get("minimum_independent_seeds", 5)
    candidate = plan.get("candidate", "B4")
    references = plan.get("references", ["B3", "B5"])
    cases = _cases(matrix)
    regimes = list(dict.fromkeys(c["regime"] for c in cases))
    expected = {(method, case["regime"], case["trace_variant"], batch, seed, repetition)
                for method, case, batch, seed, repetition in itertools.product(
                    matrix["methods"], cases, matrix["batch_tokens"], matrix["seeds"], range(matrix["repetitions"]))}
    indexed, excluded, duplicates = {}, [], []
    for record in records:
        if record.get("config_hash") != digest or record.get("split") != manifest["split"]:
            excluded.append({"source": record.get("_source"), "reason": "different_manifest_or_split"})
            continue
        try:
            key = (record["method"], record["regime"], record.get("trace_variant", "core"),
                   record["batch_tokens"], record["seed"], record["repetition"])
            if key not in expected:
                excluded.append({"source": record.get("_source"), "reason": "outside_declared_matrix"})
                continue
        except (KeyError, TypeError):
            excluded.append({"source": record.get("_source"), "reason": "invalid_pair_key"})
            continue
        if key in indexed:
            duplicates.append(list(key))
        else:
            indexed[key] = {**record, "_contract_issues": _record_contract_issues(record, manifest)}
    if duplicates:
        raise ValueError(f"duplicate paired run records: {duplicates[:5]}")
    if plan.get("require_execution_metadata"):
        groups = {}
        for key, record in indexed.items():
            if record.get("status") == "complete":
                groups.setdefault(key[1:], []).append(record)
        for group in groups.values():
            tapes = {r.get("trace_sha256") for r in group}
            weights = {r.get("weights_source", {}).get("weights_sha256")
                       for r in group if isinstance(r.get("weights_source"), dict)}
            if len(tapes) > 1 or len(weights) > 1:
                for record in group:
                    record["_contract_issues"].append("paired_trace_or_weights_hash_mismatch")
    missing = sorted(expected - indexed.keys())
    invalid = [{"key": list(key), "status": rec.get("status"),
                "correctness_pass": rec.get("correctness_pass"),
                "evidence_eligible": rec.get("evidence_eligible"),
                "contract_issues": rec.get("_contract_issues", []),
                "measurement_kind": rec.get("measurement_kind")}
               for key, rec in indexed.items() if not _eligible(rec)]
    cells, cell_logs = [], {}
    for reference, case, batch in itertools.product(references, cases, matrix["batch_tokens"]):
        regime, variant = case["regime"], case["trace_variant"]
        seed_logs, byte_deltas = {}, []
        pair_count = 0
        for seed in matrix["seeds"]:
            logs, deltas = [], []
            for repetition in range(matrix["repetitions"]):
                a = indexed.get((candidate, regime, variant, batch, seed, repetition), {})
                b = indexed.get((reference, regime, variant, batch, seed, repetition), {})
                if not (_eligible(a) and _eligible(b)):
                    continue
                logs.append(math.log(a["elapsed_seconds"] / b["elapsed_seconds"]))
                if all(isinstance(r.get("migration_bytes"), (int, float)) for r in (a, b)):
                    deltas.append(a["migration_bytes"] - b["migration_bytes"])
            pair_count += len(logs)
            # Avoid silently changing repetition weighting if one member failed.
            if len(logs) == matrix["repetitions"]:
                seed_logs[seed] = statistics.mean(logs)
                if len(deltas) == len(logs):
                    byte_deltas.append(statistics.mean(deltas))
        estimate = paired_cluster_estimate(seed_logs, bootstrap_samples=samples,
                                           bootstrap_seed=plan.get("bootstrap_seed", 471), minimum_seeds=minimum)
        complete = len(seed_logs) == len(matrix["seeds"])
        cell = {"candidate": candidate, "reference": reference, "regime": regime,
                "trace_variant": variant, "batch_tokens": batch, "paired_runs": pair_count,
                "expected_paired_runs": len(matrix["seeds"]) * matrix["repetitions"],
                "complete": complete, **estimate,
                "mean_migration_bytes_difference": statistics.mean(byte_deltas) if byte_deltas else None}
        if not complete:
            cell["status"] = "incomplete"
        cell["ratio_limit"] = _ratio_limit(plan, regime, reference)
        cell["verdict"] = _verdict(estimate, complete, manifest["split"], cell["ratio_limit"])
        cells.append(cell)
        cell_logs[(reference, regime, variant, batch)] = seed_logs
    aggregates = []
    for reference, regime in itertools.product(references, regimes):
        # Equal weight over declared batch cells, then resample entire seeds.
        blocks = {}
        for seed in matrix["seeds"]:
            logs = [cell_logs[(reference, regime, case["trace_variant"], batch)].get(seed)
                    for case in cases if case["regime"] == regime for batch in matrix["batch_tokens"]]
            if all(v is not None for v in logs):
                blocks[seed] = statistics.mean(logs)
        estimate = paired_cluster_estimate(blocks, bootstrap_samples=samples,
                                           bootstrap_seed=plan.get("bootstrap_seed", 471), minimum_seeds=minimum)
        complete = len(blocks) == len(matrix["seeds"])
        threshold = _ratio_limit(plan, regime, reference)
        verdict = _verdict(estimate, complete, manifest["split"], threshold)
        aggregates.append({"candidate": candidate, "reference": reference, "regime": regime,
                           "weighting": "equal declared variant/batch cells, then equal independent seeds",
                           "complete": complete, **estimate, "ratio_limit": threshold, "verdict": verdict})
    required = [a for a in aggregates if a["ratio_limit"] is not None]
    required += [c for c in cells if c["regime"] in plan.get("required_cell_regimes", ["stationary"])
                 and c["ratio_limit"] is not None]
    ready = not missing and not invalid and manifest["split"] == "final"
    overall = "not_evaluable"
    if ready and required:
        outcomes = {a["verdict"] for a in required}
        overall = ("not_evaluable" if "not_evaluable" in outcomes else
                   "pass" if outcomes == {"pass"} else
                   "fail" if "fail" in outcomes else "inconclusive")
    return {"schema_version": 1, "manifest_sha256": digest, "split": manifest["split"],
            "overall_verdict": overall, "expected_records": len(expected), "present_records": len(indexed),
            "missing_records": [list(k) for k in missing], "invalid_records": invalid,
            "excluded_records": excluded, "bootstrap_samples": samples,
            "analysis_unit": "independent trace seed; repetitions averaged before seed-block bootstrap",
            "inference_scope": "predeclared cells on the frozen model, hardware and implementation only",
            "decision_rule": "All required regime aggregates and designated per-cell limits must pass; no cross-regime average",
            "verdict_meaning": "fail means a predeclared criterion is missed, not necessarily that the candidate is slower than its reference",
            "cells": cells, "regime_aggregates": aggregates,
            "method_descriptives": _method_descriptives(indexed, matrix, cases)}


def _method_descriptives(indexed: dict, matrix: dict, cases: list[dict]) -> list[dict]:
    metrics = ("elapsed_seconds", "migration_bytes", "migration_seconds", "accepted", "rejected",
               "planner_seconds", "controller_seconds", "observation_seconds")
    rows = []
    for method, case, batch in itertools.product(matrix["methods"], cases, matrix["batch_tokens"]):
        blocks = []
        for seed in matrix["seeds"]:
            records = [indexed.get((method, case["regime"], case["trace_variant"], batch, seed, rep), {})
                       for rep in range(matrix["repetitions"])]
            if all(_eligible(record) for record in records):
                blocks.append(records)
        row = {"method": method, "regime": case["regime"], "trace_variant": case["trace_variant"],
               "batch_tokens": batch, "independent_seeds": len(blocks),
               "complete": len(blocks) == len(matrix["seeds"]), "inferential_status": "descriptive_only"}
        for metric in metrics:
            values = []
            for block in blocks:
                xs = [record.get(metric) for record in block]
                if all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) and x >= 0 for x in xs):
                    values.append(statistics.mean(xs))
            # Missing component measurements are not silently averaged away.
            row[f"mean_{metric}"] = statistics.mean(values) if values and len(values) == len(blocks) else None
        rows.append(row)
    return rows


def _ratio_limit(plan: dict, regime: str, reference: str) -> float | None:
    required_references = plan.get("regime_references", {}).get(regime, plan.get("references", ["B3", "B5"]))
    return plan["regime_ratio_limits"].get(regime) if reference in required_references else None


def _verdict(estimate: dict, complete: bool, split: str, threshold: float | None) -> str:
    if split != "final" or not complete or not estimate["ci95"] or threshold is None:
        return "not_evaluable"
    low, high = estimate["ci95"]
    return "pass" if high <= threshold else ("fail" if low > threshold else "inconclusive")


def write_report(summary: dict, output: Path, *, plots: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    columns = ["candidate", "reference", "regime", "trace_variant", "batch_tokens", "paired_runs", "expected_paired_runs",
               "independent_seeds", "complete", "status", "ratio", "ci95_low", "ci95_high",
               "mean_migration_bytes_difference", "ratio_limit", "verdict"]
    with (output / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cell in summary["cells"]:
            row = {key: cell.get(key) for key in columns}
            ci = cell.get("ci95") or [None, None]
            row.update(ci95_low=ci[0], ci95_high=ci[1])
            writer.writerow(row)
    descriptives = summary.get("method_descriptives", [])
    if descriptives:
        with (output / "methods.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(descriptives[0]))
            writer.writeheader()
            writer.writerows(descriptives)
    if plots:
        plot_ratios(summary, output / "runtime-ratios.png")


def plot_ratios(summary: dict, destination: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for figures; use --no-plots for JSON/CSV only") from exc
    cells = summary["cells"]
    fig, ax = plt.subplots(figsize=(9, max(3.2, .48 * len(cells))))
    for y, cell in enumerate(cells):
        ratio, ci = cell["ratio"], cell["ci95"]
        color = "#176B87" if cell["complete"] else "#B96A20"
        if ratio is None:
            ax.text(1.01, y, "missing", color="#666666", va="center")
        elif ci:
            ax.errorbar(ratio, y, xerr=[[max(0, ratio-ci[0])], [max(0, ci[1]-ratio)]], fmt="o", color=color, capsize=3)
        else:
            ax.plot(ratio, y, "x", color=color)
    ax.set_yticks(range(len(cells)), [f"B4/{c['reference']} · {c['regime']}/{c['trace_variant']} · {c['batch_tokens']} tokens" for c in cells])
    ax.axvline(1, color="#444444", linewidth=1)
    ax.axvline(.95, color="#999999", linestyle=":", linewidth=1)
    ax.axvline(1.02, color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("Paired geometric mean runtime ratio (lower is faster); 95% seed-block bootstrap CI")
    ax.set_title(f"SkewBench {summary['split']} · {summary['overall_verdict'].replace('_', ' ')}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def analyze_branches(records: list[dict], manifest: dict) -> dict:
    """Describe paired fixed-horizon replays; never infer population accuracy."""
    digest = verify_frozen_manifest(manifest)
    diagnostic = manifest.get("diagnostics", {})
    horizon = diagnostic.get("horizon_steps", diagnostic.get("future_steps", 128))
    expected = set(itertools.product(
        diagnostic.get("analysis_regimes", ["stationary", "burst", "persistent", "periodic"]),
        diagnostic.get("analysis_trace_variants", ["core"]), manifest["matrix"]["batch_tokens"],
        diagnostic.get("analysis_seeds", [manifest["matrix"]["seeds"][0]]),
        diagnostic.get("analysis_repetitions", [0]), diagnostic.get("decision_steps", [63])))
    if not expected or horizon < 1:
        raise ValueError("branch plan needs nonempty strata and a positive horizon")
    indexed, excluded = {}, []
    for record in records:
        if record.get("config_hash") != digest or record.get("split") != manifest["split"]:
            excluded.append({"source": record.get("_source"), "reason": "different_manifest_or_split"})
            continue
        key = tuple(record.get(k, "core" if k == "trace_variant" else None)
                    for k in ("regime", "trace_variant", "batch_tokens", "seed", "repetition", "decision_step"))
        if key not in expected:
            excluded.append({"source": record.get("_source"), "reason": "outside_declared_branch_strata"})
            continue
        if key in indexed:
            raise ValueError(f"duplicate branch decision state: {key}")
        indexed[key] = record
    rows = []
    for key in sorted(expected):
        regime, variant, batch, seed, repetition, decision_step = key
        row = {"regime": regime, "trace_variant": variant, "batch_tokens": batch, "seed": seed,
               "repetition": repetition, "decision_step": decision_step, "future_steps": horizon,
               "stratum": f"{regime}/{variant}/{batch}", "status": "missing", "issues": [],
               "b3_would_allow": None, "b4_allowed": None, "tv": None, "predicted_gain": None,
               "allow_elapsed_seconds": None, "skip_elapsed_seconds": None, "payback_seconds": None,
               "allow_migration_bytes": None, "skip_migration_bytes": None,
               "allow_max_abs_error": None, "skip_max_abs_error": None,
               "numeric_correctness_pass": None, "observed_payback": None,
               "source_map_sha256": None, "source_weights_sha256": None,
               "candidate_map_sha256": None, "trace_sha256": None}
        record = indexed.get(key)
        if record is None:
            rows.append(row)
            continue
        issues = []
        for field, value in (("schema_version", 1), ("measurement_kind", "gpu_branch"),
                             ("status", "complete"), ("correctness_pass", True),
                             ("evidence_eligible", True), ("future_steps", horizon)):
            if record.get(field) != value:
                issues.append(f"{field}_mismatch")
        if manifest.get("analysis", {}).get("require_execution_metadata"):
            env = manifest["environment"]
            for field, value in (("native_vllm_commit", env["vllm_commit"]),
                                 ("world_size", env["world_size"]), ("dtype", env["dtype"]),
                                 ("dispatch_backend", env["dispatch_backend"]), ("eplb_mode", env["eplb_mode"])):
                if record.get(field) != value:
                    issues.append(f"{field}_mismatch")
            if env.get("harness_sha256") and record.get("harness_sha256") != env["harness_sha256"]:
                issues.append("harness_sha256_mismatch")
            source = record.get("weights_source", {})
            for field, value in (("source_model", env["model_id"]), ("source_revision", env["model_revision"]),
                                 ("layer", env["layer_id"]), ("weights_kind", "pretrained")):
                if not isinstance(source, dict) or source.get(field) != value:
                    issues.append(f"weights_{field}_mismatch")
        branches = record.get("branches", [])
        if (not isinstance(branches, list) or len(branches) != 2
                or not all(isinstance(b, dict) for b in branches)
                or {b.get("branch") for b in branches} != {"allow", "skip"}):
            issues.append("need_exactly_one_allow_and_one_skip")
            branches = []
        hash_fields = ("source_map_sha256", "source_weights_sha256", "candidate_map_sha256", "trace_sha256")
        for field in hash_fields:
            row[field] = record.get(field)
            if not _sha256(record.get(field)):
                issues.append(f"missing_{field}")
            for branch in branches:
                if branch.get(field) != record.get(field):
                    issues.append(f"{branch['branch']}_{field}_mismatch")
        for branch in branches:
            name = branch["branch"]
            if branch.get("correctness_pass") is not True:
                issues.append(f"{name}_correctness_not_attested")
            if branch.get("future_steps") != horizon:
                issues.append(f"{name}_horizon_mismatch")
            for field in ("elapsed_seconds", "migration_bytes", "correctness_max_abs_error"):
                value = branch.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (field == "elapsed_seconds" and value == 0):
                    issues.append(f"{name}_invalid_{field}")
        decision = record.get("gate_decision", {})
        if not isinstance(decision, dict):
            decision = {}
        if not isinstance(record.get("b3_would_allow"), bool) or not isinstance(decision.get("allowed"), bool):
            issues.append("missing_admission_decision")
        for field in ("tv", "predicted_gain"):
            if not isinstance(decision.get(field), (int, float)) or not math.isfinite(decision[field]):
                issues.append(f"invalid_{field}")
        row.update(status="invalid" if issues else "complete", issues=issues,
                   b3_would_allow=record.get("b3_would_allow"), b4_allowed=decision.get("allowed"),
                   tv=decision.get("tv"), predicted_gain=decision.get("predicted_gain"))
        if not issues:
            by_name = {b["branch"]: b for b in branches}
            allow, skip = by_name["allow"], by_name["skip"]
            payback = skip["elapsed_seconds"] - allow["elapsed_seconds"]
            row.update(allow_elapsed_seconds=allow["elapsed_seconds"], skip_elapsed_seconds=skip["elapsed_seconds"],
                       payback_seconds=payback, allow_migration_bytes=allow["migration_bytes"],
                       skip_migration_bytes=skip["migration_bytes"],
                       allow_max_abs_error=allow["correctness_max_abs_error"],
                       skip_max_abs_error=skip["correctness_max_abs_error"], numeric_correctness_pass=True,
                       observed_payback="migration_faster" if payback > 0 else "migration_slower" if payback < 0 else "equal")
        rows.append(row)
    valid = sum(row["status"] == "complete" for row in rows)
    return {"schema_version": 1, "manifest_sha256": digest, "expected_pairs": len(expected),
            "complete_pairs": valid, "status": "complete" if valid == len(expected) else "incomplete",
            "fixed_future_steps": horizon, "rows": rows, "excluded_records": excluded,
            "interpretation": "Descriptive paired future replays from fixed decision states. Positive payback_seconds means migration finished the fixed future horizon faster, including its transfer cost. These strata share trace seeds and are not independent population samples; no significance test, selector-accuracy estimate, or generalization claim is computed.",
            "correctness_scope": "Runtime numerical checks and equality/hash attestations are retained; scalar summaries do not independently re-run tensor comparisons."}


def write_branch_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "branch-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "branches.csv").open("w", newline="") as handle:
        rows = summary["rows"]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "issues": ";".join(row["issues"])})


def select_cooldown(pilot_runs: list[tuple[dict, list[dict]]]) -> dict:
    """Choose one B5 setting using pilot data; no cross-manifest run pairing."""
    if not pilot_runs:
        raise ValueError("at least one pilot manifest/record set is required")
    first = pilot_runs[0][0]
    selection = first["pilot_selection"]
    grid = selection["cooldown_steps_grid"]
    candidates = []
    seen = set()
    def comparable(manifest):
        return {"environment": manifest.get("environment"), "trace": manifest.get("trace"),
                "matrix": {k: v for k, v in manifest["matrix"].items() if k != "methods"},
                "policy": {k: v for k, v in manifest["policy"].items() if k != "cooldown_steps"},
                "pilot_selection": manifest["pilot_selection"]}
    for manifest, records in pilot_runs:
        digest = verify_frozen_manifest(manifest)
        if manifest["split"] != "pilot":
            raise ValueError("cooldown selection must never consume final data")
        if comparable(manifest) != comparable(first):
            raise ValueError("cooldown pilot manifests differ beyond methods and cooldown")
        cooldown = manifest["policy"]["cooldown_steps"]
        if cooldown not in grid or cooldown in seen:
            raise ValueError("cooldown grid has an undeclared or duplicate candidate")
        seen.add(cooldown)
        if not {"B3", "B5"}.issubset(manifest["matrix"]["methods"]):
            raise ValueError("every cooldown pilot needs B3 and B5 within its own manifest")
        # Reuse execution-contract and duplicate validation. Its B4 results are
        # irrelevant here: the B5 comparison below has a separate declared role.
        audit = analyze(records, manifest)
        row = {"cooldown_steps": cooldown, "manifest_sha256": digest,
               "status": "incomplete", "all_case_runtime_ratio": None,
               "worst_stationary_batch_ratio": None, "stationary_batch_ratios": {},
               "stable_eligible": False, "missing_records": len(audit["missing_records"]),
               "invalid_records": len(audit["invalid_records"])}
        if not audit["missing_records"] and not audit["invalid_records"]:
            matrix = manifest["matrix"]
            indexed = {(r["method"], r["regime"], r.get("trace_variant", "core"), r["batch_tokens"],
                        r["seed"], r["repetition"]): r
                       for r in records if r.get("config_hash") == digest and r.get("split") == "pilot"}
            cell_logs, stable_logs = [], {}
            for case, batch in itertools.product(_cases(matrix), matrix["batch_tokens"]):
                seed_logs = []
                for seed in matrix["seeds"]:
                    repeat_logs = []
                    for rep in range(matrix["repetitions"]):
                        tail = (case["regime"], case["trace_variant"], batch, seed, rep)
                        a, b = indexed[("B5", *tail)], indexed[("B3", *tail)]
                        repeat_logs.append(math.log(a["elapsed_seconds"] / b["elapsed_seconds"]))
                    seed_logs.append(statistics.mean(repeat_logs))
                mean = statistics.mean(seed_logs)
                cell_logs.append(mean)
                if case["regime"] == "stationary":
                    stable_logs.setdefault(batch, []).append(mean)
            if not stable_logs:
                raise ValueError("cooldown selection needs a stationary pilot stratum")
            stable = {str(batch): math.exp(statistics.mean(logs)) for batch, logs in stable_logs.items()}
            worst = max(stable.values())
            row.update(status="complete", all_case_runtime_ratio=math.exp(statistics.mean(cell_logs)),
                       stationary_batch_ratios=stable, worst_stationary_batch_ratio=worst,
                       stable_eligible=worst <= 1.02)
        candidates.append(row)
    candidates.sort(key=lambda c: c["cooldown_steps"])
    result = {"schema_version": 1, "status": "not_evaluable", "selected_cooldown_steps": None,
              "stationary_constraint_met": None, "candidates": candidates,
              "missing_grid_candidates": sorted(set(grid) - seen), "selection_rule": selection,
              "interpretation": "Descriptive pilot-only parameter selection. B5/B3 runs are paired within each frozen manifest. No confidence interval or final-performance claim is made."}
    if result["missing_grid_candidates"] or any(c["status"] != "complete" for c in candidates):
        return result
    eligible = [c for c in candidates if c["stable_eligible"]]
    constraint_met = bool(eligible)
    if not eligible:
        best_stable = min(c["worst_stationary_batch_ratio"] for c in candidates)
        eligible = [c for c in candidates if c["worst_stationary_batch_ratio"] <= best_stable * 1.01]
    best = min(c["all_case_runtime_ratio"] for c in eligible)
    tied = [c for c in eligible if c["all_case_runtime_ratio"] <= best * 1.01]
    chosen = max(tied, key=lambda c: c["cooldown_steps"])
    result.update(status="selected", selected_cooldown_steps=chosen["cooldown_steps"],
                  stationary_constraint_met=constraint_met)
    return result


def write_cooldown_selection(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "cooldown-selection.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = summary["candidates"]
    if rows:
        with (output / "cooldown-candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**r, "stationary_batch_ratios": json.dumps(r["stationary_batch_ratios"], sort_keys=True)} for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze an immutable pilot/final manifest")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("destination", type=Path)
    freeze.add_argument("--pilot", type=Path)
    report = commands.add_parser("report", help="analyze JSONL GPU run records")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--records", type=Path, nargs="+", required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--no-plots", action="store_true")
    branch = commands.add_parser("branch-report", help="describe fixed-horizon allow/skip diagnostic pairs")
    branch.add_argument("--manifest", type=Path, required=True)
    branch.add_argument("--records", type=Path, nargs="+", required=True)
    branch.add_argument("--output-dir", type=Path, required=True)
    cooldown = commands.add_parser("cooldown-select", help="select one B5 cooldown using three frozen pilot datasets")
    cooldown.add_argument("--manifests", type=Path, nargs="+", required=True)
    cooldown.add_argument("--records", type=Path, nargs="+", required=True)
    cooldown.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        frozen = freeze_manifest(args.source, args.destination, args.pilot)
        print(frozen["freeze"]["sha256"])
    elif args.command == "cooldown-select":
        if len(args.manifests) != len(args.records):
            parser.error("provide one records file per manifest, in matching order")
        summary = select_cooldown([(json.loads(m.read_text()), load_records([r])) for m, r in zip(args.manifests, args.records)])
        write_cooldown_selection(summary, args.output_dir)
        print(json.dumps({k: summary[k] for k in ("status", "selected_cooldown_steps", "stationary_constraint_met")}))
    elif args.command == "branch-report":
        summary = analyze_branches(load_records(args.records), json.loads(args.manifest.read_text()))
        write_branch_report(summary, args.output_dir)
        print(json.dumps({"status": summary["status"], "complete_pairs": summary["complete_pairs"],
                          "expected_pairs": summary["expected_pairs"]}))
    else:
        summary = analyze(load_records(args.records), json.loads(args.manifest.read_text()))
        write_report(summary, args.output_dir, plots=not args.no_plots)
        print(json.dumps({"overall_verdict": summary["overall_verdict"],
                          "missing_records": len(summary["missing_records"]),
                          "invalid_records": len(summary["invalid_records"])}))


if __name__ == "__main__":
    main()
