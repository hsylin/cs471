#!/usr/bin/env python3
"""Independently audit raw old/new kernel profiles and padding diagnostics.

This script imports no optimization/runtime/analysis module. It reports
descriptive kernel residence and tile-row proxies, never bandwidth, wall-time
shares, inferential intervals or adaptive tuning decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-7)


def parse_trace(path, forwards, expected_sha256=None):
    path = Path(path)
    digest = file_hash(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"Trace SHA mismatch: {path.name}")
    trace = json.loads(path.read_text())
    groups, gemm_launches = {}, []
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name, duration = event["name"], event["dur"]
        require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and math.isfinite(duration) and duration >= 0, "Invalid device kernel duration")
        if name.startswith("fused_moe_kernel"):
            group = "fused_expert_gemm"
            gemm_launches.append({"name": name, "duration_us": duration,
                                 "block": event.get("args", {}).get("block"),
                                 "grid": event.get("args", {}).get("grid")})
        elif "nccl" in name.lower() and "allreduce" in name.lower():
            group = "nccl_allreduce"
        elif "_eplb_map_and_record_" in name:
            group = "physical_mapping"
        else:
            group = "other_kernels"
        values = groups.setdefault(group, {"calls": 0, "total_us": 0.0})
        values["calls"] += 1
        values["total_us"] += duration
    require(len(gemm_launches) == 2 * forwards, "Expected two actual fused expert kernels per forward")
    require(groups.get("physical_mapping", {}).get("calls") == forwards,
            "Expected one physical-map kernel per forward")
    require(groups.get("nccl_allreduce", {}).get("calls") == forwards,
            "Expected one output-allreduce kernel per forward")
    for values in groups.values():
        values["us_per_forward"] = values["total_us"] / forwards
    return {"trace_file": str(path.resolve()), "trace_sha256": digest,
            "trace_bytes": path.stat().st_size, "expected_hash_verified": expected_sha256 is not None,
            "event_filter": "cat == kernel and ph == X", "forwards": forwards,
            "kernel_groups": groups, "gemm_launches": gemm_launches}


def physical_diagnostics(values, batch, block_m, forwards):
    counts = np.asarray(values)
    require(counts.dtype.kind in "iu" and counts.shape == (forwards, 2, 80),
            "Expected integer physical counts [forwards,2,80]")
    require(np.all(counts >= 0), "Negative physical assignment count")
    assigned = counts.sum(axis=2)
    require(np.all(assigned.sum(axis=1) == batch * 8), "Physical assignment conservation failed")
    require(type(block_m) is int and block_m > 0, "Invalid M tile")
    active = (counts > 0).sum(axis=2)
    padded = ((counts + block_m - 1) // block_m * block_m).sum(axis=2)
    return {"physical_counts_sha256": array_hash(counts), "block_m": block_m,
            "owner_assignments_per_step": assigned.tolist(),
            "active_physical_experts_per_rank_per_step": active.tolist(),
            "local_padded_rows_per_step": padded.tolist(),
            "padded_to_useful_rows_by_rank": (padded.sum(axis=0) / assigned.sum(axis=0)).tolist(),
            "mean_active_physical_experts_by_rank": active.mean(axis=0).tolist(),
            "mean_padded_rows_by_rank": padded.mean(axis=0).tolist(),
            "mean_peak_rank_assignments": float(assigned.max(axis=1).mean())}


def audit_old(directory, expected_evidence=None):
    directory = Path(directory)
    calibration_path = directory / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    expected = {}
    if expected_evidence:
        evidence = json.loads(Path(expected_evidence).read_text())
        expected = {Path(row["file"]).name: row["sha256"] for row in evidence["profiles"]}
    result = []
    for row in calibration["rows"]:
        batch, domain, forwards = row["batch_tokens"], row["domain"], row["profile_forward_count"]
        counts = np.asarray(row["physical_tokens_per_slot"])
        require(counts.shape == (2, 80), "Old calibration physical count shape changed")
        repeated = np.repeat(counts[None], forwards, axis=0)
        estimates = [physical_diagnostics(repeated, batch, m, forwards) for m in (16, 32, 64, 128)]
        require(estimates[0]["owner_assignments_per_step"][0] == row["tokens_by_rank"],
                "Old owner-assignment summary differs from raw per-slot counts")
        require(estimates[0]["active_physical_experts_per_rank_per_step"][0]
                == row["active_physical_experts_by_rank"], "Old active-expert count mismatch")
        for rank in (0, 1):
            name = f"batch{batch}-domain{domain}-rank{rank}.json"
            if expected_evidence:
                require(name in expected, f"Old evidence inventory lacks {name}")
            trace = parse_trace(directory / name, forwards, expected.get(name))
            result.append({"batch_tokens": batch, "domain": domain, "rank": rank,
                           "capture_sha256": calibration["capture_sha256"],
                           "fixed_map_sha256": array_hash(row["fixed_map"]),
                           "raw_kernel_audit": trace, "padding_estimates": estimates})
    require(len(result) == 8, "Expected eight old calibration rank traces")
    return {"calibration_file": str(calibration_path.resolve()),
            "calibration_sha256": file_hash(calibration_path), "rows": result,
            "preexisting_trace_hash_inventory_verified": expected_evidence is not None,
            "scope": "Original fixed-map repeated-input calibration; different sources/maps from new profiles, so not a paired speedup baseline"}


def _verify_input_artifact(directory, row):
    recorded = row["input_artifact"]
    path = directory / "tuning-inputs" / Path(recorded["path"]).name
    require(file_hash(path) == recorded["sha256"] == row["replay_sha256"], "Profile input artifact SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        indices = archive["indices"]
        metadata = json.loads(str(archive["metadata"]))
    require(indices.dtype == np.int64 and indices.shape == (48, row["batch_tokens"]),
            "Profile input index tensor shape/dtype mismatch")
    require(array_hash(indices) == recorded["indices_sha256"], "Input index bytes mismatch")
    require(metadata == recorded["metadata"] and metadata["capture_sha256"] == row["capture_sha256"],
            "Profile input provenance mismatch")
    require(all(len(set(x)) == row["batch_tokens"] for x in indices.tolist()),
            "Input batch samples captured records with replacement")
    return {"path": str(path.resolve()), "sha256": recorded["sha256"],
            "bytes": path.stat().st_size, "indices_sha256": recorded["indices_sha256"]}


def audit_new(directory, manifest_path=None):
    directory = Path(directory)
    summary_path = directory / "profile-summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary.get("initial_profiler_session_discarded") is True, "First-use profiler warmup not declared")
    require(len(summary["rows"]) == 16, "Expected16 new rank/variant profile rows")
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    if manifest:
        digest = canonical_hash({k: v for k, v in manifest.items() if k != "freeze"})
        require(manifest["freeze"]["sha256"] == digest and manifest["phase"] == "validation"
                and manifest["split"] == "pilot", "Expected intact frozen pilot validation manifest")
    rows, original, trace_files = {}, {}, set()
    blocks = {r["block_id"] for r in summary["rows"]}
    require(len(blocks) == 1, "Profile comparison must use one declared pilot bank")
    for row in summary["rows"]:
        key = (row["batch_tokens"], row["domain"], row["rank"], row["kernel_variant"])
        require(key not in rows, "Duplicate new rank profile")
        require(row["phase"] == "profiling" and row["measurement_kind"] == "gpu_kernel_diagnostic"
                and row["evidence_eligible"] is False and row["correctness_pass"] is True,
                "Profile must be a correct diagnostic, not primary timing evidence")
        require(row["world_size"] == 2 and row["profile_forwards"] == 8
                and row["profile_raw_range_half_open"] == [16, 24]
                and row["conditioning_forwards"] == 16, "Changed profile execution geometry")
        require(row["kernel_config_rank_agreement"] and row["initial_profiler_session_discarded"],
                "Missing configuration consensus or profiler warmup")
        override = row["kernel_override_config"]
        require(canonical_hash(override) == row["kernel_config_sha256"], "Kernel configuration fingerprint mismatch")
        if manifest:
            require(row["config_hash"] == digest, "Profile does not match declared manifest")
            selection = manifest["selected_configs"][str(row["batch_tokens"])]
            expected_id = "DEFAULT" if row["kernel_variant"] == "DEFAULT" else selection["candidate_id"]
            expected_config = None if row["kernel_variant"] == "DEFAULT" else selection["config"]
            require(row["candidate_id"] == expected_id and override == expected_config,
                    "Profile candidate differs from frozen selection")
            for field in ("optimization_sha256", "followup_sha256", "legacy_harness_sha256"):
                require(row[field] == manifest["environment"][field], f"Profile source pin mismatch: {field}")
        resolved = row["resolved_kernel_config"]
        if override is not None:
            require(resolved == override, "Explicit configuration did not resolve exactly")
        else:
            expected = {"BLOCK_SIZE_M": 64 if row["batch_tokens"] == 128 else 128,
                        "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
                        "num_warps": 4 if row["batch_tokens"] == 128 else 8, "num_stages": 3}
            require(resolved == expected, "DEFAULT is not the declared untuned native fallback")
        path = directory / Path(row["trace_file"]).name
        require(path.name not in trace_files, "Reused trace file for different profile rows")
        trace_files.add(path.name)
        trace = parse_trace(path, 8, row["trace_sha256"])
        require(set(trace["kernel_groups"]) == set(row["kernel_event_groups"]), "Kernel categories differ from raw trace")
        for category, values in trace["kernel_groups"].items():
            recorded = row["kernel_event_groups"][category]
            require(values["calls"] == recorded["calls"] and close(values["total_us"], recorded["total_us"])
                    and close(values["us_per_forward"], recorded["us_per_forward"]),
                    "Recorded kernel sums differ from independent raw-event calculation")
        for launch in trace["gemm_launches"]:
            require(launch["block"] is not None and math.prod(launch["block"]) == 32 * resolved["num_warps"],
                    "Actual GEMM launch thread count differs from resolved warp count")
        physical = physical_diagnostics(row["physical_counts"], row["batch_tokens"], resolved["BLOCK_SIZE_M"], 8)
        for field in ("physical_counts_sha256", "owner_assignments_per_step",
                      "active_physical_experts_per_rank_per_step", "local_padded_rows_per_step"):
            require(physical[field] == row[field], f"Raw physical-count recomputation mismatch: {field}")
        require(all(close(a, b) for a, b in zip(physical["padded_to_useful_rows_by_rank"],
                                               row["padded_to_useful_rows_by_rank"])), "Padding ratio mismatch")
        artifact = _verify_input_artifact(directory, row)
        rows[key] = {"batch_tokens": key[0], "domain": key[1], "rank": key[2], "kernel_variant": key[3],
                     "candidate_id": row["candidate_id"], "resolved_kernel_config": resolved,
                     "raw_kernel_audit": trace, "physical_diagnostics": physical, "input_artifact": artifact}
        original[key] = row
    expected = set(itertools.product((128, 1024), (0, 1), (0, 1), ("DEFAULT", "TUNED")))
    require(set(rows) == expected, "New profiles do not cover exact2batches/2domains/2ranks/2variants")
    comparisons = []
    for batch, domain, rank in itertools.product((128, 1024), (0, 1), (0, 1)):
        default, tuned = original[(batch, domain, rank, "DEFAULT")], original[(batch, domain, rank, "TUNED")]
        for field in ("source_map_sha256", "source_weights_sha256", "capture_sha256", "replay_sha256",
                      "source_prompt_ids", "allocation_id", "hostname", "process_run_id", "physical_counts_sha256",
                      "physical_counts", "variant_order", "config_hash"):
            require(default[field] == tuned[field], f"DEFAULT/TUNED comparison is not paired: {field}")
        peer = original[(batch, domain, 1 - rank, "DEFAULT")]
        require(default["physical_counts_sha256"] == peer["physical_counts_sha256"]
                and default["source_map_sha256"] == peer["source_map_sha256"], "Ranks disagree on shared map/dispatch")
        dr, tr = rows[(batch, domain, rank, "DEFAULT")], rows[(batch, domain, rank, "TUNED")]
        dgroups, tgroups = dr["raw_kernel_audit"]["kernel_groups"], tr["raw_kernel_audit"]["kernel_groups"]
        ratios = {category: {"default_us_per_forward": dgroups[category]["us_per_forward"],
                             "tuned_us_per_forward": tgroups[category]["us_per_forward"],
                             "tuned_over_default": tgroups[category]["us_per_forward"] / dgroups[category]["us_per_forward"]}
                  for category in dgroups if dgroups[category]["us_per_forward"] > 0}
        dp, tp = dr["physical_diagnostics"], tr["physical_diagnostics"]
        comparisons.append({"batch_tokens": batch, "domain": domain, "rank": rank,
                            "selected_candidate": tuned["candidate_id"], "kernel_residence": ratios,
                            "default_padded_rows_per_input": dp["mean_padded_rows_by_rank"][rank],
                            "tuned_padded_rows_per_input": tp["mean_padded_rows_by_rank"][rank],
                            "padded_rows_tuned_over_default": tp["mean_padded_rows_by_rank"][rank] / dp["mean_padded_rows_by_rank"][rank],
                            "active_experts_per_input": dp["mean_active_physical_experts_by_rank"][rank]})
    return {"summary_file": str(summary_path.resolve()), "summary_sha256": file_hash(summary_path),
            "manifest_checked": manifest is not None,
            "manifest_sha256": manifest["freeze"]["sha256"] if manifest else None,
            "rows": [rows[key] for key in sorted(rows)], "paired_rank_comparisons": comparisons,
            "checks": {"trace_sha256": 16, "input_artifact_sha256": 16, "physical_count_conservation": 16,
                       "padding_recomputed": 16, "actual_launch_warp_count": 256,
                       "same_state_variant_pairs": 8, "nested_annotations_excluded": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-profile-dir", type=Path)
    parser.add_argument("--old-evidence-json", type=Path, help="Optional previously saved old trace-hash inventory")
    parser.add_argument("--new-profile-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Frozen pilot validation manifest for new profiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.old_profile_dir is not None or args.new_profile_dir is not None, "Supply old and/or new profiles")
    require(not args.output.exists(), "Refusing to overwrite audit output")
    result = {"schema_version": 1, "status": "pass", "descriptive_only": True,
              "old": audit_old(args.old_profile_dir, args.old_evidence_json) if args.old_profile_dir else None,
              "new": audit_new(args.new_profile_dir, args.manifest) if args.new_profile_dir else None,
              "limits": ["Device kernel sums are residence times, not exclusive wall-time shares or a bandwidth measurement.",
                         "Padding is an integer tile-row proxy, not a measured instruction count or promised speedup.",
                         "Only new DEFAULT/TUNED runs share inputs/maps; old/new calibration datasets are not paired.",
                         "No confidence intervals, adaptive selection or policy recommendation is computed.",
                         "Output correctness is a recorded runtime attestation; this offline audit does not rerun expert weights."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "old_traces": len(result["old"]["rows"]) if result["old"] else 0,
                      "new_traces": len(result["new"]["rows"]) if result["new"] else 0}))


if __name__ == "__main__":
    main()
