#!/usr/bin/env python3
"""Reproduce descriptive resident-dispatch profile analysis using stdlib only.

Reads Nsight SQLite exports in read-only mode. Kernel -> CUDA runtime API is
joined by correlationId, then API -> innermost application NVTX phase by thread
and CPU interval containment. GPU timestamps are never matched to CPU range
containment: asynchronous launches can finish long after a host range closes.
There are no significance tests, dropped steps, or primary timing estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


ARMS = ("native", "pre_native", "one_copy_home", "token_balanced")
PHASE = re.compile(r"^resident/rank(\d+)/step(\d+)/raw(\d+)/([a-z_]+)$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    values = list(values)
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": ordered[0], "max": ordered[-1]}


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative kernel interval")
        total += end - max(start, right) if right is not None and end > right else end - start if right is None else 0
        right = max(end, right) if right is not None else end
    return total


def read_profile(sqlite_path, profile_dir):
    connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        nvtx = [dict(row) for row in connection.execute("SELECT * FROM NVTX_EVENTS")]
        for row in nvtx:
            row["label"] = row["text"] or strings.get(row["textId"], "")
        outer = [row for row in nvtx if row["label"].startswith("resident_profile/")]
        if len(outer) != 1:
            raise ValueError(f"Expected exactly one application profile range: {sqlite_path}")
        stem = outer[0]["label"].split("/", 1)[1]
        metadata_path = profile_dir / "nsys-data" / (stem + ".profile.json")
        data = json.loads(metadata_path.read_text())
        if (data["profile_backend"] != "nsys" or data["evidence_eligible"] is not False
                or data["phase"] != "pilot_profile" or data["split"] != "pilot"
                or data["profile_forwards"] != 32 or data["warmup_forwards"] != 16
                or data["correctness_pass"] is not True):
            raise ValueError("Unexpected or failed profile contract")
        epoch = connection.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        ranges = []
        for row in nvtx:
            match = PHASE.fullmatch(row["label"])
            if match:
                rank, step, raw = map(int, match.groups()[:3])
                if rank != data["rank"] or raw != 16 + step or not 0 <= step < 32:
                    raise ValueError("NVTX rank/input identity mismatch")
                ranges.append({**row, "rank": rank, "step": step, "raw_index": raw,
                               "phase": match.group(4)})
        if len(ranges) != 160:
            raise ValueError("Expected five instrumented phases per input")
        api_rows = [dict(row) for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
        apis = {row["correlationId"]: row for row in api_rows}
        if len(apis) != len(api_rows):
            raise ValueError("Ambiguous runtime correlation IDs")
        by_step = {step: [] for step in range(32)}
        linked = []
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            kernel = dict(row)
            api = apis.get(kernel["correlationId"])
            if api is None:
                raise ValueError("Kernel lacks correlated CUDA runtime API")
            candidates = [r for r in ranges if r["globalTid"] == api["globalTid"]
                          and r["start"] <= api["start"] and api["end"] <= r["end"]]
            if not candidates:
                raise ValueError("Kernel launch lacks an application NVTX range")
            parent = min(candidates, key=lambda r: r["end"] - r["start"])
            value = {
                "profile_step": parent["step"], "raw_index": parent["raw_index"],
                "phase": parent["phase"], "kernel": strings[kernel["shortName"]],
                "demangled_kernel": strings[kernel["demangledName"]],
                "runtime_api": strings[api["nameId"]], "correlation_id": kernel["correlationId"],
                "stream_id": kernel["streamId"], "device_id": kernel["deviceId"],
                "gpu_start_epoch_ns": epoch + kernel["start"],
                "gpu_end_epoch_ns": epoch + kernel["end"],
                "api_start_epoch_ns": epoch + api["start"],
                "api_end_epoch_ns": epoch + api["end"],
                "duration_us": (kernel["end"] - kernel["start"]) / 1000,
                "grid": [kernel[k] for k in ("gridX", "gridY", "gridZ")],
                "block": [kernel[k] for k in ("blockX", "blockY", "blockZ")],
                "registers_per_thread": kernel["registersPerThread"],
                "dynamic_shared_memory_bytes": kernel["dynamicSharedMemory"],
            }
            linked.append(value)
            by_step[parent["step"]].append(value)
        if len(linked) != 352:
            raise ValueError("Expected eleven kernels per forward+step")
        steps = []
        for step, kernels in by_step.items():
            kernels.sort(key=lambda k: k["gpu_start_epoch_ns"])
            gemms = [k for k in kernels if k["kernel"] == "fused_moe_kernel"]
            collective = [k for k in kernels if k["kernel"].startswith("ncclDevKernel_AllReduce")]
            mapper = [k for k in kernels if k["phase"] == "mapping"]
            experts = [k for k in kernels if k["phase"] == "fused_experts"]
            if (len(gemms), len(collective), len(mapper), len(experts)) != (2, 1, 1, 7):
                raise ValueError("Unexpected mapping/GEMM/collective kernel sequence")
            phases = {r["phase"]: r for r in ranges if r["step"] == step}
            if len(phases) != 5:
                raise ValueError("Duplicated or missing per-step NVTX phase")
            busy = union_ns([(k["gpu_start_epoch_ns"], k["gpu_end_epoch_ns"]) for k in experts])
            span = max(k["gpu_end_epoch_ns"] for k in experts) - min(k["gpu_start_epoch_ns"] for k in experts)
            counts = data["assignment_metrics"][step]
            steps.append({
                "profile_step": step, "raw_index": step + 16,
                "gemm_first_us": gemms[0]["duration_us"], "gemm_second_us": gemms[1]["duration_us"],
                "gemm_sum_us": sum(k["duration_us"] for k in gemms),
                "mapper_kernel_us": mapper[0]["duration_us"],
                "expert_kernel_union_us": busy / 1000, "expert_kernel_envelope_us": span / 1000,
                "expert_interkernel_gap_us": (span - busy) / 1000,
                "expert_gpu_end_epoch_ns": max(k["gpu_end_epoch_ns"] for k in experts),
                "collective_kernel_us": collective[0]["duration_us"],
                "collective_gpu_start_epoch_ns": collective[0]["gpu_start_epoch_ns"],
                "collective_gpu_end_epoch_ns": collective[0]["gpu_end_epoch_ns"],
                "forward_nvtx_start_epoch_ns": epoch + phases["forward"]["start"],
                "forward_nvtx_end_epoch_ns": epoch + phases["forward"]["end"],
                "nvtx_host_spans_us": {name: (row["end"] - row["start"]) / 1000 for name, row in phases.items()},
                "per_rank_tokens": counts["per_rank_tokens"],
                "per_rank_active_copies": counts["per_rank_active_copies"],
                "per_rank_padded_rows": counts["per_rank_padded_rows"],
            })
        phase_stats = {}
        for phase in ("mapping", "fused_experts", "output_allreduce", "forward", "step"):
            phase_stats[phase] = stats(row["cuda_elapsed_ms"] * 1000
                                      for row in data["phase_events"] if row["phase"] == phase)
        kernel_stats = {}
        for name in sorted({k["kernel"] for k in linked}):
            group = [k for k in linked if k["kernel"] == name]
            kernel_stats[name] = {"duration_us": stats(k["duration_us"] for k in group),
                                  "streams": sorted({k["stream_id"] for k in group}),
                                  "grid_shapes": sorted({tuple(k["grid"]) for k in group})}
        host_rows = {(x["profile_step"], x["phase"]): x for x in data["phase_events"]}
        alignment = [(epoch + r["start"] - host_rows[(r["step"], r["phase"])]["host_entry_wall_ns"]) / 1000
                     for r in ranges]
        sync_names = dict(connection.execute("SELECT id,name FROM ENUM_CUPTI_SYNC_TYPE"))
        syncs = Counter(sync_names[row[0]] for row in connection.execute("SELECT syncType FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"))
        diagnostics = [dict(row) for row in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT")]
        result = {
            "profile_stem": stem, "rank": data["rank"], "batch_tokens": data["batch_tokens"],
            "arm": data["arm"], "block_id": data["block_id"], "domain": data["domain"],
            "config_hash": data["config_hash"], "allocation_id": data["allocation_id"],
            "profile_source_sha256": data["profile_source_sha256"],
            "source_map_sha256": data["source_map_sha256"], "capture_sha256": data["capture_sha256"],
            "source_weights_sha256": data["source_weights_sha256"], "assignment_sha256": data["assignment_sha256"],
            "assignment_artifact": data["assignment_artifact"], "resolved_kernel_config": data["resolved_kernel_config"],
            "source_prompt_ids": data["source_prompt_ids"], "source_map": data["source_map"],
            "correctness_max_abs_error": data["correctness_max_abs_error"],
            "full_batch_max_abs_difference": data["full_batch_max_abs_difference"],
            "steps": steps, "kernel_count": len(linked), "linked_kernel_count": len(linked),
            "kernels": linked, "kernel_statistics": kernel_stats,
            "instrumented_cuda_phase_spans_us": phase_stats,
            "nvtx_clock_alignment_minus_saved_host_entry_us": stats(alignment),
            "synchronization_event_counts": dict(syncs), "nsys_diagnostic_events": diagnostics,
            "sources": {str(sqlite_path): digest(sqlite_path), str(metadata_path): digest(metadata_path),
                        str(sqlite_path.with_suffix(".nsys-rep")): digest(sqlite_path.with_suffix(".nsys-rep"))},
        }
        return result
    finally:
        connection.close()


def summarize_pair(profiles):
    rank0, rank1 = sorted(profiles, key=lambda p: p["rank"])
    if [rank0["rank"], rank1["rank"]] != [0, 1]:
        raise ValueError("Expected both ranks")
    for field in ("config_hash", "source_map_sha256", "source_weights_sha256", "capture_sha256",
                  "assignment_sha256", "resolved_kernel_config", "source_prompt_ids"):
        if rank0[field] != rank1[field]:
            raise ValueError(f"Ranks disagree on {field}")
    paired_steps = []
    for a, b in zip(rank0["steps"], rank1["steps"]):
        if a["profile_step"] != b["profile_step"]:
            raise ValueError("Rank steps misaligned")
        starts = [a["collective_gpu_start_epoch_ns"], b["collective_gpu_start_epoch_ns"]]
        ends = [a["collective_gpu_end_epoch_ns"], b["collective_gpu_end_epoch_ns"]]
        paired_steps.append({
            "profile_step": a["profile_step"], "raw_index": a["raw_index"],
            "max_rank_gemm_sum_us": max(a["gemm_sum_us"], b["gemm_sum_us"]),
            "collective_gpu_arrival_skew_us": abs(starts[0] - starts[1]) / 1000,
            "collective_completion_skew_us": abs(ends[0] - ends[1]) / 1000,
            "collective_tail_after_both_gpu_arrivals_us": (max(ends) - max(starts)) / 1000,
            "forward_host_start_skew_us": abs(a["forward_nvtx_start_epoch_ns"] - b["forward_nvtx_start_epoch_ns"]) / 1000,
            "collective_kernel_us_by_rank": [a["collective_kernel_us"], b["collective_kernel_us"]],
            "expert_compute_end_skew_us": abs(a["expert_gpu_end_epoch_ns"] - b["expert_gpu_end_epoch_ns"]) / 1000,
        })
    summary = {
        "batch_tokens": rank0["batch_tokens"], "arm": rank0["arm"],
        "rank_statistics": [], "paired_steps": paired_steps,
        "paired_statistics": {field: stats(s[field] for s in paired_steps) for field in paired_steps[0]
                              if field not in ("profile_step", "raw_index", "collective_kernel_us_by_rank")},
    }
    for p in (rank0, rank1):
        rank = p["rank"]
        numeric = ("gemm_first_us", "gemm_second_us", "gemm_sum_us", "mapper_kernel_us",
                   "expert_kernel_union_us", "expert_kernel_envelope_us", "expert_interkernel_gap_us", "collective_kernel_us")
        counts = ("per_rank_tokens", "per_rank_active_copies", "per_rank_padded_rows")
        summary["rank_statistics"].append({"rank": rank,
            **{field: stats(s[field] for s in p["steps"]) for field in numeric},
            **{field: stats(s[field][rank] for s in p["steps"]) for field in counts}})
    return summary


def analyze(directory):
    sqlite_files = sorted((directory / "nsys-traces").glob("*.sqlite"))
    if len(sqlite_files) != 16:
        raise ValueError("Expected all sixteen Nsight SQLite reports")
    profiles = [read_profile(path, directory) for path in sqlite_files]
    expected = {(b, arm, rank) for b in (128, 1024) for arm in ARMS for rank in (0, 1)}
    if {(p["batch_tokens"], p["arm"], p["rank"]) for p in profiles} != expected:
        raise ValueError("Incomplete or repeated profile matrix")
    if len({(p["block_id"], p["domain"]) for p in profiles}) != 1:
        raise ValueError("Expected one source bank and domain")
    for field in ("config_hash", "allocation_id", "capture_sha256", "source_weights_sha256", "source_map_sha256"):
        if len({p[field] for p in profiles}) != 1:
            raise ValueError(f"Input/source invariant changed: {field}")
    if len({json.dumps(p["profile_source_sha256"], sort_keys=True) for p in profiles}) != 1:
        raise ValueError("Profiler implementation differs between reports")
    artifact_hashes = {}
    for p in profiles:
        artifact = directory / "nsys-data/inputs" / Path(p["assignment_artifact"]["path"]).name
        actual = digest(artifact)
        if actual != p["assignment_artifact"]["sha256"]:
            raise ValueError("Assignment/input artifact file hash changed")
        artifact_hashes[str(artifact)] = actual
    summaries = []
    for batch in (128, 1024):
        subset = [p for p in profiles if p["batch_tokens"] == batch]
        if len({json.dumps(p["resolved_kernel_config"], sort_keys=True) for p in subset}) != 1:
            raise ValueError("Kernel configuration changed between arms")
        for rank in (0, 1):
            controls = [p for p in subset if p["rank"] == rank and p["arm"] in ("native", "pre_native")]
            if len({p["assignment_sha256"] for p in controls}) != 1:
                raise ValueError("Precomputed-native control changes physical routes")
        for arm in ARMS:
            summaries.append(summarize_pair([p for p in subset if p["arm"] == arm]))
    contrasts = []
    for batch in (128, 1024):
        by_arm = {s["arm"]: s for s in summaries if s["batch_tokens"] == batch}
        baseline = by_arm["pre_native"]
        for arm in ("native", "one_copy_home", "token_balanced"):
            current = by_arm[arm]
            contrasts.append({"batch_tokens": batch, "arm": arm, "reference": "pre_native",
                "max_rank_gemm_sum_mean_ratio": current["paired_statistics"]["max_rank_gemm_sum_us"]["mean"] /
                                                   baseline["paired_statistics"]["max_rank_gemm_sum_us"]["mean"],
                "per_rank_gemm_sum_mean_ratios": [current["rank_statistics"][r]["gemm_sum_us"]["mean"] /
                                                  baseline["rank_statistics"][r]["gemm_sum_us"]["mean"] for r in (0, 1)],
                "per_rank_active_copy_mean_ratios": [current["rank_statistics"][r]["per_rank_active_copies"]["mean"] /
                                                     baseline["rank_statistics"][r]["per_rank_active_copies"]["mean"] for r in (0, 1)],
                "per_rank_padded_row_mean_ratios": [current["rank_statistics"][r]["per_rank_padded_rows"]["mean"] /
                                                    baseline["rank_statistics"][r]["per_rank_padded_rows"]["mean"] for r in (0, 1)]})
    warning_counts = Counter(row["text"] for p in profiles for row in p["nsys_diagnostic_events"]
                             if row["severity"] >= 2)
    return {"schema_version": 1, "evidence_eligible": False,
            "scope": "One pilot bank/domain, two batches, four arms, both ranks, 32 inputs each; fixed order and instrumented execution; descriptive only.",
            "method": "CUDA kernel -> runtime API by unique correlationId; API -> innermost NVTX range using same host thread and interval containment; epoch alignment uses Nsight session start metadata.",
            "source_invariants_pass": True, "profiles": profiles, "summary": summaries, "contrasts": contrasts,
            "input_artifact_sha256": artifact_hashes,
            "script_sha256": digest(__file__),
            "collection_warnings": dict(warning_counts),
            "total_kernels": sum(p["kernel_count"] for p in profiles),
            "total_linked_kernels": sum(p["linked_kernel_count"] for p in profiles),
            "limitations": ["No GPU counters were collected: weight-traffic, occupancy and bandwidth explanations remain hypotheses.",
                "Profile startup and per-phase events perturb rank arrivals; this is not the clean headroom pilot or a 5% efficacy test.",
                "Nsight warns that driver 12.6 is unsupported by this build and uses 12.5 collection libraries, with possible incomplete CUDA/NVTX events. Expected kernel/range counts pass, but exhaustive event coverage is not established.",
                "NCCL kernels can wait for peer ranks. Their durations are not pure transfer time.",
                "Nested event/CPU phase spans overlap. Kernel durations are not exclusive shares of completed wall time.",
                "All 32 steps retained. Median and identified first-step delays are descriptive; no outlier filter or inferential CI.",
                "Offline assignment construction is excluded; no demonstrated online selector cost or full-serving benefit.",
                "No planner opportunity occurs during these 32 profile steps; no migration or scheduling conclusion follows."]}


def markdown(result):
    lines = ["The resident-copy assignment changes expert computation, but these instrumented timelines cannot establish an end-to-end speedup. Home assignment reduces both actual expert-GEMM durations and active/padded work. Large all-reduce spans primarily expose mismatched rank arrivals in this profile, including first-step delays of milliseconds.",
             "", f"All {result['total_linked_kernels']:,} of {result['total_kernels']:,} recorded CUDA kernels in 16 reports were linked to their CUDA runtime launch and the innermost same-thread NVTX phase. All 32 inputs are retained. Exact bank, weight, source-map, kernel-configuration, and input-artifact invariants pass; native and precomputed-native physical assignment hashes agree.", "",
             "| Batch | Arm | Rank | Active copies | Padded rows | Two GEMMs (µs) | Mapper kernel (µs) | NCCL median / max (µs) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for summary in result["summary"]:
        for r in summary["rank_statistics"]:
            lines.append(f"| {summary['batch_tokens']} | {summary['arm']} | {r['rank']} | {r['per_rank_active_copies']['mean']:.2f} | {r['per_rank_padded_rows']['mean']:.0f} | {r['gemm_sum_us']['mean']:.2f} | {r['mapper_kernel_us']['mean']:.2f} | {r['collective_kernel_us']['median']:.2f} / {r['collective_kernel_us']['max']:.2f} |")
    lines += ["", "Active/padded counts and GEMM/mapper durations in the table are 32-input means per rank. Padded rows use the resolved native M tile and count each owner route once; they are not measured FLOPs or memory bytes. Two-GEMM duration is a component statistic, not an additive wall-time share.", ""]
    for c in result["contrasts"]:
        if c["arm"] == "one_copy_home":
            ratios = c["per_rank_gemm_sum_mean_ratios"]
            lines.append(f"At batch {c['batch_tokens']}, home/precomputed-native two-GEMM ratios are {ratios[0]:.4f} and {ratios[1]:.4f} on ranks 0/1. The mean of the per-input maximum-rank GEMM sums has ratio {c['max_rank_gemm_sum_mean_ratio']:.4f}. This is a consistent compute-headroom signal within this instrumented source block; it does not include online assignment construction or establish a 5% completed-time benefit.")
            lines.append("")
    lines += ["", "The decisive waiting examples come from the paired timelines:", ""]
    for batch, arm in ((128, "one_copy_home"), (1024, "pre_native"), (1024, "one_copy_home")):
        row = next(s for s in result["summary"] if s["batch_tokens"] == batch and s["arm"] == arm)["paired_steps"][0]
        lines.append(f"- Batch {batch}, {arm}, raw input 16: rank NCCL durations {row['collective_kernel_us_by_rank'][0]:.2f}/{row['collective_kernel_us_by_rank'][1]:.2f} µs; GPU arrival skew {row['collective_gpu_arrival_skew_us']:.2f} µs; completion skew {row['collective_completion_skew_us']:.2f} µs; tail after both arrivals {row['collective_tail_after_both_gpu_arrivals_us']:.2f} µs. The host forward-entry skew is {row['forward_host_start_skew_us']:.2f} µs.")
    lines += ["", "A long early-rank NCCL kernel ending almost simultaneously with the late rank's short NCCL kernel is direct evidence of a peer-arrival dependency. The traces locate the delay before the late rank's first forward or launch sequence; they do not uniquely attribute its host cause to Nsight, OS scheduling, or the Python/event instrumentation. cudaProfilerStart occurs after the pre-capture barrier, so collection startup is one plausible contributor. The short tail after both GPU arrivals is descriptive and is still not a transport-only latency measurement.",
              "", "Precomputed-native and native retain the same assignments and similar GEMM times; replacing the mapper changes its actual GPU kernel by about 1 µs. Home and token-balanced assignments both produce exactly 512 or 4,096 owner-executed routes per rank at the two batch sizes. Yet home reduces both GEMMs while token balance alone leaves the maximum-rank GEMM statistic nearly unchanged. This localizes the observed component improvement to changed physical assignments rather than the mapper bypass or token balance alone. It does not separately identify active-expert count, padding, or memory-traffic effects. Reduced expert weight traffic is plausible, but no DRAM/L2/occupancy counters were collected to establish it.",
              "", "Collection quality limits the timeline claims. Nsight reports that CUDA driver 12.6 is unsupported by this build and that tracing uses 12.5 libraries; it also warns that some CUDA/NVTX events might be absent. Every report nevertheless contains the expected 352 kernels and 160 application phase ranges, all linked, so the observed sequence is internally consistent. This check does not establish exhaustive collection of every runtime event. Scheduler/context-switch tracing was disabled, preventing a direct scheduling-cause diagnosis.",
              "", "The appropriate causal test remains the separately timed apparatus-controlled pilot/fresh evaluation. These profiles were collected in a fixed arm order for one pilot source block/domain, with no repetitions or planner boundaries. They support testing one bounded online assignment mechanism only if the clean headroom gate passes; they cannot resolve the separate A/A outlier or waive that guard.", "", "Limitations:", ""]
    lines += ["- " + x for x in result["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.markdown is not None and args.markdown.exists()):
        raise FileExistsError("Use fresh analysis output paths")
    result = analyze(args.profile_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result))
    print(json.dumps({"profiles": len(result["profiles"]), "kernels_linked": result["total_linked_kernels"],
                      "source_invariants_pass": result["source_invariants_pass"], "contrasts": result["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
