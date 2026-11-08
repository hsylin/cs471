"""Run actual multi-GPU layer measurements, once per frozen paired case.

Example (two GPUs):
  torchrun --standalone --nproc-per-node=2 -m skewbench.runner \
    --mode smoke --weights-dir data/qwen-layer0 --output results/smoke.jsonl

Matrix mode requires a frozen manifest. No CPU performance fallback exists.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from .native import NativeLayer, Runtime, Weights, PINNED_COMMIT, fill_logical_histogram
from .traces import TraceConfig, generate_trace

_TRACE_ARTIFACT_CACHE = {}


@lru_cache(maxsize=1)
def cached_trace(config):
    return generate_trace(config)


def persist_raw_trace(rt, trace, hidden):
    """Persist exact replay tensors outside timings, once per trace fingerprint."""
    fingerprint = trace.fingerprint()
    result = None
    if rt.rank == 0:
        if fingerprint in _TRACE_ARTIFACT_CACHE:
            result = _TRACE_ARTIFACT_CACHE[fingerprint]
        else:
            directory = rt.artifact_dir
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{fingerprint}.npz"
            if not target.exists():
                temporary = target.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, expert_ids=trace.expert_ids,
                                        combining_weights=trace.combining_weights,
                                        hidden_states=hidden.float().cpu().numpy(),
                                        metadata=np.asarray(json.dumps(trace.metadata, sort_keys=True)),
                                        trace_sha256=np.asarray(fingerprint))
                temporary.replace(target)
            with np.load(target, allow_pickle=False) as saved:
                digest = hashlib.sha256(str(saved["metadata"]).encode())
                digest.update(memoryview(saved["expert_ids"]).cast("B"))
                digest.update(memoryview(saved["combining_weights"]).cast("B"))
                if digest.hexdigest() != fingerprint or str(saved["trace_sha256"]) != fingerprint:
                    raise AssertionError("Persisted routing tape fingerprint mismatch")
                if not np.array_equal(saved["hidden_states"], hidden.float().cpu().numpy()):
                    raise AssertionError("Persisted hidden states differ from replay inputs")
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result = {"path": str(target), "sha256": digest.hexdigest(),
                      "trace_sha256": fingerprint, "bytes": target.stat().st_size,
                      "hidden_storage": "float32 exact representation of replayed bfloat16 inputs"}
            _TRACE_ARTIFACT_CACHE[fingerprint] = result
    payload = [result]
    rt.torch.distributed.broadcast_object_list(payload, src=0, group=rt.ep_group.cpu_group)
    return payload[0]


def finalize_decision_log(rt, layer):
    """Materialize per-decision committed-map evidence after device completion."""
    local = []
    for decision in layer.decisions:
        row = {}
        for key, value in decision.items():
            if key == "committed_map_device":
                value = value.cpu().numpy()
                row["committed_map"] = value.tolist()
                row["committed_map_sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
            elif key == "physical_load_window_device":
                raw = value.cpu().numpy()
                cursor = decision["physical_window_next_write"]
                chronological = np.concatenate((raw[cursor:], raw[:cursor]), axis=0)
                row["physical_slot_assignments_window"] = chronological.tolist()
                row["physical_slot_assignments_sha256"] = hashlib.sha256(chronological.tobytes()).hexdigest()
            else:
                row[key] = value.tolist() if isinstance(value, np.ndarray) else value
        local.append(row)
    all_ranks = [None] * rt.world_size
    rt.torch.distributed.all_gather_object(all_ranks, local, group=rt.ep_group.cpu_group)
    if any(len(rows) != len(local) for rows in all_ranks):
        raise AssertionError("Ranks disagree on number of admission decisions")
    combined = []
    for index, row in enumerate(local):
        for other in all_ranks:
            peer = other[index]
            for key in ("step", "allowed", "current_map", "proposed_map", "committed_map_sha256",
                        "physical_slot_assignments_sha256"):
                if peer[key] != row[key]:
                    raise AssertionError(f"Ranks disagree on decision {index}: {key}")
        expected_map = row["proposed_map"] if row["allowed"] else row["current_map"]
        if row["committed_map"] != expected_map:
            raise AssertionError("Native committed map differs from admitted map")
        payloads = [other[index]["transfer_sent_bytes_local"] for other in all_ranks]
        row.pop("transfer_sent_bytes_local", None)
        row["transfer_sent_bytes_by_rank"] = payloads
        row["transfer_sent_bytes"] = sum(payloads)
        row["commit_completed"] = True
        row["candidate_changed_slots"] = int(np.count_nonzero(
            np.asarray(row["current_map"]) != np.asarray(row["proposed_map"])))
        physical = np.asarray(row["physical_slot_assignments_window"], dtype=np.int64)
        counts_from_physical = np.zeros(layer.num_experts, dtype=np.int64)
        np.add.at(counts_from_physical, np.asarray(row["current_map"]), physical.sum(axis=0))
        if not np.array_equal(counts_from_physical * rt.world_size, np.asarray(row["current_counts"])):
            raise AssertionError("Physical counts under the old map do not recover native logical counts")
        row["physical_counter_scope"] = "one copy of identical replicated-input counters; rank-owned slot blocks identify actual expert work"
        row["physical_counter_input_multiplicity"] = rt.world_size
        row["physical_owner_assignments_per_step"] = physical.reshape(
            layer.window_steps, rt.world_size, layer.slots).sum(axis=2).tolist()
        row["physical_window_measured_step_range"] = [row["step"] - layer.window_steps + 1, row["step"]]
        combined.append(row)
    return combined


def snapshot_weight_hash(rt, tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                                 sort_keys=True).encode())
        raw = tensor.detach().cpu().contiguous().view(rt.torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
    ranks = [None] * rt.world_size
    rt.torch.distributed.all_gather_object(ranks, digest.hexdigest(), group=rt.ep_group.cpu_group)
    return hashlib.sha256(json.dumps(ranks).encode()).hexdigest(), ranks


def build_jobs(manifest):
    """Randomize paired methods reproducibly; keep all declared cells visible."""
    matrix = manifest["matrix"]
    cases = matrix.get("cases") or [{"regime": r, "trace_variant": "core", "phase_steps": 0}
                                    for r in matrix["regimes"]]
    jobs = []
    repetitions = matrix["repetitions"]
    rep_ids = range(repetitions) if isinstance(repetitions, int) else repetitions
    for seed in matrix["seeds"]:
        for repetition in rep_ids:
            for case in cases:
                for batch in matrix["batch_tokens"]:
                    methods = list(matrix["methods"])
                    order_key = json.dumps([seed, repetition, case, batch], sort_keys=True)
                    rng = random.Random(int(hashlib.sha256(order_key.encode()).hexdigest()[:16], 16))
                    rng.shuffle(methods)
                    for method in methods:
                        jobs.append({**case, "method": method, "seed": seed,
                                     "repetition": repetition, "batch_tokens": batch})
    return jobs


def _trace_config(manifest, job, weights):
    trace = dict(manifest["trace"])
    for key in ("num_experts", "top_k"):
        if int(trace[key]) != int(weights.config[key]):
            raise ValueError(f"Manifest {key} does not match actual checkpoint layer")
    fields = {key: value for key, value in trace.items() if key in TraceConfig.__dataclass_fields__}
    fields.update({"seed": int(job["seed"]), "regime": job["regime"],
                   "tokens_per_step": int(job["batch_tokens"]),
                   "phase_steps": int(job.get("phase_steps", 0))})
    # Manifest num_steps is the measured horizon. Two earlier windows are real,
    # untimed observations shared by all methods and included in the raw tape.
    fields["warmup_steps"] = 2 * int(trace["window_steps"])
    fields["num_steps"] = int(trace["num_steps"]) + fields["warmup_steps"]
    return TraceConfig(**fields)


def _make_layer(rt, weights, manifest, method, seed):
    policy = manifest["policy"]
    return NativeLayer(rt, weights, method=method, replicas=int(policy["replica_budget"]),
                       window_steps=int(manifest["trace"]["window_steps"]),
                       rebalance_interval=int(policy["rebalance_interval"]),
                       epsilon=float(policy["epsilon"]), tau=float(policy["tau"]),
                       cooldown_steps=int(policy["cooldown_steps"]), seed=seed)


def warm_native_transport(rt, weights, manifest):
    """Warm real P2P connections in this process before any measured method.

    The throwaway layer uses the same persistent native EPLB ProcessGroup as
    every subsequent layer. It moves actual checkpoint weights across ranks,
    commits each map, and restores the original arrangement. All initialization
    and validation work here is outside every primary/branch timing interval.
    """
    if rt.rank == 0:
        print(json.dumps({"event": "initialization", "phase": "warm_native_weight_transport"}), flush=True)
    layer = _make_layer(rt, weights, manifest, "B2", seed=0)
    original = layer.map_snapshot()
    rotated = np.roll(original, layer.slots)
    rt.barrier()
    rt.sync()
    started = time.perf_counter()
    try:
        layer.force_migration(rotated)
        layer.assert_maps_and_weights()
        first_bytes = rt.gather_scalars(layer.counter.sent_bytes)
        layer.force_migration(original)
        layer.assert_maps_and_weights()
        if not np.array_equal(layer.map_snapshot(), original):
            raise AssertionError("Transport warmup did not restore the original map")
        layer.close()
        elapsed = rt.gather_scalars(time.perf_counter() - started)
        total_bytes = rt.gather_scalars(layer.counter.sent_bytes)
        reverse_bytes = [total - first for total, first in zip(total_bytes, first_bytes)]
        if any(value <= 0 for value in first_bytes + reverse_bytes):
            raise AssertionError("Transport warmup must send actual payloads in both moves on every rank")
        messages = rt.gather_scalars(layer.counter.sent_messages)
        metadata = {
            "completed": True, "migration_count": 2,
            "sent_bytes": int(sum(total_bytes)),
            "sent_bytes_by_rank": [int(value) for value in total_bytes],
            "forward_sent_bytes_by_rank": [int(value) for value in first_bytes],
            "rollback_sent_bytes_by_rank": [int(value) for value in reverse_bytes],
            "sent_messages": int(sum(messages)),
            "elapsed_seconds_slowest_rank": max(elapsed),
            "replica_budget_extra": layer.replicas,
            "world_size": rt.world_size,
            "shared_native_eplb_process_group_verified": True,
            "original_map_and_weights_restored": True,
            "scope": "one process-local cyclic cross-rank native migration and rollback on a throwaway layer; same persistent EPLB device group as all measured methods; excluded from primary and branch timing",
        }
    finally:
        del layer
        gc.collect()
        rt.sync()
    rt.barrier()
    rt.metadata["process_transport_warmup"] = metadata
    if rt.rank == 0:
        print(json.dumps({"event": "initialization", "phase": "native_weight_transport_warm",
                          "process_transport_warmup": metadata}), flush=True)
    return metadata


def _inputs(rt, weights, trace):
    torch = rt.torch
    generator = torch.Generator(device=rt.device).manual_seed(trace.config.seed)
    hidden = torch.randn((trace.config.tokens_per_step, weights.config["hidden_size"]),
                         generator=generator, device=rt.device, dtype=torch.bfloat16)
    ids = torch.from_numpy(trace.expert_ids.copy()).to(rt.device, torch.int32)
    combine = torch.from_numpy(trace.combining_weights.copy()).to(rt.device, torch.float32)
    return hidden, ids, combine


def _record_base(rt, weights, manifest, config_hash, job, trace):
    return {
        "schema_version": 1, "measurement_kind": "gpu", "status": "incomplete",
        "config_hash": config_hash, "split": manifest["split"], **job,
        "record_id": hashlib.sha256(json.dumps([config_hash, job], sort_keys=True).encode()).hexdigest(),
        "trace_sha256": trace.fingerprint(), "trace_config": asdict(trace.config),
        "policy_config": manifest["policy"], **rt.metadata,
        "weights_source": weights.metadata,
        "evidence_eligible": weights.metadata.get("weights_kind") == "pretrained",
        "hidden_input_source": "seeded Gaussian synthetic hidden states, fixed across paired methods",
        "logical_routing_source": "controlled trace, identical IDs/order/combining weights across paired methods",
        "timing_scope": "completed distributed MoE layer horizon, native EPLB monitoring/planning/transfers/commit included",
        "prefill_decode_scope": "layer token-batch experiment; no full-model serving latency claim",
        "startup_contract": "2W observed untimed steps, forced no migration, then native cadence counter reset to zero",
        "observation_warmup_steps": 2 * int(manifest["trace"]["window_steps"]),
        "raw_trace_steps": trace.config.num_steps,
        "measured_horizon_steps": int(manifest["trace"]["num_steps"]),
        "native_threshold_active": False, "dtype": "bfloat16",
    }


def run_case(rt, weights, manifest, config_hash, job, *, diagnostic_checks=False):
    torch = rt.torch
    trace = cached_trace(_trace_config(manifest, job, weights))
    record = _record_base(rt, weights, manifest, config_hash, job, trace)
    if diagnostic_checks:
        record["measurement_kind"] = "gpu_smoke"
        record["evidence_eligible"] = False
    hidden, ids, combine = _inputs(rt, weights, trace)
    record["trace_artifact"] = persist_raw_trace(rt, trace, hidden)
    layer = _make_layer(rt, weights, manifest, job["method"], job["seed"])
    # Compile/warm kernels without advancing observation windows or placement.
    for _ in range(3):
        layer.forward(hidden, ids[0], combine[0], observe=False)
    rt.sync()
    forced = None
    histogram_proof = None
    if diagnostic_checks:
        histogram = torch.empty(weights.config["num_experts"], dtype=torch.int64, device=rt.device)
        ones = torch.ones(ids[0].numel(), dtype=torch.int64, device=rt.device)
        frames = sorted({0, min(2 * trace.config.window_steps + trace.config.phase_steps,
                               trace.config.num_steps - 1), trace.config.num_steps - 1})
        previous_debug_mode = torch.cuda.get_sync_debug_mode()
        for frame in frames:
            try:
                torch.cuda.set_sync_debug_mode("error")
                fill_logical_histogram(histogram, ids[frame], ones)
            finally:
                torch.cuda.set_sync_debug_mode(previous_debug_mode)
            rt.sync()
            expected = np.bincount(trace.expert_ids[frame].reshape(-1),
                                   minlength=weights.config["num_experts"])
            if not np.array_equal(histogram.cpu().numpy(), expected):
                raise AssertionError("Fixed CUDA histogram differs from independent CPU counts")
        histogram_proof = {"passed": True, "frames_checked": frames,
                           "histogram_update_sync_debug_mode": "error",
                           "prior_debug_mode_restored": previous_debug_mode,
                           "reference": "independent CPU numpy.bincount outside guard and timing"}
        original = layer.map_snapshot()
        before = layer.forward(hidden[:4], ids[0, :4], combine[0, :4], observe=False).clone()
        layer.force_migration(np.roll(original, layer.slots))
        layer.assert_maps_and_weights()
        after = layer.forward(hidden[:4], ids[0, :4], combine[0, :4], observe=False)
        torch.testing.assert_close(before, after, atol=0.02, rtol=0.03)
        forced_bytes = sum(rt.gather_scalars(layer.counter.sent_bytes))
        layer.force_migration(original)
        layer.assert_maps_and_weights()
        before_bytes = layer.counter.sent_bytes
        layer.force_migration(original)  # native unchanged-map rejection contract
        layer.assert_maps_and_weights()
        if layer.counter.sent_bytes != before_bytes:
            raise AssertionError("Unchanged map generated expert payload traffic")
        forced = {"forced_allow_pass": True, "forced_skip_pass": True,
                  "forced_allow_sent_bytes": int(forced_bytes),
                  "scope": "native transfer and commit helper; scheduled admission is checked separately in decision logs"}
        if forced_bytes <= 0:
            raise AssertionError("Forced distributed migration did not send expert tensors")
    warmup_steps = 2 * trace.config.window_steps
    measured_steps = int(manifest["trace"]["num_steps"])
    layer.warming_up = True
    for step in range(warmup_steps):
        layer.forward(hidden, ids[step], combine[step], observe=True)
        layer.step(step)
    layer.warming_up = False
    layer.state.expert_rearrangement_step = 0
    layer.state._update_layer_should_record(log_stats=False)
    layer.planner_seconds = layer.controller_seconds = 0.0
    layer.decisions.clear()
    layer.last_accepted_step = None
    layer.counter.sent_bytes = layer.counter.sent_messages = 0
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(measured_steps)]
    torch.cuda.reset_peak_memory_stats(rt.device)
    rt.barrier()
    rt.sync()
    started = time.perf_counter()
    for step in range(measured_steps):
        trace_step = step + warmup_steps
        events[step][0].record()
        layer.forward(hidden, ids[trace_step], combine[trace_step], observe=True)
        layer.end_of_input = step == measured_steps - 1
        layer.step(step)
        events[step][1].record()
    layer.close()
    elapsed = time.perf_counter() - started
    rank_elapsed = rt.gather_scalars(elapsed)
    rank_step_times = [None] * rt.world_size
    local_step_times = [a.elapsed_time(b) / 1000.0 for a, b in events]
    torch.distributed.all_gather_object(rank_step_times, local_step_times,
                                        group=rt.ep_group.cpu_group)
    step_times = np.max(np.asarray(rank_step_times), axis=0)
    payload_bytes = int(sum(rt.gather_scalars(layer.counter.sent_bytes)))
    sent_messages = int(sum(rt.gather_scalars(layer.counter.sent_messages)))
    planner_seconds = max(rt.gather_scalars(layer.planner_seconds))
    controller_seconds = max(rt.gather_scalars(layer.controller_seconds))
    peak_memory = max(rt.gather_scalars(torch.cuda.max_memory_allocated(rt.device)))
    decision_log = finalize_decision_log(rt, layer)
    # Independent conservation check after timing: every replay token contributes
    # top_k assignments on each replicated-input rank. Never feed this oracle
    # total back into the policy's online observations.
    expected_window_assignments = (trace.config.window_steps * trace.config.tokens_per_step
                                   * trace.config.top_k * rt.world_size)
    observation_checks = 0
    for decision in layer.decisions:
        for key in ("current_counts", "previous_counts"):
            counts = decision.get(key)
            if counts is not None:
                if int(np.sum(counts)) != expected_window_assignments:
                    raise AssertionError(f"{key} does not conserve replay assignments")
                end = warmup_steps + decision["step"] + 1
                if key == "previous_counts":
                    end -= trace.config.window_steps
                expected_vector = trace.counts(end - trace.config.window_steps, end) * rt.world_size
                if not np.array_equal(np.asarray(counts), expected_vector):
                    raise AssertionError(f"{key} differs from exact executed logical window at step {decision['step']}")
                observation_checks += 1
    layer.assert_maps_and_weights()
    correctness = layer.check_outputs(hidden, ids[-1], combine[-1])
    record.update({
        "status": "complete", "correctness_pass": True, **correctness,
        "steps_completed": measured_steps,
        "elapsed_seconds": max(rank_elapsed), "per_rank_elapsed_seconds": rank_elapsed,
        "step_seconds_median": float(np.median(step_times)),
        "step_seconds_p95": float(np.quantile(step_times, .95)),
        "step_seconds_p99": float(np.quantile(step_times, .99)),
        "step_seconds_slowest_rank": step_times.tolist(),
        "migration_bytes": payload_bytes, "migration_sent_messages": sent_messages,
        "migration_bytes_scope": "sum of sender payload tensor bytes posted by native communicator; excludes protocol headers and local copies",
        "planner_seconds": planner_seconds, "controller_seconds": controller_seconds,
        "phase_timing_caveat": "CPU wall subregions can overlap GPU work; do not sum as a latency decomposition",
        "peak_gpu_memory_bytes": int(peak_memory),
        "accepted": sum(d["allowed"] for d in layer.decisions),
        "rejected": sum(not d["allowed"] for d in layer.decisions),
        "decision_log": decision_log, "final_map": layer.map_snapshot().tolist(),
        "replica_budget_actual": layer.replicas,
        "forced_contract_checks": forced,
        "histogram_cuda_validation": histogram_proof,
        "observation_window_assignment_total_expected": expected_window_assignments,
        "observation_count_multiplicity": rt.world_size,
        "observation_conservation_checks": observation_checks,
        "terminal_proposals_suppressed": layer.terminal_proposals_suppressed,
        "end_of_input_contract": "Record final-step load but suppress any new terminal proposal before planning; finish already accepted work",
    })
    del layer, hidden, ids, combine, events
    gc.collect()
    torch.cuda.empty_cache()
    rt.barrier()
    return record


def branch_case(rt, weights, manifest, config_hash, job):
    """Paired real migration versus keep replay from a common saved state.

    The common state is captured from an actual B3 trajectory immediately before
    a native candidate is applied. No later rearrangements occur in either
    future branch. This is a bounded diagnostic, not an oracle serving policy.
    """
    torch = rt.torch
    trace = cached_trace(_trace_config(manifest, job, weights))
    hidden, ids, combine = _inputs(rt, weights, trace)
    trace_artifact = persist_raw_trace(rt, trace, hidden)
    w = trace.config.window_steps
    measured_steps = int(manifest["trace"]["num_steps"])
    cadence = int(manifest["policy"]["rebalance_interval"])
    decision_step = int(job.get("decision_step", cadence - 1))
    if not 0 <= decision_step < measured_steps - 1 or (decision_step + 1) % cadence:
        raise ValueError("Branch decision must be a scheduled measured step with a nonempty future")
    diagnostic_config = manifest.get("diagnostics", {})
    horizon = int(diagnostic_config.get("horizon_steps", diagnostic_config.get("future_steps", 128)))
    if horizon <= 0 or decision_step + 1 + horizon > measured_steps:
        raise ValueError("Branch diagnostic requires its entire declared future horizon")
    scout = _make_layer(rt, weights, manifest, "B3", job["seed"])
    scout.collect_history = True  # diagnostic prefix overhead is outside both branch timings
    scout.warming_up = True
    for step in range(2 * w):
        scout.forward(hidden, ids[step], combine[step], observe=True)
        scout.step(step)
    scout.warming_up = False
    scout.state.expert_rearrangement_step = 0
    scout.state._update_layer_should_record(log_stats=False)
    scout.capture_decision_step = decision_step
    for step in range(decision_step + 1):
        trace_step = 2 * w + step
        scout.forward(hidden, ids[trace_step], combine[trace_step], observe=True)
        scout.step(step)
    rt.sync()
    snapshot = scout.captured_candidate
    if snapshot is None or snapshot["previous_counts"] is None:
        raise AssertionError("Scheduled native candidate was not captured with both complete windows")
    source_map, proposal = snapshot["current_map"], snapshot["proposed_map"]
    current_counts, previous_counts = snapshot["current_counts"], snapshot["previous_counts"]
    absolute_end = 2 * w + decision_step + 1
    if not np.array_equal(current_counts, trace.counts(absolute_end - w, absolute_end) * rt.world_size):
        raise AssertionError("Branch current window differs from executed routing tape")
    if not np.array_equal(previous_counts, trace.counts(absolute_end - 2 * w, absolute_end - w) * rt.world_size):
        raise AssertionError("Branch previous window differs from executed routing tape")
    if not np.array_equal(scout.map_snapshot(), source_map):
        raise AssertionError("Diagnostic intercepted candidate changed the common placement")
    scout.assert_maps_and_weights()
    source_weights = (scout.w13.clone(), scout.w2.clone())
    source_weights_hash, source_rank_hashes = snapshot_weight_hash(rt, source_weights)
    hashes = {"source_map_sha256": hashlib.sha256(source_map.tobytes()).hexdigest(),
              "candidate_map_sha256": hashlib.sha256(proposal.tobytes()).hexdigest(),
              "source_weights_sha256": source_weights_hash,
              "trace_sha256": trace.fingerprint()}
    from .policies import evaluate_admission
    decision = evaluate_admission("stability", current_counts, previous_counts,
                                   source_map, proposal, rt.world_size,
                                   epsilon=float(manifest["policy"]["epsilon"]),
                                   tau=float(manifest["policy"]["tau"]),
                                   step=decision_step, windows_complete=True)
    results = []
    order = ["allow", "skip"]
    random.Random(job["seed"] + job["repetition"]).shuffle(order)
    for branch in order:
        layer = _make_layer(rt, weights, manifest, "B4", job["seed"])
        layer.restore_snapshot(source_map, *source_weights)
        if not torch.equal(layer.w13, source_weights[0]) or not torch.equal(layer.w2, source_weights[1]):
            raise AssertionError("Branch restoration did not reproduce source tensors exactly")
        layer.assert_maps_and_weights()
        future_start = 2 * w + decision_step + 1
        for _ in range(2):
            layer.forward(hidden, ids[future_start], combine[future_start], observe=False)
        rt.barrier()
        rt.sync()
        start = time.perf_counter()
        if branch == "allow":
            layer.force_migration(proposal)
        future_stop = future_start + horizon
        for step in range(future_start, future_stop):
            layer.forward(hidden, ids[step], combine[step], observe=False)
        layer.close()
        elapsed = max(rt.gather_scalars(time.perf_counter() - start))
        migration_bytes = int(sum(rt.gather_scalars(layer.counter.sent_bytes)))
        layer.assert_maps_and_weights()
        correctness = layer.check_outputs(hidden, ids[future_stop - 1], combine[future_stop - 1])
        results.append({"branch": branch, "elapsed_seconds": elapsed,
                        "migration_bytes": migration_bytes, "correctness_pass": True,
                        "future_steps": horizon, **hashes, **correctness})
        del layer
    record = _record_base(rt, weights, manifest, config_hash, job, trace)
    record.update({
        "measurement_kind": "gpu_branch", "status": "complete", "correctness_pass": True,
        "trace_artifact": trace_artifact, **hashes,
        "source_weight_rank_sha256": source_rank_hashes,
        "decision_step": decision_step, "decision_trace_step": 2 * w + decision_step,
        "future_steps": horizon,
        "current_map": source_map.tolist(), "proposed_map": proposal.tolist(),
        "current_counts": current_counts.tolist(), "previous_counts": previous_counts.tolist(),
        "gate_decision": asdict(decision), "branches": results,
        "b3_would_allow": snapshot["b3_would_allow"], "b3_reason": snapshot["b3_reason"],
        "snapshot_scope": "actual B3 trajectory at a native proposal boundary; exact committed map and cloned local weights restored in both branches",
        "timing_scope": "future layer replay plus actual candidate migration in allow branch; shared prefix and candidate planning excluded",
        "future_rebalancing_enabled": False,
    })
    del scout, source_weights, hidden, ids, combine
    gc.collect()
    torch.cuda.empty_cache()
    rt.barrier()
    return record


def _smoke_manifest(weights, args):
    return {
        "schema_version": 1, "split": "pilot", "name": "gpu-smoke",
        "environment": {"vllm_commit": PINNED_COMMIT},
        "matrix": {"methods": [args.method], "cases": [{"regime": args.regime,
                    "trace_variant": "core", "phase_steps": 0}],
                   "batch_tokens": [args.batch_tokens], "seeds": [args.seed], "repetitions": 1},
        "trace": {"num_experts": weights.config["num_experts"], "top_k": weights.config["top_k"],
                  "num_steps": args.num_steps, "window_steps": args.window_steps,
                  "hotspot_lifetime_steps": args.window_steps, "zipf_exponent": 1.2},
        "policy": {"epsilon": .1, "tau": .05, "cooldown_steps": 4 * args.window_steps,
                   "rebalance_interval": 2 * args.window_steps, "replica_budget": args.replicas},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "matrix", "branch"), default="smoke")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--method", default="B4")
    parser.add_argument("--regime", default="periodic")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--window-steps", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rt = Runtime()
    try:
        _execute(args, rt)
    finally:
        rt.close()


def _execute(args, rt):
    weights = Weights(args.weights_dir)
    from .analysis import manifest_hash, verify_frozen_manifest
    if args.config:
        manifest = json.loads(Path(args.config).read_text())
        config_hash = verify_frozen_manifest(manifest)
    else:
        if args.mode == "matrix":
            raise ValueError("Matrix mode requires --config with a frozen manifest")
        manifest = _smoke_manifest(weights, args)
        config_hash = manifest_hash(manifest)
    expected_harness = manifest.get("environment", {}).get("harness_sha256")
    if expected_harness is not None and expected_harness != rt.metadata["harness_sha256"]:
        raise ValueError("Actual benchmark harness differs from frozen environment.harness_sha256")
    jobs = build_jobs(manifest)
    if args.mode == "branch":
        diag = manifest.get("diagnostics", {})
        if any(key in diag for key in ("cases", "batch_tokens", "seeds", "repetitions")):
            diagnostic_manifest = dict(manifest)
            diagnostic_matrix = dict(manifest["matrix"])
            for key in ("cases", "batch_tokens", "seeds", "repetitions"):
                if key in diag:
                    diagnostic_matrix[key] = diag[key]
            diagnostic_manifest["matrix"] = diagnostic_matrix
            jobs = build_jobs(diagnostic_manifest)
        # One same-state branch pair per trace/repetition, not one per policy.
        seen = set()
        unique = []
        for job in jobs:
            key = tuple((k, v) for k, v in job.items() if k != "method")
            if key not in seen:
                seen.add(key)
                steps = manifest.get("diagnostics", {}).get("decision_steps", [
                    int(manifest["policy"]["rebalance_interval"]) - 1,
                    2 * int(manifest["policy"]["rebalance_interval"]) - 1,
                ])
                unique.extend({**job, "method": "BRANCH", "decision_step": int(step)}
                              for step in steps if int(step) < int(manifest["trace"]["num_steps"]) - 1)
        jobs = unique
    output = Path(args.output)
    rt.artifact_dir = output.parent / "traces"
    completed = set()
    if args.resume and output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "complete" and row.get("correctness_pass"):
                completed.add(row.get("record_id"))
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    if rt.rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    # A separate preflight torchrun does not initialize these process-local
    # NCCL P2P connections. Warm them here, even when resuming or running branches.
    warm_native_transport(rt, weights, manifest)
    for index, job in enumerate(jobs):
        record_id = hashlib.sha256(json.dumps([config_hash, job], sort_keys=True).encode()).hexdigest()
        if record_id in completed:
            continue
        if rt.rank == 0:
            print(json.dumps({"event": "start", "job": index + 1, "total": len(jobs), **job}), flush=True)
        try:
            if args.mode == "branch":
                record = branch_case(rt, weights, manifest, config_hash, job)
            else:
                record = run_case(rt, weights, manifest, config_hash, job,
                                  diagnostic_checks=args.mode == "smoke")
        except Exception as error:
            if rt.rank == 0:
                failure = {"schema_version": 1, "measurement_kind": "gpu", "status": "failed",
                           "config_hash": config_hash, "split": manifest["split"], **job,
                           "record_id": record_id, "correctness_pass": False,
                           "error": repr(error), "traceback": traceback.format_exc()}
                with output.open("a") as handle:
                    handle.write(json.dumps(failure) + "\n")
            raise
        if rt.rank == 0:
            with output.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps({"event": "complete", "record_id": record["record_id"],
                              "elapsed_seconds": record.get("elapsed_seconds"),
                              "migration_bytes": record.get("migration_bytes")}), flush=True)
    rt.barrier()


if __name__ == "__main__":
    main()
