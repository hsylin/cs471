"""Fixed-map, captured-input tuning trials for the bounded kernel candidate set."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np

from followup.router_tapes import load_router_bank
from followup.payback import _materialize
from optimization.common import agree_config, array_hash, canonical_hash, make_layer, manifest_hash
from optimization.kernel_config import CANDIDATES, config_fingerprint, kernel_override, resolve_kernel_config
from skewbench.runner import snapshot_weight_hash

WARMUP_FORWARDS = 16
TIMED_FORWARDS = 32
INPUT_SEED = 10471


def input_indices(domain_ids, block_id, domain, batch_tokens):
    """Forty-eight batches; no replacement within a batch, reuse across inputs."""
    key = json.dumps([str(block_id), int(domain), int(batch_tokens)], separators=(",", ":"))
    seed = (INPUT_SEED + int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)) % (2 ** 64)
    pool = np.flatnonzero(np.asarray(domain_ids) == domain)
    if batch_tokens <= 0 or len(pool) < batch_tokens:
        raise ValueError("Insufficient captured domain records for without-replacement batch")
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.choice(pool, size=batch_tokens, replace=False)
                        for _ in range(WARMUP_FORWARDS + TIMED_FORWARDS)]).astype(np.int64)
    return indices, seed


def tuning_groups(manifest):
    matrix = manifest["matrix"]
    candidates = list(matrix["candidate_ids"])
    if candidates != list(CANDIDATES) or manifest["tuning"]["candidate_configs"] != CANDIDATES:
        raise ValueError("Tuning candidate order/configuration differs from bounded implementation")
    if manifest["tuning"]["forwards_per_trial"] != TIMED_FORWARDS:
        raise ValueError("This tuning protocol requires32 timed forwards")
    groups = [dict(zip(("block_id", "domain", "batch_tokens", "repetition"), values))
              for values in itertools.product(matrix["block_ids"], matrix["domains"],
                                               matrix["batch_tokens"], range(matrix["repetitions"]))]
    orders = [candidates[i % len(candidates):] + candidates[:i % len(candidates)]
              for i in range(len(groups))]
    random.Random(manifest["execution"]["order_seed"]).shuffle(orders)
    for group, order in zip(groups, orders):
        group["candidate_order"] = order
    return groups


def _persist_indices(rt, output, bank, indices, seed, group):
    directory = output.parent / "tuning-inputs"
    metadata = {"schema": "optimization_tuning_inputs_v1", "block_id": group["block_id"],
                "domain": group["domain"], "batch_tokens": group["batch_tokens"],
                "capture_sha256": bank.sha256, "source_prompt_ids": bank.metadata["source_prompt_ids"],
                "seed": seed, "warmup_range_half_open": [0, WARMUP_FORWARDS],
                "timed_range_half_open": [WARMUP_FORWARDS, WARMUP_FORWARDS + TIMED_FORWARDS],
                "sampling": "without replacement within each batch; captured records may repeat across inputs",
                "selection_independent_of_candidate_and_repetition": True}
    identity = canonical_hash(metadata)
    path = directory / f"{identity}.npz"
    result = None
    if rt.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, indices=indices,
                                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            temporary.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["indices"], indices)
                    or json.loads(str(saved["metadata"])) != metadata):
                raise AssertionError("Saved tuning inputs differ from deterministic capture selection")
        result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "bytes": path.stat().st_size, "indices_sha256": array_hash(indices),
                  "metadata": metadata}
    payload = [result]
    rt.torch.distributed.broadcast_object_list(payload, src=0, group=rt.ep_group.cpu_group)
    return payload[0]


def _validate_bank(bank, entry, manifest, weights):
    if bank.sha256 != entry["capture_sha256"] or bank.metadata.get("correctness_pass") is not True:
        raise ValueError("Capture hash or native correctness attestation mismatch")
    expected = {"block_id": entry["block_id"], "split": "pilot",
                **{field: weights.config[field] for field in (
                    "source_model", "source_revision", "layer", "num_experts", "top_k", "hidden_size")}}
    for field, value in expected.items():
        if bank.metadata.get(field) != value:
            raise ValueError(f"Capture differs from tuning/checkpoint identity: {field}")
    block = next(b for b in manifest["blocks"] if b["block_id"] == entry["block_id"])
    if bank.metadata["source_prompt_ids"] != block["source_document_ids"]:
        raise ValueError("Capture source documents differ from frozen tuning block")


def _write_row(rt, output, row):
    if rt.rank == 0:
        with output.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def run_tuning(rt, weights, manifest, args):
    """Run the complete declared tuning matrix; Runtime metadata is pre-attested."""
    from optimization.common import build_references, check_references
    if manifest["phase"] != "tuning" or manifest["split"] != "pilot":
        raise ValueError("Kernel candidate search is restricted to the tuning pilot")
    digest = manifest_hash(manifest)
    if manifest.get("freeze", {}).get("sha256") != digest:
        raise ValueError("Tuning manifest must be frozen")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite/append tuning output: {output}")
    if rt.rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    groups = tuning_groups(manifest)
    cells = {}
    for group in groups:
        key = (group["block_id"], group["domain"], group["batch_tokens"])
        cells.setdefault(key, []).append(group)
    banks = {entry["block_id"]: entry for entry in manifest["banks"]}
    base = Path(args.banks_dir)
    rows_written = 0
    for (block, domain, batch), repetitions in cells.items():
        cell_setup_started = time.perf_counter()
        entry = banks[block]
        path = Path(entry["capture_path"])
        bank = load_router_bank(path if path.is_absolute() else base / path)
        _validate_bank(bank, entry, manifest, weights)
        indices, seed = input_indices(bank.domain_ids, block, domain, batch)
        artifact = _persist_indices(rt, output, bank, indices, seed, repetitions[0])
        tensors = _materialize(rt, SimpleNamespace(bank=bank, indices=indices))
        hidden, ids, combining = tensors
        layer = make_layer(rt, weights, manifest, "SKIP")
        layer.assert_maps_and_weights()
        source_map = layer.map_snapshot()
        source_weight_hash, source_rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
        with kernel_override(None):
            default_config = resolve_kernel_config(layer, batch)
            agree_config(rt, default_config)
            references = build_references(rt, layer, tensors,
                                          [WARMUP_FORWARDS, WARMUP_FORWARDS + TIMED_FORWARDS - 1])
        cell_setup_seconds = max(rt.gather_scalars(time.perf_counter() - cell_setup_started))
        verified = {}
        for group in repetitions:
            for order_position, candidate in enumerate(group["candidate_order"]):
                config = CANDIDATES[candidate]
                identity = {"config_hash": digest, **group, "candidate_id": candidate}
                row = {"schema_version": 1, "optimization_schema_version": 1,
                       "phase": "tuning", "split": "pilot", "measurement_kind": "gpu_kernel_tuning",
                       "evidence_eligible": True, **rt.metadata, **identity,
                       "record_id": canonical_hash(identity), "candidate_order_position": order_position,
                       "kernel_config_id": candidate, "kernel_override_config": config,
                       "kernel_config_sha256": config_fingerprint(config),
                       "default_resolved_kernel_config": default_config,
                       "source_map": source_map.tolist(), "source_map_sha256": array_hash(source_map),
                       "source_weights_sha256": source_weight_hash, "source_weight_rank_sha256": source_rank_hashes,
                       "weights_source": weights.metadata, "source_prompt_ids": bank.metadata["source_prompt_ids"],
                       "capture_sha256": bank.sha256, "replay_sha256": artifact["sha256"],
                       "input_artifact": artifact, "raw_input_positions": len(indices),
                       "cell_setup_seconds_excluded": cell_setup_seconds,
                       "cell_setup_scope": "bank verification, saved input selection, materialization, layer/map/weight verification and DEFAULT references; shared across cell trials",
                       "warmup_forwards": WARMUP_FORWARDS, "timed_range_half_open": [16, 48],
                       "policy": "SKIP", "native_observation_enabled": False,
                       "native_rebalance_steps_called": 0, "replica_budget_actual": manifest["policy"]["replica_budget"],
                       "timing_scope": "32 complete fixed-map forwards, actual captured inputs, slowest rank; conditioning/reference/materialization excluded",
                       "input_scope": "same saved48 batches for every candidate and technical repetition;16 conditioning then32 timed"}
                if rt.rank == 0:
                    print(json.dumps({"event": "tuning_start", "block_id": block, "domain": domain,
                                      "batch_tokens": batch, "candidate": candidate,
                                      "repetition": group["repetition"]}), flush=True)
                try:
                    with kernel_override(config):
                        candidate_setup_started = time.perf_counter()
                        row["resolved_kernel_config"] = resolve_kernel_config(layer, batch)
                        agree_config(rt, row["resolved_kernel_config"])
                        checked_now = candidate not in verified
                        if checked_now:
                            verified[candidate] = check_references(rt, layer, tensors, references)
                        rt.sync()
                        candidate_setup_seconds = max(rt.gather_scalars(time.perf_counter() - candidate_setup_started))
                        conditioning_started = time.perf_counter()
                        for step in range(WARMUP_FORWARDS):
                            layer.forward(hidden[step], ids[step], combining[step], observe=False)
                        rt.sync()
                        conditioning_seconds = max(rt.gather_scalars(time.perf_counter() - conditioning_started))
                        rt.barrier()
                        rt.sync()
                        started = time.perf_counter()
                        for step in range(WARMUP_FORWARDS, WARMUP_FORWARDS + TIMED_FORWARDS):
                            layer.forward(hidden[step], ids[step], combining[step], observe=False)
                        layer.close()
                        elapsed = time.perf_counter() - started
                        rank_times = rt.gather_scalars(elapsed)
                    if not np.array_equal(layer.map_snapshot(), source_map):
                        raise AssertionError("Fixed-map tuning changed expert placement")
                    layer.assert_maps_and_weights()
                    if not np.array_equal(ids.cpu().numpy(), bank.expert_ids[indices]):
                        raise AssertionError("Tuning changed captured logical routing IDs")
                    if layer.counter.sent_bytes or layer.counter.sent_messages or layer.decisions:
                        raise AssertionError("Fixed-map tuning performed a native migration/decision")
                    row.update(verified[candidate])
                    row.update(status="complete", correctness_pass=True, steps_completed=TIMED_FORWARDS,
                               elapsed_seconds=max(rank_times), per_rank_elapsed_seconds=rank_times,
                               elapsed_seconds_per_forward=max(rank_times) / TIMED_FORWARDS,
                               migration_bytes=0, source_restore_verified=True, expected_arm_weights_verified=True,
                               logical_routing_preserved=True, kernel_config_rank_agreement=True,
                               candidate_setup_seconds_excluded=candidate_setup_seconds,
                               conditioning_seconds_excluded=conditioning_seconds,
                               commit_completed=True, correctness_checked_this_trial=checked_now,
                               correctness_scope="cached once per candidate/cell on timed endpoints; full DEFAULT outputs plus16 evenly-spaced independent FP32 samples per endpoint")
                    _write_row(rt, output, row)
                    rows_written += 1
                    if rt.rank == 0:
                        print(json.dumps({"event": "tuning_complete", "record_id": row["record_id"],
                                          "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
                except Exception as error:
                    row.update(status="failed", correctness_pass=False, evidence_eligible=False,
                               error=repr(error), traceback=traceback.format_exc())
                    _write_row(rt, output, row)
                    raise
        layer.close()
        del layer, tensors, hidden, ids, combining, references, bank
        gc.collect()
        rt.torch.cuda.empty_cache()
        rt.barrier()
    return {"records_written": rows_written, "cells": len(cells), "output": str(output),
            "profiling_performed": False}


def kernel_event_summary(trace, forwards):
    """Sum actual device kernel events only; exclude nested operator annotations."""
    groups = {}
    for event in trace["traceEvents"]:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name = event["name"]
        category = ("fused_expert_gemm" if name.startswith("fused_moe_kernel") else
                    "nccl_allreduce" if "nccl" in name.lower() and "allreduce" in name.lower() else
                    "physical_mapping" if "_eplb_map_and_record_" in name else "other_kernels")
        group = groups.setdefault(category, {"calls": 0, "total_us": 0.0})
        group["calls"] += 1
        group["total_us"] += float(event["dur"])
    for group in groups.values():
        group["us_per_forward"] = group["total_us"] / forwards
    if groups.get("fused_expert_gemm", {}).get("calls") != 2 * forwards:
        raise AssertionError("Expected exactly two real fused expert GEMM kernels per forward")
    return groups


def _profile_physical_counts(rt, layer, ids, start, stop):
    torch, state = rt.torch, layer.layer_state
    buffer = torch.zeros_like(state.expert_load_view)
    record = torch.ones((), dtype=torch.bool, device=rt.device)
    unpadded = torch.tensor(ids.shape[1], dtype=torch.int32, device=rt.device)
    rows = []
    for raw in range(start, stop):
        buffer.zero_()
        layer.mapping_kernel(ids[raw], buffer, state.logical_to_physical_map,
                             state.logical_replica_count, record, unpadded)
        rows.append(buffer.clone())
    counts = torch.stack(rows).cpu().numpy().reshape(stop - start, rt.world_size, layer.slots)
    if not np.all(counts.sum(axis=(1, 2)) == ids.shape[1] * ids.shape[2]):
        raise AssertionError("Profile mapping diagnostics violate assignment conservation")
    digest = array_hash(counts)
    agree_config(rt, {"physical_counts_sha256": digest})
    return counts, digest


def profile_selected(rt, weights, manifest, args):
    """Separate DEFAULT/selected diagnostics on the first declared pilot bank.

No policy runs or selection are performed. Profiling is excluded from all tuning
and validation measurements, and device kernel sums are not wall-time fractions.
    """
    from optimization.common import build_references, check_references
    if manifest["phase"] != "validation" or manifest["split"] != "pilot":
        raise ValueError("Selected-kernel profiling requires the frozen pilot validation manifest")
    if manifest.get("freeze", {}).get("sha256") != manifest_hash(manifest):
        raise ValueError("Profiling manifest must be frozen")
    directory = Path(args.profile_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a fresh directory for selected-kernel profiles")
    if rt.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    rt.barrier()
    block = manifest["matrix"]["block_ids"][0]
    entries = manifest.get("banks") or manifest["tapes"]
    entry = next(e for e in entries if e["block_id"] == block)
    path = Path(entry["capture_path"])
    bank = load_router_bank(path if path.is_absolute() else Path(args.banks_dir) / path)
    _validate_bank(bank, entry, manifest, weights)
    torch = rt.torch
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    cells = list(itertools.product((0, 1), manifest["matrix"]["batch_tokens"]))
    orders = [["DEFAULT", "TUNED"] if i % 2 == 0 else ["TUNED", "DEFAULT"] for i in range(len(cells))]
    random.Random(manifest["execution"]["order_seed"]).shuffle(orders)
    summaries, profiler_warmed = [], False
    for (domain, batch), order in zip(cells, orders):
        indices, seed = input_indices(bank.domain_ids, block, domain, batch)
        artifact = _persist_indices(rt, directory / "profile-summary.json", bank, indices, seed,
                                    {"block_id": block, "domain": domain, "batch_tokens": batch})
        tensors = _materialize(rt, SimpleNamespace(bank=bank, indices=indices))
        hidden, ids, combining = tensors
        layer = make_layer(rt, weights, manifest, "SKIP")
        layer.assert_maps_and_weights()
        source_map = layer.map_snapshot()
        weight_hash, rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
        references = build_references(rt, layer, tensors, [16, 23])
        counts, counts_hash = _profile_physical_counts(rt, layer, ids, 16, 24)
        if not profiler_warmed:
            with kernel_override(None):
                # Explicitly discard the first profiler session, whose startup
                # overhead was substantial in the earlier diagnostic traces.
                with torch.profiler.profile(activities=activities):
                    layer.forward(hidden[0], ids[0], combining[0], observe=False)
                    rt.sync()
            profiler_warmed = True
            rt.barrier()
        for variant in order:
            selection = manifest["selected_configs"][str(batch)]
            candidate = "DEFAULT" if variant == "DEFAULT" else selection["candidate_id"]
            config = None if variant == "DEFAULT" else selection["config"]
            with kernel_override(config):
                resolved = resolve_kernel_config(layer, batch)
                agree_config(rt, resolved)
                correctness = check_references(rt, layer, tensors, references)
                for raw in range(16):
                    layer.forward(hidden[raw], ids[raw], combining[raw], observe=False)
                rt.barrier()
                rt.sync()
                with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                    for raw in range(16, 24):
                        with torch.profiler.record_function("optimization_fixed_map_moe"):
                            layer.forward(hidden[raw], ids[raw], combining[raw], observe=False)
                    rt.sync()
            stem = f"{block}-domain{domain}-batch{batch}-{variant}-rank{rt.rank}"
            trace_path = directory / f"{stem}.json"
            profiler.export_chrome_trace(str(trace_path))
            (directory / f"{stem}.txt").write_text(profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=40))
            kernel_groups = kernel_event_summary(json.loads(trace_path.read_text()), 8)
            tile = resolved["BLOCK_SIZE_M"]
            padded = ((counts + tile - 1) // tile * tile).sum(axis=2)
            assigned = counts.sum(axis=2)
            layer.assert_maps_and_weights()
            if not np.array_equal(source_map, layer.map_snapshot()):
                raise AssertionError("Profiling changed the fixed placement")
            summary = {**rt.metadata, "optimization_schema_version": 1,
                       "phase": "profiling", "measurement_kind": "gpu_kernel_diagnostic",
                       "evidence_eligible": False, "config_hash": manifest_hash(manifest),
                       "block_id": block, "domain": domain, "batch_tokens": batch,
                       "rank": rt.rank, "kernel_variant": variant, "candidate_id": candidate,
                       "kernel_override_config": config, "resolved_kernel_config": resolved,
                       "kernel_config_sha256": config_fingerprint(config), "kernel_config_rank_agreement": True,
                       "source_map_sha256": array_hash(source_map), "source_weights_sha256": weight_hash,
                       "source_weight_rank_sha256": rank_hashes, "capture_sha256": bank.sha256,
                       "replay_sha256": artifact["sha256"], "source_prompt_ids": bank.metadata["source_prompt_ids"],
                       "input_artifact": artifact, "profile_forwards": 8, "profile_raw_range_half_open": [16, 24],
                       "conditioning_forwards": 16, "variant_order": order,
                       "initial_profiler_session_discarded": True, "kernel_event_filter": "cat=kernel,ph=X",
                       "kernel_event_groups": kernel_groups, "trace_file": str(trace_path),
                       "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                       "physical_counts_sha256": counts_hash, "physical_counts": counts.tolist(),
                       "owner_assignments_per_step": assigned.tolist(),
                       "active_physical_experts_per_rank_per_step": (counts > 0).sum(axis=2).tolist(),
                       "local_padded_rows_per_step": padded.tolist(),
                       "padded_to_useful_rows_by_rank": (padded.sum(axis=0) / assigned.sum(axis=0)).tolist(),
                       "padding_scope": "integer tile-row count from exact native physical assignments; not hardware instruction or traffic measurement",
                       "timing_scope": "diagnostic profiled kernel residence only; no unprofiled latency or additive wall-time decomposition",
                       **correctness}
            peers = [None] * rt.world_size
            torch.distributed.all_gather_object(peers, summary, group=rt.ep_group.cpu_group)
            if rt.rank == 0:
                summaries.extend(peers)
            rt.barrier()
        layer.close()
        del layer, tensors, hidden, ids, combining, references, counts
        gc.collect()
        torch.cuda.empty_cache()
    if rt.rank == 0:
        with (directory / "profile-summary.json").open("x") as handle:
            json.dump({"schema_version": 1, "scope": "selected pilot kernel diagnostics only",
                       "initial_profiler_session_discarded": True, "rows": summaries}, handle,
                      indent=2, allow_nan=False)
            handle.write("\n")
    rt.barrier()
    return {"profile_cells": len(cells), "variants_per_cell": 2, "output_dir": str(directory)}
