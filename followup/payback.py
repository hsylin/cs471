"""Same-state KEEP / PREINSTALLED / CHARGED native MoE layer replay.

This is a conditional action diagnostic. Proposal construction and the common
B3 prefix are sunk, excluded work; it is not an end-to-end online controller.
The completed skewbench package is imported without modifying its source.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np

from skewbench.native import NativeLayer, Runtime, Weights
from skewbench.policies import predicted_rank_loads
from skewbench.runner import finalize_decision_log, snapshot_weight_hash, warm_native_transport

ARMS = ("KEEP", "PREINSTALLED", "CHARGED")
CANDIDATES = ("native", "one_copy")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def manifest_hash(manifest):
    return canonical_hash({k: v for k, v in manifest.items() if k != "freeze"})


def array_hash(value, dtype=np.int64):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def followup_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "followup").glob("*.py"))
    for name in ("capture_router_tapes.py", "compose_router_tapes.py"):
        capture = root / "scripts" / name
        if capture.exists():
            paths.append(capture)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return canonical_hash(files), files


def replay_bounds(window_steps, decision_step, horizon_steps, raw_steps, rebalance_interval):
    """The decision input belongs to the prefix; future starts at the next input."""
    if window_steps <= 0 or rebalance_interval < 2 * window_steps:
        raise ValueError("Require positive W and R >= 2W")
    if decision_step < 0 or (decision_step + 1) % rebalance_interval:
        raise ValueError("Decision must be a scheduled measured native boundary")
    if horizon_steps <= 0:
        raise ValueError("Future horizon must be positive")
    start = 2 * window_steps + decision_step + 1
    stop = start + horizon_steps
    if stop > raw_steps:
        raise ValueError("Tape does not contain the complete declared future")
    return start, stop


def balanced_arm_orders(count, seed):
    """Latin-square order and its reverse: every six groups use all permutations.

    Position totals over the declared group set differ by at most one.
    Shuffling assignment to groups does not use measured outcomes.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    cycle = [ARMS, ARMS[1:] + ARMS[:1], ARMS[2:] + ARMS[:2]]
    cycle += [tuple(reversed(order)) for order in cycle]
    orders = [list(cycle[i % 6]) for i in range(count)]
    random.Random(seed).shuffle(orders)
    return orders


def validate_maps(current, proposed, num_experts, world_size):
    old, new = np.asarray(current), np.asarray(proposed)
    for name, mapping in (("source", old), ("candidate", new)):
        if (mapping.ndim != 1 or mapping.dtype.kind not in "iu" or not mapping.size
                or mapping.size % world_size or np.any(mapping < 0)
                or np.any(mapping >= num_experts)
                or np.any(np.bincount(mapping, minlength=num_experts) == 0)):
            raise ValueError(f"Invalid {name} physical map")
    if old.shape != new.shape:
        raise ValueError("Candidate must retain the fixed slot/replica budget")
    return old.astype(np.int64, copy=False), new.astype(np.int64, copy=False)


def predicted_gain(counts, current, proposed, world_size):
    old = float(predicted_rank_loads(counts, current, world_size).max())
    new = float(predicted_rank_loads(counts, proposed, world_size).max())
    return (old - new) / old if old else 0.0


def one_copy_candidate(counts, current, world_size):
    """Best legal one-slot remote copy under the current token-count proxy.

    Remove only a redundant copy. The new expert must be absent from the
    destination rank and present elsewhere. Lexicographic (slot, expert) order
    breaks exact score ties. A non-improving best legal candidate is retained.
    """
    counts = np.asarray(counts, dtype=np.float64)
    current, _ = validate_maps(current, current, len(counts), world_size)
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("Counts must be finite and nonnegative")
    slots = len(current) // world_size
    copies = np.bincount(current, minlength=len(counts))
    best = None
    for slot, donor in enumerate(current):
        if copies[donor] < 2:
            continue
        rank = slot // slots
        resident = set(current[rank * slots:(rank + 1) * slots].tolist())
        for incoming in range(len(counts)):
            if incoming in resident:
                continue
            candidate = current.copy()
            candidate[slot] = incoming
            gain = predicted_gain(counts, current, candidate, world_size)
            if best is None or gain > best["predicted_gain"]:
                best = {"map": candidate, "predicted_gain": gain,
                        "replacement_slot": slot, "removed_logical_expert": int(donor),
                        "incoming_logical_expert": incoming, "destination_rank": rank}
    if best is not None:
        best["candidate_improves_proxy"] = best["predicted_gain"] > 0
        best["candidate_reason"] = ("best_legal_one_copy" if best["candidate_improves_proxy"]
                                    else "no_improving_candidate")
    return best


def build_groups(manifest):
    matrix = manifest["matrix"]
    if tuple(matrix["arms"]) != ARMS:
        raise ValueError("Declared arms must be KEEP, PREINSTALLED, CHARGED")
    if not set(matrix["candidate_ids"]) <= set(CANDIDATES):
        raise ValueError("Unknown candidate family")
    tapes = {(str(t["block_id"]), t["regime"], int(t["batch_tokens"])): t
             for t in manifest["tapes"]}
    if len(tapes) != len(manifest["tapes"]):
        raise ValueError("Duplicate replay tape cell")
    groups = []
    for block in matrix["block_ids"]:
        for regime in matrix["regimes"]:
            for batch in matrix["batch_tokens"]:
                key = (str(block), regime, int(batch))
                if key not in tapes:
                    raise ValueError(f"Missing declared tape {key}")
                for candidate in matrix["candidate_ids"]:
                    for horizon in matrix["horizons"]:
                        for repetition in range(matrix["repetitions"]):
                            groups.append({"block_id": str(block), "regime": regime,
                                           "batch_tokens": int(batch), "candidate_id": candidate,
                                           "horizon_steps": int(horizon), "repetition": repetition,
                                           "tape": tapes[key]})
    return groups


def select_groups(groups, *, stage=None, candidates=None, repetitions=None,
                  horizons=None, regimes=None, blocks=None, order_seed=471):
    result = []
    for group in groups:
        if stage == "screening" and group["repetition"] != 0:
            continue
        if stage == "confirmation" and not (group["repetition"] in (1, 2)
                and group["regime"] in ("transient", "persistent", "switching")
                and group["horizon_steps"] == 512):
            continue
        if any(values is not None and group[field] not in values for field, values in (
                ("candidate_id", candidates), ("repetition", repetitions),
                ("horizon_steps", horizons), ("regime", regimes), ("block_id", blocks))):
            continue
        result.append(dict(group))
    for group, order in zip(result, balanced_arm_orders(len(result), order_seed)):
        group["arm_order"] = order
    return result


def validate_execution_metadata(manifest, metadata, weights_config):
    """Fail before measurement on any pinned runtime or checkpoint mismatch."""
    environment = manifest.get("environment", {})
    aliases = {"vllm_commit": "native_vllm_commit"}
    for field in ("world_size", "dtype", "dispatch_backend", "eplb_mode", "vllm_commit",
                  "legacy_harness_sha256", "followup_sha256"):
        if field in environment and environment[field] != metadata.get(aliases.get(field, field)):
            raise ValueError(f"Runtime differs from declared {field}")
    for field, key in (("model_id", "source_model"), ("model_revision", "source_revision"),
                       ("layer_id", "layer"), ("hidden_size", "hidden_size"),
                       ("intermediate_size", "intermediate_size")):
        if field in environment and environment[field] != weights_config.get(key):
            raise ValueError(f"Checkpoint differs from declared {field}")
    if environment.get("gpu_name_contains", "") not in metadata.get("gpu_name", ""):
        raise ValueError("GPU differs from declared hardware")


def validate_replay_contract(manifest, tape, weights_config):
    replay = manifest["replay"]
    if tape.bank.metadata.get("correctness_pass") is not True:
        raise ValueError("Capture has no passing native-router correctness attestation")
    checks = ("native_boundary_exact", "native_router_exact", "manual_router_exact", "bf16_roundtrip_exact")
    prompts = tape.bank.metadata.get("prompts", [])
    if len(prompts) != 2 or not all(prompt.get(check) is True for prompt in prompts for check in checks):
        raise ValueError("Capture prompt checks are incomplete")
    if tape.metadata["num_steps"] != replay["raw_steps"]:
        raise ValueError("Replay raw length differs from declared geometry")
    if tape.metadata["warmup_steps"] != 2 * replay["window_steps"]:
        raise ValueError("Replay observation warmup differs from executed prefix")
    case = next(c for c in replay["cases"] if c["regime"] == tape.metadata["regime"])
    for field, key in (("phase_steps", "phase_steps"), ("hotspot_lifetime_steps", "duration_steps"),
                       ("repeat_steps", "repeat_steps")):
        if tape.metadata[key] != case[field]:
            raise ValueError(f"Replay differs from declared {field}")
    for field in ("num_experts", "hidden_size", "top_k"):
        if tape.bank.metadata[field] != weights_config[field]:
            raise ValueError(f"Capture bank differs from checkpoint {field}")
    block = next(b for b in manifest["blocks"] if b["block_id"] == tape.metadata["block_id"])
    if sorted(block["source_document_ids"]) != sorted(tape.metadata["source_prompt_ids"]):
        raise ValueError("Replay source documents differ from frozen block")
    if tape.metadata["split"] != manifest["split"]:
        raise ValueError("Replay split differs from manifest")


def _make_layer(rt, weights, manifest):
    policy, replay = manifest["policy"], manifest["replay"]
    return NativeLayer(rt, weights, method="B3", replicas=policy["replica_budget"],
                       window_steps=replay["window_steps"],
                       rebalance_interval=replay["rebalance_interval"],
                       epsilon=policy.get("epsilon", .02), tau=policy.get("tau", .1),
                       cooldown_steps=policy.get("cooldown_steps", 256), seed=0)


def _agree(rt, value, label):
    peers = [None] * rt.world_size
    rt.torch.distributed.all_gather_object(peers, value, group=rt.ep_group.cpu_group)
    if any(peer != peers[0] for peer in peers):
        raise AssertionError(f"Ranks disagree: {label}")
    return value


def _materialize(rt, tape):
    """Materialize all actual per-step inputs before any measurement."""
    bank, torch = tape.bank, rt.torch
    raw_hidden = np.asarray(bank.hidden_states_bf16)
    if raw_hidden.dtype != np.uint16 or raw_hidden.ndim != 2:
        raise ValueError("Capture bank must preserve raw BF16 bits as uint16[N,H]")
    indices = torch.from_numpy(np.asarray(tape.indices).copy()).to(rt.device, torch.long)
    hidden = torch.from_numpy(raw_hidden.copy()).view(torch.bfloat16).to(rt.device)[indices]
    ids = torch.from_numpy(np.asarray(bank.expert_ids).copy()).to(rt.device, torch.int32)[indices]
    combine = torch.from_numpy(np.asarray(bank.combining_weights).copy()).to(rt.device, torch.float32)[indices]
    rt.sync()
    return hidden.contiguous(), ids.contiguous(), combine.contiguous()


def _counts(tape, start, stop, num_experts):
    ids = np.asarray(tape.bank.expert_ids)[np.asarray(tape.indices)[start:stop]]
    return np.bincount(ids.reshape(-1), minlength=num_experts)


def _physical_diagnostic(rt, layer, tape, ids, start, stop):
    """Untimed native mapping-only replay, with no duplicate expert GEMMs."""
    torch, ls = rt.torch, layer.layer_state
    buffer = torch.zeros_like(ls.expert_load_view)
    record = torch.ones((), dtype=torch.bool, device=rt.device)
    unpadded = torch.tensor(ids.shape[1], dtype=torch.int32, device=rt.device)
    rows = []
    for step in range(start, stop):
        buffer.zero_()
        layer.mapping_kernel(ids[step], buffer, ls.logical_to_physical_map,
                             ls.logical_replica_count, record, unpadded)
        rows.append(buffer.clone())
    physical = torch.stack(rows).cpu().numpy().reshape(stop - start, -1)
    mapping = layer.map_snapshot()
    for index, row in enumerate(physical):
        logical = np.zeros(layer.num_experts, dtype=np.int64)
        np.add.at(logical, mapping, row)
        if not np.array_equal(logical, _counts(tape, start + index, start + index + 1,
                                              layer.num_experts)):
            raise AssertionError("Untimed native dispatch does not conserve exact tape assignments")
    digest = array_hash(physical, np.int32)
    _agree(rt, digest, "physical diagnostic")
    owner = physical.reshape(stop - start, rt.world_size, layer.slots)
    return {"scope": "untimed native mapping-kernel replay only; one identical replicated-input copy",
            "physical_counts_sha256": digest, "physical_map_sha256": array_hash(mapping),
            "raw_step_range_half_open": [start, stop],
            "owner_assignments_per_step": owner.sum(axis=2).tolist(),
            "active_physical_experts_per_rank_per_step": (owner > 0).sum(axis=2).tolist(),
            "logical_conservation_checks": stop - start,
            "input_rank_multiplicity": rt.world_size}


def _source_state(rt, weights, manifest, tape, tensors):
    hidden, ids, combine = tensors
    replay = manifest["replay"]
    w, decision = replay["window_steps"], replay["decision_step"]
    future, _ = replay_bounds(w, decision, 1, len(tape.indices), replay["rebalance_interval"])
    layer = _make_layer(rt, weights, manifest)
    for _ in range(3):
        layer.forward(hidden[0], ids[0], combine[0], observe=False)
    rt.sync()
    started = time.perf_counter()
    layer.warming_up = True
    for step in range(2 * w):
        layer.forward(hidden[step], ids[step], combine[step])
        layer.step(step)
    layer.warming_up = False
    layer.state.expert_rearrangement_step = 0
    layer.state._update_layer_should_record(log_stats=False)
    layer.decisions.clear()
    layer.counter.sent_bytes = layer.counter.sent_messages = 0
    layer.planner_seconds = layer.controller_seconds = 0.0
    layer.capture_decision_step = decision
    target_planner_seconds = 0.0
    for step in range(decision + 1):
        raw = 2 * w + step
        layer.forward(hidden[raw], ids[raw], combine[raw])
        planner_before = layer.planner_seconds
        layer.step(step)
        if step == decision:
            target_planner_seconds = layer.planner_seconds - planner_before
    layer.close()
    prefix_seconds = max(rt.gather_scalars(time.perf_counter() - started))
    snapshot = layer.captured_candidate
    if snapshot is None:
        raise AssertionError("Native proposal boundary was not intercepted")
    current = snapshot["current_map"]
    expected = _counts(tape, future - w, future, layer.num_experts) * rt.world_size
    if not np.array_equal(snapshot["current_counts"], expected):
        raise AssertionError("Source current window differs from executed tape")
    if not np.array_equal(layer.map_snapshot(), current):
        raise AssertionError("Intercepted proposal changed source placement")
    # The target was considered by B3 but deliberately not committed. Preserve
    # both facts before applying the legacy log's actual-commit assertions.
    target = layer.decisions[-1]
    target["would_allow"] = target["allowed"]
    target["allowed"] = False
    target["intercepted_for_branch"] = True
    prefix_log = finalize_decision_log(rt, layer)
    for row in prefix_log:
        end = 2 * w + row["step"] + 1
        if not np.array_equal(row["current_counts"],
                              _counts(tape, end - w, end, layer.num_experts) * rt.world_size):
            raise AssertionError("A prefix native observation window differs from the exact executed tape")
    layer.assert_maps_and_weights()
    local_weights = (layer.w13.clone(), layer.w2.clone())
    weight_hash, rank_hashes = snapshot_weight_hash(rt, local_weights)
    _agree(rt, {"map": current.tolist(), "proposal": snapshot["proposed_map"].tolist(),
                "counts": expected.tolist()}, "common source state")
    return layer, local_weights, {
        "source_map": current.tolist(), "source_map_sha256": array_hash(current),
        "source_weights_sha256": weight_hash, "source_weight_rank_sha256": rank_hashes,
        "current_counts": expected.tolist(), "prefix_decisions": prefix_log,
        "prefix_seconds_excluded": prefix_seconds,
        "candidate_planner_seconds_excluded": max(rt.gather_scalars(target_planner_seconds)),
        "prefix_planner_seconds_excluded": max(rt.gather_scalars(layer.planner_seconds)),
        "prefix_controller_seconds_excluded": max(rt.gather_scalars(layer.controller_seconds)),
        "prefix_migration_bytes_excluded": sum(d["transfer_sent_bytes"] for d in prefix_log),
        "b3_would_allow_native": snapshot["b3_would_allow"], "b3_native_reason": snapshot["b3_reason"],
        "native_proposal": snapshot["proposed_map"].tolist(),
        "decision_step": decision, "decision_trace_step": future - 1,
        "future_start_raw_step": future, "source_state_verified": True,
        "source_scope": "actual B3 prefix; earlier scheduled commits execute; target candidate intercepted before commit",
    }


def _candidate(rt, layer, source, family):
    started = time.perf_counter()
    current = np.asarray(source["source_map"], dtype=np.int64)
    if family == "native":
        proposed = np.asarray(source["native_proposal"], dtype=np.int64)
        gain = predicted_gain(source["current_counts"], current, proposed, rt.world_size)
        data = {"map": proposed, "predicted_gain": gain,
                "candidate_improves_proxy": gain > 0, "candidate_reason": "native_proposal"}
    else:
        data = one_copy_candidate(source["current_counts"], current, rt.world_size)
        if data is None:
            return None
    validate_maps(current, data["map"], layer.num_experts, rt.world_size)
    proposed = data.pop("map")
    data.update(candidate_id=family, candidate_map=proposed.tolist(),
                candidate_map_sha256=array_hash(proposed),
                candidate_changed_slots=int(np.count_nonzero(current != proposed)),
                candidate_selection_seconds_excluded=max(rt.gather_scalars(time.perf_counter() - started)))
    _agree(rt, {k: v for k, v in data.items() if not k.endswith("seconds_excluded")}, "candidate")
    return data


def _candidate_preflight(rt, layer, local_weights, source, candidate, tensors, conditioning_steps):
    hidden, ids, combine = tensors
    before = layer.counter.sent_bytes
    started = time.perf_counter()
    layer.force_migration(candidate["candidate_map"])
    layer.assert_maps_and_weights()
    for raw in range(source["future_start_raw_step"] - conditioning_steps,
                     source["future_start_raw_step"]):
        layer.forward(hidden[raw], ids[raw], combine[raw], observe=False)
    layer.close()
    payload = [int(x) for x in rt.gather_scalars(layer.counter.sent_bytes - before)]
    elapsed = max(rt.gather_scalars(time.perf_counter() - started))
    layer.restore_snapshot(source["source_map"], *local_weights)
    layer.assert_maps_and_weights()
    if not all(rt.torch.equal(a, b) for a, b in zip((layer.w13, layer.w2), local_weights)):
        raise AssertionError("Candidate preflight failed to restore source tensors")
    return {"candidate_weights_verified": True, "candidate_preflight_seconds_excluded": elapsed,
            "candidate_preflight_payload_by_rank": payload,
            "candidate_preflight_scope": "untimed native install, full weight/map validation, observed-prefix kernel warmup, exact source restore"}


def _run_arm(rt, weights, manifest, tape, tensors, source, local_weights,
             candidate, preflight, group, arm, diagnostics):
    hidden, ids, combine = tensors
    start, stop = replay_bounds(manifest["replay"]["window_steps"], source["decision_step"],
                                group["horizon_steps"], len(tape.indices),
                                manifest["replay"]["rebalance_interval"])
    layer = _make_layer(rt, weights, manifest)
    layer.restore_snapshot(source["source_map"], *local_weights)
    if not all(rt.torch.equal(a, b) for a, b in zip((layer.w13, layer.w2), local_weights)):
        raise AssertionError("Arm did not restore exact source tensors")
    layer.assert_maps_and_weights()
    untimed_install = 0
    if arm == "PREINSTALLED":
        layer.force_migration(candidate["candidate_map"])
        layer.assert_maps_and_weights()
        untimed_install = int(sum(rt.gather_scalars(layer.counter.sent_bytes)))
    conditioning_steps = manifest["execution"].get("conditioning_steps", 2)
    for raw in range(start - conditioning_steps, start):
        layer.forward(hidden[raw], ids[raw], combine[raw], observe=False)
    layer.counter.sent_bytes = layer.counter.sent_messages = 0
    rt.barrier()
    rt.sync()
    started = time.perf_counter()
    migration_time = 0.0
    if arm == "CHARGED":
        migration_started = time.perf_counter()
        layer.force_migration(candidate["candidate_map"])
        migration_time = time.perf_counter() - migration_started
    for raw in range(start, stop):
        layer.forward(hidden[raw], ids[raw], combine[raw], observe=False)
    layer.close()
    elapsed = time.perf_counter() - started
    rank_elapsed = rt.gather_scalars(elapsed)
    payload = [int(x) for x in rt.gather_scalars(layer.counter.sent_bytes)]
    messages = int(sum(rt.gather_scalars(layer.counter.sent_messages)))
    migration_seconds = max(rt.gather_scalars(migration_time))
    expected_map = source["source_map"] if arm == "KEEP" else candidate["candidate_map"]
    if layer.map_snapshot().tolist() != expected_map:
        raise AssertionError("Arm committed an unexpected map")
    layer.assert_maps_and_weights()
    if arm == "CHARGED" and payload != preflight["candidate_preflight_payload_by_rank"]:
        raise AssertionError("Charged payload differs from the same candidate preflight")
    if arm != "CHARGED" and any(payload):
        raise AssertionError("Keep/preinstalled arm moved weights during measurement")
    checks = []
    for raw in sorted({start, stop - 1}):
        checks.append({"raw_step": raw, **layer.check_outputs(hidden[raw], ids[raw], combine[raw])})
    key = (array_hash(expected_map), start, stop)
    if key not in diagnostics:
        diagnostics[key] = _physical_diagnostic(rt, layer, tape, ids, start, stop)
    final_map = _agree(rt, layer.map_snapshot().tolist(), "arm final map")
    data = {
        "elapsed_seconds": max(rank_elapsed), "per_rank_elapsed_seconds": rank_elapsed,
        "migration_seconds": migration_seconds, "migration_bytes": sum(payload),
        "migration_sent_bytes_by_rank": payload, "migration_sent_messages": messages,
        "untimed_install_bytes": untimed_install, "steps_completed": stop - start,
        "future_start_raw_step": start, "future_stop_raw_step_exclusive": stop,
        "final_map": final_map, "committed_map_sha256": array_hash(final_map),
        "commit_completed": True, "future_rebalancing_enabled": False,
        "source_restore_verified": True, "expected_arm_weights_verified": True,
        "correctness_pass": True, "correctness_checks": checks,
        "correctness_max_abs_error": max(x["correctness_max_abs_error"] for x in checks),
        "correctness_forward_batch_tokens": group["batch_tokens"],
        "correctness_reference_tokens": 4, "correctness_atol": .02, "correctness_rtol": .03,
        "physical_dispatch_diagnostic": diagnostics[key],
        "cache_conditioning": {
            "steps": conditioning_steps, "raw_indices": list(range(start - conditioning_steps, start)),
            "map": "candidate" if arm == "PREINSTALLED" else "source",
            "scope": "identical observed-prefix inputs; no future inputs; mapping is the arm's pre-timer placement; no explicit L2 flush",
            "charged_minus_preinstalled": "includes native transfer/commit and placement-transition/cache effects; not a pure network decomposition",
        },
        "timing_scope": "synchronized complete fixed-horizon layer replay; CHARGED includes native weight transfer+commit; common proposal/prefix excluded",
        "migration_bytes_scope": "actual native sender tensor payload; excludes protocol headers and local copies",
    }
    del layer
    gc.collect()
    rt.sync()
    rt.barrier()
    return data


def _resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def _record_id(config_hash, group, arm):
    return canonical_hash({"config_hash": config_hash, **{k: v for k, v in group.items() if k != "tape"}, "arm": arm})


def execute(args, rt):
    from followup.router_tapes import load_router_tape
    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    config_hash = manifest_hash(manifest)
    if manifest.get("freeze", {}).get("sha256") != config_hash:
        raise ValueError("Follow-up manifest must be frozen before execution")
    groups = select_groups(build_groups(manifest), stage=args.stage, candidates=args.candidate_family,
                           repetitions=args.repetition, horizons=args.horizon,
                           regimes=args.regime, blocks=args.block_id,
                           order_seed=manifest["execution"].get("order_seed", 471))
    if not groups:
        raise ValueError("Requested subset has no declared arm groups")
    weights = Weights(args.weights_dir)
    digest, files = followup_fingerprint()
    environment = manifest.get("environment", {})
    if environment.get("followup_sha256") != digest:
        raise ValueError("Follow-up code differs from frozen manifest")
    if environment.get("legacy_harness_sha256") != rt.metadata["harness_sha256"]:
        raise ValueError("Legacy harness differs from frozen manifest")
    rt.metadata.update(followup_sha256=digest, followup_file_sha256=files,
                       legacy_harness_sha256=rt.metadata["harness_sha256"], dtype="bfloat16")
    rt.metadata.update(allocation_id=os.environ.get("SLURM_JOB_ID", "unallocated"),
                       hostname=socket.gethostname(),
                       process_run_id=os.environ.get("TORCHELASTIC_RUN_ID", f"pid-{os.getpid()}"))
    validate_execution_metadata(manifest, rt.metadata, weights.config)
    warm_manifest = {"policy": {**manifest["policy"],
                               "rebalance_interval": manifest["replay"]["rebalance_interval"]},
                     "trace": {"window_steps": manifest["replay"]["window_steps"]}}
    warm_native_transport(rt, weights, warm_manifest)
    output = Path(args.output)
    if rt.rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "complete" and row.get("correctness_pass"):
                completed.add(row["record_id"])
    elif output.exists():
        raise ValueError("Refusing to append without --resume")
    for group in groups:
        present = sum(_record_id(config_hash, group, arm) in completed for arm in ARMS)
        if present not in (0, len(ARMS)):
            raise ValueError("Cannot resume a partially measured three-arm group; preserve the partial attempt and rerun the complete group in a fresh output")
    states = {}
    for group in groups:
        key = (group["block_id"], group["regime"], group["batch_tokens"])
        states.setdefault(key, []).append(group)
    base = Path(args.banks_dir) if args.banks_dir else path.parent
    for state_index, (key, state_groups) in enumerate(states.items()):
        if args.max_states is not None and state_index >= args.max_states:
            break
        if all(_record_id(config_hash, g, arm) in completed for g in state_groups for arm in ARMS):
            continue
        entry = state_groups[0]["tape"]
        tape = load_router_tape(_resolve(entry["replay_path"], base),
                                _resolve(entry["capture_path"], base))
        if tape.sha256 != entry["replay_sha256"] or tape.metadata["capture_sha256"] != entry["capture_sha256"]:
            raise ValueError("Tape/capture hash differs from frozen manifest")
        for field, expected in (("block_id", key[0]), ("regime", key[1]), ("batch_tokens", key[2])):
            if str(tape.metadata[field]) != str(expected):
                raise ValueError(f"Replay metadata differs from declared {field}")
        for field, expected in (("source_model", weights.config["source_model"]),
                                ("source_revision", weights.config["source_revision"]),
                                ("layer", weights.config["layer"])):
            if tape.metadata[field] != expected:
                raise ValueError(f"Capture differs from checkpoint {field}")
        validate_replay_contract(manifest, tape, weights.config)
        tensors = _materialize(rt, tape)
        for group in state_groups:
            replay_bounds(manifest["replay"]["window_steps"], manifest["replay"]["decision_step"],
                          group["horizon_steps"], len(tape.indices), manifest["replay"]["rebalance_interval"])
        layer, local_weights, source = _source_state(rt, weights, manifest, tape, tensors)
        state_id = canonical_hash({"block_id": key[0], "regime": key[1], "batch_tokens": key[2],
                                   "replay_sha256": tape.sha256, "source_map_sha256": source["source_map_sha256"]})
        candidates, preflights, diagnostics = {}, {}, {}
        for family in dict.fromkeys(g["candidate_id"] for g in state_groups):
            candidate = _candidate(rt, layer, source, family)
            if candidate is None:
                raise ValueError(f"No legal one-copy candidate for state {state_id}; explicit unavailable state, no claimed move")
            candidates[family] = candidate
            preflights[family] = _candidate_preflight(rt, layer, local_weights, source, candidate,
                                                      tensors, manifest["execution"].get("conditioning_steps", 2))
        del layer
        for group in state_groups:
            candidate, preflight = candidates[group["candidate_id"]], preflights[group["candidate_id"]]
            for arm_index, arm in enumerate(group["arm_order"]):
                record_id = _record_id(config_hash, group, arm)
                if record_id in completed:
                    continue
                common = {"schema_version": 1, "payback_schema_version": 1,
                          "measurement_kind": "gpu_payback",
                          "evidence_eligible": manifest.get("study_role") != "smoke",
                          "study_role": manifest.get("study_role", "payback_study"),
                          "split": manifest["split"], "config_hash": config_hash,
                          **{k: v for k, v in group.items() if k != "tape"},
                          "arm": arm, "arm_index": arm_index, "record_id": record_id,
                          "state_id": state_id, "source_prompt_ids": tape.metadata["source_prompt_ids"],
                          "capture_sha256": tape.metadata["capture_sha256"],
                          "replay_sha256": tape.sha256, "trace_sha256": tape.sha256,
                          "replay_metadata": tape.metadata, "weights_source": weights.metadata,
                          **rt.metadata, **source, **candidate, **preflight,
                          "replica_budget_actual": manifest["policy"]["replica_budget"],
                          "execution_selection": {"stage": args.stage,
                              "candidate_families": args.candidate_family, "repetitions": args.repetition,
                              "horizons": args.horizon, "regimes": args.regime, "blocks": args.block_id,
                              "order_seed": manifest["execution"].get("order_seed", 471),
                              "order_scope": "balanced over declared filtered groups before resume or max-states filtering"},
                          "input_materialization": "per-step captured BF16 hidden/IDs/weights gathered from bank before timing",
                          "proposal_scope": "shared sunk proposal cost excluded from every arm; conditional action diagnostic"}
                if rt.rank == 0:
                    print(json.dumps({"event": "payback_start", "state_id": state_id, "arm": arm,
                                      "candidate_id": group["candidate_id"], "horizon": group["horizon_steps"],
                                      "repetition": group["repetition"]}), flush=True)
                try:
                    measured = _run_arm(rt, weights, manifest, tape, tensors, source, local_weights,
                                        candidate, preflight, group, arm, diagnostics)
                    row = {**common, **measured, "status": "complete"}
                except Exception as error:
                    row = {**common, "status": "failed", "correctness_pass": False,
                           "error": repr(error), "traceback": traceback.format_exc()}
                    if rt.rank == 0:
                        with output.open("a") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                    raise
                if rt.rank == 0:
                    with output.open("a") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({"event": "payback_complete", "record_id": record_id,
                                      "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        del tensors, local_weights, tape, diagnostics
        gc.collect()
        rt.torch.cuda.empty_cache()
        rt.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--banks-dir")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-family", action="append", choices=CANDIDATES)
    parser.add_argument("--stage", choices=("screening", "confirmation"))
    parser.add_argument("--repetition", action="append", type=int)
    parser.add_argument("--horizon", action="append", type=int)
    parser.add_argument("--regime", action="append")
    parser.add_argument("--block-id", action="append")
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rt = Runtime()
    try:
        execute(args, rt)
    finally:
        rt.close()


if __name__ == "__main__":
    main()
