"""Matched captured-token occurrences, with explicitly constructed ordering.

Document continuity is limited to the captured prompt prefix. Replaying these
vectors does not execute requests, autoregressive decoding, or a serving batcher.
The mixed intervention changes both batch composition and temporal ordering.
"""
from pathlib import Path
import json
import math

import numpy as np

from followup.router_tapes import load_router_bank, file_sha256
from stock_comparison.data import digest
from stock_comparison.inputs import array_digest, BATCH, WARMUP

SEED = 471902
MODES = ("contiguous", "mixed", "batch_shuffle")


def _rng(seed, block_id, purpose):
    key = int(digest([seed, block_id, purpose])[:16], 16)
    return np.random.Generator(np.random.PCG64(key))


def validate_records(prompt_index, token_position):
    prompt_index, token_position = np.asarray(prompt_index), np.asarray(token_position)
    if (prompt_index.ndim != 1 or token_position.shape != prompt_index.shape
            or prompt_index.dtype.kind not in "iu" or token_position.dtype.kind not in "iu"
            or len(prompt_index) < BATCH or (prompt_index < 0).any() or (token_position < 0).any()):
        raise ValueError("Need at least128 valid captured prompt-token records")
    prompts = np.unique(prompt_index)
    if not np.array_equal(prompts, np.arange(len(prompts))):
        raise ValueError("Prompt indices must be contiguous and start at zero")
    records = []
    for prompt in prompts:
        selected = np.flatnonzero(prompt_index == prompt)
        selected = selected[np.argsort(token_position[selected], kind="stable")]
        if not np.array_equal(token_position[selected], np.arange(len(selected))):
            raise ValueError("Each document must contain one complete captured prefix without gaps or duplicates")
        records.append(selected)
    return records


def sequence_properties(indices, prompt_index, token_position):
    """Raw input structure only; no captured/HF routing is called native routing."""
    indices = np.asarray(indices)
    flat = indices.ravel()
    prompts = np.asarray(prompt_index)[flat]
    positions = np.asarray(token_position)[flat]
    same = prompts[1:] == prompts[:-1]
    following = same & (positions[1:] == positions[:-1] + 1)
    distinct_records = np.asarray([len(np.unique(row)) for row in indices])
    distinct_docs = np.asarray([len(np.unique(np.asarray(prompt_index)[row])) for row in indices])
    changes = np.flatnonzero(np.r_[True, prompts[1:] != prompts[:-1], True])
    return dict(
        same_document_adjacent_fraction=float(same.mean()),
        next_position_same_document_fraction=float(following.mean()),
        repeated_record_adjacent_fraction=float((flat[1:] == flat[:-1]).mean()),
        mean_distinct_documents_per_job=float(distinct_docs.mean()),
        minimum_distinct_records_per_job=int(distinct_records.min()),
        jobs_with_repeated_record=int((distinct_records < BATCH).sum()),
        repeated_record_occurrences_within_jobs=int((BATCH - distinct_records).sum()),
        longest_same_document_run_tokens=int(np.diff(changes).max()),
        interpreted_as_natural_serving_order=False,
    )


def repair_collisions(indices, rng):
    """Deterministic cross-job swaps, preserving the exact occurrence bag.

Only record identities enter the repair; no routing, workload outcome, or
timing is consulted. A swap must remove a duplicate in the current row and
introduce no duplicate in either row. Failure is explicit, never resampling.
"""
    result = np.asarray(indices, dtype=np.int64).copy()
    if result.ndim != 2 or result.shape[1] != BATCH or (result < 0).any():
        raise ValueError("Collision repair expects nonnegative [jobs,128] record IDs")
    jobs = len(result)
    counts = [dict(zip(*np.unique(row, return_counts=True))) for row in result]
    before = sum(BATCH - len(c) for c in counts)
    occurrences = np.bincount(result.ravel())
    if occurrences.max() > jobs:
        raise ValueError("Occurrence bag cannot avoid repeated records within a job")
    swaps = fallback_searches = 0
    for row in range(jobs):
        for col in range(BATCH):
            x = int(result[row, col])
            if counts[row][x] == 1:
                continue
            target = None
            for _ in range(64):
                other, position = divmod(int(rng.integers(jobs * BATCH)), BATCH)
                y = int(result[other, position])
                if other != row and x not in counts[other] and y not in counts[row]:
                    target = (other, position, y)
                    break
            if target is None:
                fallback_searches += 1
                start = int(rng.integers(jobs))
                for step in range(jobs):
                    other = (start + step) % jobs
                    if other == row or x in counts[other]:
                        continue
                    for position, value in enumerate(result[other]):
                        y = int(value)
                        if y not in counts[row]:
                            target = (other, position, y)
                            break
                    if target is not None:
                        break
            if target is None:
                raise RuntimeError("Bounded deterministic collision repair failed; no trace replacement permitted")
            other, position, y = target
            result[row, col], result[other, position] = y, x
            counts[row][x] -= 1
            counts[row][y] = 1
            counts[other][y] -= 1
            if counts[other][y] == 0:
                del counts[other][y]
            counts[other][x] = 1
            swaps += 1
    after = sum(BATCH - len(c) for c in counts)
    if after or not np.array_equal(np.bincount(result.ravel()), occurrences):
        raise AssertionError("Collision repair failed uniqueness or exact multiset preservation")
    return result, dict(algorithm="global_shuffle_then_deterministic_cross_job_swaps_v1",
        original_duplicate_occurrences=before, final_duplicate_occurrences=after,
        swaps=swaps, fallback_searches=fallback_searches,
        random_candidates_per_duplicate=64,
        fallback_bound="At most jobs*128 candidates per duplicate; fail without resampling",
        consulted_routing_or_timing=False, exact_occurrence_multiset_preserved=True)


def build_indices(prompt_index, token_position, block_id, mode, *, jobs=4096, seed=SEED):
    if mode not in MODES or type(jobs) is not int or jobs < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid ordering mode, job count, or seed")
    records = validate_records(prompt_index, token_position)
    prompt_index = np.asarray(prompt_index)
    token_position = np.asarray(token_position)
    order = _rng(seed, block_id, "document_order").permutation(len(records))
    one_cycle = np.concatenate([records[int(p)] for p in order])
    needed = jobs * BATCH
    continuous = np.tile(one_cycle, math.ceil(needed / len(one_cycle)))[:needed].copy()
    # Repetition is by whole captured bank, never by grouping all repeated
    # copies of the same token/document into an artificial long hotspot.
    repair = None
    if mode == "contiguous":
        measured = continuous.reshape(jobs, BATCH)
        batch_permutation = None
    elif mode == "mixed":
        measured = _rng(seed, block_id, "mixed_occurrence_permutation").permutation(continuous).reshape(jobs, BATCH)
        measured, repair = repair_collisions(measured, _rng(seed, block_id, "mixed_collision_repair"))
        batch_permutation = None
    else:
        batch_permutation = _rng(seed, block_id, "whole_batch_permutation").permutation(jobs)
        measured = continuous.reshape(jobs, BATCH)[batch_permutation]
    warmup = np.stack([_rng(seed, block_id, ["warmup", i]).permutation(len(one_cycle))[:BATCH]
                       for i in range(WARMUP)])
    indices = np.concatenate([warmup, measured]).astype(np.int64, copy=False)
    uses = np.bincount(measured.ravel(), minlength=len(one_cycle)).astype(np.int64)
    expected = np.bincount(continuous, minlength=len(one_cycle))
    if not np.array_equal(uses, expected):
        raise AssertionError("Ordering changed the captured-token occurrence multiset")
    metadata = dict(
        schema="sequence_hidden_jobs_v1", block_id=block_id, mode=mode,
        batch_tokens=BATCH, warmup_jobs=WARMUP, measured_jobs=jobs, seed=seed,
        source_records=len(one_cycle), source_documents=len(records),
        document_order=order.tolist(), one_cycle_sha256=array_digest(one_cycle),
        complete_bank_cycles=needed // len(one_cycle),
        final_partial_cycle_tokens=needed % len(one_cycle),
        document_order_repeated_each_cycle=True,
        final_truncation="Stop after exactly jobs*128 token occurrences; final captured document may be partial",
        indices_sha256=array_digest(indices), warmup_indices_sha256=array_digest(warmup),
        measured_occurrence_multiset_sha256=array_digest(uses),
        measured_record_occurrences=needed, token_use_counts=uses.tolist(),
        distinct_measured_records=int((uses > 0).sum()),
        minimum_record_uses=int(uses.min()), maximum_record_uses=int(uses.max()),
        batch_permutation_sha256=None if batch_permutation is None else array_digest(batch_permutation),
        measured_sequence_properties=sequence_properties(measured, prompt_index, token_position),
        not_independent_requests=True, natural_batch_trace=False,
        fresh_content=False, fresh_arrival_block=False, new_ordering_only=True,
        mixed_may_repeat_captured_record_within_job=False,
        collision_repair=repair,
        composition=(
            "Full captured document prefixes in fixed seeded document order, repeated as whole-bank cycles"
            if mode == "contiguous" else
            "Seeded global shuffle of exactly the contiguous sequence's token-occurrence multiset, then deterministic cross-job swaps remove within-job record collisions; this constrained shuffle is not the previous independent per-job sampler"
            if mode == "mixed" else
            "Seeded permutation of complete contiguous batches; every batch membership and within-batch order is unchanged"
        ),
        causal_scope=("Temporal batch ordering only relative to contiguous" if mode == "batch_shuffle"
                      else "Mixed versus contiguous jointly changes batching and temporal order"),
    )
    return indices, metadata


def load_inputs(rt, entry, banks_dir, output_dir, *, mode, jobs=4096, seed=SEED):
    bank = load_router_bank(Path(banks_dir) / entry["capture_path"])
    m = bank.metadata
    if (m["selection_content_sha256"] != entry["selection_content_sha256"]
            or bank.sha256 != entry["capture_sha256"] or m.get("study_id") != "resident_residual_v1"
            or m["block_id"] != entry["block_id"] or m["dataset"] != entry["dataset"]
            or not m["correctness_pass"]):
        raise ValueError("Wrong or changed source input bank")
    if len(m["prompts"]) != 32 or any(not p["native_boundary_exact"] or not p["bf16_roundtrip_exact"] for p in m["prompts"]):
        raise ValueError("Expected32 authenticated documents")
    if (np.asarray(bank.hidden_states_bf16, dtype=np.uint16) & 0x7f80 == 0x7f80).any():
        raise ValueError("Nonfinite BF16 input")
    indices, meta = build_indices(bank.prompt_index, bank.token_position, entry["block_id"], mode, jobs=jobs, seed=seed)
    hidden = rt.torch.empty((len(indices), BATCH, 2048), device=rt.device, dtype=rt.torch.bfloat16)
    for i, idx in enumerate(indices):
        bits = np.asarray(bank.hidden_states_bf16[idx], dtype=np.uint16).copy()
        hidden[i].copy_(rt.torch.from_numpy(bits).view(rt.torch.bfloat16))
    rt.sync()
    target = Path(output_dir) / "inputs" / (entry["block_id"] + "-" + mode + ".npz")
    if rt.rank == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, indices=indices, metadata=np.asarray(json.dumps(meta, sort_keys=True)))
    rt.barrier()
    return hidden, dict(**meta, capture_sha256=bank.sha256, source_prompt_ids=m["source_prompt_ids"],
        gate_weight_sha256=m["source_tensor_sha256"]["model.layers.0.mlp.gate.weight"],
        index_artifact=str(target), index_artifact_sha256=file_sha256(target),
        executable_input="Exact captured BF16 hidden states; actual native gate/top-k computed inside each measured forward")
