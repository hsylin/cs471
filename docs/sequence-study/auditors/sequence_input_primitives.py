"""Independent reconstruction of the frozen sequence-study input schedules.

No production module is imported.  The SHA256/PCG64 convention and random-draw
order are part of the published input contract, so this implementation retains
them while reconstructing the schedules and their metadata from source record
identities.  Captured bank bytes/provenance must be authenticated by the caller.
"""
from collections import Counter
import hashlib
import json

import numpy as np

BATCH = 128
WARMUP = 16
SEED = 471902
MODES = ('contiguous', 'mixed', 'batch_shuffle')


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256(digest([value.dtype.str, list(value.shape)]).encode())
    h.update(value.tobytes(order='C'))
    return h.hexdigest()


def _random(seed, block_id, domain):
    integer = int(digest([seed, block_id, domain])[:16], 16)
    return np.random.Generator(np.random.PCG64(integer))


def _documents(prompt_index, token_position):
    prompt_index, token_position = np.asarray(prompt_index), np.asarray(token_position)
    _require(prompt_index.ndim == 1 and token_position.shape == prompt_index.shape,
             'Captured prompt/position arrays differ')
    _require(prompt_index.dtype.kind in 'iu' and token_position.dtype.kind in 'iu'
             and len(prompt_index) >= BATCH and np.all(prompt_index >= 0)
             and np.all(token_position >= 0), 'Invalid captured record identities')
    documents = {}
    for record, (prompt, position) in enumerate(zip(prompt_index, token_position)):
        documents.setdefault(int(prompt), []).append((int(position), record))
    _require(sorted(documents) == list(range(len(documents))),
             'Captured document indices must start at zero with no gaps')
    result = []
    for prompt in range(len(documents)):
        pairs = sorted(documents[prompt])
        _require([p for p, _ in pairs] == list(range(len(pairs))),
                 'Captured document is not one complete unique prefix')
        result.append(np.array([record for _, record in pairs], dtype=np.int64))
    return result


def _repair(shuffled, generator):
    """Replay the specified swaps, using positive Counter membership checks."""
    result = np.array(shuffled, dtype=np.int64, copy=True)
    height, width = result.shape
    _require(width == BATCH and height > 0, 'Invalid constrained-shuffle shape')
    original_bag = Counter(map(int, result.ravel()))
    _require(max(original_bag.values()) <= height, 'Unique-per-batch bag is infeasible')
    frequencies = [Counter(map(int, row)) for row in result]
    initial_duplicates = sum(width - len(f) for f in frequencies)
    swaps = fallbacks = 0
    for offset in range(result.size):
        row, col = divmod(offset, width)
        value = int(result[row, col])
        if frequencies[row][value] <= 1:
            continue

        def acceptable(other, position):
            candidate = int(result[other, position])
            return (other != row and frequencies[other][value] == 0
                    and frequencies[row][candidate] == 0)

        selected = None
        for _ in range(64):
            candidate = divmod(int(generator.integers(result.size)), width)
            if acceptable(*candidate):
                selected = candidate
                break
        if selected is None:
            fallbacks += 1
            first = int(generator.integers(height))
            for delta in range(height):
                other = (first + delta) % height
                if other == row or frequencies[other][value] != 0:
                    continue
                for position in range(width):
                    if acceptable(other, position):
                        selected = (other, position)
                        break
                if selected is not None:
                    break
        _require(selected is not None, 'Frozen bounded collision repair has no legal swap')
        other, position = selected
        replacement = int(result[other, position])
        result[row, col], result[other, position] = replacement, value
        frequencies[row][value] -= 1
        frequencies[row][replacement] += 1
        frequencies[other][replacement] -= 1
        frequencies[other][value] += 1
        swaps += 1
    final_duplicates = sum(width - sum(count > 0 for count in f.values()) for f in frequencies)
    _require(final_duplicates == 0 and Counter(map(int, result.ravel())) == original_bag,
             'Independent repair failed occurrence conservation/uniqueness')
    return result, dict(
        algorithm='global_shuffle_then_deterministic_cross_job_swaps_v1',
        original_duplicate_occurrences=initial_duplicates,
        final_duplicate_occurrences=final_duplicates, swaps=swaps,
        fallback_searches=fallbacks, random_candidates_per_duplicate=64,
        fallback_bound='At most jobs*128 candidates per duplicate; fail without resampling',
        consulted_routing_or_timing=False, exact_occurrence_multiset_preserved=True)


def _properties(measured, prompts, positions):
    sequence = measured.ravel()
    docs = prompts[sequence]
    offsets = positions[sequence]
    transitions = docs[1:] == docs[:-1]
    runs = np.diff(np.r_[0, np.flatnonzero(~transitions) + 1, len(sequence)])
    unique_records = [len(set(map(int, row))) for row in measured]
    unique_docs = [len(set(map(int, prompts[row]))) for row in measured]
    return dict(
        same_document_adjacent_fraction=float(np.count_nonzero(transitions) / len(transitions)),
        next_position_same_document_fraction=float(np.count_nonzero(
            transitions & (offsets[1:] == offsets[:-1] + 1)) / len(transitions)),
        repeated_record_adjacent_fraction=float(np.count_nonzero(sequence[1:] == sequence[:-1]) / len(transitions)),
        mean_distinct_documents_per_job=float(sum(unique_docs) / len(unique_docs)),
        minimum_distinct_records_per_job=min(unique_records),
        jobs_with_repeated_record=sum(x < BATCH for x in unique_records),
        repeated_record_occurrences_within_jobs=sum(BATCH - x for x in unique_records),
        longest_same_document_run_tokens=int(max(runs)),
        interpreted_as_natural_serving_order=False)


def reconstruct(prompt_index, token_position, block_id, jobs=4096, seed=SEED):
    """Return ``(indices_by_mode, checked_metadata_by_mode)`` for one bank.

    Metadata is intentionally a checked subset: factual numerical fields,
    hashes, constraints and scope flags. Human-readable composition prose and
    outer capture-file provenance are not regenerated here.
    """
    _require(type(jobs) is int and jobs > 0 and type(seed) is int and seed >= 0,
             'Invalid schedule size/seed')
    prompts, positions = np.asarray(prompt_index), np.asarray(token_position)
    documents = _documents(prompts, positions)
    order = _random(seed, block_id, 'document_order').permutation(len(documents))
    cycle = np.concatenate([documents[int(p)] for p in order])
    length = jobs * BATCH
    flat = cycle[np.arange(length) % len(cycle)]
    contiguous = flat.reshape(jobs, BATCH).copy()
    permutation = _random(seed, block_id, 'whole_batch_permutation').permutation(jobs)
    mixed = _random(seed, block_id, 'mixed_occurrence_permutation').permutation(flat).reshape(jobs, BATCH)
    mixed, repair = _repair(mixed, _random(seed, block_id, 'mixed_collision_repair'))
    warm = np.empty((WARMUP, BATCH), dtype=np.int64)
    for row in range(WARMUP):
        warm[row] = _random(seed, block_id, ['warmup', row]).permutation(len(cycle))[:BATCH]
    measured = dict(contiguous=contiguous, mixed=mixed,
                    batch_shuffle=contiguous[permutation])
    occurrence_counts = np.bincount(flat, minlength=len(cycle)).astype(np.int64)
    arrays, metadata = {}, {}
    for mode in MODES:
        observed = measured[mode]
        _require(np.array_equal(np.bincount(observed.ravel(), minlength=len(cycle)), occurrence_counts),
                 'Reconstructed modes changed the token-occurrence multiset')
        full = np.vstack((warm, observed)).astype(np.int64)
        arrays[mode] = full
        metadata[mode] = dict(
            schema='sequence_hidden_jobs_v1', block_id=block_id, mode=mode,
            batch_tokens=BATCH, warmup_jobs=WARMUP, measured_jobs=jobs, seed=seed,
            source_records=len(cycle), source_documents=len(documents),
            document_order=order.tolist(), one_cycle_sha256=array_hash(cycle),
            complete_bank_cycles=length // len(cycle), final_partial_cycle_tokens=length % len(cycle),
            document_order_repeated_each_cycle=True,
            final_truncation='Stop after exactly jobs*128 token occurrences; final captured document may be partial',
            indices_sha256=array_hash(full), warmup_indices_sha256=array_hash(warm),
            measured_occurrence_multiset_sha256=array_hash(occurrence_counts),
            measured_record_occurrences=length, token_use_counts=occurrence_counts.tolist(),
            distinct_measured_records=int(np.count_nonzero(occurrence_counts)),
            minimum_record_uses=int(occurrence_counts.min()), maximum_record_uses=int(occurrence_counts.max()),
            batch_permutation_sha256=array_hash(permutation) if mode == 'batch_shuffle' else None,
            measured_sequence_properties=_properties(observed, prompts, positions),
            not_independent_requests=True, natural_batch_trace=False,
            fresh_content=False, fresh_arrival_block=False, new_ordering_only=True,
            mixed_may_repeat_captured_record_within_job=False,
            collision_repair=repair if mode == 'mixed' else None,
            causal_scope=('Temporal batch ordering only relative to contiguous' if mode == 'batch_shuffle'
                          else 'Mixed versus contiguous jointly changes batching and temporal order'))
    return arrays, metadata


def check_inputs(indices, metadata, prompt_index, token_position, block_id, *, reconstruction=None):
    """Verify saved arrays and factual metadata; allow outer provenance keys.

    Pass ``reconstruction=reconstruct(...)`` to reuse one reconstruction across
    the modes from a bank. The caller must only cache by authenticated bank,
    block ID, seed and job count.
    """
    mode = metadata.get('mode')
    _require(mode in MODES, 'Unknown saved input mode')
    if reconstruction is None:
        reconstruction = reconstruct(prompt_index, token_position, block_id,
                                     jobs=metadata['measured_jobs'], seed=metadata['seed'])
    expected, factual = reconstruction
    value = np.asarray(indices)
    _require(value.dtype == np.dtype('int64') and np.array_equal(value, expected[mode]),
             'Saved indices differ from independent frozen reconstruction')
    for key, wanted in factual[mode].items():
        _require(metadata.get(key) == wanted, 'Input metadata mismatch: ' + key)
    return dict(passed=True, mode=mode, block_id=block_id, prepared_batches=len(value),
                measured_batches=len(value) - WARMUP,
                independently_reconstructed_indices_sha256=array_hash(value),
                measured_occurrence_multiset_sha256=factual[mode]['measured_occurrence_multiset_sha256'],
                checked_metadata_fields=len(factual[mode]),
                actual_sequence_properties=factual[mode]['measured_sequence_properties'],
                bank_authentication_scope='Caller authenticates capture-file bytes and source provenance')
