import numpy as np
import pytest

from sequence_study.inputs import build_indices, validate_records, repair_collisions, MODES, WARMUP


def records(lengths=(37, 81, 144, 56)):
    return (np.concatenate([np.full(n, i, np.int64) for i, n in enumerate(lengths)]),
            np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]))


def test_ordering_intervention_preserves_exact_occurrences_and_common_warmup():
    docs, positions = records()
    outputs = {m: build_indices(docs, positions, "bundle-a", m, jobs=61) for m in MODES}
    first, first_meta = outputs["contiguous"]
    counts = np.bincount(first[WARMUP:].ravel(), minlength=len(docs))
    assert len(set(counts.tolist())) == 2  # Exact full cycles plus one prefix.
    for indices, meta in outputs.values():
        assert indices.shape == (61 + WARMUP, 128)
        np.testing.assert_array_equal(indices[:WARMUP], first[:WARMUP])
        np.testing.assert_array_equal(np.bincount(indices[WARMUP:].ravel(), minlength=len(docs)), counts)
        assert meta["measured_occurrence_multiset_sha256"] == first_meta["measured_occurrence_multiset_sha256"]
        assert meta["token_use_counts"] == counts.tolist()
        assert not meta["natural_batch_trace"] and not meta["fresh_content"]
    assert not np.array_equal(outputs["mixed"][0], first)


def test_contiguous_repeats_whole_bank_and_preserves_capture_order():
    docs, positions = records()
    indices, meta = build_indices(docs, positions, "bundle-a", "contiguous", jobs=61)
    flat = indices[WARMUP:].ravel()
    expected = np.concatenate([np.flatnonzero(docs == i) for i in meta["document_order"]])
    np.testing.assert_array_equal(flat, np.tile(expected, 25)[:len(flat)])
    assert meta["measured_sequence_properties"]["longest_same_document_run_tokens"] == 144
    assert meta["measured_sequence_properties"]["jobs_with_repeated_record"] == 0
    assert meta["complete_bank_cycles"] == len(flat) // len(docs)


def test_whole_batch_shuffle_changes_only_batch_order():
    docs, positions = records()
    continuous, cm = build_indices(docs, positions, "bundle-a", "contiguous", jobs=61)
    shuffled, sm = build_indices(docs, positions, "bundle-a", "batch_shuffle", jobs=61)
    assert sorted(row.tobytes() for row in continuous[WARMUP:]) == sorted(row.tobytes() for row in shuffled[WARMUP:])
    assert not np.array_equal(continuous, shuffled)
    assert sm["batch_permutation_sha256"] is not None
    assert cm["measured_sequence_properties"]["mean_distinct_documents_per_job"] == sm["measured_sequence_properties"]["mean_distinct_documents_per_job"]


def test_seed_reproducibility_and_duplicate_exposure_are_explicit():
    docs, positions = records()
    a, am = build_indices(docs, positions, "bundle-a", "mixed", jobs=61)
    b, bm = build_indices(docs, positions, "bundle-a", "mixed", jobs=61)
    c, _ = build_indices(docs, positions, "bundle-b", "mixed", jobs=61)
    np.testing.assert_array_equal(a, b)
    assert am == bm
    assert not np.array_equal(a, c)
    assert am["measured_sequence_properties"]["jobs_with_repeated_record"] == 0
    assert not am["mixed_may_repeat_captured_record_within_job"]
    assert am["collision_repair"]["original_duplicate_occurrences"] > 0
    assert am["collision_repair"]["final_duplicate_occurrences"] == 0


def test_collision_repair_dense128_record_boundary_and_impossible_bag():
    original = np.repeat(np.arange(128), 8).reshape(8, 128)
    fixed, meta = repair_collisions(original, np.random.default_rng(97))
    assert all(len(set(row)) == 128 for row in fixed)
    np.testing.assert_array_equal(np.bincount(fixed.ravel()), np.bincount(original.ravel()))
    assert meta["swaps"] > 0 and meta["final_duplicate_occurrences"] == 0
    with pytest.raises(ValueError, match="cannot avoid"):
        repair_collisions(np.zeros((2, 128), np.int64), np.random.default_rng(97))


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "negative", "bad_prompt", "short", "float"])
def test_incomplete_or_ambiguous_capture_records_rejected(mutation):
    docs, positions = records()
    if mutation == "gap": positions[5] = 999
    elif mutation == "duplicate": positions[5] = 4
    elif mutation == "negative": positions[5] = -1
    elif mutation == "bad_prompt": docs[0] = 10
    elif mutation == "short": docs, positions = docs[:100], positions[:100]
    elif mutation == "float": positions = positions.astype(float)
    with pytest.raises(ValueError): validate_records(docs, positions)


@pytest.mark.parametrize("kwargs", [dict(mode="unknown"), dict(jobs=0), dict(jobs=True), dict(seed=-1)])
def test_invalid_design_fails_without_silent_repair(kwargs):
    docs, positions = records()
    values = dict(mode="mixed", jobs=8, seed=471902)
    values.update(kwargs)
    with pytest.raises(ValueError): build_indices(docs, positions, "bundle-a", **values)
