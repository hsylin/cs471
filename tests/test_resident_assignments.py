"""Independent small-case oracles for offline physical-copy assignments."""
import itertools

import numpy as np
import pytest

from resident_dispatch.assignments import generate_assignment, validate_assignment


def _native_scalar(ids, mapping):
    # Deliberately scalar and separate from selector vectorization.
    out = np.empty(ids.shape, dtype=np.int64)
    for token in range(len(ids)):
        for choice in range(ids.shape[1]):
            copies = [slot for slot, expert in enumerate(mapping)
                      if expert == ids[token, choice]]
            hash_value = (token * 2654435769) & 0xFFFFFFFF
            out[token, choice] = copies[hash_value % len(copies)]
    return out


def _brute_home_objectives(ids, mapping):
    active = np.unique(ids)
    homes = [[i for i, value in enumerate(mapping) if value == expert]
             for expert in active]
    best = None
    for choices in itertools.product(*homes):
        chosen = dict(zip(active, choices))
        physical = np.array([[chosen[e] for e in row] for row in ids])
        counts = np.bincount(physical.reshape(-1), minlength=len(mapping)).reshape(2, -1)
        key = (int((counts > 0).sum(1).max()), int(counts.sum(1).max()))
        best = key if best is None else min(best, key)
    return best


def test_native_exact_pinned_hash_real_geometry_and_row_not_route_index():
    mapping = np.r_[np.arange(128), np.arange(32)]
    ids = np.tile(np.array([0, 1, 31, 32, 79, 80, 127, 5]), (257, 1))
    got = generate_assignment(ids, mapping)
    np.testing.assert_array_equal(got.physical_ids, _native_scalar(ids, mapping))
    assert got.physical_ids[0, 0] == 0
    assert got.physical_ids[1, 0] == 128
    # Both shared experts in the same token use the same hash, not flat position.
    assert got.physical_ids[1, 1] == 129
    assert sum(got.diagnostics["per_rank_tokens"]) == ids.size


def test_native_three_copies_and_asymmetric_noncontiguous_ownership():
    mapping = np.array([0, 1, 0, 2, 0, 3, 1, 4])
    ids = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 0]])
    np.testing.assert_array_equal(generate_assignment(ids, mapping).physical_ids,
                                  _native_scalar(ids, mapping))


@pytest.mark.parametrize("seed", range(12))
def test_home_matches_exhaustive_count_and_token_lexicographic_oracle(seed):
    rng = np.random.default_rng(seed)
    # Singleton owners, shared experts, and extra copies on one rank.
    mapping = np.array([0, 1, 2, 0, 0, 1, 3, 4])
    ids = rng.integers(0, 5, size=(13, 2), dtype=np.int64)
    got = generate_assignment(ids, mapping, method="one_copy_home")
    objective = (got.diagnostics["max_rank_active_copies"],
                 got.diagnostics["max_rank_tokens"])
    assert objective == _brute_home_objectives(ids, mapping)
    for expert in np.unique(ids):
        assert len(np.unique(got.physical_ids[ids == expert])) == 1
    np.testing.assert_array_equal(got.physical_ids,
        generate_assignment(ids, mapping, method="one_copy_home").physical_ids)


def test_hot_shared_expert_exposes_count_vs_token_objective_tradeoff():
    mapping = np.array([0, 1, 2, 3, 0, 1, 4, 5])
    ids = np.array([0] * 99 + [1] + [2, 3, 4, 5]).reshape(-1, 1)
    home = generate_assignment(ids, mapping, method="one_copy_home")
    balanced = generate_assignment(ids, mapping, method="token_balanced")
    assert home.diagnostics["max_rank_active_copies"] == 3
    assert home.diagnostics["max_rank_tokens"] == 101
    assert balanced.diagnostics["max_rank_tokens"] == 52
    assert balanced.diagnostics["max_rank_active_copies"] == 4
    assert home.diagnostics["runtime_optimality_claimed"] is False


@pytest.mark.parametrize("fixed0,fixed1,shared", [(20, 1, 3), (1, 20, 3),
                                                (2, 3, 8), (0, 0, 7), (8, 1, 0)])
def test_token_balanced_attains_unit_route_peak_with_fixed_owner_constraints(
        fixed0, fixed1, shared):
    mapping = np.array([0, 1, 0, 2])
    ids = np.array([1]*fixed0 + [2]*fixed1 + [0]*shared).reshape(-1, 1)
    got = generate_assignment(ids, mapping, method="token_balanced")
    oracle = min(max(fixed0+x, fixed1+shared-x) for x in range(shared+1))
    assert got.diagnostics["max_rank_tokens"] == oracle


def test_one_copy_home_can_differ_from_token_only_greedy_assignment():
    mapping = np.array([0, 1, 2, 3, 0, 1, 2, 4])
    ids = np.array([0]*11 + [1]*8 + [2]*3 + [3]*2 + [4]*12).reshape(-1, 1)
    result = generate_assignment(ids, mapping, method="one_copy_home")
    assert (result.diagnostics["max_rank_active_copies"],
            result.diagnostics["max_rank_tokens"]) == _brute_home_objectives(ids, mapping)


def test_counts_and_m_tiles_at_zero_exact_and_overflow_boundaries():
    mapping = np.arange(4)
    ids = np.array([0]*64 + [1]*65 + [2]*1).reshape(-1, 1)
    got = generate_assignment(ids, mapping, block_size_m=64)
    assert got.diagnostics["physical_counts"] == [64, 65, 1, 0]
    assert got.diagnostics["per_rank_tokens"] == [129, 1]
    assert got.diagnostics["per_rank_active_copies"] == [2, 1]
    assert got.diagnostics["per_rank_m_tiles"] == [3, 1]
    assert got.diagnostics["per_rank_padded_rows"] == [192, 64]


@pytest.mark.parametrize("method", ["native", "one_copy_home", "token_balanced"])
def test_preserves_inputs_route_order_shape_and_independent_conservation(method):
    mapping = np.r_[np.arange(128), np.arange(32)]
    ids = np.random.default_rng(7).integers(0, 128, size=(128, 8), dtype=np.int32)
    saved_ids, saved_mapping = ids.copy(), mapping.copy()
    got = generate_assignment(ids, mapping, method=method)
    np.testing.assert_array_equal(ids, saved_ids)
    np.testing.assert_array_equal(mapping, saved_mapping)
    np.testing.assert_array_equal(mapping[got.physical_ids], ids)
    assert got.physical_ids.shape == ids.shape
    assert not got.physical_ids.flags.writeable
    assert not np.shares_memory(got.physical_ids, ids)
    checked = validate_assignment(ids, got.physical_ids, mapping)
    assert checked["logical_counts"] == np.bincount(ids.ravel(), minlength=128).tolist()
    assert checked["total_routes"] == 1024


@pytest.mark.parametrize("method", ["native", "one_copy_home", "token_balanced"])
def test_empty_batch_is_conserved(method):
    got = generate_assignment(np.empty((0, 8), dtype=np.int64),
                              np.array([0, 1]), method=method)
    assert got.physical_ids.shape == (0, 8)
    assert got.diagnostics["per_rank_tokens"] == [0, 0]
    assert got.diagnostics["minimum_possible_max_rank_active_copies"] == 0


@pytest.mark.parametrize("ids,mapping,kwargs", [
    (np.array([0]), np.array([0, 1]), {}),
    (np.array([[0.0]]), np.array([0, 1]), {}),
    (np.array([[-1]]), np.array([0, 1]), {}),
    (np.array([[2]]), np.array([0, 1]), {}),
    (np.array([[0]]), np.array([0, 2]), {}),
    (np.array([[0]]), np.array([0, 1, 0]), {}),
    (np.array([[0]]), np.array([0, -1]), {}),
    (np.array([[0]]), np.array([0, 1]), {"world_size": 1}),
    (np.array([[0]]), np.array([0, 1]), {"world_size": 2.0}),
    (np.array([[0]]), np.array([0, 1]), {"block_size_m": 0}),
    (np.array([[0]]), np.array([0, 1]), {"block_size_m": True}),
    (np.array([[0]]), np.array([0, 1]), {"method": "runtime_oracle"}),
])
def test_invalid_inputs_fail_closed(ids, mapping, kwargs):
    with pytest.raises(ValueError):
        generate_assignment(ids, mapping, **kwargs)


@pytest.mark.parametrize("physical", [np.array([[1, 0]]), np.array([[0, 2]]),
                                      np.array([[0.0, 1.0]]), np.array([0, 1])])
def test_independent_validator_rejects_ineligible_or_malformed_assignment(physical):
    with pytest.raises(ValueError):
        validate_assignment(np.array([[0, 1]]), physical, np.array([0, 1]))
