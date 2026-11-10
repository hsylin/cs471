import numpy as np
import pytest

from full_serving.resident_policy import balanced_map, cpu_reference, validate_map


def test_balanced_residency_has_exact_budget_and_distinct_copy_ranks():
    mapping = np.asarray(balanced_map())
    counts = np.bincount(mapping, minlength=128)
    assert len(mapping) == 160 and sorted(counts.tolist()) == [1] * 96 + [2] * 32
    for rank in range(8):
        assert mapping.reshape(8, 20)[rank, :16].tolist() == list(range(16 * rank, 16 * rank + 16))
    for expert in np.flatnonzero(counts == 2):
        assert len(set(np.flatnonzero(mapping == expert) // 20)) == 2
    validate_map(mapping.tolist())
    with pytest.raises(ValueError):
        validate_map(list(range(128)) + list(range(32)))


@pytest.mark.parametrize("tokens", [0, 1, 17, 513, 4097])
def test_route_identity_conservation_padding_and_one_home(tokens):
    rng = np.random.default_rng(471 + tokens)
    ids = rng.integers(0, 128, (tokens, 8))
    ids[::3] = -1
    result = cpu_reference(ids)
    out = result["physical_ids"]
    valid = ids >= 0
    assert np.array_equal(out == -1, ~valid)
    assert np.array_equal(np.asarray(balanced_map())[out[valid]], ids[valid])
    assert result["physical_counts"].sum() == valid.sum()
    for expert in np.unique(ids[valid]):
        assert len(np.unique(out[ids == expert])) == 1


def test_uniform_work_balances_active_copies_and_tokens():
    ids = np.tile(np.arange(128).reshape(16, 8), (3, 1))
    result = cpu_reference(ids)
    assert result["active_per_rank"].tolist() == [16] * 8
    assert result["tokens_per_rank"].tolist() == [48] * 8


def test_native_hash_obeys_sorted_native_replica_order_and_local_token_index():
    ids = np.full((7, 8), 0, dtype=np.int32)
    result = cpu_reference(ids, mode="native_hash")["physical_ids"]
    assert result[:, 0].tolist() == [0, 36, 0, 36, 0, 36, 0]


@pytest.mark.parametrize("bad", [-2, 128])
def test_invalid_real_logical_routes_are_rejected(bad):
    with pytest.raises(ValueError):
        cpu_reference(np.full((1, 8), bad))
