"""CPU contract/oracle tests; importing the policy does not load GPU libraries."""
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resident_dispatch.online import (
    _check_adversarial_result, adversarial_inputs, cpu_greedy_assignment, fixed_map,
    validate_fixed_map, validate_native_metadata,
)


def native_maps():
    logical = np.full((128, 33), -1, dtype=np.int64)
    logical[:, 0] = np.arange(128)
    logical[:32, 1] = np.arange(128, 160)
    replicas = np.r_[np.full(32, 2), np.ones(96)].astype(np.int64)
    return logical, replicas


def test_cpu_import_does_not_load_torch_vllm_or_triton(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'triton'):
            raise AssertionError('GPU import occurred in CPU policy module')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    # A second named import must execute the module, rather than use its cache.
    import sys
    path = Path(__file__).parents[1] / 'resident_dispatch' / 'online.py'
    spec = importlib.util.spec_from_file_location('resident_online_cpu_import_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.POLICY_ID == 'current_batch_one_home_lpt_active_caps_v1'


def test_fixed_residency_checks_fail_on_same_shape_map_change():
    mapping = fixed_map()
    validate_fixed_map(mapping)
    logical, replicas = native_maps()
    assert validate_native_metadata(logical, replicas, mapping)
    mapping[[0, 1]] = mapping[[1, 0]]
    with pytest.raises(ValueError): validate_fixed_map(mapping)
    logical[0, 1] = 129
    with pytest.raises(ValueError): validate_native_metadata(logical, replicas)


@pytest.mark.parametrize('kind', ['count', 'extra_copy', 'padding', 'shape'])
def test_native_metadata_changes_are_rejected(kind):
    logical, replicas = native_maps()
    if kind == 'count': replicas[0] = 1
    elif kind == 'extra_copy': logical[32, 1] = 128
    elif kind == 'padding': logical[0, -1] = 0
    else: logical = logical.T
    with pytest.raises(ValueError): validate_native_metadata(logical, replicas)


def test_equal_count_ties_are_expert_id_then_rank_zero():
    ids = np.tile(np.arange(8), (4, 1))
    result = cpu_greedy_assignment(ids, fixed_map())
    assert result.diagnostics['greedy_decisions'] == [[e, e % 2] for e in range(8)]
    assert result.homes[:8].tolist() == [0, 129, 2, 131, 4, 133, 6, 135]
    assert result.diagnostics['per_rank_tokens'] == [16, 16]


def test_hot_shared_expert_respects_remaining_active_capacity():
    # All four shared experts must use rank1: rank0 already owns four active
    # singleton experts and the exact active-count optimum is four per rank.
    ids = np.array([0]*100 + [1]*4 + [2]*3 + [3]*2 + [32, 33, 34, 35]).reshape(-1, 1)
    result = cpu_greedy_assignment(ids, fixed_map())
    assert result.homes[:4].tolist() == [128, 129, 130, 131]
    assert result.diagnostics['per_rank_active_copies'] == [4, 4]
    assert result.diagnostics['per_rank_tokens'] == [4, 109]
    assert result.diagnostics['runtime_optimality_claimed'] is False


def test_descending_load_order_accounts_for_asymmetric_singletons():
    ids = np.array([0]*3 + [1]*11 + [2]*5 + [32]*7 + [80]).reshape(-1, 1)
    result = cpu_greedy_assignment(ids, fixed_map())
    assert [e for e, _ in result.diagnostics['greedy_decisions']] == [1, 2, 0]
    # The first heavy expert goes to the less-loaded singleton owner.
    assert result.diagnostics['greedy_decisions'][0] == [1, 1]


@pytest.mark.parametrize('batch', [128, 1024])
def test_adversarial_cases_preserve_routes_and_attain_count_bound(batch):
    for case in adversarial_inputs(batch):
        ids, real = case['logical_ids'], case['real_tokens']
        saved = ids.copy()
        result = cpu_greedy_assignment(ids, fixed_map(), real_tokens=real)
        np.testing.assert_array_equal(ids, saved)
        np.testing.assert_array_equal(fixed_map()[result.physical_ids[:real]], ids[:real])
        counts = np.bincount(ids[:real].ravel(), minlength=128)
        np.testing.assert_array_equal(result.logical_counts, counts)
        active = set(ids[:real].ravel().tolist())
        f0 = len(active & set(range(32,80)))
        f1 = len(active & set(range(80,128)))
        optimum = max(f0, f1, (len(active)+1)//2)
        assert max(result.diagnostics['per_rank_active_copies']) == optimum
        assert sum(result.diagnostics['per_rank_tokens']) == real*8
        for expert in active:
            assert len(np.unique(result.physical_ids[:real][ids[:real] == expert])) == 1
        again = cpu_greedy_assignment(ids, fixed_map(), real_tokens=real)
        np.testing.assert_array_equal(again.physical_ids, result.physical_ids)
        assert not result.physical_ids.flags.writeable


def test_padding_never_changes_current_batch_counts_or_real_route_homes():
    ids = np.tile(np.arange(8), (128,1))
    initial = cpu_greedy_assignment(ids, fixed_map(), real_tokens=64)
    ids[64:] = np.arange(8)+80
    changed = cpu_greedy_assignment(ids, fixed_map(), real_tokens=64)
    np.testing.assert_array_equal(initial.homes, changed.homes)
    np.testing.assert_array_equal(initial.logical_counts, changed.logical_counts)
    ids[64:] = -1
    masked = cpu_greedy_assignment(ids, fixed_map(), real_tokens=64)
    assert np.all(masked.physical_ids[64:] == -1)


@pytest.mark.parametrize('batch,expected', [(128,[63,64,65]),
                                          (1024,[63,64,65,127,128,129])])
def test_adversarial_fixture_hits_exact_m_tile_boundaries(batch, expected):
    case = next(c for c in adversarial_inputs(batch) if c['name'] == 'tile_boundary_counts')
    counts = np.bincount(case['logical_ids'].ravel(), minlength=128)
    assert counts[:len(expected)].tolist() == expected


def test_unsigned_valid_input_agrees_with_signed_input():
    ids = np.tile(np.arange(8), (4,1))
    a = cpu_greedy_assignment(ids, fixed_map())
    b = cpu_greedy_assignment(ids.astype(np.uint64), fixed_map())
    np.testing.assert_array_equal(a.physical_ids, b.physical_ids)


@pytest.mark.parametrize('enabled', [False, True])
def test_adversarial_checker_accepts_exact_outputs_and_sentinel_recording(enabled):
    case = next(c for c in adversarial_inputs(128) if c['name'] == 'partial_real_rows')
    ref = cpu_greedy_assignment(case['logical_ids'], fixed_map(), real_tokens=case['real_tokens'])
    snapshot = {'physical_ids':ref.physical_ids.copy(), 'homes':ref.homes.copy(),
                'logical_counts':ref.logical_counts.copy(),
                'summary':np.array(ref.diagnostics['summary'])}
    load = np.full(160,7) + (np.array(ref.diagnostics['physical_counts']) if enabled else 0)
    proof = _check_adversarial_result(ref,snapshot,load,sentinel=7,record_enabled=enabled)
    assert len(proof) == 5 and all(len(v) == 64 for v in proof.values())


@pytest.mark.parametrize('corruption', ['physical_ids','homes','logical_counts','summary','load'])
def test_adversarial_checker_rejects_each_gpu_output_corruption(corruption):
    ref = cpu_greedy_assignment(np.tile(np.arange(8),(128,1)), fixed_map())
    snapshot = {'physical_ids':ref.physical_ids.copy(), 'homes':ref.homes.copy(),
                'logical_counts':ref.logical_counts.copy(),
                'summary':np.array(ref.diagnostics['summary'])}
    load = np.full(160,7)
    if corruption == 'load': load[0] += 1
    else: snapshot[corruption].flat[0] += 1
    with pytest.raises(AssertionError):
        _check_adversarial_result(ref,snapshot,load,sentinel=7,record_enabled=False)


@pytest.mark.parametrize('seed', range(8))
def test_random_current_batches_have_exact_coverage_and_feasible_caps(seed):
    ids = np.random.default_rng(seed).integers(0,128,size=(128,8))
    result = cpu_greedy_assignment(ids, fixed_map())
    np.testing.assert_array_equal(fixed_map()[result.physical_ids], ids)
    assert sum(result.diagnostics['physical_counts']) == ids.size
    assert max(result.diagnostics['per_rank_active_copies']) == result.diagnostics[
        'minimum_possible_max_rank_active_copies']


@pytest.mark.parametrize('ids,real', [(np.array([[0.0]]),None),
    (np.array([0]),None), (np.array([[-1]]),1), (np.array([[128]]),1),
    (np.array([[0]]),-1), (np.array([[0]]),2), (np.array([[0]]),True)])
def test_oracle_rejects_invalid_real_routes_and_masks(ids, real):
    with pytest.raises(ValueError):
        cpu_greedy_assignment(ids, fixed_map(), real_tokens=real)
