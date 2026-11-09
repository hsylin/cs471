"""CPU contracts for matched policy runs; these are not GPU benchmark evidence."""
from collections import Counter
from copy import deepcopy
import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.runner import groups, trajectory_hash, validate_observations


def design(blocks=8, repetitions=2, seed=10471):
    return {
        'matrix': {
            'kernel_variants': ['DEFAULT', 'TUNED'],
            'policies': ['B3', 'SKIP'],
            'block_ids': [f'block-{i:02d}' for i in range(blocks)],
            'regimes': ['stable', 'transient', 'persistent', 'switching'],
            'batch_tokens': [128, 1024], 'repetitions': repetitions,
        },
        'execution': {'order_seed': seed},
    }


@pytest.mark.parametrize('blocks,repetitions,expected_groups', [(2, 1, 16), (8, 2, 128)])
def test_declared_matrix_has_one_complete_balanced_quartet_per_key(blocks, repetitions, expected_groups):
    manifest = design(blocks, repetitions)
    actual = groups(manifest)
    matrix = manifest['matrix']
    keys = [tuple(group[name] for name in ('block_id', 'regime', 'batch_tokens', 'repetition'))
            for group, _ in actual]
    expected_keys = set(itertools.product(matrix['block_ids'], matrix['regimes'],
                                         matrix['batch_tokens'], range(repetitions)))
    assert len(actual) == len(set(keys)) == expected_groups
    assert set(keys) == expected_keys
    arms = set(itertools.product(matrix['kernel_variants'], matrix['policies']))
    for _, order in actual:
        assert len(order) == 4 and set(order) == arms
    # Every intervention occupies every execution position equally often for
    # the actual pilot and final matrix sizes, not merely in expectation.
    for arm in arms:
        positions = Counter(order.index(arm) for _, order in actual)
        assert positions == {position: expected_groups // 4 for position in range(4)}


def test_group_order_is_reproducible_and_does_not_use_global_rng_or_mutate_manifest():
    manifest = design()
    before = deepcopy(manifest)
    state = random.getstate()
    first = groups(manifest)
    assert random.getstate() == state
    assert groups(manifest) == first and manifest == before
    other = groups(design(seed=10472))
    assert [g for g, _ in other] == [g for g, _ in first]
    assert [o for _, o in other] != [o for _, o in first]


def test_group_generator_rejects_nonfactorial_arm_count():
    manifest = design()
    manifest['matrix']['policies'] = ['B3']
    with pytest.raises(ValueError, match='two-kernel/two-policy'):
        groups(manifest)


@pytest.fixture
def decision_log():
    return [
        {'step': 63, 'allowed': True, 'reason': 'allowed',
         'current_counts': [10, 20, 30, 40], 'current_map': [0, 1, 2, 3, 0, 1],
         'proposed_map': [0, 1, 2, 3, 2, 1], 'committed_map': [0, 1, 2, 3, 2, 1],
         'transfer_sent_bytes': 9437184,
         'physical_slot_assignments_window': [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
         'planner_seconds': .02, 'controller_seconds': .001,
         'migration_seconds': .008, 'elapsed_seconds': .5},
        {'step': 127, 'allowed': False, 'reason': 'below_improvement_threshold',
         'current_counts': [40, 30, 20, 10], 'current_map': [0, 1, 2, 3, 2, 1],
         'proposed_map': [0, 1, 2, 3, 0, 1], 'committed_map': [0, 1, 2, 3, 2, 1],
         'transfer_sent_bytes': 0,
         'physical_slot_assignments_window': [[2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2]],
         'planner_seconds': .01, 'controller_seconds': .001,
         'migration_seconds': 0., 'elapsed_seconds': .4},
    ]


def test_trajectory_hash_ignores_elapsed_measurements_and_dictionary_insertion_order(decision_log):
    changed = deepcopy(decision_log)
    for row in changed:
        for field in ('planner_seconds', 'controller_seconds', 'migration_seconds', 'elapsed_seconds'):
            row[field] += 100.
        row['per_rank_elapsed_seconds'] = [9., 10.]
        row['new_untimed_diagnostic'] = {'irrelevant': True}
    assert trajectory_hash(changed) == trajectory_hash(decision_log)
    reordered = [dict(reversed(list(row.items()))) for row in decision_log]
    assert trajectory_hash(reordered) == trajectory_hash(decision_log)


@pytest.mark.parametrize('field', ['current_map', 'proposed_map', 'committed_map'])
def test_trajectory_hash_detects_a_physical_slot_change(decision_log, field):
    changed = deepcopy(decision_log)
    changed[0][field][0], changed[0][field][1] = changed[0][field][1], changed[0][field][0]
    assert trajectory_hash(changed) != trajectory_hash(decision_log)


@pytest.mark.parametrize('field', ['step', 'allowed', 'reason', 'current_counts',
                                   'transfer_sent_bytes', 'physical_slot_assignments_window'])
def test_trajectory_hash_detects_policy_observation_and_payload_changes(decision_log, field):
    changed = deepcopy(decision_log)
    if field == 'step': changed[0][field] += 1
    elif field == 'allowed': changed[0][field] = False
    elif field == 'reason': changed[0][field] = 'skip'
    elif field == 'current_counts': changed[0][field][0] += 1
    elif field == 'transfer_sent_bytes': changed[0][field] += 9437184
    else: changed[0][field][0][0] += 1
    assert trajectory_hash(changed) != trajectory_hash(decision_log)


def test_trajectory_hash_detects_reordered_or_missing_decisions(decision_log):
    assert trajectory_hash(list(reversed(decision_log))) != trajectory_hash(decision_log)
    assert trajectory_hash(decision_log[:-1]) != trajectory_hash(decision_log)


@pytest.fixture
def observation_fixture():
    # Each successive input changes its logical histogram. Thus an off-by-one
    # endpoint or omitted warmup offset cannot accidentally pass a total check.
    logical = ((np.arange(544)[:, None, None] +
                np.arange(4)[None, :, None] * 7 + np.arange(2)[None, None, :]) % 128).astype(np.int64)
    tape = SimpleNamespace(expert_ids=logical)
    decisions = []
    for step in range(63, 511, 64):
        first, last = 32 + step - 15, 32 + step + 1
        counts = np.zeros(128, np.int64)
        # Deliberately derive the reference by an explicit per-assignment count,
        # independently of the implementation's flattened bincount expression.
        for assigned in logical[first:last].flat:
            counts[int(assigned)] += 2
        decisions.append({'step': step, 'current_counts': counts.tolist()})
    return tape, decisions


def test_exact_native_windows_include_current_input_and_replicated_rank_factor(observation_fixture):
    tape, decisions = observation_fixture
    assert validate_observations(decisions, tape, warmup=32, window=16, world_size=2) == 7
    assert all(sum(d['current_counts']) == 16 * 4 * 2 * 2 for d in decisions)


@pytest.mark.parametrize('warmup,window,world_size', [(31, 16, 2), (33, 16, 2),
                                                   (32, 15, 2), (32, 17, 2), (32, 16, 1)])
def test_wrong_window_alignment_length_or_replicated_input_scale_fails(observation_fixture, warmup, window, world_size):
    tape, decisions = observation_fixture
    with pytest.raises(AssertionError, match='exact replay'):
        validate_observations(decisions, tape, warmup, window, world_size)


def test_per_expert_conservation_is_checked_not_only_total_assignments(observation_fixture):
    tape, decisions = observation_fixture
    altered = deepcopy(decisions)
    nonzero = next(i for i, count in enumerate(altered[0]['current_counts']) if count)
    altered[0]['current_counts'][nonzero] -= 1
    altered[0]['current_counts'][(nonzero + 1) % 128] += 1
    assert sum(altered[0]['current_counts']) == sum(decisions[0]['current_counts'])
    with pytest.raises(AssertionError, match='exact replay'):
        validate_observations(altered, tape, 32, 16, 2)


def test_future_and_unobserved_inputs_do_not_enter_a_current_window(observation_fixture):
    tape, decisions = observation_fixture
    changed = tape.expert_ids.copy()
    first_window_start, first_window_stop = 80, 96
    changed[:first_window_start] = 0
    changed[first_window_stop:] = 127
    assert validate_observations(decisions[:1], SimpleNamespace(expert_ids=changed), 32, 16, 2) == 1
    changed[first_window_stop - 1, 0, 0] = (changed[first_window_stop - 1, 0, 0] + 1) % 128
    with pytest.raises(AssertionError, match='exact replay'):
        validate_observations(decisions[:1], SimpleNamespace(expert_ids=changed), 32, 16, 2)
