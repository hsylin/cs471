from dataclasses import replace

import numpy as np
import pytest

from skewbench.traces import TraceConfig, generate_trace, hotspot_schedule, split_seed


def config(**overrides):
    values = dict(seed=471, num_experts=8, top_k=2, tokens_per_step=17,
                  num_steps=12, window_steps=2, regime="stationary")
    values.update(overrides)
    return TraceConfig(**values)


@pytest.mark.parametrize("regime", ["uniform", "stationary", "burst", "persistent", "periodic"])
def test_trace_contract_unique_ids_weights_and_count_conservation(regime):
    trace = generate_trace(config(regime=regime))
    assert trace.expert_ids.shape == (12, 17, 2)
    assert trace.expert_ids.dtype == np.int64
    assert trace.combining_weights.dtype == np.float32
    assert np.all(trace.expert_ids >= 0)
    assert np.all(trace.expert_ids < 8)
    assert np.all(np.diff(np.sort(trace.expert_ids, axis=-1), axis=-1) > 0)
    assert np.all(np.isfinite(trace.combining_weights))
    assert np.all(trace.combining_weights >= 0)
    np.testing.assert_allclose(trace.combining_weights.sum(axis=-1), 1, atol=1e-7)
    np.testing.assert_array_equal(trace.logical_counts.sum(axis=1), np.full(12, 34))
    for step in range(12):
        np.testing.assert_array_equal(trace.logical_counts[step], np.bincount(trace.expert_ids[step].ravel(), minlength=8))
    np.testing.assert_allclose(trace.probabilities.sum(axis=1), 1)


def test_same_seed_replays_exactly_and_different_seed_changes_actual_routes():
    first = generate_trace(config())
    repeated = generate_trace(config())
    other = generate_trace(config(seed=472))
    np.testing.assert_array_equal(first.expert_ids, repeated.expert_ids)
    np.testing.assert_array_equal(first.combining_weights, repeated.combining_weights)
    assert first.fingerprint() == repeated.fingerprint()
    assert not np.array_equal(first.expert_ids, other.expert_ids)
    assert first.fingerprint() != other.fingerprint()


def test_pilot_final_seed_domains_are_disjoint_and_change_realizations():
    pilot_seeds = {split_seed("pilot", i) for i in range(10)}
    final_seeds = {split_seed("final", i) for i in range(10)}
    assert pilot_seeds.isdisjoint(final_seeds)
    pilot = generate_trace(config(seed=split_seed("pilot", 0)))
    final = generate_trace(config(seed=split_seed("final", 0)))
    assert not np.array_equal(pilot.expert_ids, final.expert_ids)


def test_warmup_burst_reversion_and_persistent_change_are_distinct():
    burst = hotspot_schedule(config(regime="burst", hotspot_lifetime_steps=3))
    persistent = hotspot_schedule(config(regime="persistent", hotspot_lifetime_steps=3))
    np.testing.assert_array_equal(burst, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(persistent, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


def test_phase_changes_window_alignment_without_changing_lifetime():
    base = config(regime="periodic", hotspot_lifetime_steps=2)
    phase_zero = hotspot_schedule(base)
    phase_one = hotspot_schedule(replace(base, phase_steps=1))
    np.testing.assert_array_equal(phase_zero, [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(phase_one, [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])


def test_repeated_bursts_have_fixed_onsets_and_return_to_baseline():
    trace_config = config(regime="burst", num_steps=22, phase_steps=1,
                          hotspot_lifetime_steps=2, burst_repeat_steps=6)
    schedule = hotspot_schedule(trace_config)
    expected = np.zeros(22, dtype=np.int8)
    expected[[5, 6, 11, 12, 17, 18]] = 1
    np.testing.assert_array_equal(schedule, expected)
    assert np.count_nonzero(schedule) == 3 * 2
    # Omission preserves the original one-burst contract.
    single = hotspot_schedule(replace(trace_config, burst_repeat_steps=None))
    np.testing.assert_array_equal(np.flatnonzero(single), [5, 6])


def test_pilot_schedule_has_declared_phase_not_implicit_cadence_alignment():
    trace_config = config(regime="burst", num_steps=512, window_steps=16,
                          phase_steps=40, hotspot_lifetime_steps=16,
                          burst_repeat_steps=128)
    schedule = hotspot_schedule(trace_config)
    onsets = np.flatnonzero(np.diff(np.r_[0, schedule]) == 1)
    np.testing.assert_array_equal(onsets, [72, 200, 328, 456])
    for onset in onsets:
        assert schedule[onset:onset + 16].sum() == 16
        assert schedule[onset + 16] == 0


def test_common_warmup_and_reset_cadence_have_explicit_burst_alignment():
    # The runner contract is 32 untimed common steps, then 512 measured steps.
    # With counter reset, opportunities after 64 measured steps are global 95,
    # 159, 223, ... . This geometry is chosen before observing any policy gain.
    trace_config = config(regime="burst", num_steps=544, window_steps=16,
                          phase_steps=56, hotspot_lifetime_steps=16,
                          burst_repeat_steps=128)
    schedule = hotspot_schedule(trace_config)
    onsets = np.flatnonzero(np.diff(np.r_[0, schedule]) == 1)
    np.testing.assert_array_equal(onsets, [88, 216, 344, 472])
    assert not schedule[:32].any()
    for proposal_step in (95, 223, 351, 479):
        assert schedule[proposal_step - 7:proposal_step + 1].sum() == 8
        assert schedule[proposal_step + 1:proposal_step + 9].sum() == 8


def test_repeat_field_does_not_change_nonburst_regimes():
    for regime in ("stationary", "uniform", "persistent", "periodic"):
        base = config(regime=regime)
        np.testing.assert_array_equal(hotspot_schedule(base),
                                      hotspot_schedule(replace(base, burst_repeat_steps=6)))


def test_zipf_hot_experts_rotate_while_stationary_and_uniform_do_not():
    moving = generate_trace(config(regime="persistent"))
    assert np.argmax(moving.probabilities[0]) == 0
    assert np.argmax(moving.probabilities[4]) == 4
    np.testing.assert_allclose(moving.probabilities[4], np.roll(moving.probabilities[0], 4))
    stationary = generate_trace(config())
    np.testing.assert_array_equal(stationary.probabilities[0], stationary.probabilities[-1])
    uniform = generate_trace(config(regime="uniform"))
    np.testing.assert_allclose(uniform.probabilities, 1 / 8)


def test_warmup_does_not_mean_identical_sampled_windows():
    trace = generate_trace(config(tokens_per_step=100, regime="stationary"))
    np.testing.assert_array_equal(trace.probabilities[0], trace.probabilities[3])
    assert not np.array_equal(trace.counts(0, 2), trace.counts(2, 4))


def test_read_only_replay_and_interval_counts():
    trace = generate_trace(config())
    for array in (trace.expert_ids, trace.combining_weights, trace.logical_counts,
                  trace.probabilities, trace.hotspot_states):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    assert trace.counts().sum() == 12 * 17 * 2
    assert trace.counts(2, 6).sum() == 4 * 17 * 2
    np.testing.assert_array_equal(trace.counts(3, 3), np.zeros(8))
    for interval in ((-1, 3), (3, 2), (0, 13)):
        with pytest.raises(ValueError):
            trace.counts(*interval)


def test_all_experts_selected_once_when_top_k_equals_expert_count():
    trace = generate_trace(config(top_k=8))
    np.testing.assert_array_equal(np.sort(trace.expert_ids, axis=-1),
                                  np.broadcast_to(np.arange(8), (12, 17, 8)))
    np.testing.assert_array_equal(trace.logical_counts, np.full((12, 8), 17))


def test_single_expert_is_a_valid_degenerate_trace():
    trace = generate_trace(config(num_experts=1, top_k=1))
    assert np.all(trace.expert_ids == 0)
    assert np.all(trace.combining_weights == 1)


@pytest.mark.parametrize("overrides", [
    {"seed": -1}, {"seed": True}, {"num_experts": 0}, {"top_k": 0},
    {"top_k": 9}, {"tokens_per_step": 0}, {"num_steps": 0},
    {"window_steps": 0}, {"regime": "unknown"}, {"phase_steps": -1},
    {"hotspot_lifetime_steps": 0}, {"zipf_exponent": np.nan},
    {"zipf_exponent": -1}, {"warmup_steps": -1},
    {"burst_repeat_steps": 0}, {"burst_repeat_steps": 2},
])
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides)
