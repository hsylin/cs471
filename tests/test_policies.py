import numpy as np
import pytest

from skewbench.policies import evaluate_admission, predicted_rank_loads, total_variation


CURRENT = np.array([400, 300, 150, 150])
PREVIOUS = np.array([390, 305, 155, 150])
OLD = np.array([0, 1, 2, 3])
NEW = np.array([0, 2, 1, 3])


def decide(mode="stability", **overrides):
    args = dict(
        current_counts=CURRENT,
        previous_counts=PREVIOUS,
        current_map=OLD,
        proposed_map=NEW,
        world_size=2,
        epsilon=0.1,
        tau=0.05,
        step=20,
        windows_complete=True,
    )
    args.update(overrides)
    return evaluate_admission(mode, **args)


def test_proposal_worked_example_and_changed_history():
    decision = decide()
    assert decision.allowed
    assert decision.tv == pytest.approx(0.01)
    assert decision.predicted_gain == pytest.approx(150 / 700)
    changed = decide(previous_counts=[100, 150, 500, 250])
    assert not changed.allowed
    assert changed.reason == "unstable_demand"
    assert changed.tv == pytest.approx(0.45)
    assert changed.predicted_gain == pytest.approx(decision.predicted_gain)


def test_replicas_share_one_experts_demand_without_inventing_work():
    # E0 is duplicated once per rank, E1 exists only on rank 0, E2 on rank 1.
    result = predicted_rank_loads([600, 300, 100], [0, 1, 0, 2], 2)
    np.testing.assert_allclose(result, [0.6, 0.4])
    assert result.sum() == pytest.approx(1)
    # Two replicas on the same rank still conserve the expert's total work.
    same_rank = predicted_rank_loads([600, 300, 100], [0, 0, 1, 2], 2)
    np.testing.assert_allclose(same_rank, [0.6, 0.4])


def test_gain_and_tv_are_invariant_to_absolute_assignment_scale():
    small = decide()
    large = decide(current_counts=CURRENT * 100, previous_counts=PREVIOUS * 100)
    assert small == large


def test_expert_relabeling_preserves_admission():
    permutation = np.array([2, 0, 3, 1])  # old logical ID -> new ID
    counts = np.empty_like(CURRENT)
    previous = np.empty_like(PREVIOUS)
    counts[permutation] = CURRENT
    previous[permutation] = PREVIOUS
    remapped = decide(current_counts=counts, previous_counts=previous,
                     current_map=permutation[OLD], proposed_map=permutation[NEW])
    assert remapped == decide()


def test_thresholds_are_inclusive_and_reject_immediately_beyond_boundary():
    measured = decide()
    assert decide(epsilon=measured.predicted_gain, tau=measured.tv).allowed
    assert not decide(epsilon=np.nextafter(measured.predicted_gain, np.inf)).allowed
    assert not decide(tau=np.nextafter(measured.tv, -np.inf)).allowed


def test_improvement_does_not_require_a_previous_window():
    assert decide("improvement", previous_counts=None, windows_complete=(False, True)).allowed
    blocked = decide("stability", previous_counts=None, windows_complete=(False, True))
    assert not blocked.allowed
    assert blocked.reason == "incomplete_previous_window"
    for mode in ("improvement", "stability", "cooldown"):
        assert not decide(mode, windows_complete=(True, False)).allowed


def test_zero_windows_do_not_produce_nan_decisions():
    for mode in ("improvement", "stability", "cooldown"):
        decision = decide(mode, current_counts=np.zeros(4))
        assert decision.reason == "zero_current_load"
        assert not decision.allowed
        assert decision.predicted_gain is None
    assert decide(previous_counts=np.zeros(4)).reason == "zero_previous_load"
    with pytest.raises(ValueError, match="zero-total"):
        total_variation(CURRENT, np.zeros(4))


def test_native_and_always_skip_do_not_add_observation_requirements():
    assert decide("native", windows_complete=False, current_counts=np.zeros(4)).allowed
    skip = decide("skip", windows_complete=False, current_counts=np.zeros(4))
    assert not skip.allowed
    assert skip.reason == "always_skip"


def test_cooldown_counts_engine_steps_since_last_acceptance():
    assert decide("cooldown", cooldown_steps=8, last_accepted_step=None).allowed
    assert not decide("cooldown", cooldown_steps=8, last_accepted_step=13).allowed
    assert decide("cooldown", cooldown_steps=8, last_accepted_step=12).allowed
    # Passing a cooldown cannot override an unprofitable proposal.
    assert not decide("cooldown", cooldown_steps=8, last_accepted_step=0,
                      proposed_map=OLD).allowed


def test_unmodified_or_worse_candidate_is_rejected_by_improvement_rule():
    assert decide("improvement", proposed_map=OLD).reason == "below_improvement_threshold"
    worse = decide("improvement", current_map=NEW, proposed_map=OLD)
    assert not worse.allowed
    assert worse.predicted_gain < 0


@pytest.mark.parametrize("overrides", [
    {"current_counts": [-1, 2, 3, 4]},
    {"current_counts": [np.nan, 2, 3, 4]},
    {"current_counts": [np.inf, 2, 3, 4]},
    {"current_counts": [[1, 2, 3, 4]]},
    {"previous_counts": [1, 2, 3]},
    {"current_map": [0, 0, 2, 3]},
    {"proposed_map": [0, 1, 2, -1]},
    {"proposed_map": [0, 1, 2, 4]},
    {"proposed_map": [0.0, 1.0, 2.0, 3.0]},
    {"proposed_map": [0, 1, 2, 3, 0, 1]},
    {"world_size": 3},
    {"world_size": 0},
    {"epsilon": 0},
    {"epsilon": np.nan},
    {"tau": -0.1},
    {"tau": 1.1},
    {"step": -1},
    {"last_accepted_step": 21},
    {"cooldown_steps": -1},
    {"windows_complete": (True, 1)},
])
def test_invalid_inputs_fail_explicitly(overrides):
    with pytest.raises(ValueError):
        decide(**overrides)


def test_decision_does_not_mutate_observations_or_maps():
    arrays = [CURRENT.copy(), PREVIOUS.copy(), OLD.copy(), NEW.copy()]
    copies = [a.copy() for a in arrays]
    evaluate_admission("stability", *arrays, world_size=2)
    for actual, expected in zip(arrays, copies):
        np.testing.assert_array_equal(actual, expected)
