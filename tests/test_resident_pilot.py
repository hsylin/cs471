"""CPU checks of executed pilot grouping and local order counterbalancing."""
import copy
import itertools

import pytest

from resident_dispatch.pilot import ARMS, pilot_groups


def manifest():
    return {
        "matrix": {"arms": list(ARMS), "repetitions": 2,
                   "block_ids": ["pilot-a", "pilot-b"], "domains": [0, 1],
                   "batch_tokens": [128, 1024]},
        "execution": {"order_seed": 471},
    }


def test_every_declared_cell_arm_repeat_appears_exactly_once():
    groups = list(pilot_groups(manifest()))
    jobs = [(g["block_id"], g["domain"], g["batch_tokens"], g["repetition"], arm)
            for g, order in groups for arm in order]
    expected = set(itertools.product(["pilot-a", "pilot-b"], [0, 1], [128, 1024],
                                     [0, 1], ARMS))
    assert len(jobs) == len(set(jobs)) == 80
    assert set(jobs) == expected


def test_local_pair_orders_reverse_every_arm_pair_and_center_positions():
    groups = list(pilot_groups(manifest()))
    for i in range(0, len(groups), 2):
        first, forward = groups[i]
        second, reverse = groups[i + 1]
        # Adjacent cell repeats permit a single immutable input artifact/cache.
        assert {k: v for k, v in first.items() if k != "repetition"} == {
            k: v for k, v in second.items() if k != "repetition"}
        assert (first["repetition"], second["repetition"]) == (0, 1)
        assert reverse == list(reversed(forward))
        for arm in ARMS:
            assert forward.index(arm) + reverse.index(arm) == len(ARMS) - 1
        for left, right in itertools.combinations(ARMS, 2):
            assert (forward.index(left) < forward.index(right)) != (
                reverse.index(left) < reverse.index(right))


def test_order_is_reproducible_without_mutating_frozen_input():
    m = manifest()
    original = copy.deepcopy(m)
    a = list(pilot_groups(m))
    assert a == list(pilot_groups(m))
    assert m == original
    m["execution"]["order_seed"] += 1
    assert [g for g, _ in a] != [g for g, _ in pilot_groups(m)]


@pytest.mark.parametrize("change", ["arm_order", "missing_control", "repetitions"])
def test_unreviewed_pilot_design_is_rejected(change):
    m = manifest()
    if change == "arm_order":
        m["matrix"]["arms"] = list(reversed(ARMS))
    elif change == "missing_control":
        m["matrix"]["arms"].remove("pre_native")
    else:
        m["matrix"]["repetitions"] = 1
    with pytest.raises(ValueError):
        list(pilot_groups(m))
