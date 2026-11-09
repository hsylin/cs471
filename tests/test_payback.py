"""Causal orchestration contracts; CPU fixtures are never GPU evidence."""
from collections import Counter
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

import followup.payback as p


def matrix_manifest():
    blocks, regimes, batches = ["pilot-01", "pilot-02"], ["stable", "transient", "persistent", "switching"], [128, 1024]
    return {"matrix": {"arms": list(p.ARMS), "block_ids": blocks, "regimes": regimes,
                       "batch_tokens": batches, "horizons": [128, 512],
                       "candidate_ids": ["native"], "repetitions": 3},
            "tapes": [{"block_id": b, "regime": r, "batch_tokens": n}
                      for b in blocks for r in regimes for n in batches],
            "execution": {"order_seed": 471}}


def test_future_starts_after_second_observed_decision_and_excludes_prefix():
    assert p.replay_bounds(16, 127, 128, 672, 64) == (160, 288)
    assert p.replay_bounds(16, 127, 512, 672, 64) == (160, 672)
    assert 159 not in range(*p.replay_bounds(16, 127, 128, 672, 64))
    with pytest.raises(ValueError, match="scheduled"):
        p.replay_bounds(16, 128, 128, 672, 64)
    with pytest.raises(ValueError, match="complete declared future"):
        p.replay_bounds(16, 127, 513, 672, 64)
    with pytest.raises(ValueError, match="R >= 2W"):
        p.replay_bounds(16, 15, 10, 672, 16)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 7, 24, 32, 96])
def test_order_is_deterministic_position_balanced_and_uses_all_six_permutations(count):
    orders = p.balanced_arm_orders(count, 471)
    assert orders == p.balanced_arm_orders(count, 471)
    assert all(set(order) == set(p.ARMS) for order in orders)
    for position in range(3):
        totals = Counter(o[position] for o in orders)
        assert max(totals[a] for a in p.ARMS) - min(totals[a] for a in p.ARMS) <= 1
    if count % 6 == 0:
        assert Counter(map(tuple, orders)) == {order: count // 6 for order in permutations(p.ARMS)}


def test_pilot_filter_precedes_balance_and_resume_does_not_redraw_orders():
    groups = p.build_groups(matrix_manifest())
    screen = p.select_groups(groups, stage="screening")
    confirmation = p.select_groups(groups, stage="confirmation")
    assert len(screen) * 3 == 96
    assert len(confirmation) * 3 == 72
    assert {g["repetition"] for g in screen} == {0}
    assert {g["repetition"] for g in confirmation} == {1, 2}
    assert {g["horizon_steps"] for g in confirmation} == {512}
    for selected in (screen, confirmation):
        for position in range(3):
            totals = Counter(g["arm_order"][position] for g in selected)
            assert max(totals.values()) - min(totals.values()) <= 1
    # Resume drops already-completed IDs only after this deterministic selection.
    completed = {p._record_id("config", screen[0], arm) for arm in p.ARMS}
    again = p.select_groups(groups, stage="screening")
    assert [g for g in again if p._record_id("config", g, "KEEP") not in completed] == screen[1:]
    assert all("arm_order" not in g for g in groups)


def test_one_copy_keeps_coverage_slots_and_requires_remote_new_resident():
    old = np.array([0, 0, 1, 2])
    candidate = p.one_copy_candidate([100, 1, 1], old, 2)
    new = candidate["map"]
    assert np.count_nonzero(new != old) == 1 and len(new) == len(old)
    assert set(new) == {0, 1, 2}
    slot = candidate["replacement_slot"]
    rank = candidate["destination_rank"]
    assert candidate["incoming_logical_expert"] not in old[rank * 2:(rank + 1) * 2]
    assert np.bincount(old)[old[slot]] >= 2
    assert np.isclose(candidate["predicted_gain"], p.predicted_gain([100, 1, 1], old, new, 2))


def test_one_copy_retains_nonimproving_best_legal_and_breaks_ties_lexicographically():
    candidate = p.one_copy_candidate([0, 0, 0], [0, 0, 1, 2], 2)
    assert candidate["replacement_slot"] == 0 and candidate["incoming_logical_expert"] == 1
    assert candidate["predicted_gain"] == 0 and not candidate["candidate_improves_proxy"]
    assert candidate["candidate_reason"] == "no_improving_candidate"
    assert p.one_copy_candidate([1, 1], [0, 1], 2) is None
    with pytest.raises(ValueError, match="fixed slot"):
        p.validate_maps([0, 1], [0, 1, 0, 1], 2, 2)


def test_manifest_hash_excludes_only_freeze_and_runtime_pins_fail_closed():
    m = {"environment": {"world_size": 2, "dtype": "bfloat16", "followup_sha256": "a"},
         "policy": {"replica_budget": 32}}
    assert p.manifest_hash(m) == p.manifest_hash({**m, "freeze": {"sha256": "ignored"}})
    assert p.manifest_hash(m) != p.manifest_hash({**m, "policy": {"replica_budget": 64}})
    metadata = {"world_size": 2, "dtype": "bfloat16", "followup_sha256": "a"}
    p.validate_execution_metadata(m, metadata, {})
    with pytest.raises(ValueError, match="followup_sha256"):
        p.validate_execution_metadata(m, {**metadata, "followup_sha256": "b"}, {})


@pytest.mark.parametrize("arm", p.ARMS)
def test_each_arm_restores_same_source_and_times_only_prescribed_commit_and_future(monkeypatch, arm):
    events = []
    class Layer:
        def __init__(self):
            self.counter = SimpleNamespace(sent_bytes=0, sent_messages=0)
            self.w13, self.w2 = np.array([4]), np.array([8])
            self.mapping = [0, 1]
        def restore_snapshot(self, mapping, w13, w2):
            events.append("restore")
            self.mapping = list(mapping)
        def assert_maps_and_weights(self):
            events.append("validate")
        def force_migration(self, mapping):
            events.append("migrate")
            self.mapping = list(mapping)
            self.counter.sent_bytes += 10
            self.counter.sent_messages += 1
        def forward(self, hidden, ids, weights, observe=False):
            assert observe is False
            events.append(("forward", int(hidden)))
        def close(self):
            events.append("complete")
        def map_snapshot(self):
            return np.asarray(self.mapping)
        def check_outputs(self, hidden, ids, weights):
            return {"correctness_max_abs_error": 0}
    rt = SimpleNamespace(torch=SimpleNamespace(equal=np.array_equal),
                         barrier=lambda: events.append("barrier"), sync=lambda: None,
                         gather_scalars=lambda x: [x])
    def clock():
        events.append("clock")
        return float(len(events))
    monkeypatch.setattr(p, "_make_layer", lambda *a: Layer())
    monkeypatch.setattr(p.time, "perf_counter", clock)
    monkeypatch.setattr(p, "_agree", lambda rt, value, label: value)
    monkeypatch.setattr(p, "_physical_diagnostic", lambda *a: {"verified": True})
    manifest = {"replay": {"window_steps": 1, "rebalance_interval": 2},
                "execution": {"conditioning_steps": 2}}
    tape = SimpleNamespace(indices=np.zeros((8, 1), np.int64))
    source = {"source_map": [0, 1], "decision_step": 3}
    candidate = {"candidate_map": [1, 0]}
    data = p._run_arm(rt, None, manifest, tape, (np.arange(8),) * 3, source,
                      (np.array([4]), np.array([8])), candidate,
                      {"candidate_preflight_payload_by_rank": [10]},
                      {"horizon_steps": 2, "batch_tokens": 128}, arm, {})
    first_clock = events.index("clock")
    assert events.index("restore") < first_clock
    assert [e for e in events if isinstance(e, tuple)] == [("forward", 4), ("forward", 5),
                                                         ("forward", 6), ("forward", 7)]
    if arm == "PREINSTALLED":
        assert events.index("migrate") < first_clock
    elif arm == "CHARGED":
        assert first_clock < events.index("migrate") < events.index(("forward", 6))
    else:
        assert "migrate" not in events
    assert data["migration_bytes"] == (10 if arm == "CHARGED" else 0)
    assert data["steps_completed"] == 2 and data["source_restore_verified"]
    assert data["future_start_raw_step"] == 6 and data["future_stop_raw_step_exclusive"] == 8
