"""CPU schedule/state contracts; no GPU benchmark data or execution."""
import copy
import itertools
from types import SimpleNamespace
import unittest

from resident_dispatch.runner import groups, reset_observation


def manifest(final=True):
    return {"matrix": {"block_ids": [f"block-{i}" for i in range(8 if final else 2)],
                       "regimes": ["stable", "transient", "persistent", "switching"] if final else ["domain0", "domain1"],
                       "batch_tokens": [128, 1024], "repetitions": 2,
                       "arms": ["native", "online"]},
            "execution": {"order_seed": 20471}}


class Buffer:
    def __init__(self):
        self.value = "stale"

    def zero_(self):
        self.value = 0

    def fill(self, value):
        self.value = value


class ResidentRunnerTests(unittest.TestCase):
    def test_final_allocations_cover_exact_256_runs_with_local_reverse_order(self):
        m = manifest()
        first, second = list(groups(m, 0)), list(groups(m, 1))
        self.assertEqual(len(first), 64)
        self.assertEqual(len(second), 64)
        keys = set()
        for (a, oa), (b, ob) in zip(first, second):
            self.assertEqual({k: a[k] for k in a if k != "repetition"},
                             {k: b[k] for k in b if k != "repetition"})
            self.assertEqual(oa, list(reversed(ob)))
            for group, order in ((a, oa), (b, ob)):
                for arm in order:
                    keys.add((*[group[k] for k in ("block_id", "regime", "batch_tokens", "repetition")], arm))
        expected = set(itertools.product(m["matrix"]["block_ids"], m["matrix"]["regimes"],
                                         (128, 1024), (0, 1), ("native", "online")))
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), 256)
        for schedule in (first, second):
            self.assertEqual(sum(order[0] == "native" for _, order in schedule), 32)

    def test_development_has_32_runs_and_adjacent_repeated_states(self):
        schedule = list(groups(manifest(False)))
        self.assertEqual(sum(len(order) for _, order in schedule), 32)
        for index in range(0, len(schedule), 2):
            a, oa = schedule[index]
            b, ob = schedule[index + 1]
            self.assertEqual(a["repetition"], 0)
            self.assertEqual(b["repetition"], 1)
            self.assertEqual(tuple(a[k] for k in ("block_id", "regime", "batch_tokens")),
                             tuple(b[k] for k in ("block_id", "regime", "batch_tokens")))
            self.assertEqual(oa, list(reversed(ob)))

    def test_schedule_deterministic_and_rejects_missing_repetition_or_extra_arm(self):
        m = manifest()
        self.assertEqual(list(groups(m)), list(groups(copy.deepcopy(m))))
        for count in (0, 1, 3):
            invalid = copy.deepcopy(m)
            invalid["matrix"]["repetitions"] = count
            with self.assertRaises(ValueError):
                list(groups(invalid))
        for rep in (-1, 2):
            with self.assertRaises(ValueError):
                list(groups(m, rep))
        m["matrix"]["arms"].append("posthoc-choice")
        with self.assertRaises(ValueError):
            list(groups(m))

    def test_reset_clears_observation_history_and_timing_but_not_resident_state(self):
        updates = []
        state = SimpleNamespace(expert_rearrangement_step=64, expert_load_window_step=7,
                                _update_layer_should_record=lambda **kw: updates.append(kw))
        layer = SimpleNamespace(state=state, native_valid_steps=64, observation_steps=64,
                                ms=SimpleNamespace(expert_load_window=Buffer(), expert_load_pass=Buffer()),
                                logical_ring=Buffer(), logical_ring_timestamps=Buffer(),
                                histogram_capture_count=64, previous_window_timestamp_checks=1,
                                decisions=[{"step": 63}], last_accepted_step=63, planner_seconds=1.,
                                controller_seconds=2., end_of_input=True, terminal_proposals_suppressed=1,
                                counter=SimpleNamespace(sent_bytes=4, sent_messages=1), step_index=63,
                                resident_state="must remain unchanged")
        reset_observation(layer)
        self.assertEqual(layer.resident_state, "must remain unchanged")
        self.assertEqual(state.expert_rearrangement_step, 0)
        self.assertEqual(state.expert_load_window_step, 0)
        self.assertEqual(layer.native_valid_steps, 0)
        self.assertEqual(layer.logical_ring.value, 0)
        self.assertEqual(layer.logical_ring_timestamps.value, -1)
        self.assertEqual(layer.ms.expert_load_window.value, 0)
        self.assertEqual(layer.ms.expert_load_pass.value, 0)
        self.assertEqual(layer.decisions, [])
        self.assertEqual(layer.counter.sent_bytes, 0)
        self.assertEqual(layer.planner_seconds, 0)
        self.assertEqual(layer.controller_seconds, 0)
        self.assertEqual(layer.step_index, -1)
        self.assertFalse(layer.end_of_input)
        self.assertEqual(layer.terminal_proposals_suppressed, 0)
        self.assertEqual(updates, [{"log_stats": False}])


if __name__ == "__main__":
    unittest.main()
