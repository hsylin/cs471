"""Boundary checks for the independent supplemental statistics reconstruction."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'sequence_auditor', Path(__file__).with_name('audit_sequence_statistics.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(differences, treatment_drain=0., baseline_drain=10.):
    index, metrics = {}, {}
    for mode in ('mixed', 'contiguous'):
        for rep in (0, 1):
            for arm in audit.ARMS:
                key = ('bundle', mode, rep, arm)
                service = [10] * 4096 if arm == 'stockoff' else [10 + x for x in differences]
                starts = list(range(0, 4096 * 100, 100))
                index[key] = dict(dataset='gsm8k_train', coordinator_start_ns=starts,
                                  coordinator_finish_ns=[s + d for s, d in zip(starts, service)])
                metrics[key] = {m: 10. for m in audit.METRICS}
                metrics[key]['drain_ns'] = baseline_drain if arm == 'stockoff' else treatment_drain
    return audit.independent_statistics(index, metrics, [{'block_id': 'bundle'}])


class IndependentStatisticsTests(unittest.TestCase):
    def test_zero_drain_is_a_valid_ratio_and_zero_denominator_interaction_is_undefined(self):
        _, bundles, interactions, breaks = fixture([-2] * 4096)
        self.assertTrue(all(b['ratios']['drain_ns'] == 0. for b in bundles))
        self.assertTrue(all(x['contiguous_over_mixed_relative_effect']['drain_ns'] is None for x in interactions))
        self.assertTrue(all(x['terminal_difference_ns'] == -8192
                            and x['first_job_remaining_nonpositive'] == 0 for x in breaks))

    def test_zero_baseline_drain_does_not_invent_a_ratio(self):
        _, bundles, _, _ = fixture([0] * 4096, baseline_drain=0.)
        self.assertTrue(all(b['ratios']['drain_ns'] is None for b in bundles))

    def test_break_even_requires_the_entire_remaining_suffix(self):
        # Cumulative differences: -1, +1, 0, -1, -2, ... . The initial
        # negative prefix is not lasting payback because the next is positive.
        _, _, _, breaks = fixture([-1, 2, -1] + [-1] * 4093)
        self.assertTrue(all(x['first_job_remaining_nonpositive'] == 2 for x in breaks))

    def test_positive_terminal_cost_has_no_lasting_break_even(self):
        _, _, _, breaks = fixture([0] * 4095 + [1])
        self.assertTrue(all(x['first_job_remaining_nonpositive'] is None for x in breaks))


if __name__ == '__main__':
    unittest.main()
