"""Analysis checks on synthetic timestamps, never GPU observations."""
import copy
import itertools
import unittest

from arrival_study.analysis import analyze, _hash, _spans, _state, _schedule_name
from arrival_study.schedules import calibration_capacity, fcfs_metrics, generate_schedule
from tests.test_arrival_schedules import row_fixture


def fixture(phase='evaluation'):
    matrix = dict(block_ids=['fixture-a', 'fixture-b'], regimes=['domain0', 'domain1'],
                  batch_tokens=[128, 1024], arms=['native', 'online'], repetitions=2)
    plan = dict(phase=phase, matrix=matrix, arrival_source_sha256='fixture-source', schedules={}, calibration={})
    states = list(itertools.product(matrix['block_ids'], matrix['regimes'], matrix['batch_tokens']))
    for state in states:
        plan['calibration'][_state(state)] = dict(allocation_id='fixture-allocation')
        plan['schedules'][_state(state)] = [generate_schedule(pattern, rho=rho, calibration_elapsed_ns=256000)
            for pattern, rho in itertools.product(('constant', 'burst8'), (.5, .85))]
    plan['freeze'] = dict(sha256=_hash(plan))
    rows = []
    for state in states:
        schedules = [None] if phase == 'calibration' else plan['schedules'][_state(state)]
        for schedule in schedules:
            for rep, arm in ([(0, 'native')] if phase == 'calibration' else itertools.product((0, 1), ('native', 'online'))):
                offsets = [0]*256 if schedule is None else schedule['arrival_offsets_ns']
                horizon = 1 if schedule is None else schedule['horizon_ns']
                service = 1000 if arm == 'native' else 800
                row = row_fixture(offsets, [service]*256, horizon=horizon)
                row.update(dict(zip(('block_id', 'regime', 'batch_tokens'), state)))
                row.update(phase=phase, arm=arm, repetition=rep, allocation_id='fixture-allocation',
                    schedule_name='calibration_saturated' if schedule is None else _schedule_name(schedule),
                    arrival_schedule=schedule, config_hash=plan['freeze']['sha256'], arrival_source_sha256='fixture-source',
                    status='complete', evidence_eligible=False, correctness_pass=True, preflight_correctness={'correctness_pass': True},
                    normal_completion_verified=True, resident_state_verified=True, rank_assignment_agreement=True,
                    migration_bytes=0, migration_sent_messages=0, terminal_proposals_suppressed=1, policy='SKIP',
                    source_map=[0, 1], final_map=[0, 1], arrival_clock_contract=dict(implementation='clock_gettime(CLOCK_MONOTONIC)',
                    monotonic=True, adjustable=False), arm_order=['native', 'online'] if rep == 0 else ['online', 'native'],
                    capture_sha256='capture', replay_sha256='replay', source_prompt_ids=['doc-a', 'doc-b'],
                    source_weights_sha256='weights', source_weight_rank_sha256=['rank0', 'rank1'],
                    resolved_kernel_config={}, logical_routing_sha256='logical', decision_trajectory_sha256='trajectory',
                    warmup_raw_indices=list(range(16)), conditioning_raw_indices=list(range(64)))
                row['queue_metrics'] = fcfs_metrics(row)
                row['observed_window_spans'] = _spans(row)
                if schedule is None:
                    row['calibration_capacity'] = calibration_capacity(row['coordinator_start_ns'], row['coordinator_finish_ns'])
                rows.append(row)
    return plan, rows


class ArrivalAnalysisTests(unittest.TestCase):
    def test_complete_evaluation_preserves_all_repeats_without_inference(self):
        plan, rows = fixture()
        result = analyze(plan, rows)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['expected_records'], 128)
        self.assertEqual(len(result['trial_summaries']), 128)
        self.assertEqual(len(result['paired_cases']), 64)
        self.assertEqual(len(result['repeat_averaged_cases']), 32)
        for pair in result['paired_cases']:
            self.assertAlmostEqual(pair['contrasts']['service_mean_ns']['online_over_native'], .8)
        self.assertNotIn('ci95', result)
        self.assertNotIn('acceptance', result)

    def test_calibration_all_eight_states_uses_raw_cycle_denominator(self):
        plan, rows = fixture('calibration')
        result = analyze(plan, rows)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['expected_records'], 8)
        self.assertEqual(result['paired_cases'], [])
        self.assertEqual(result['trial_summaries'][0]['calibration_capacity']['native_mean_cycle_ns'], 1000)

    def test_missing_duplicate_and_mutated_summary_never_become_valid(self):
        for mutation in ('missing', 'duplicate', 'summary', 'calendar', 'trajectory', 'eligibility', 'order'):
            plan, rows = fixture()
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'duplicate':
                rows.append(copy.deepcopy(rows[0]))
            elif mutation == 'summary':
                rows[0]['queue_metrics']['summary_ns']['response']['p95'] += 1
            elif mutation == 'calendar':
                rows[0]['arrival_offsets_ns'] = rows[0]['arrival_offsets_ns'][:]
                rows[0]['arrival_offsets_ns'][1] += 1
            elif mutation == 'trajectory':
                rows[0]['decision_trajectory_sha256'] = 'changed'
            elif mutation == 'eligibility':
                rows[0]['evidence_eligible'] = True
            else:
                rows[2]['arm_order'] = rows[0]['arm_order']
                rows[3]['arm_order'] = rows[0]['arm_order']
            with self.subTest(mutation=mutation):
                result = analyze(plan, rows)
                self.assertFalse(result['valid'])

    def test_zero_wait_ratio_is_undefined_not_infinite_or_fake_gain(self):
        plan, rows = fixture()
        result = analyze(plan, rows)
        pair = next(p for p in result['paired_cases'] if p['pattern'] == 'constant' and p['rho'] == .5)
        self.assertIsNone(pair['contrasts']['queue_wait_mean_ns']['online_over_native'])
        self.assertEqual(pair['contrasts']['queue_wait_mean_ns']['online_minus_native'], 0)

    def test_nonfrozen_plan_is_rejected(self):
        plan, rows = fixture()
        plan['matrix']['repetitions'] = 3
        with self.assertRaises(ValueError):
            analyze(plan, rows)

    def test_calibration_and_evaluation_must_share_allocation(self):
        plan, rows = fixture()
        for calibration in plan['calibration'].values():
            calibration['allocation_id'] = 'other-allocation'
        plan['freeze']['sha256'] = _hash({key: value for key, value in plan.items() if key != 'freeze'})
        for row in rows:
            row['config_hash'] = plan['freeze']['sha256']
        result = analyze(plan, rows)
        self.assertFalse(result['valid'])
        self.assertTrue(all('calibration_evaluation_allocation_mismatch' in error['issues']
            for error in result['errors'] if isinstance(error, dict)))

    def test_busy_fraction_is_coordinator_occupancy(self):
        plan, rows = fixture()
        result = analyze(plan, rows)
        pair = next(p for p in result['paired_cases'] if p['pattern'] == 'constant' and p['rho'] == .5)
        self.assertEqual(pair['contrasts']['coordinator_server_busy_fraction']['native'], .5)
        self.assertEqual(pair['contrasts']['coordinator_server_busy_fraction']['online'], .4)


if __name__ == '__main__':
    unittest.main()
