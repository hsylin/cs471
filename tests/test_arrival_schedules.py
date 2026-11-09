"""Deterministic CPU clock fixtures; none are measured benchmark evidence."""
import copy
import unittest

from arrival_study.schedules import calibration_capacity, fcfs_metrics, generate_schedule, validate_schedule


def row_fixture(offsets, services, *, horizon, epoch=10**18, admission_lag=0, dispatch_lag=0):
    admits, starts, finishes = [], [], []
    previous_finish = epoch
    for offset, service in zip(offsets, services):
        admit = max(epoch + offset + admission_lag, previous_finish)
        start = admit + dispatch_lag
        finish = start + service
        admits.append(admit)
        starts.append(start)
        finishes.append(finish)
        previous_finish = finish
    return dict(n_jobs=len(offsets), arrival_offsets_ns=offsets, horizon_ns=horizon,
                epoch_monotonic_ns=epoch, coordinator_admit_ns=admits,
                coordinator_start_ns=starts, coordinator_finish_ns=finishes)


class ArrivalSchedulesTests(unittest.TestCase):
    def test_calibration_includes_interjob_gaps(self):
        c = calibration_capacity([0, 12, 24], [10, 22, 34], expected_jobs=3)
        self.assertEqual(c['calibration_elapsed_ns'], 34)
        self.assertAlmostEqual(c['native_mean_cycle_ns'], 34 / 3)
        self.assertEqual(c['mean_calibration_service_ns'], 10)
        self.assertAlmostEqual(c['native_capacity_jobs_per_second'], 3e9 / 34)

    def test_calibration_rejects_omission_overlap_and_empty(self):
        for starts, finishes, expected in (([0], [1], 2), ([0, 2], [3, 4], 2), ([], [], 0), ([1], [1], 1)):
            with self.subTest(starts=starts, expected=expected), self.assertRaises(ValueError):
                calibration_capacity(starts, finishes, expected_jobs=expected)

    def test_constant_and_burst_have_identical_rate_count_and_horizon(self):
        kwargs = dict(rho=.85, calibration_elapsed_ns=256_000_003)
        constant = generate_schedule('constant', **kwargs)
        burst = generate_schedule('burst8', **kwargs)
        for key in ('n_jobs', 'horizon_ns', 'arrival_rate_jobs_per_second', 'native_mean_cycle_ns'):
            self.assertEqual(constant[key], burst[key])
        self.assertEqual(len(set(constant['arrival_offsets_ns'])), 256)
        self.assertEqual(len(set(burst['arrival_offsets_ns'])), 32)
        for group in range(32):
            self.assertEqual(burst['arrival_offsets_ns'][8*group:8*group+8], [constant['arrival_offsets_ns'][8*group]] * 8)
        self.assertEqual(constant, generate_schedule('constant', **kwargs))
        self.assertEqual(validate_schedule(constant), constant['schedule_sha256'])

    def test_fraction_rounding_is_exact_ties_to_even(self):
        plan = generate_schedule('constant', rho=.8, calibration_elapsed_ns=8, calibration_jobs=8, n_jobs=8)
        self.assertEqual(plan['arrival_offsets_ns'], [0, 1, 2, 4, 5, 6, 8, 9])
        self.assertEqual(plan['horizon_ns'], 10)

    def test_frozen_schedule_rejects_changed_calendar_and_parameters(self):
        plan = generate_schedule('constant', rho=.5, calibration_elapsed_ns=25600)
        for change in ('calendar', 'rho', 'hash', 'missing'):
            altered = copy.deepcopy(plan)
            if change == 'calendar':
                altered['arrival_offsets_ns'][5] += 1
            elif change == 'rho':
                altered['rho'] = .85
            elif change == 'hash':
                altered['schedule_sha256'] = '0' * 64
            else:
                del altered['calibration_jobs']
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_schedule(altered)

    def test_rejects_invalid_calendar_parameters(self):
        for rho in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                generate_schedule('constant', rho=rho, calibration_elapsed_ns=25600)
        for overrides in ({'pattern': 'poisson'}, {'n_jobs': 7}, {'burst_size': 4}, {'calibration_elapsed_ns': 0}):
            args = dict(pattern='burst8', rho=.5, calibration_elapsed_ns=25600)
            args.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                generate_schedule(**args)

    def test_constant_jobs_have_zero_queue_and_exact_large_epoch_arithmetic(self):
        result = fcfs_metrics(row_fixture([0, 10, 20, 30], [4]*4, horizon=40))
        self.assertTrue(result['complete'])
        self.assertEqual(result['vectors_ns']['response'], [4]*4)
        self.assertEqual(result['vectors_ns']['queue_wait'], [0]*4)
        self.assertEqual(result['queue_occupancy']['integral_job_ns'], 0)
        self.assertEqual(result['system_occupancy']['integral_job_ns'], 16)
        self.assertEqual(result['system_occupancy']['mean_jobs'], .4)
        self.assertEqual(result['trailing_scheduled_idle_ns'], 6)

    def test_burst_queue_contains_all_simultaneously_arrived_jobs(self):
        result = fcfs_metrics(row_fixture([0]*8, [3]*8, horizon=30))
        self.assertEqual(result['vectors_ns']['queue_wait'], list(range(0, 24, 3)))
        self.assertEqual(result['vectors_ns']['response'], list(range(3, 27, 3)))
        self.assertEqual(result['queue_occupancy']['peak_jobs'], 7)
        self.assertEqual(result['system_occupancy']['peak_jobs'], 8)
        self.assertEqual(result['queue_occupancy']['integral_job_ns'], 84)
        self.assertEqual(result['system_occupancy']['integral_job_ns'], 108)

    def test_service_spike_does_not_move_arrivals_or_omit_waiting_jobs(self):
        row = row_fixture([0, 10, 20, 30], [35, 5, 5, 5], horizon=40)
        original_offsets = row['arrival_offsets_ns'][:]
        result = fcfs_metrics(row)
        self.assertEqual(row['arrival_offsets_ns'], original_offsets)
        self.assertEqual(result['vectors_ns']['response'], [35, 30, 25, 20])
        self.assertEqual(result['vectors_ns']['queue_wait'], [0, 25, 20, 15])
        self.assertEqual(result['backlog_at_horizon'], 2)
        self.assertEqual(result['waiting_jobs_at_horizon'], 1)
        self.assertEqual(result['drain_ns'], 10)
        self.assertEqual(result['required_observation_extent_ns'], 50)

    def test_admission_lateness_is_part_of_response_not_external_arrival(self):
        result = fcfs_metrics(row_fixture([0], [8], horizon=30, admission_lag=15, dispatch_lag=2))
        self.assertEqual(result['vectors_ns'], {'response': [25], 'queue_wait': [17], 'service': [8],
                                              'admission_delay': [15], 'dispatch_delay': [2]})

    def test_all_finished_by_horizon_means_equal_throughput_at_fixed_offered_rate(self):
        fast = fcfs_metrics(row_fixture([0, 10, 20, 30], [2]*4, horizon=40))
        slow = fcfs_metrics(row_fixture([0, 10, 20, 30], [8]*4, horizon=40))
        self.assertEqual(fast['completed_by_horizon_per_second'], slow['completed_by_horizon_per_second'])
        self.assertEqual(fast['completed_per_evaluation_second'], slow['completed_per_evaluation_second'])
        self.assertLess(fast['summary_ns']['response']['mean'], slow['summary_ns']['response']['mean'])

    def test_explicit_observation_overshoot_is_not_hidden(self):
        row = row_fixture([0, 10], [3, 3], horizon=20)
        row['observation_end_ns'] = row['epoch_monotonic_ns'] + 25
        result = fcfs_metrics(row)
        self.assertEqual(result['required_observation_extent_ns'], 20)
        self.assertEqual(result['observation_overshoot_ns'], 5)
        self.assertEqual(result['completed_per_evaluation_second'], 2e9 / 25)

    def test_incomplete_run_retains_unfinished_and_unreleased_jobs(self):
        row = row_fixture([0, 10, 20, 30], [4]*4, horizon=40)
        epoch = row['epoch_monotonic_ns']
        row['coordinator_admit_ns'][1:] = [epoch + 10, None, None]
        row['coordinator_start_ns'][1:] = [epoch + 10, None, None]
        row['coordinator_finish_ns'][1:] = [None]*3
        row['observation_end_ns'] = epoch + 15
        result = fcfs_metrics(row)
        self.assertFalse(result['complete'])
        self.assertFalse(result['horizon_observed'])
        self.assertEqual(result['unfinished_jobs'], 3)
        self.assertEqual(result['system_occupancy']['jobs_at_end'], 1)
        self.assertEqual(result['unfinished_due_latency_lower_bound_ns'], [None, 5, None, None])
        self.assertIsNone(result['summary_ns'])
        self.assertIsNone(result['backlog_at_horizon'])
        self.assertIsNone(result['completed_by_horizon_per_second'])

    def test_incomplete_after_horizon_reports_real_backlog(self):
        row = row_fixture([0, 10], [5, 5], horizon=20)
        row['coordinator_finish_ns'][1] = None
        row['observation_end_ns'] = row['epoch_monotonic_ns'] + 25
        result = fcfs_metrics(row)
        self.assertTrue(result['horizon_observed'])
        self.assertEqual(result['backlog_at_horizon'], 1)
        self.assertEqual(result['unfinished_due_latency_lower_bound_ns'], [None, 15])

    def test_rejects_omitted_jobs_overlapping_service_and_early_admission(self):
        original = row_fixture([0, 10], [5, 5], horizon=20)
        for error in ('omitted', 'overlap', 'early', 'finish_before_start', 'end_before_event'):
            row = copy.deepcopy(original)
            if error == 'omitted':
                row['coordinator_finish_ns'].pop()
            elif error == 'overlap':
                row['arrival_offsets_ns'][1] = 0
                row['coordinator_admit_ns'][1] = row['epoch_monotonic_ns'] + 4
            elif error == 'early':
                row['coordinator_admit_ns'][1] -= 1
            elif error == 'finish_before_start':
                row['coordinator_finish_ns'][1] = row['coordinator_start_ns'][1] - 1
            else:
                row['observation_end_ns'] = row['epoch_monotonic_ns'] + 14
            with self.subTest(error=error), self.assertRaises(ValueError):
                fcfs_metrics(row)

    def test_rank_execution_must_fit_coordinator_service(self):
        row = row_fixture([0, 10], [5, 5], horizon=20)
        row['per_rank_start_ns'] = [row['coordinator_start_ns'][:], row['coordinator_start_ns'][:]]
        row['per_rank_gpu_complete_ns'] = [row['coordinator_finish_ns'][:], row['coordinator_finish_ns'][:]]
        self.assertTrue(fcfs_metrics(row)['complete'])
        row['per_rank_gpu_complete_ns'][1][1] += 1
        with self.assertRaises(ValueError):
            fcfs_metrics(row)


if __name__ == '__main__':
    unittest.main()
