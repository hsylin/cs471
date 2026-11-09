"""Hand-checkable CPU fixtures for the independent auditor, never GPU evidence."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/verify_instructor_arrivals.py'
SPEC = importlib.util.spec_from_file_location('independent_arrival_verifier', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def four_jobs():
    # Two simultaneous-release groups. Queue waits 1,5,1,6; service 3,5,4,5.
    return dict(n_jobs=4, epoch_monotonic_ns=100, horizon_ns=40,
        arrival_offsets_ns=[0, 0, 20, 20], coordinator_admit_ns=[100, 104, 120, 125],
        coordinator_start_ns=[101, 105, 121, 126], coordinator_finish_ns=[104, 110, 125, 131],
        per_rank_start_ns=[[102, 106, 122, 127], [102, 106, 122, 127]],
        per_rank_gpu_complete_ns=[[103, 109, 124, 130], [103, 109, 124, 130]],
        observation_end_ns=142)


def schedule(pattern='constant', rho=.5):
    # Native cycle = 10 ns: exact spacings 20 ns at load .5, independent fixture.
    group = 1 if pattern == 'constant' else 8
    value = dict(schedule_schema_version=1, pattern=pattern, rho=rho,
        calibration_jobs=256, calibration_elapsed_ns=2560,
        native_capacity_jobs_per_second=100_000_000., native_mean_cycle_ns=10.,
        arrival_rate_jobs_per_second=50_000_000., n_jobs=256, burst_size=8,
        horizon_ns=5120, arrival_offsets_ns=[(i//group)*group*20 for i in range(256)],
        rounding='nearest_nanosecond_ties_to_even',
        arrival_unit='synthetic preassembled B-token layer job')
    value['schedule_sha256'] = audit.digest(value)
    return value


def test_known_queue_service_response_and_idle():
    result = audit.timing_metrics(four_jobs())
    assert result['vectors_ns']['queue_wait'] == [1, 5, 1, 6]
    assert result['vectors_ns']['service'] == [3, 5, 4, 5]
    assert result['vectors_ns']['response'] == [4, 10, 5, 11]
    assert result['summary_ns']['response']['mean'] == 7.5
    assert result['summary_ns']['response']['p50'] == 7.5
    assert result['summary_ns']['response']['p95'] == pytest.approx(10.85)
    assert result['summary_ns']['response']['p99'] == pytest.approx(10.97)
    assert result['queue_occupancy']['integral_job_ns'] == 13
    assert result['system_occupancy']['integral_job_ns'] == 30
    assert result['queue_occupancy']['peak_jobs'] == 2
    assert result['system_occupancy']['peak_jobs'] == 2
    assert result['drain_ns'] == 0
    assert result['trailing_scheduled_idle_ns'] == 9
    assert result['observation_overshoot_ns'] == 2
    assert result['required_observation_extent_ns'] == 40
    assert result['completed_per_evaluation_second'] == pytest.approx(4e9/42)
    assert audit.flat_metrics(result)['coordinator_server_busy_fraction'] == pytest.approx(17/42)


def test_drain_is_not_censored_at_release_horizon():
    row = four_jobs()
    row['horizon_ns'] = 25
    result = audit.timing_metrics(row)
    assert result['drain_ns'] == 6
    assert result['backlog_at_horizon'] == 1
    assert result['waiting_jobs_at_horizon'] == 1
    assert result['completed_jobs'] == 4
    assert result['vectors_ns']['response'][-1] == 11


@pytest.mark.parametrize('field,index,value', [
    ('coordinator_admit_ns', 2, 119),  # Earlier than immutable release.
    ('coordinator_start_ns', 1, 103),  # Earlier than admission.
    ('coordinator_admit_ns', 1, 103),  # Overlap with first completed service.
    ('coordinator_finish_ns', 0, 101),  # Nonpositive service.
    ('coordinator_finish_ns', 3, None),  # Dropped/unfinished tail.
    ('coordinator_start_ns', 0, True),  # A flag is not a timestamp.
])
def test_invalid_fcfs_or_omitted_jobs_rejected(field, index, value):
    row = four_jobs()
    row[field][index] = value
    with pytest.raises(AssertionError):
        audit.timing_metrics(row)


def test_shortened_arrays_rejected():
    row = four_jobs()
    row['coordinator_finish_ns'].pop()
    with pytest.raises(AssertionError, match='missing jobs'):
        audit.timing_metrics(row)


def test_ack_before_rank_gpu_completion_rejected():
    row = four_jobs()
    row['per_rank_gpu_complete_ns'][1][0] = 105
    with pytest.raises(AssertionError, match='envelope'):
        audit.timing_metrics(row)


def test_horizon_observation_cannot_end_early():
    row = four_jobs()
    row['observation_end_ns'] = 131
    with pytest.raises(AssertionError, match='horizon'):
        audit.timing_metrics(row)


def test_native_capacity_includes_between_job_gaps():
    result = audit.capacity(four_jobs())
    assert result['calibration_elapsed_ns'] == 30
    assert result['native_mean_cycle_ns'] == 7.5
    assert result['mean_calibration_service_ns'] == 4.25
    assert result['native_capacity_jobs_per_second'] == pytest.approx(4e9/30)


@pytest.mark.parametrize('pattern', ['constant', 'burst8'])
def test_known_equal_horizon_calendars(pattern):
    result = audit.calendar(schedule(pattern))
    assert result['horizon_ns'] == 5120
    assert len(result['arrival_offsets_ns']) == 256
    assert result['arrival_offsets_ns'][8] == 160
    assert result['arrival_offsets_ns'][-1] == (5100 if pattern == 'constant' else 4960)


@pytest.mark.parametrize('numerator,denominator,expected', [(1, 2, 0), (3, 2, 2), (5, 2, 2), (7, 2, 4), (8, 3, 3)])
def test_integer_ties_to_even(numerator, denominator, expected):
    assert audit.nearest_even(numerator, denominator) == expected


def test_changed_calendar_rejected_even_with_updated_checksum():
    changed = schedule()
    changed['arrival_offsets_ns'][3] += 1
    changed['schedule_sha256'] = audit.digest({k: v for k, v in changed.items() if k != 'schedule_sha256'})
    with pytest.raises(AssertionError):
        audit.calendar(changed)


def test_production_summary_corruption_detected():
    expected = audit.timing_metrics(four_jobs())
    bad = copy.deepcopy(expected)
    bad['summary_ns']['response']['p95'] += .01
    with pytest.raises(AssertionError, match='p95'):
        audit.compare(bad, expected, 'summary')


def test_native_window_spans_use_raw_endpoints():
    row = dict(arrival_offsets_ns=list(range(0, 512, 2)),
               coordinator_start_ns=list(range(1000, 1512, 2)),
               coordinator_finish_ns=list(range(1001, 1513, 2)))
    expected = [dict(end_step=e, jobs=16, scheduled_arrival_span_ns=30,
                     service_coverage_span_ns=31, completion_span_ns=30)
                for e in (63, 127, 191, 255)]
    assert audit.window_spans(row) == expected


def test_compressed_raw_record_identity_preserved(tmp_path):
    raw = b'{"example":1}\n{"example":2}\n'
    plain, compressed = tmp_path/'records.jsonl', tmp_path/'records.jsonl.gz'
    plain.write_bytes(raw)
    with gzip.open(compressed, 'wb') as out:
        out.write(raw)
    assert audit.original_raw_hash(plain) == audit.original_raw_hash(compressed) == hashlib.sha256(raw).hexdigest()
    assert audit.read_records([compressed]) == [{'example': 1}, {'example': 2}]
