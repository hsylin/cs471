"""Immutable wall-clock calendars and FCFS accounting for synthetic layer jobs.

An arrival is a scheduled layer-job eligibility time, not a real chat request.
Coordinator admission is an observation of the implicit queue, not an external
network enqueue. These functions perform no waits, GPU work or measurements.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral

import numpy as np


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def calibration_capacity(start_ns, finish_ns, *, expected_jobs=256):
    """Saturated cycle time includes inter-job dispatch/bookkeeping gaps."""
    expected_jobs = _integer(expected_jobs, "expected_jobs", minimum=1)
    if len(start_ns) != expected_jobs or len(finish_ns) != expected_jobs:
        raise ValueError("calibration must contain every declared job")
    starts = [_integer(x, "calibration start") for x in start_ns]
    finishes = [_integer(x, "calibration finish") for x in finish_ns]
    if any(f <= s for s, f in zip(starts, finishes)) or any(starts[i] < finishes[i - 1] for i in range(1, expected_jobs)):
        raise ValueError("calibration must be positive-duration, nonoverlapping FCFS work")
    elapsed = finishes[-1] - starts[0]
    return {"calibration_jobs": expected_jobs, "calibration_elapsed_ns": elapsed,
            "native_capacity_jobs_per_second": expected_jobs * 1e9 / elapsed,
            "native_mean_cycle_ns": elapsed / expected_jobs,
            "mean_calibration_service_ns": sum(f - s for s, f in zip(starts, finishes)) / expected_jobs,
            "definition": "jobs/(last coordinator finish-first coordinator command start); includes inter-job gaps"}


def generate_schedule(pattern, *, rho, calibration_elapsed_ns,
                      calibration_jobs=256, n_jobs=256, burst_size=8):
    """Freeze equal-count/equal-horizon constant or simultaneous-burst arrivals.

    Fractions preserve the declared decimal rho; nanoseconds use nearest-integer
    rounding with ties to even. The first arrival is zero. No completion time
    participates in generating or changing this calendar.
    """
    if pattern not in {"constant", "burst8"}:
        raise ValueError("pattern must be constant or burst8")
    elapsed = _integer(calibration_elapsed_ns, "calibration_elapsed_ns", minimum=1)
    calibration_jobs = _integer(calibration_jobs, "calibration_jobs", minimum=1)
    n_jobs = _integer(n_jobs, "n_jobs", minimum=1)
    burst_size = _integer(burst_size, "burst_size", minimum=1)
    if burst_size != 8 or n_jobs % burst_size:
        raise ValueError("the declared calendar requires complete bursts of eight")
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    fraction = Fraction(str(rho))
    interval = Fraction(elapsed, calibration_jobs) / fraction
    group = 1 if pattern == "constant" else burst_size
    offsets = [round((i // group) * group * interval) for i in range(n_jobs)]
    horizon = round(n_jobs * interval)
    if offsets[0] != 0 or horizon <= offsets[-1] or horizon >= 2 ** 63 or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("calendar cannot be represented safely in integer nanoseconds")
    if pattern == "constant" and len(set(offsets)) != n_jobs:
        raise ValueError("constant arrival spacing is below nanosecond resolution")
    capacity = calibration_jobs * 1e9 / elapsed
    plan = {"schedule_schema_version": 1, "pattern": pattern, "rho": float(rho),
            "calibration_jobs": calibration_jobs, "calibration_elapsed_ns": elapsed,
            "native_capacity_jobs_per_second": capacity,
            "native_mean_cycle_ns": elapsed / calibration_jobs,
            "arrival_rate_jobs_per_second": float(fraction) * capacity,
            "n_jobs": n_jobs, "burst_size": burst_size, "horizon_ns": horizon,
            "arrival_offsets_ns": offsets, "rounding": "nearest_nanosecond_ties_to_even",
            "arrival_unit": "synthetic preassembled B-token layer job"}
    plan["schedule_sha256"] = _digest(plan)
    return plan


def validate_schedule(plan):
    """Reject altered parameters/calendar; caller archives the returned hash."""
    try:
        expected = generate_schedule(plan["pattern"], rho=plan["rho"],
            calibration_elapsed_ns=plan["calibration_elapsed_ns"], calibration_jobs=plan["calibration_jobs"],
            n_jobs=plan["n_jobs"], burst_size=plan["burst_size"])
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed frozen arrival parameters") from exc
    if plan != expected:
        raise ValueError("changed or incomplete frozen arrival calendar")
    return plan["schedule_sha256"]


def _occupancy(arrivals, departures, epoch, end):
    changes = {}
    for times, sign in ((arrivals, 1), (departures, -1)):
        for timestamp in times:
            if timestamp is not None and timestamp <= end:
                changes[timestamp] = changes.get(timestamp, 0) + sign
    area, count, peak, previous = 0, 0, 0, epoch
    for timestamp in sorted(changes):
        area += count * (timestamp - previous)
        count += changes[timestamp]
        if count < 0:
            raise ValueError("a departure precedes its arrival")
        peak = max(peak, count)
        previous = timestamp
    area += count * (end - previous)
    return {"peak_jobs": peak, "jobs_at_end": count, "integral_job_ns": area,
            "mean_jobs": area / (end - epoch) if end > epoch else 0.}


def _summary(values):
    return {"mean": float(np.mean(values)), "p50": float(np.quantile(values, .50, method="linear")),
            "p95": float(np.quantile(values, .95, method="linear")),
            "p99": float(np.quantile(values, .99, method="linear")), "max": max(values)}


def fcfs_metrics(row):
    """Validate rank-0 FCFS timing and account for every scheduled layer job.

    Exact row fields: arrival_offsets_ns, epoch_monotonic_ns, horizon_ns,
    coordinator_admit_ns, coordinator_start_ns, coordinator_finish_ns; optional
    per_rank_start_ns and per_rank_gpu_complete_ns. Missing finishes must form
    a suffix and require observation_end_ns. Incomplete runs have no complete-
    run latency summary, so timeouts cannot disappear from reported percentiles.
    """
    offsets = [_integer(x, "arrival offset") for x in row["arrival_offsets_ns"]]
    n = _integer(row.get("n_jobs", 256), "n_jobs", minimum=1)
    epoch = _integer(row["epoch_monotonic_ns"], "epoch_monotonic_ns")
    horizon = _integer(row["horizon_ns"], "horizon_ns", minimum=1)
    if len(offsets) != n or offsets[0] != 0 or offsets[-1] >= horizon or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("wrong count or nonmonotone/out-of-horizon arrival calendar")
    arrays = []
    for field in ("coordinator_admit_ns", "coordinator_start_ns", "coordinator_finish_ns"):
        if len(row[field]) != n:
            raise ValueError("missing scheduled jobs in " + field)
        arrays.append([None if value is None else _integer(value, field) for value in row[field]])
    admits, starts, finishes = arrays
    arrivals = [epoch + offset for offset in offsets]
    incomplete_seen = False
    for i, (a, e, s, f) in enumerate(zip(arrivals, admits, starts, finishes)):
        if e is not None and e < a:
            raise ValueError("admission before scheduled arrival")
        if s is not None and (e is None or s < e):
            raise ValueError("start before admission")
        if f is not None and (s is None or f < s):
            raise ValueError("finish before start")
        if i and e is not None and (finishes[i - 1] is None or e < finishes[i - 1]):
            raise ValueError("overlap or out-of-order FCFS admission")
        if f is None:
            incomplete_seen = True
        elif incomplete_seen:
            raise ValueError("completed jobs are not an FCFS prefix")
    complete = all(value is not None for value in finishes)
    last_finish = max((value for value in finishes if value is not None), default=epoch)
    if not complete and "observation_end_ns" not in row:
        raise ValueError("incomplete runs require an actual observation end")
    end = _integer(row.get("observation_end_ns", max(epoch + horizon, last_finish)), "observation_end_ns")
    if end < max(last_finish, epoch) or any(t is not None and t > end for t in admits + starts):
        raise ValueError("observation end precedes a recorded event")
    if complete and end < epoch + horizon:
        raise ValueError("complete exposure must include the full offered-load horizon")
    rank_fields = ("per_rank_start_ns", "per_rank_gpu_complete_ns")
    if any(field in row for field in rank_fields):
        if not all(field in row and len(row[field]) == 2 for field in rank_fields):
            raise ValueError("both ranks' timing arrays are required")
        for rank in range(2):
            rs, rf = (row[field][rank] for field in rank_fields)
            if len(rs) != n or len(rf) != n:
                raise ValueError("rank timing arrays omit scheduled jobs")
            for i in range(n):
                if finishes[i] is not None:
                    rstart = _integer(rs[i], "rank start")
                    rfinish = _integer(rf[i], "rank GPU complete")
                    if not starts[i] <= rstart <= rfinish <= finishes[i]:
                        raise ValueError("rank execution lies outside coordinator service interval")
    response = [None if f is None else f - a for a, f in zip(arrivals, finishes)]
    waiting = [None if s is None else s - a for a, s in zip(arrivals, starts)]
    service = [None if f is None else f - s for s, f in zip(starts, finishes)]
    admission_delay = [None if e is None else e - a for a, e in zip(arrivals, admits)]
    dispatch_delay = [None if s is None else s - e for e, s in zip(admits, starts)]
    vectors = dict(response=response, queue_wait=waiting, service=service,
                   admission_delay=admission_delay, dispatch_delay=dispatch_delay)
    queue = _occupancy(arrivals, starts, epoch, end)
    system = _occupancy(arrivals, finishes, epoch, end)
    if complete and (queue["integral_job_ns"] != sum(waiting) or system["integral_job_ns"] != sum(response)):
        raise AssertionError("event-sweep occupancy differs from per-job latency accounting")
    cutoff = epoch + horizon
    horizon_observed = end >= cutoff
    by_horizon = sum(f is not None and f <= cutoff for f in finishes)
    started_by_horizon = sum(s is not None and s <= cutoff for s in starts)
    completed = sum(f is not None for f in finishes)
    return {"complete": complete, "scheduled_jobs": n, "completed_jobs": completed,
            "unfinished_jobs": n - completed, "evaluation_end_ns": end,
            "horizon_observed": horizon_observed,
            "required_observation_extent_ns": max(horizon, last_finish - epoch) if complete else None,
            "observation_overshoot_ns": end - max(cutoff, last_finish) if complete else None,
            "horizon_ns": horizon, "drain_ns": max(0, last_finish - cutoff) if complete else None,
            "trailing_scheduled_idle_ns": max(0, cutoff - last_finish) if complete else None,
            "completed_by_horizon": by_horizon if horizon_observed else None,
            "backlog_at_horizon": n - by_horizon if horizon_observed else None,
            "waiting_jobs_at_horizon": n - started_by_horizon if horizon_observed else None,
            "offered_jobs_per_second": n * 1e9 / horizon,
            "completed_by_horizon_per_second": by_horizon * 1e9 / horizon if horizon_observed else None,
            "completed_per_evaluation_second": completed * 1e9 / (end - epoch) if end > epoch else 0.,
            "vectors_ns": vectors, "summary_ns": {key: _summary(values) for key, values in vectors.items()} if complete else None,
            "queue_occupancy": queue, "system_occupancy": system,
            "unfinished_due_latency_lower_bound_ns": [max(0, end - a) if f is None and a <= end else None for a, f in zip(arrivals, finishes)],
            "latency_definition": "coordinator completion minus immutable scheduled arrival; includes queue/admission delay and CPU command/ACK coordination",
            "arrival_unit": "synthetic preassembled B-token layer job; not a user/chat request",
            "percentile_scope": "descriptive within-run percentiles; jobs/cycles are not independent source-block replicates"}
