"""Immutable normalized gamma calendars, generated before GPU measurements."""
from fractions import Fraction
import hashlib
import json
import math
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def arrival_blocks():
    states = [('opt-pilot-01', 'domain0'), ('opt-pilot-01', 'domain1'),
              ('opt-pilot-02', 'domain0'), ('opt-pilot-02', 'domain1')]
    return [dict(id=f'arrival-fresh-{i+1:02d}', seed=2026110901+i, block_id=state[0], regime=state[1])
            for i, state in enumerate(states*2)]


def frozen_weights(seed, pattern):
    if pattern not in ('spread', 'cluster'):
        raise ValueError('Unknown arrival pattern')
    shape = 4.0 if pattern == 'spread' else .25
    rng_seed = int(seed)*2 + (pattern == 'cluster')
    rng = random.Random(rng_seed)
    draws = [rng.gammavariate(shape, 1.0) for _ in range(256)]
    if any(not math.isfinite(x) or x <= 0 for x in draws):
        raise ValueError('Gamma sampler returned an invalid positive gap')
    result = dict(seed=int(seed), rng_seed=rng_seed, pattern=pattern, shape=shape,
                  generator='Python Random.gammavariate; exact binary64 draws archived as hex',
                  gamma_draws_hex=[x.hex() for x in draws], n_jobs=256)
    result['weights_sha256'] = digest(result)
    return result


def make_calendar(weights, rho, calibration_elapsed_ns, calibration_jobs=256):
    if weights.get('weights_sha256') != digest({k:v for k,v in weights.items() if k != 'weights_sha256'}):
        raise ValueError('Changed gamma weight draw')
    if rho not in (.5, .85) or type(calibration_elapsed_ns) is not int or calibration_elapsed_ns <= 0 or calibration_jobs != 256:
        raise ValueError('Invalid fixed calibration/load contract')
    if weights['n_jobs'] != 256 or len(weights['gamma_draws_hex']) != 256:
        raise ValueError('Exactly 256 gap weights required')
    values = [Fraction.from_float(float.fromhex(x)) for x in weights['gamma_draws_hex']]
    if any(x <= 0 for x in values):
        raise ValueError('Nonpositive gap')
    # N=calibration_jobs. Last weight is the final release-to-horizon interval.
    ideal_horizon = Fraction(calibration_elapsed_ns, 1) / Fraction(str(rho))
    total = sum(values)
    cumulative = Fraction(0)
    offsets = []
    for value in values:
        offsets.append(round(ideal_horizon*cumulative/total))
        cumulative += value
    horizon = round(ideal_horizon)
    # Some tiny gamma gaps round to zero. They are legitimate simultaneous jobs.
    # A release rounded to H is capped at H-1ns because the all-job accounting
    # contract requires every release strictly before H. All jobs are retained.
    offsets = [min(x, horizon-1) for x in offsets]
    if horizon <= 1 or offsets[0] != 0 or offsets != sorted(offsets):
        raise ValueError('Calendar does not fit the integer clock')
    result = dict(schedule_schema_version=2, pattern=weights['pattern'], rho=float(rho),
                  seed=weights['seed'], shape=weights['shape'], weights_sha256=weights['weights_sha256'],
                  calibration_jobs=256, calibration_elapsed_ns=calibration_elapsed_ns,
                  n_jobs=256, horizon_ns=horizon, arrival_offsets_ns=offsets,
                  arrival_rate_jobs_per_second=256*1e9/horizon,
                  rounding='exact archived binary64 weights; nearest nanosecond ties-to-even; releases capped at H-1ns',
                  arrival_unit='synthetic preassembled B-token layer job',
                  process='fixed-count normalized gamma-gap calendar, not an unconditioned renewal process')
    result['schedule_sha256'] = digest(result)
    return result


def validate_calendar(calendar):
    if calendar.get('schedule_sha256') != digest({k:v for k,v in calendar.items() if k != 'schedule_sha256'}):
        raise ValueError('Calendar checksum changed')
    offsets = calendar['arrival_offsets_ns']
    if (len(offsets) != 256 or offsets[0] != 0 or offsets != sorted(offsets)
            or any(type(x) is not int or x < 0 for x in offsets)
            or offsets[-1] >= calendar['horizon_ns']):
        raise ValueError('Calendar count/order/horizon changed')
    return True
