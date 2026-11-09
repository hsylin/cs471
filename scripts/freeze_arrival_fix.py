#!/usr/bin/env python3
"""Freeze the single no-op optimization and previously unseen arrival calendars."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrival_fix.common import file_hash, source_identity, load_plan, GPU_UUIDS
from arrival_fix.calendars import arrival_blocks, frozen_weights, make_calendar
from arrival_fix.analysis import ANALYSIS_DEFAULTS
from arrival_study.schedules import calibration_capacity, fcfs_metrics
from optimization.common import manifest_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'calibration', 'final'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-plan', type=Path)
    parser.add_argument('--calibration-records', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase == 'final':
        if args.calibration_plan is None or args.calibration_records is None:
            raise ValueError('Final requires complete native calibration')
        previous, _ = load_plan(args.calibration_plan)
        if previous['phase'] != 'calibration':
            raise ValueError('Wrong input phase')
        plan = copy.deepcopy(previous)
        plan['phase'] = 'final'
        rows = [json.loads(x) for x in args.calibration_records.read_text().splitlines() if x.strip()]
        expected = set(itertools.product(('opt-pilot-01', 'opt-pilot-02'), ('domain0', 'domain1'), (128, 1024)))
        index = {tuple(row[k] for k in ('block_id', 'regime', 'batch_tokens')):row for row in rows}
        if len(rows) != 8 or set(index) != expected or len({r['allocation_id'] for r in rows}) != 1:
            raise ValueError('Eight native states in one allocation required')
        calibration = {}
        for state, row in index.items():
            if (row['phase'] != 'calibration' or row['arm'] != 'native' or row['status'] != 'complete'
                    or row['config_hash'] != manifest_hash(previous) or row['arrival_fix_source_sha256'] != previous['arrival_fix_source_sha256']
                    or not row['correctness_pass'] or not row['normal_completion_verified'] or not row['resident_state_verified']
                    or not row['rank_assignment_agreement'] or row['noop_guard_ranks'] or row['migration_bytes']
                    or sorted(row['physical_gpu_uuids']) != sorted(GPU_UUIDS) or row['hostname'] != 'kh083'
                    or row['arrival_offsets_ns'] != [0]*256 or row['horizon_ns'] != 1 or not fcfs_metrics(row)['complete']):
                raise ValueError('Invalid baseline-only calibration')
            capacity = calibration_capacity(row['coordinator_start_ns'], row['coordinator_finish_ns'])
            if capacity != row['calibration_capacity']:
                raise ValueError('Calibration summary does not match raw timestamps')
            key = f'{state[0]}/{state[1]}/B{state[2]}'
            calibration[key] = {**capacity, 'record_id':row['record_id'], 'allocation_id':row['allocation_id'],
                                'capture_sha256':row['capture_sha256'], 'replay_sha256':row['replay_sha256']}
        schedules = {}
        for block in plan['matrix']['arrival_blocks']:
            for B in plan['matrix']['batches']:
                cap = calibration[f"{block['block_id']}/{block['regime']}/B{B}"]
                schedules[f"{block['id']}/B{B}"] = [make_calendar(plan['gamma_weights'][block['id']][pattern], rho, cap['calibration_elapsed_ns'])
                    for rho in plan['matrix']['rhos'] for pattern in plan['matrix']['patterns']]
        plan.update(calibration=calibration, schedules=schedules,
            calibration_manifest_sha256=manifest_hash(previous),
            calibration_records_sha256=file_hash(args.calibration_records))
    else:
        base_path = ROOT/'configs/resident-online-validation-frozen-v1.json'
        base = json.loads(base_path.read_text())
        identity, files = source_identity()
        blocks = arrival_blocks()
        plan = dict(arrival_fix_schema_version=1, phase=args.phase,
            base_manifest=dict(path=str(base_path.relative_to(ROOT)), sha256=base['freeze']['sha256']),
            arrival_fix_source_sha256=identity, arrival_fix_source_files=files,
            protocol_file_sha256=file_hash(ROOT/'docs/arrival-response-fix-protocol.md'),
            analysis_file_sha256=file_hash(ROOT/'arrival_fix/analysis.py'),
            hardware=dict(hostname='kh083', gpu_uuids=GPU_UUIDS, world_size=2, interconnect='NV6'),
            order_seed=471421, jobs_per_trial=256,
            matrix=dict(arrival_blocks=blocks, batches=[128, 1024], patterns=['spread', 'cluster'],
                        rhos=[.5, .85], reps=2, arms=['native', 'fast']),
            analysis=ANALYSIS_DEFAULTS, payload_freshness='reused archived development inputs; only arrival calendars are fresh',
            gamma_weights={b['id']:{p:frozen_weights(b['seed'], p) for p in ('spread', 'cluster')} for b in blocks},
            calendar_generator_python=platform.python_version(),
            timing_contract='unchanged absolute-clock FCFS wrapper; all jobs drained; command/ACK/GPU completion and fast-path agreement charged')
        if args.phase == 'pilot':
            previous_path = ROOT/'results/instructor/arrivals-v1/evaluation-plan.json'
            previous = json.loads(previous_path.read_text())
            plan['matrix'].update(arrival_blocks=[dict(id='pilot-old-calendar', seed=0, block_id='opt-pilot-01', regime='domain1')],
                                  patterns=['constant', 'burst8'], rhos=[.85], arms=['native', 'native_alias', 'fast'])
            plan['schedules'] = {f'pilot-old-calendar/B{B}':[x for x in previous['schedules'][f'opt-pilot-01/domain1/B{B}'] if x['rho'] == .85] for B in (128, 1024)}
            plan['pilot_calendar_source_sha256'] = file_hash(previous_path)
            plan['pilot_timing_inference'] = 'descriptive A/A and treatment; no timing eligibility threshold or parameter selection'
            del plan['gamma_weights']
    plan['freeze'] = dict(algorithm='sha256', sha256=manifest_hash(plan))
    args.output.write_text(json.dumps(plan, indent=2)+'\n')
    print(plan['freeze']['sha256'])


if __name__ == '__main__':
    main()
