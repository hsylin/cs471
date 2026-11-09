#!/usr/bin/env python3
"""Freeze an arrival calibration or its deterministic baseline-derived calendars."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrival_study.common import source_identity,file_hash,load_plan
from arrival_study.schedules import calibration_capacity,generate_schedule,fcfs_metrics
from optimization.common import manifest_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('calibration','evaluation'),required=True)
    p.add_argument('--calibration-plan',type=Path)
    p.add_argument('--calibration-records',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    base_path=ROOT/'configs/resident-online-validation-frozen-v1.json'
    base=json.loads(base_path.read_text())
    digest,files=source_identity()
    plan=dict(arrival_schema_version=1,phase=a.phase,
        base_manifest=dict(path=str(base_path.relative_to(ROOT)),sha256=base['freeze']['sha256']),
        arrival_source_sha256=digest,arrival_source_files=files,
        protocol_file_sha256=file_hash(ROOT/'docs/instructor-arrival-protocol.md'),
        order_seed=30471,matrix=dict(block_ids=['opt-pilot-01','opt-pilot-02'],
            regimes=['domain0','domain1'],batch_tokens=[128,1024],arms=['native','online'],repetitions=2),
        jobs_per_trial=256,offered_load_fractions=[.5,.85],arrival_patterns=['constant','burst8'],burst_size=8,
        control_contract='one FCFS layer job in flight; rank0 CPU command/ACK coordination; same-host CLOCK_MONOTONIC',
        inference='descriptive supplementary reuse of two development blocks; no new confirmatory acceptance or policy selection')
    analysis=ROOT/'arrival_study/analysis.py'
    if analysis.exists():plan['analysis_file_sha256']=file_hash(analysis)
    if a.phase=='evaluation':
        if a.calibration_records is None or a.calibration_plan is None:raise ValueError('Evaluation requires complete frozen native-only calibration')
        previous,_=load_plan(a.calibration_plan)
        if previous['phase']!='calibration':raise ValueError('Wrong calibration phase')
        rows=[json.loads(line) for line in a.calibration_records.read_text().splitlines() if line.strip()]
        expected=set(itertools.product(plan['matrix']['block_ids'],plan['matrix']['regimes'],plan['matrix']['batch_tokens']))
        index={tuple(r[k] for k in ('block_id','regime','batch_tokens')):r for r in rows}
        if len(rows)!=8 or set(index)!=expected:raise ValueError('Complete eight-state calibration required')
        schedules={};calibration={}
        for state,r in index.items():
            if (r['config_hash']!=manifest_hash(previous) or r['arrival_source_sha256']!=digest
                or r['phase']!='calibration' or r['arm']!='native' or r['policy']!='SKIP'
                or r['status']!='complete' or not r['correctness_pass'] or not r['normal_completion_verified']
                or not r['resident_state_verified'] or r['migration_bytes']!=0 or not fcfs_metrics(r)['complete']):
                raise ValueError('Invalid native calibration record')
            cap=calibration_capacity(r['coordinator_start_ns'],r['coordinator_finish_ns'])
            if cap!=r['calibration_capacity']:raise ValueError('Calibration summary differs from timestamps')
            key=f'{state[0]}/{state[1]}/B{state[2]}'
            calibration[key]={**cap,'record_id':r['record_id'],'allocation_id':r['allocation_id'],
                'capture_sha256':r['capture_sha256'],'replay_sha256':r['replay_sha256']}
            schedules[key]=[generate_schedule(pattern,rho=rho,calibration_elapsed_ns=cap['calibration_elapsed_ns'])
                for rho in plan['offered_load_fractions'] for pattern in plan['arrival_patterns']]
        plan.update(calibration_manifest_sha256=manifest_hash(previous),
            calibration_records_file_sha256=file_hash(a.calibration_records),calibration=calibration,schedules=schedules)
    plan['freeze']=dict(algorithm='sha256',sha256=manifest_hash(plan))
    a.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(plan['freeze']['sha256'])


if __name__=='__main__':main()
