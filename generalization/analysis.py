"""Validate complete external-arrival trials before descriptive or final inference."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np
from arrival_fix.analysis import canonical_hash,_flat_metrics,_window_spans,_guard_errors,MATCH_FIELDS
from arrival_study.schedules import fcfs_metrics,calibration_capacity
from generalization.common import validate_calendar
from generalization.inference import infer_endpoints,choose_final_sample_size,DATASETS

ROOT=Path(__file__).resolve().parents[1]
ARMS=('native','shortcut','combined')
CONTRASTS={'combined_over_shortcut':('shortcut','combined'),'combined_over_native':('native','combined'),'shortcut_over_native':('native','shortcut')}
PAIR_FIELDS=tuple(k for k in MATCH_FIELDS if k not in ('arrival_seed','assignment_sha256'))
KEY_FIELDS=('content_group_id','arrival_type','rate_kind','arm')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plan(plan):
    phase=plan.get('phase')
    if phase not in ('calibration','pilot','final') or plan.get('generalization_schema_version')!=1:
        raise ValueError('Declared generalization calibration, pilot or final phase required')
    ph=canonical_hash({k:v for k,v in plan.items() if k!='freeze'})
    if plan.get('freeze',{}).get('sha256')!=ph:raise ValueError('Plan is unfrozen or changed')
    for name,key in [('analysis.py','analysis_sha256'),('inference.py','inference_sha256')]:
        if plan.get(key)!=_sha(ROOT/'generalization'/name):raise ValueError(f'Changed frozen {name}')
    files=plan.get('execution_source_files',{})
    if not files or canonical_hash(files)!=plan.get('execution_source_sha256'):
        raise ValueError('Missing or inconsistent execution source inventory')
    for name,sha in {**files,**plan.get('protocol_files',{})}.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or _sha(p)!=sha:
            raise ValueError(f'Changed frozen source/protocol: {name}')
    if plan.get('arms')!=list(ARMS) or plan.get('batch_tokens')!=128 or plan.get('physical_to_logical_map')!=list(range(128))+list(range(32)):
        raise ValueError('Frozen three-arm B128 fixed-map design changed')
    groups={g['id']:g['dataset'] for g in plan['groups']}
    if len(groups)!=len(plan['groups']) or set(groups.values())!=set(DATASETS):raise ValueError('Content group inventory invalid')
    counts=Counter(groups.values())
    allowed=(2,) if phase=='pilot' or plan.get('split')=='pilot' else (6,8)
    if len(set(counts.values()))!=1 or next(iter(counts.values())) not in allowed:raise ValueError('Content allocation differs')
    banks={b['block_id']:b for b in plan['banks']}
    if len(banks)!=len(plan['banks']) or set(banks)!=set(groups):raise ValueError('Bank/group inventory mismatch')
    cases={}
    for case in plan['cases']:
        gid=case['block_id'];b=case['source_block'];c=case['calendar'];key=(gid,b['trace'],c['rate_kind'])
        validate_calendar(c)
        if key in cases or groups.get(gid)!=case['dataset'] or sorted(case['arm_order'])!=sorted(ARMS):raise ValueError('Duplicate or malformed case')
        if c['source_date_utc']!=b['source_date_utc'] or c['source_rows_sha256']!=b['source_rows_sha256']:raise ValueError('Calendar/source block mismatch')
        cases[key]=case
    if phase=='calibration':
        if cases:raise ValueError('Calibration has treatment cases')
        expected={(g,None,'saturated','native') for g in groups}
    else:
        if set(cases)!=set(itertools.product(groups,('code','conversation'),('calibrated','common'))):raise ValueError('Four explicit cases per content group required')
        orders=Counter(tuple(c['arm_order']) for c in cases.values())
        required={p:len(cases)//6 for p in itertools.permutations(ARMS)}
        if len(cases)%6 or orders!=Counter(required):raise ValueError('Exact balance of all six arm permutations required')
        expected={(*k,a) for k in cases for a in ARMS}
    return ph,groups,banks,cases,expected


def _physical_issues(row,mapping):
    issues=[];trajectory=[];logical_windows=[]
    decisions=row.get('decision_log',[])
    if [d.get('step') for d in decisions]!=[63,127,191]:return ['decision_cadence'],None
    def collapse(raw):
        a=np.asarray(raw)
        if a.shape!=(16,160) or not np.issubdtype(a.dtype,np.integer) or (a<0).any() or not np.all(a.sum(axis=1)==1024):
            raise ValueError('physical window shape, integer counts or token total differs')
        return (np.bincount(mapping,weights=a.sum(axis=0),minlength=128)*2).astype(np.int64).tolist()
    try:
        for d in decisions:
            if d.get('allowed') is not False or d.get('current_map')!=mapping or d.get('committed_map')!=mapping:
                issues.append('decision_committed_state')
            candidate=d.get('proposed_map',[])
            if len(candidate)!=160 or set(candidate)!=set(range(128)):issues.append('illegal_candidate_map')
            logical=collapse(d['physical_slot_assignments_window'])
            if logical!=d['current_counts']:issues.append('physical_logical_window_conservation')
            logical_windows.append(logical)
            trajectory.append({k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')})
        terminal=collapse(row['terminal_physical_window'])
        if canonical_hash(trajectory)!=row.get('decision_trajectory_sha256'):issues.append('trajectory_hash')
    except (KeyError,TypeError,ValueError) as exc:
        return issues+['physical_windows:'+str(exc)],None
    return issues,terminal


def _physical_signature(row):
    return [d.get('physical_slot_assignments_window') for d in row.get('decision_log',[])]+[row.get('terminal_physical_window')]


def analyze(plan,records):
    ph,groups,banks,cases,expected=_plan(plan);phase=plan['phase']
    errors=[];index={};flat={};summaries=[];actual_orders=defaultdict(list);terminals={}
    for row in records:
        key=tuple(row.get(k) for k in KEY_FIELDS)
        if key not in expected or key in index:
            errors.append({'key':list(key),'issues':['duplicate_or_unexpected_record']});continue
        index[key]=row;actual_orders[key[:-1]].append(row['arm']);gid=key[0];arm=key[-1];combined=arm=='combined';issues=[]
        checks=dict(status='complete',phase=phase,config_hash=ph,generalization_schema_version=1,
            execution_source_sha256=plan['execution_source_sha256'],base_config_hash=plan['base_manifest']['sha256'],
            block_id=gid,dataset=groups[gid],regime='newcontent',batch_tokens=128,repetition=0,
            correctness_pass=True,evidence_eligible=phase=='final',diagnostic_only=False,guard_symbols_restored=True,
            normal_completion_verified=True,resident_state_verified=True,rank_assignment_agreement=True,
            migration_bytes=0,migration_sent_messages=0,terminal_proposals_suppressed=1,policy='SKIP',
            capture_sha256=banks[gid]['capture_sha256'],content_selection_sha256=plan['content_selection_sha256'],
            source_map=plan['physical_to_logical_map'],final_map=plan['physical_to_logical_map'],
            dispatch_mode='online' if combined else 'native',noop_enabled=arm!='native',assignment_selection_cost_included=True,
            online_residency_metadata_verified=combined,mapper_calls_measured=256 if combined else 0,
            measurement_setup_revision=2,gc_enabled_during_measurement=True,
            warmup_raw_indices=list(range(16)),conditioning_raw_indices=[i%16 for i in range(64)],
            measured_raw_indices=list(range(16,272)),full_unique_preflight_gpu_routes_checked=272)
        for field,value in checks.items():
            if row.get(field)!=value:issues.append(field)
        for scope,n in [(row,2),(row.get('preflight_correctness',{}),272)]:
            if any(scope.get(k)!=v for k,v in dict(correctness_pass=True,correctness_atol=.02,correctness_rtol=.03,full_batch_inputs_checked=n).items()):issues.append('numerical_check_scope')
            if any(not isinstance(scope.get(k),(float,int)) or not math.isfinite(scope[k]) or scope[k]<0 for k in ('correctness_max_abs_error','full_batch_max_abs_difference')):issues.append('numerical_error_record')
        hw=plan['hardware']
        if str(row.get('allocation_id'))!=str(hw['allocation_id']) or str(row.get('hostname','')).split('.')[0]!=hw['hostname']:
            issues.append('hardware_allocation')
        if any(sorted(row.get(k,[]))!=sorted(hw['gpu_uuids']) for k in ('physical_gpu_uuids','rank_gpu_uuids')):issues.append('gpu_identity')
        clock=row.get('arrival_clock_contract',{})
        if clock.get('monotonic') is not True or clock.get('adjustable') is not False or 'CLOCK_MONOTONIC' not in clock.get('implementation',''):issues.append('clock_contract')
        if np.asarray(row.get('per_rank_start_ns')).shape!=(2,256) or np.asarray(row.get('per_rank_gpu_complete_ns')).shape!=(2,256):issues.append('rank_timing_arrays')
        issues.extend(_guard_errors(row))
        if len(row.get('assignment_metrics',[]))!=256:issues.append('assignment_metrics')
        reuse=row.get('job_reuse',{})
        if reuse.get('unique_all_job_combinations')!=272 or reuse.get('unique_measured_job_combinations')!=256:issues.append('unique_job_combinations')
        if phase=='calibration':
            if row.get('arrival_schedule') is not None or row.get('arrival_offsets_ns')!=[0]*256 or row.get('horizon_ns')!=1 or row.get('arm_order')!=['native'] or row.get('arrival_date') is not None:issues.append('saturated_calibration')
        else:
            case=cases[key[:-1]];cal=case['calendar'];block=case['source_block']
            for field,value in dict(arrival_schedule=cal,arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],arrival_block_id=block['id'],arrival_date=block['source_date_utc'],arrival_seed=plan['arrival_seed'],schedule_name=cal['name'],pattern=cal['pattern'],rho=cal['rho'],arm_order=case['arm_order']).items():
                if row.get(field)!=value:issues.append(field)
            if str(row.get('allocation_id'))!=str(plan['calibration'][gid]['allocation_id']):issues.append('calibration_allocation')
        physical,terminal=_physical_issues(row,plan['physical_to_logical_map']);issues.extend(physical);terminals[key]=terminal
        try:
            metrics=fcfs_metrics(row)
            if not metrics['complete'] or metrics['scheduled_jobs']!=256:issues.append('incomplete_trial')
            if row.get('queue_metrics')!=metrics:issues.append('stored_metrics_differ_from_raw_clocks')
            if row.get('observed_window_spans')!=_window_spans(row):issues.append('window_spans')
            if phase=='calibration' and row.get('calibration_capacity')!=calibration_capacity(row['coordinator_start_ns'],row['coordinator_finish_ns']):issues.append('calibration_capacity')
        except (KeyError,ValueError,TypeError,IndexError) as exc:issues.append('raw_timing:'+str(exc))
        identity=('config_hash','phase','arrival_block_id','block_id','regime','batch_tokens','arm','repetition','schedule_name')
        if row.get('record_id')!=canonical_hash({k:row.get(k) for k in identity}):issues.append('record_id')
        if issues:
            errors.append(dict(key=list(key),issues=issues));continue
        values=_flat_metrics(row,metrics);flat[key]=values
        summaries.append({**{k:row[k] for k in KEY_FIELDS},'dataset':groups[gid],'arrival_date':row['arrival_date'],
            'record_id':row['record_id'],'allocation_id':row['allocation_id'],'metrics':values,
            'observed_window_spans':row['observed_window_spans']})
    missing=sorted(expected-set(index),key=str)
    pairs=[];positions={arm:[0,0,0] for arm in ARMS}
    if phase!='calibration':
        for key,case in sorted(cases.items()):
            keys={a:(*key,a) for a in ARMS}
            if any(k not in flat for k in keys.values()):continue
            rows={a:index[k] for a,k in keys.items()}
            for field in PAIR_FIELDS:
                if any(field not in r for r in rows.values()) or len({canonical_hash(r.get(field)) for r in rows.values()})!=1:errors.append(dict(key=list(key),issues=['triplet_'+field]))
            if rows['native'].get('assignment_sha256') is None or rows['native']['assignment_sha256']!=rows['shortcut'].get('assignment_sha256'):errors.append(dict(key=list(key),issues=['native_shortcut_assignment']))
            if canonical_hash(_physical_signature(rows['native']))!=canonical_hash(_physical_signature(rows['shortcut'])):errors.append(dict(key=list(key),issues=['native_shortcut_physical_windows']))
            if len({canonical_hash(terminals[k]) for k in keys.values()})!=1:errors.append(dict(key=list(key),issues=['triplet_terminal_logical']))
            if actual_orders[key]!=case['arm_order']:errors.append(dict(key=list(key),issues=['actual_arm_order']))
            else:
                for pos,arm in enumerate(case['arm_order']):positions[arm][pos]+=1
            for contrast,(control,treatment) in CONTRASTS.items():
                a,b=flat[keys[control]],flat[keys[treatment]]
                logs={m:math.log(b[m])-math.log(a[m]) for m in a if a[m]>0 and b[m]>0}
                if any(m not in logs for m in ('response_mean_ns','response_p95_ns','service_mean_ns')):errors.append(dict(key=list(key),issues=['nonpositive_endpoint']))
                pairs.append(dict(content_group_id=key[0],dataset=groups[key[0]],arrival_type=key[1],rate_kind=key[2],arrival_date=case['source_block']['source_date_utc'],contrast=contrast,control_arm=control,treatment_arm=treatment,control=a,treatment=b,log_ratios=logs,
                    ratios={m:b[m]/a[m] if a[m]>0 else None for m in a},differences={m:b[m]-a[m] for m in a}))
        # Inputs and planner trajectory cannot change with rate, trace, or arm.
        for gid in groups:
            members=[r for key,r in index.items() if key[0]==gid]
            for field in PAIR_FIELDS:
                if len({canonical_hash(r.get(field)) for r in members})!=1:errors.append(dict(group=gid,issues=['cross_case_'+field]))
            for dispatch in ('native','online'):
                subset=[r for r in members if r.get('dispatch_mode')==dispatch]
                if len({canonical_hash(_physical_signature(r)) for r in subset})!=1:errors.append(dict(group=gid,issues=['cross_case_'+dispatch+'_physical_windows']))
                for field in ('assignment_sha256','terminal_physical_window','assignment_metrics'):
                    if len({canonical_hash(r.get(field)) for r in subset})!=1:errors.append(dict(group=gid,issues=['cross_case_'+dispatch+'_'+field]))
    valid=not errors and not missing
    result=dict(analysis_schema_version=1,phase=phase,valid=valid,expected_records=len(expected),observed_records=len(records),valid_records=len(flat),
        matched_triplets=len(pairs)//3,matched_pairs=len(pairs),pairs_per_contrast=dict(Counter(p['contrast'] for p in pairs)),
        plan_sha256=ph,execution_source_sha256=plan['execution_source_sha256'],missing_keys=[list(k) for k in missing],errors=errors,
        trial_summaries=summaries,paired_cases=pairs,arm_position_counts=positions,confirmatory=None,allocation_decision=None,
        scope='Content-bundle inference conditional on the selected external trace material and one physical allocation; not production-arrival or full-model inference.',
        limitations=['Raw counters are checked for conservation and paired identities. Independent authenticated-bank/oracle audit is still required.',
            'Full-batch preflight covers272 unique inputs; postflight covers two endpoints; FP32 checks sample16 positions at each endpoint.',
            'Zero-denominator queue/drain ratios are null; differences remain available. Jobs are not independent inference units.'])
    if valid and phase=='final':result['confirmatory']=infer_endpoints(pairs,plan['groups'])
    if valid and phase=='pilot':result['allocation_decision']=choose_final_sample_size(pairs,plan['groups'])
    if valid and phase=='calibration':result['calibration']={r['block_id']:r['calibration_capacity'] for r in records}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','records','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--allocation-output',type=Path)
    args=p.parse_args();raw=args.records.read_bytes();payload=gzip.decompress(raw) if args.records.suffix=='.gz' else raw
    records=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    result=analyze(json.loads(args.plan.read_text()),records)
    result.update(analysis_file_sha256=_sha(__file__),records_file_sha256=hashlib.sha256(raw).hexdigest(),
        records_jsonl_payload_sha256=hashlib.sha256(payload).hexdigest(),records_compressed=args.records.suffix=='.gz')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.allocation_output:
        if not result['valid'] or result['phase']!='pilot':raise ValueError('Allocation output requires a complete valid pilot')
        allocation={**result['allocation_decision'],'plan_sha256':result['plan_sha256'],'records_jsonl_payload_sha256':result['records_jsonl_payload_sha256'],'analysis_file_sha256':result['analysis_file_sha256']}
        args.allocation_output.parent.mkdir(parents=True,exist_ok=True);args.allocation_output.write_text(json.dumps(allocation,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('phase','valid','expected_records','observed_records','matched_pairs')}))
    if not result['valid']:raise SystemExit(1)

if __name__=='__main__':main()
