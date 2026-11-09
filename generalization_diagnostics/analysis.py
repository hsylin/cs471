"""Descriptive diagnostics only: no selection, confidence interval or acceptance."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from generalization_diagnostics.plan import CONDITIONS,sequence_for,validate,sha
from optimization.common import canonical_hash,manifest_hash
from arrival_study.schedules import fcfs_metrics
from arrival_fix.analysis import _flat_metrics,_guard_errors,_window_spans
from generalization.analysis import _physical_issues


def _architecture(rows):
    out={}
    for key in ('per_rank_tokens','per_rank_active_copies','per_rank_m_tiles','per_rank_padded_rows'):
        values=np.asarray([r[key] for r in rows],dtype=float)
        if values.shape!=(256,2) or not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid architecture array')
        out[key+'_mean_rank_max']=float(values.max(axis=1).mean())
        out[key+'_mean_total']=float(values.sum(axis=1).mean())
    return out


def analyze(plan,records):
    pilot,_=validate(plan);expected={(s['content_group_id'],c) for s in plan['states'] for c in s['condition_order']}
    expected_order=[(s['content_group_id'],c) for s in plan['states'] for c in s['condition_order']]
    state_info={s['content_group_id']:s for s in plan['states']};index={};flat={};errors=[];actual_order=[];trials=[]
    for row in records:
        key=(row.get('content_group_id'),row.get('condition'));actual_order.append(key)
        if key not in expected or key in index:errors.append(dict(key=key,issues=['duplicate_or_unexpected']));continue
        index[key]=row;state=state_info[key[0]];seq_name,mode=CONDITIONS[key[1]];seq=sequence_for(seq_name)
        cal=state['source_case']['calendar'];block=state['source_case']['source_block'];issues=[]
        bank=next(b for b in pilot['banks'] if b['block_id']==key[0])
        checks=dict(generalization_diagnostic_schema_version=1,instrumentation_revision=2,
            executed_combining_weights_encoding='little-endian float32; C-order; no header',
            phase='diagnostic',status='complete',evidence_eligible=False,
            diagnostic_only=True,diagnostic_config_hash=manifest_hash(plan),diagnostic_source_sha256=plan['diagnostic_source_sha256'],
            main_execution_source_sha256=pilot['execution_source_sha256'],pilot_plan_sha256=pilot['freeze']['sha256'],
            block_id=key[0],dataset=state['dataset'],regime='newcontent',batch_tokens=128,policy='SKIP',
            sequence_name=seq_name,dispatch_mode=mode,condition_order=state['condition_order'],
            capture_sha256=bank['capture_sha256'],arrival_block_id=block['id'],arrival_type='code',arrival_date=block['source_date_utc'],rate_kind='calibrated',
            arrival_schedule=cal,arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],
            source_map=plan['physical_to_logical_map'],final_map=plan['physical_to_logical_map'],
            warmup_raw_indices=list(range(16)),conditioning_raw_indices=[i%16 for i in range(64)],measured_raw_indices=seq,
            unique_measured_job_combinations=len(set(seq)),raw_job_frequency={str(i):seq.count(i) for i in sorted(set(seq))},
            migration_bytes=0,migration_sent_messages=0,terminal_proposals_suppressed=1,normal_completion_verified=True,
            resident_state_verified=True,rank_assignment_agreement=True,guard_symbols_restored=True,
            online_selection_cost_included=mode=='online',precomputed_selection_cost_excluded=mode.startswith('pre_'),
            precomputed_adapter_cost_included=mode.startswith('pre_'),mapper_calls_measured=0 if mode=='native' else 256,
            precomputed_gpu_adapter_checks=544,full_unique_preflight_gpu_routes_checked=272,postflight_reference_raw_indices=[16,271],
            measurement_setup_revision=2,gc_enabled_during_measurement=True)
        issues += [k for k,v in checks.items() if row.get(k)!=v]
        weight_digest=row.get('executed_combining_weights_sha256','')
        if not isinstance(weight_digest,str) or len(weight_digest)!=64 or any(c not in '0123456789abcdef' for c in weight_digest):issues.append('float_weight_digest_format')
        for obj,n in [(row,2),(row.get('preflight_correctness',{}),272)]:
            if any(obj.get(k)!=v for k,v in dict(correctness_pass=True,correctness_atol=.02,correctness_rtol=.03,full_batch_inputs_checked=n).items()):issues.append('numerical_scope')
            for metric in ('correctness_max_abs_error','full_batch_max_abs_difference'):
                if not isinstance(obj.get(metric),(float,int)) or not math.isfinite(obj[metric]) or obj[metric]<0:issues.append(metric)
        hw=plan['hardware'];clock=row.get('arrival_clock_contract',{})
        if str(row.get('allocation_id'))!=str(hw['allocation_id']) or str(row.get('hostname','')).split('.')[0]!=hw['hostname']:issues.append('hardware_allocation')
        if any(sorted(row.get(k,[]))!=sorted(hw['gpu_uuids']) for k in ('physical_gpu_uuids','rank_gpu_uuids')):issues.append('gpu_identity')
        if clock.get('monotonic') is not True or clock.get('adjustable') is not False or 'CLOCK_MONOTONIC' not in clock.get('implementation',''):issues.append('clock')
        if any(np.asarray(row.get(k)).shape!=(2,256) for k in ('per_rank_start_ns','per_rank_gpu_complete_ns')):issues.append('rank_envelopes')
        issues += _guard_errors({**row,'arm':'shortcut'})
        physical,_=_physical_issues(row,plan['physical_to_logical_map']);issues+=physical
        try:
            metrics=fcfs_metrics(row)
            if not metrics['complete'] or metrics['scheduled_jobs']!=256 or row.get('queue_metrics')!=metrics:issues.append('raw_completion_metrics')
            if row.get('observed_window_spans')!=_window_spans(row):issues.append('window_spans')
            values=_flat_metrics(row,metrics);service=metrics['vectors_ns']['service'];boundaries={63,127,191,255}
            values.update(ordinary_step_service_mean_ns=statistics.mean(v for i,v in enumerate(service) if i not in boundaries),
                nonterminal_boundary_service_mean_ns=statistics.mean(service[i] for i in (63,127,191)),terminal_service_ns=service[255])
            architecture=_architecture(row['assignment_metrics'])
        except (KeyError,ValueError,TypeError,IndexError) as exc:issues.append('raw_statistics:'+str(exc))
        ident={k:row.get(k) for k in ('diagnostic_config_hash','phase','content_group_id','condition')}
        if row.get('record_id')!=canonical_hash(ident):issues.append('record_identity')
        if issues:errors.append(dict(key=key,issues=issues));continue
        flat[key]=values;trials.append(dict(content_group_id=key[0],condition=key[1],dataset=state['dataset'],metrics=values,architecture=architecture))
    if actual_order!=expected_order:errors.append(dict(issues=['actual_global_condition_order']))
    missing=sorted(expected-set(index));pairs=[]
    shared=('replay_sha256','source_prompt_ids','source_weights_sha256','source_weight_rank_sha256','resolved_kernel_config','logical_routing_sha256')
    same_sequence=('executed_logical_routing_sha256','executed_combining_weights_sha256','decision_trajectory_sha256')
    comparisons={**plan['comparison_pairs'],
        'cycle0_over_main_shortcut':['main_shortcut','cycle0_shortcut'],
        'cycle0_over_main_combined':['main_combined','cycle0_combined'],
        'cycle16_over_cycle0_shortcut':['cycle0_shortcut','cycle16_shortcut'],
        'cycle16_over_cycle0_combined':['cycle0_combined','cycle16_combined'],
        'online_over_precomputed_online':['offline_online','main_combined'],
        'native_over_precomputed_native':['offline_native','main_shortcut']}
    for state in plan['states']:
        gid=state['content_group_id'];members=[index[gid,c] for c in state['condition_order'] if (gid,c) in index]
        for field in shared:
            if len({canonical_hash(r.get(field)) for r in members})!=1:errors.append(dict(group=gid,issues=['cross_condition_'+field]))
        for seq in ('main','cycle0','cycle16'):
            subset=[r for r in members if r['sequence_name']==seq]
            for field in same_sequence:
                if len({canonical_hash(r.get(field)) for r in subset})!=1:errors.append(dict(group=gid,issues=['same_sequence_'+field]))
        for names in [('main_shortcut','offline_native'),('main_combined','offline_online')]:
            if all((gid,n) in index for n in names):
                a,b=(index[gid,n] for n in names)
                for field in ('assignment_sha256','terminal_physical_window','assignment_metrics'):
                    if a.get(field)!=b.get(field):errors.append(dict(group=gid,issues=['same_assignment_'+field]))
                if [d['physical_slot_assignments_window'] for d in a['decision_log']]!=[d['physical_slot_assignments_window'] for d in b['decision_log']]:errors.append(dict(group=gid,issues=['same_assignment_physical_windows']))
        for name,(control,treatment) in comparisons.items():
            if (gid,control) not in flat or (gid,treatment) not in flat:continue
            a,b=flat[gid,control],flat[gid,treatment]
            pairs.append(dict(content_group_id=gid,dataset=state['dataset'],comparison=name,control_condition=control,treatment_condition=treatment,
                control=a,treatment=b,ratios={k:b[k]/a[k] if a[k]>0 else None for k in a},differences={k:b[k]-a[k] for k in a}))
    positions={c:[0]*8 for c in CONDITIONS}
    for state in plan['states']:
        for i,c in enumerate(state['condition_order']):positions[c][i]+=1
    precedence={pair:dict(control_first=sum(s['condition_order'].index(names[0])<s['condition_order'].index(names[1]) for s in plan['states']),states=3) for pair,names in plan['comparison_pairs'].items()}
    return dict(schema='generalization_diagnostic_summary_v1',valid=not errors and not missing,expected_records=24,observed_records=len(records),
        valid_records=len(flat),errors=errors,missing_keys=missing,trial_summaries=trials,paired_cases=pairs,
        condition_position_counts=positions,pair_precedence=precedence,diagnostic_config_hash=manifest_hash(plan),
        diagnostic_source_sha256=plan['diagnostic_source_sha256'],confidence_intervals=None,acceptance=None,instrumentation_revision=2,
        scope='Three development states, descriptive sensitivity only; no tuning or production-population inference.',
        limitations=['Main-versus-cycle changes input coverage and repetition; phase changes also alter input-to-arrival alignment.',
            'Offline replay removes construction cost but charges the adapter; online/offline difference is not exclusive GPU selector time.',
            'No length matching, complete-request execution, new algorithm selection or final-data inspection.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','records','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();raw=a.records.read_bytes();payload=gzip.decompress(raw) if a.records.suffix=='.gz' else raw
    result=analyze(json.loads(a.plan.read_text()),[json.loads(x) for x in payload.decode().splitlines() if x.strip()])
    result.update(records_file_sha256=hashlib.sha256(raw).hexdigest(),records_jsonl_payload_sha256=hashlib.sha256(payload).hexdigest(),analysis_file_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('valid','expected_records','observed_records','valid_records')}))
    if not result['valid']:raise SystemExit(1)
if __name__=='__main__':main()
