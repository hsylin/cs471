"""Complete-record validation and prespecified resident-fix analysis.

Reuses only immutable clock/accounting and unchanged-arm validators. Fresh fix
state is checked explicitly; main inference and MBPP sentinel remain separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import itertools
import json
import math
from pathlib import Path
import numpy as np
from arrival_study.schedules import calibration_capacity
from stock_comparison.analysis import clock_metrics,read_records,paired_values,native_issues,_paired_identity
from stock_comparison.data import digest
from resident_fastpath.common import ARMS,OLD_CLOSURE,seal,validate_calendar,load_plan
from resident_fastpath.inference import DATASETS,SENTINEL,infer_endpoints,validate_arm_orders

CONTRASTS={'fix_over_stockoff':('stockoff','fix'),'fix_over_current':('current','fix'),'current_over_stockoff':('stockoff','current')}
KEY_FIELDS=('content_group_id','arrival_type','rho','arm')


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_plan(plan):
    require(plan.get('schema')=='resident_fix_plan_v1' and seal(plan)==plan,'Invalid/changed frozen plan')
    phase=plan['phase'];split=plan['split']
    require(phase in ('calibration','pilot','final') and split in ('pilot','final') and (phase=='calibration' or phase==split),'Phase/split mismatch')
    require(plan['arms']==list(ARMS) and plan['jobs']==4096 and plan['batch_tokens']==128,'Wrong frozen scope')
    require(plan['old_closure_sha256']==OLD_CLOSURE,'Old frozen closure changed')
    require(plan['source_sha256']==digest(plan['source_files']),'New source inventory digest differs')
    selected=plan['selected_fix']
    require(selected.get('selector') in ('three_kernel','fused'),'Unknown selected implementation')
    require(plan['selected_fix_content_sha256']==digest(selected),'Selected fix content changed')
    require(isinstance(plan['selected_fix_sha256'],str) and len(plan['selected_fix_sha256'])==64 and all(c in '0123456789abcdef' for c in plan['selected_fix_sha256']),'Selected fix file hash missing')
    groups={b['block_id']:b for b in plan['banks']};n=2 if split=='pilot' else 8
    require(len(groups)==len(plan['banks'])==2*n+1,'Duplicate/missing fresh groups')
    require(Counter(b['dataset'] for b in groups.values())==Counter({'gsm8k':n,'wildchat':n,'mbpp':1}),'Wrong main/sentinel allocation')
    require(all(b['split']==split for b in groups.values()),'Capture split mismatch')
    if phase=='calibration':
        require(plan['cases']==[],'Calibration has treatment cases')
        return groups,{},[]
    cases={}
    for c in plan['cases']:
        cal=c['calendar'];validate_calendar(cal,c['source_block'])
        key=(c['content_group_id'],c['arrival_type'],c['rho'])
        require(key not in cases and key[0] in groups,'Duplicate/unexpected case')
        require(c['block_id']==key[0] and c['dataset']==groups[key[0]]['dataset'],'Case content identity')
        require(c['arrival_type']==cal['arrival_type'] and c['rho']==cal['rho'],'Flattened/calendar case mismatch')
        cases[key]=c
    normalized=[dict(id=g,dataset=b['dataset']) for g,b in groups.items()]
    order_counts=validate_arm_orders(list(cases.values()),normalized,phase=phase)
    provenance=plan['calibration_provenance']
    require(set(provenance)==set(groups),'Missing baseline calibration provenance')
    for key,c in cases.items():
        p=provenance[key[0]]
        require(p['calibration_elapsed_ns']==c['calendar']['stockoff_calibration_elapsed_ns'],'Calendar calibrated from another capacity')
        require(str(p['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration allocation differs')
        require(p['source_sha256']==plan['source_sha256'] and p['record_id']==plan['calibration_record_ids'][key[0]],'Calibration source/record differs')
    return groups,cases,order_counts


def fix_native_issues(row,selector):
    """Check fixed-residency custom semantics, never relabel it stock EPLB-off."""
    issues=[];pre=row.get('preflight',{});post=row.get('postflight',{})
    for scope,check,count in (('preflight',pre,4112),('postflight',post,16)):
        if any(check.get(k)!=v for k,v in dict(correctness_pass=True,full_batch_inputs_checked=count,atol=.02,rtol=.03).items()):issues.append(scope+'_numerical_coverage')
        error=check.get('full_batch_max_abs_difference')
        if not isinstance(error,(int,float)) or not math.isfinite(error) or error<0:issues.append(scope+'_numerical_error')
    if any(pre.get(k) is not True for k in ('logical_ids_equal','combining_weights_equal','rank_routing_agreement','input_unchanged')):issues.append('fixed_logical_identity')
    if pre.get('fp32_job_indices')!=[16,765,3765,4111] or pre.get('fp32_tokens_per_job')!=16:issues.append('fp32_coverage')
    if post.get('raw_indices')!=np.linspace(16,4111,16,dtype=np.int64).tolist():issues.append('postflight_indices')
    if pre.get('online_assignment_inputs_checked')!=4112 or not pre.get('physical_assignment_digest'):issues.append('fixed_assignment_oracle')
    for field in ('logical_routing_sha256','float32_combining_weights_sha256','physical_assignment_digest'):
        if not isinstance(pre.get(field),str) or len(pre[field])!=64:issues.append('identity_'+field)
    peers=row.get('per_rank_state',[])
    if len(peers)!=2:return issues+['missing_rank_state']
    mapping=list(range(128))+list(range(32))
    expected=dict(schema_version=1,arm='fix',enable_eplb=True,replicas=32,physical_slots=160,local_slots=80,
        static_resident_mode=True,controller_active=False,upstream_eplb_disabled=False,selector_kind=selector,
        logical_router_computation_included=True,native_output_reduction=True,stock_router_unmodified=False,stock_policy_unmodified=False,
        custom_all_reduce_disabled=False,source_map=mapping,final_map=mapping,controller_step_calls=0,native_counter=0,native_valid_steps=0,
        recorded_step_indices=[],deferred_boundary_steps=[],decisions=[],async_events=[],physical_window_next_write=0,
        pending_result=False,rebalanced=False,migration_sent_bytes_local=0,migration_sent_messages_local=0,
        terminal_proposals_suppressed=0,completion_disposition='normal_commit_no_discard',guard=None)
    for rank,p in enumerate(peers):
        for scope in ('start','end','audit'):
            s=p.get(scope,{})
            if any(s.get(k)!=v for k,v in expected.items()):issues.append(f'rank{rank}_{scope}_static_state')
            calls=0 if scope=='start' else 4096
            if s.get('static_step_calls')!=calls or s.get('mapper_calls')!=calls:issues.append(f'rank{rank}_{scope}_call_count')
            if s.get('terminal_physical_window_totals_local')!=[0]*160:issues.append(f'rank{rank}_{scope}_recording_active')
            if s.get('gate_weight_sha256')!=row.get('input_meta',{}).get('gate_weight_sha256'):issues.append(f'rank{rank}_{scope}_gate_identity')
        if any(p.get('audit',{}).get(k) is not True for k in ('resident_state_verified','rank_map_agreement','normal_completion_verified','static_state_unchanged','observation_and_migration_inactive')):issues.append(f'rank{rank}_static_audit')
        if p.get('shortcut_guard') is not None:issues.append(f'rank{rank}_unexpected_shortcut')
        if selector=='fused':
            context=p.get('audit',{}).get('fused_stream_context',{})
            if (context.get('device')!=f'cuda:{rank}' or context.get('sequential_single_stream_required') is not True
                or context.get('reused_output_storage') is not True or type(context.get('stream_handle')) is not int
                or context.get('stream_handle',-1)<0):issues.append(f'rank{rank}_fused_stream_context')
    if selector=='fused':
        proof=row.get('runtime_metadata',{}).get('fused_adversarial',{})
        if any(proof.get(k)!=v for k,v in dict(passed=True,cases_checked=9,gpu_invocations_checked=18,reference_count=2,rank_agreement=True,stream_context_unchanged=True,input_ids_unchanged=True,internal_real_tokens=128).items()):issues.append('fused_adversarial_proof')
    return issues


def precision_issues(runtime):
    return (runtime.get('tf32_allowed') is not False or runtime.get('float32_matmul_precision')!='highest'
        or runtime.get('precision_per_rank')!=[{'allow_tf32':False,'matmul_precision':'highest'}]*2)


def row_issues(row,plan,group,case):
    issues=[];arm=row['arm'];phase=plan['phase']
    expected=dict(schema='resident_fix_trial_v1',phase=phase,status='complete',plan_sha256=plan['content_sha256'],
        source_sha256=plan['source_sha256'],old_closure_sha256=OLD_CLOSURE,content_group_id=group['block_id'],block_id=group['block_id'],
        dataset=group['dataset'],n_jobs=4096,evidence_eligible=phase=='final',
        inference_role='sentinel_descriptive' if group['dataset']==SENTINEL else 'main')
    if any(row.get(k)!=v for k,v in expected.items()):issues.append('row_identity_or_phase')
    if row.get('record_id')!=digest({k:v for k,v in row.items() if k!='record_id'}):issues.append('record_content_hash')
    hw=plan['hardware'];clock=row.get('clock',{})
    if str(row.get('allocation_id'))!=str(hw['allocation_id']) or str(row.get('hostname','')).split('.')[0]!=hw['hostname'] or sorted(row.get('gpu_uuids',[]))!=sorted(hw['gpu_uuids']):issues.append('hardware_identity')
    if clock.get('monotonic') is not True or clock.get('adjustable') is not False or 'CLOCK_MONOTONIC' not in clock.get('implementation',''):issues.append('shared_clock')
    meta=row.get('input_meta',{})
    required=dict(schema='resident_fix_hidden_jobs_v1',seed=471703,block_id=group['block_id'],capture_sha256=group['capture_sha256'],
        warmup_jobs=16,measured_jobs=4096,batch_tokens=128,distinct_job_combinations=4112)
    if any(meta.get(k)!=v for k,v in required.items()):issues.append('authentic_job_identity')
    if not isinstance(meta.get('indices_sha256'),str) or len(meta['indices_sha256'])!=64:issues.append('input_indices_hash')
    rt=row.get('runtime_metadata',{})
    required=dict(study_id='resident_fix_v1',selected_fix=plan['selected_fix'],native_vllm_commit=plan['vllm_commit'],
        model_revision=plan['checkpoint_revision'],world_size=2,native_output_reduction=True)
    if any(rt.get(k)!=v for k,v in required.items()):issues.append('native_runtime_identity')
    if precision_issues(rt):issues.append('dense_fp32_precision')
    if row.get('source_weights',{}).get('source_revision')!=plan['checkpoint_revision'] or row.get('source_weights',{}).get('layer')!=0:issues.append('checkpoint_weights')
    if case is None:
        if arm!='stockoff' or row.get('calendar') is not None or row.get('arrival_offsets_ns')!=[0]*4096 or row.get('horizon_ns')!=1 or row.get('arm_order')!=['stockoff']:issues.append('saturated_calibration')
        try:
            if row.get('calibration_capacity')!=calibration_capacity(row['coordinator_start_ns'],row['coordinator_finish_ns'],expected_jobs=4096):issues.append('calibration_capacity')
        except (KeyError,ValueError,TypeError) as exc:issues.append('calibration:'+str(exc))
    else:
        cal=case['calendar']
        if any(row.get(k)!=v for k,v in dict(calendar=cal,arrival_type=cal['arrival_type'],arrival_date=cal['arrival_date'],rho=cal['rho'],arrival_offsets_ns=cal['arrival_offsets_ns'],horizon_ns=cal['horizon_ns'],arm_order=case['arm_order']).items()):issues.append('calendar_or_order')
    if arm=='fix':issues.extend(fix_native_issues(row,plan['selected_fix']['selector']))
    else:
        alias=dict(row,arm='ours' if arm=='current' else 'stockoff')
        issues.extend(native_issues(alias))
    return issues


def analyze(plan,records):
    groups,cases,order_counts=validate_plan(plan);phase=plan['phase']
    expected={(g,None,None,'stockoff') for g in groups} if phase=='calibration' else {(*k,a) for k in cases for a in ARMS}
    index={};values={};errors=[];trials=[];actual_orders={k:[] for k in cases}
    for row in records:
        key=tuple(row.get(k) for k in KEY_FIELDS)
        if key not in expected or key in index:errors.append(dict(key=list(key),issues=['unexpected_or_duplicate_row']));continue
        index[key]=row
        if phase!='calibration':actual_orders[key[:3]].append(row['arm'])
        issues=[]
        try:issues.extend(row_issues(row,plan,groups[key[0]],None if phase=='calibration' else cases[key[:3]]))
        except (KeyError,ValueError,TypeError,IndexError) as exc:issues.append('record_schema:'+str(exc))
        try:flat=clock_metrics(row)
        except (KeyError,ValueError,TypeError,IndexError,AssertionError) as exc:issues.append('clock:'+str(exc));flat=None
        if issues:errors.append(dict(key=list(key),issues=issues));continue
        values[key]=flat
        trials.append({**dict(zip(KEY_FIELDS,key)),'dataset':row['dataset'],'arrival_date':row.get('arrival_date'),
            'inference_role':row['inference_role'],'record_id':row['record_id'],'metrics':flat})
    missing=expected-set(index);pairs=[]
    if phase!='calibration':
        for key,case in cases.items():
            ks={a:(*key,a) for a in ARMS}
            if any(k not in values for k in ks.values()):continue
            rows={a:index[k] for a,k in ks.items()}
            if actual_orders[key]!=case['arm_order']:errors.append(dict(key=list(key),issues=['actual_arm_order']))
            if len({digest(_paired_identity(r)) for r in rows.values()})!=1:errors.append(dict(key=list(key),issues=['paired_input_routes_gate']))
            if rows['current']['preflight']['physical_assignment_digest']!=rows['fix']['preflight']['physical_assignment_digest']:errors.append(dict(key=list(key),issues=['changed_physical_assignment_policy']))
            for contrast,(control,treatment) in CONTRASTS.items():
                pairs.append(dict(content_group_id=key[0],dataset=groups[key[0]]['dataset'],arrival_type=key[1],rho=key[2],arrival_date=case['calendar']['arrival_date'],
                    contrast=contrast,control_arm=control,treatment_arm=treatment,**paired_values(values[ks[control]],values[ks[treatment]])))
        for gid in groups:
            members=[r for k,r in index.items() if k[0]==gid]
            if len({digest(_paired_identity(r)) for r in members})!=1:errors.append(dict(group=gid,issues=['cross_case_input_routes_gate']))
    valid=not errors and not missing
    result=dict(schema='resident_fix_analysis_v1',phase=phase,valid=valid,plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],old_closure_sha256=OLD_CLOSURE,
        expected_records=len(expected),observed_records=len(records),valid_records=len(values),missing_keys=[list(k) for k in sorted(missing,key=str)],errors=errors,
        matched_triplets=len(pairs)//3,paired_comparisons=len(pairs),pairs_per_contrast=dict(Counter(p['contrast'] for p in pairs)),
        trial_summaries=trials,paired_cases=pairs,declared_order_balance=order_counts,confirmatory=None,
        scope='Main inference on two datasets conditional on selected trace/content and one allocation; MBPP is a separate descriptive operational sentinel.',
        limitations=['Current/fix physical assignment identity is audited separately; no offline routing enters final arms.',
            'Controller tails are post-output extent, not exclusive cost; complete elapsed includes horizon/drain.',
            'Numerics:4,112whole batches/group against stockoff;64FP32 positions/group;16whole postflight batches/trial plus resident-weight checks.',
            'Independent bank/route/lifecycle and statistical audits remain required.'])
    if valid and phase=='final':result['confirmatory']=infer_endpoints(pairs,[dict(id=g,dataset=b['dataset']) for g,b in groups.items()])
    if valid and phase=='calibration':result['calibration']={r['content_group_id']:r['calibration_capacity'] for r in records}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plan=load_plan(args.plan);rows,files=read_records(args.records)
    result=analyze(plan,rows);result['record_files']=files
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('phase','valid','expected_records','observed_records','paired_comparisons')}))
    if not result['valid']:raise SystemExit(1)
if __name__=='__main__':main()
