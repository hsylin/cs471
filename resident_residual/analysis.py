"""Complete-record validation and prespecified residual-path repair analysis.

Reuses only immutable clock/accounting and unchanged-arm validators. The previous/fix static
semantics and the selected alignment specialization are checked explicitly.
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
from resident_residual.common import ARMS,PREVIOUS_CLOSURE,seal,validate_calendar,load_plan
from resident_residual.inference import DATASETS,infer_endpoints,validate_arm_orders
from resident_fastpath.analysis import fix_native_issues,precision_issues
from resident_residual.alignment import EXPECTED_CONFIG,SOURCE_SHA256
from resident_residual.compiled import TRITON_VERSION,TRITON_SOURCE_SHA256,GPU_SOURCE_SHA256

CONTRASTS={'fix_over_stockoff':('stockoff','fix'),'fix_over_previous':('previous','fix'),'previous_over_stockoff':('stockoff','previous')}
KEY_FIELDS=('content_group_id','arrival_type','rho','arm')


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_plan(plan):
    require(plan.get('schema')=='resident_residual_plan_v1' and seal(plan)==plan,'Invalid/changed frozen plan')
    phase=plan['phase'];split=plan['split']
    require(phase in ('calibration','pilot','final') and split in ('pilot','final') and (phase=='calibration' or phase==split),'Phase/split mismatch')
    require(plan['arms']==list(ARMS) and plan['jobs']==4096 and plan['batch_tokens']==128,'Wrong frozen scope')
    require(plan['previous_closure_sha256']==PREVIOUS_CLOSURE,'Old frozen closure changed')
    require(plan['source_sha256']==digest(plan['source_files']),'New source inventory digest differs')
    selected=plan['selected_fix']
    require(selected.get('selector')=='fused' and selected.get('selector_launch') in ('jit','prebound_compiled')
        and all(type(selected.get(k)) is bool for k in ('alignment_trim','mapper_input_checks')),'Unknown selected implementation')
    require(plan['selected_fix_content_sha256']==digest(selected),'Selected fix content changed')
    require(isinstance(plan['selected_fix_sha256'],str) and len(plan['selected_fix_sha256'])==64 and all(c in '0123456789abcdef' for c in plan['selected_fix_sha256']),'Selected fix file hash missing')
    groups={b['block_id']:b for b in plan['banks']};n=2 if split=='pilot' else 8
    require(len(groups)==len(plan['banks'])==2*n,'Duplicate/missing fresh groups')
    require(Counter(b['dataset'] for b in groups.values())==Counter({'gsm8k_train':n,'wildchat':n}),'Wrong two-dataset allocation')
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


def geometry_issues(row,plan):
    issues=[];arm=row['arm'];selected=plan['selected_fix']
    trimmed=arm=='fix' and selected['alignment_trim'];slots=128 if arm=='stockoff' else 160
    for rank,peer in enumerate(row.get('per_rank_state',[])):
        observed=peer.get('alignment',{})
        expected=dict(schema='native_alignment_extent_v1',enabled=trimmed,probe_only=False,
            alignment_calls=4096 if trimmed else 0,native_allocation_preserved=True,
            native_alignment_algorithm_preserved=True,capacity_bound=5056,active_local_cap=64,
            global_physical_experts=slots,native_capacity=1024+slots*63,config=EXPECTED_CONFIG,
            native_source_sha256=SOURCE_SHA256,runtime_gpu_readback=False,actual_padding_verified=False)
        if any(observed.get(k)!=v for k,v in expected.items()):issues.append(f'rank{rank}_charged_alignment_contract')
    # Each arm's full prepared-input geometry proof is generated outside timing.
    probes=row.get('preflight',{}).get('alignment_probe',[])
    if len(probes)!=2:issues.append('missing_rank_alignment_preflight')
    for rank,probe in enumerate(probes):
        expected=dict(schema='native_alignment_extent_v1',enabled=trimmed,probe_only=True,
            actual_padding_verified=True,alignment_calls=4112,capacity_bound=5056,active_local_cap=64,
            global_physical_experts=slots,native_capacity=1024+slots*63,config=EXPECTED_CONFIG,
            native_source_sha256=SOURCE_SHA256,native_allocation_preserved=True,native_alignment_algorithm_preserved=True)
        if any(probe.get(k)!=v for k,v in expected.items()):issues.append(f'rank{rank}_alignment_preflight')
    return issues


def compiled_launch_issues(row,plan):
    """Bind the selected invocation path to actual compiled-kernel attestations."""
    issues=[];arm=row['arm'];mode=plan['selected_fix']['selector_launch'] if arm=='fix' else 'jit'
    stable=[]
    def valid_hash(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for rank,peer in enumerate(row.get('per_rank_state',[])):
        rank_identity=[]
        for scope in ('start','end','audit'):
            state=peer.get(scope,{});meta=state.get('compiled_launch')
            if mode=='jit':
                if meta is not None:issues.append(f'rank{rank}_{scope}_unexpected_compiled_launch')
                continue
            if not isinstance(meta,dict):issues.append(f'rank{rank}_{scope}_compiled_launch_missing');continue
            expected=dict(schema='resident_prebound_compiled_launch_v1',mode='prebound_compiled',triton_version=TRITON_VERSION,
                gpu_algorithm_changed=False,input_checks_retained=True,pointer_alignment_guard=True,
                device=f'cuda:{rank}',fixed_argument_storage_verified=True,hook_contract_verified=True,
                dynamic_launch_metadata_hooks_preserved=True,jit_prerun_hooks_required_empty=True,
                grid=[1,1,1],constexpr_args=[1024,8],gpu_source_sha256=GPU_SOURCE_SHA256,
                calls=0 if scope=='start' else 4096)
            if any(meta.get(k)!=v for k,v in expected.items()):issues.append(f'rank{rank}_{scope}_compiled_contract')
            if type(meta.get('stream_handle')) is not int or meta.get('stream_handle',-1)<0:issues.append(f'rank{rank}_{scope}_compiled_stream')
            if scope=='audit' and meta.get('stream_handle')!=state.get('fused_stream_context',{}).get('stream_handle'):issues.append(f'rank{rank}_compiled_context_disagrees')
            sources=meta.get('compiler_source_sha256',{})
            if sources!=TRITON_SOURCE_SHA256:issues.append(f'rank{rank}_{scope}_compiler_source_pins')
            kernels=meta.get('kernels',{})
            if not isinstance(kernels,dict) or set(kernels)!={'int32','int64'}:issues.append(f'rank{rank}_{scope}_compiled_specializations');continue
            for dtype,kernel in kernels.items():
                if not isinstance(kernel,dict):issues.append(f'rank{rank}_{scope}_{dtype}_compiled_kernel');continue
                if (kernel.get('baseline_jit_same_compiled_object') is not True or kernel.get('num_warps')!=4 or kernel.get('num_ctas')!=1 or kernel.get('arch')!=90
                    or any(not valid_hash(kernel.get(key)) for key in ('compiled_hash','cubin_sha256','ptx_sha256'))
                    or not kernel.get('pointer_specialization_attributes')
                    or type(kernel.get('shared')) is not int or kernel.get('shared',-1)<0):issues.append(f'rank{rank}_{scope}_{dtype}_compiled_kernel')
                signature=dict(ids='*i32' if dtype=='int32' else '*i64',real_tokens='*i32',
                    output='*i32' if dtype=='int32' else '*i64',homes='*i32',logical_counts='*i32',summary='*i32',N='constexpr',TOPK='constexpr')
                encoded=[[repr(k),repr(v)] for k,v in sorted(signature.items(),key=lambda item:repr(item[0]))]
                if kernel.get('signature')!=encoded or kernel.get('constants')!=[['(6,)','1024'],['(7,)','8']]:issues.append(f'rank{rank}_{scope}_{dtype}_compiled_signature')
            rank_identity.append(digest(dict(kernels=kernels,compiler_source_sha256=sources)))
        if mode=='prebound_compiled':
            if len(rank_identity)!=3 or len(set(rank_identity))!=1:issues.append(f'rank{rank}_compiled_identity_changed')
            stable.extend(rank_identity)
    if mode=='prebound_compiled' and len(set(stable))!=1:issues.append('rank_compiled_specialization_disagreement')
    return issues


def resident_native_issues(row,plan):
    """Validate real public arm labels before adapting immutable static checks."""
    issues=[];arm=row['arm'];selected=plan['selected_fix']
    peers=row.get('per_rank_state',[])
    for rank,peer in enumerate(peers):
        for scope in ('start','end','audit'):
            state=peer.get(scope,{})
            if state.get('arm')!=arm:issues.append(f'rank{rank}_{scope}_public_arm')
            if state.get('selector_launch')!=(selected['selector_launch'] if arm=='fix' else 'jit'):issues.append(f'rank{rank}_{scope}_selector_launch')
            if state.get('alignment_trim')!=(arm=='fix' and selected['alignment_trim']):issues.append(f'rank{rank}_{scope}_trim_selection')
            if state.get('mapper_input_checks')!=(selected['mapper_input_checks'] if arm=='fix' else True):issues.append(f'rank{rank}_{scope}_mapper_checks_selection')
    alias={**row,'arm':'fix','per_rank_state':copy.deepcopy(peers)}
    for peer in alias.get('per_rank_state',[]):
        for scope in ('start','end','audit'):peer.setdefault(scope,{})['arm']='fix'
    issues.extend(fix_native_issues(alias,'fused'))
    issues.extend(compiled_launch_issues(row,plan))
    return issues


def row_issues(row,plan,group,case):
    issues=[];arm=row['arm'];phase=plan['phase']
    expected=dict(schema='resident_residual_trial_v1',phase=phase,status='complete',plan_sha256=plan['content_sha256'],
        source_sha256=plan['source_sha256'],previous_closure_sha256=PREVIOUS_CLOSURE,content_group_id=group['block_id'],block_id=group['block_id'],
        dataset=group['dataset'],n_jobs=4096,evidence_eligible=phase=='final',
        inference_role='main')
    if any(row.get(k)!=v for k,v in expected.items()):issues.append('row_identity_or_phase')
    if row.get('record_id')!=digest({k:v for k,v in row.items() if k!='record_id'}):issues.append('record_content_hash')
    hw=plan['hardware'];clock=row.get('clock',{})
    if str(row.get('allocation_id'))!=str(hw['allocation_id']) or str(row.get('hostname','')).split('.')[0]!=hw['hostname'] or sorted(row.get('gpu_uuids',[]))!=sorted(hw['gpu_uuids']):issues.append('hardware_identity')
    if clock.get('monotonic') is not True or clock.get('adjustable') is not False or 'CLOCK_MONOTONIC' not in clock.get('implementation',''):issues.append('shared_clock')
    meta=row.get('input_meta',{})
    required=dict(schema='resident_residual_hidden_jobs_v1',seed=471803,block_id=group['block_id'],capture_sha256=group['capture_sha256'],
        warmup_jobs=16,measured_jobs=4096,batch_tokens=128,distinct_job_combinations=4112)
    if any(meta.get(k)!=v for k,v in required.items()):issues.append('authentic_job_identity')
    if not isinstance(meta.get('indices_sha256'),str) or len(meta['indices_sha256'])!=64:issues.append('input_indices_hash')
    rt=row.get('runtime_metadata',{})
    required=dict(study_id='resident_residual_v1',selected_fix=plan['selected_fix'],native_vllm_commit=plan['vllm_commit'],
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
    if arm=='stockoff':issues.extend(native_issues(row))
    else:issues.extend(resident_native_issues(row,plan))
    issues.extend(geometry_issues(row,plan))
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
            if rows['previous']['preflight']['physical_assignment_digest']!=rows['fix']['preflight']['physical_assignment_digest']:errors.append(dict(key=list(key),issues=['changed_physical_assignment_policy']))
            for contrast,(control,treatment) in CONTRASTS.items():
                pairs.append(dict(content_group_id=key[0],dataset=groups[key[0]]['dataset'],arrival_type=key[1],rho=key[2],arrival_date=case['calendar']['arrival_date'],
                    contrast=contrast,control_arm=control,treatment_arm=treatment,**paired_values(values[ks[control]],values[ks[treatment]])))
        for gid in groups:
            members=[r for k,r in index.items() if k[0]==gid]
            if len({digest(_paired_identity(r)) for r in members})!=1:errors.append(dict(group=gid,issues=['cross_case_input_routes_gate']))
    valid=not errors and not missing
    result=dict(schema='resident_residual_analysis_v1',phase=phase,valid=valid,plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],previous_closure_sha256=PREVIOUS_CLOSURE,
        expected_records=len(expected),observed_records=len(records),valid_records=len(values),missing_keys=[list(k) for k in sorted(missing,key=str)],errors=errors,
        matched_triplets=len(pairs)//3,paired_comparisons=len(pairs),pairs_per_contrast=dict(Counter(p['contrast'] for p in pairs)),
        trial_summaries=trials,paired_cases=pairs,declared_order_balance=order_counts,confirmatory=None,
        scope='Inference on GSM8K-train/WildChat content bundles conditional on selected trace/content and one allocation.',
        limitations=['Previous/fix physical assignment identity is audited separately; no offline routing enters final arms.',
            'Controller tails are post-output extent, not exclusive cost; complete elapsed includes horizon/drain.',
            'Numerics:4,112 whole batches/group against stockoff;64 FP32 positions/group;16 whole postflight batches/trial plus resident-weight checks.',
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
