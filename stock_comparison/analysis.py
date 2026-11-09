"""Complete-matrix analysis of genuine stock-layer comparator trials.

Only final verified run-level content bundles enter the frozen inference helper.
Per-job summaries and controller-tail accounting are descriptive measurements.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np
from arrival_study.schedules import calibration_capacity, fcfs_metrics
from stock_comparison.common import ARMS, seal, validate_calendar
from stock_comparison.data import digest
from stock_comparison.inference import DATASETS, infer_endpoints

CONTRASTS={'ours_over_stockoff':('stockoff','ours'),
           'ours_over_stockon32':('stockon32','ours'),
           'stockon32_over_stockoff':('stockoff','stockon32')}
KEY_FIELDS=('content_group_id','arrival_type','rho','arm')


def _require(condition,message):
    if not condition:raise ValueError(message)


def validate_plan(plan):
    """Validate sealed explicit scope and treatment-order counterbalancing."""
    _require(plan.get('schema')=='stock_comparison_plan_v1' and seal(plan)==plan,'Invalid/changed frozen plan')
    _require(plan.get('phase') in ('calibration','pilot','final'),'Unknown phase')
    phase=plan['phase'];split=plan['split']
    _require(split in ('pilot','final') and (phase=='calibration' or phase==split),'Phase/split mismatch')
    _require(plan.get('arms')==list(ARMS) and plan.get('jobs')==4096 and plan.get('batch_tokens')==128,'Wrong frozen scope')
    banks=plan['banks'];groups={b['block_id']:b for b in banks};n=2 if split=='pilot' else 6
    _require(len(groups)==len(banks)==3*n,'Duplicate/missing content groups')
    _require(Counter(b['dataset'] for b in banks)==Counter({d:n for d in DATASETS}),'Wrong dataset allocation')
    _require(all(b['split']==split for b in banks),'Capture split mismatch')
    if phase=='calibration':
        _require(plan['cases']==[],'Calibration has treatment cases')
        return groups,{}
    cases={};orders={}
    for c in plan['cases']:
        cal=c['calendar'];validate_calendar(cal,c['source_block'])
        k=(c['block_id'],cal['arrival_type'],cal['rho'])
        _require(k not in cases and k[0] in groups,'Duplicate/unexpected case')
        _require(c['dataset']==groups[k[0]]['dataset'],'Case dataset mismatch')
        _require(sorted(c['arm_order'])==sorted(ARMS),'Invalid arm permutation')
        cases[k]=c;orders[k]=tuple(c['arm_order'])
    expected=set(itertools.product(groups,('code','conversation'),(.5,.85)))
    _require(set(cases)==expected,'Incomplete declared case matrix')
    provenance=plan.get('calibration_provenance',{})
    _require(set(provenance)==set(groups),'Missing frozen calibration provenance')
    for key,case in cases.items():
        p=provenance[key[0]]
        _require(p['calibration_elapsed_ns']==case['calendar']['stockoff_calibration_elapsed_ns'],'Calendar calibration mismatch')
        _require(str(p['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration allocation mismatch')
        _require(p['source_sha256']==plan['source_sha256'] and p['record_id']==plan['calibration_record_ids'][key[0]],'Calibration source/record mismatch')
    permutations=set(itertools.permutations(ARMS))
    strata=[(d,rho) for d in (DATASETS if phase=='final' else (None,)) for rho in (.5,.85)]
    for dataset,rho in strata:
        counts=Counter(order for k,order in orders.items() if k[2]==rho and (dataset is None or groups[k[0]]['dataset']==dataset))
        _require(set(counts)==permutations and len(set(counts.values()))==1,'Arm orders not balanced within declared load/dataset strata')
    return groups,cases


def clock_metrics(row):
    """Reconstruct complete FCFS outcomes and distinguish output from quiescence."""
    _require(row.get('per_job_completion_scope')=='current_inference_stream_after_native_step','Wrong per-job completion stream scope')
    _require(row.get('n_jobs')==4096,'Wrong measured job count')
    _require(np.asarray(row.get('per_rank_start_ns')).shape==(2,4096) and np.asarray(row.get('per_rank_gpu_complete_ns')).shape==(2,4096),'Missing rank timing arrays')
    m=fcfs_metrics(row)
    _require(m['complete'] and m['completed_jobs']==4096,'Incomplete trial')
    _require(row.get('queue_metrics')==m,'Stored metrics differ from raw clocks')
    e=row['epoch_monotonic_ns'];f=row['coordinator_finish_ns'][-1];q=row['controller_quiescent_ns'];s=row['controller_quiescence_start_ns'];end=row['observation_end_ns'];h=row['horizon_ns']
    _require(all(type(v) is int for v in (e,f,q,s,end,h)) and f<=s<=q<=end,'Invalid controller/observation timestamps')
    expected={'completed_elapsed_ns':end-e,'job_completed_elapsed_ns':f-e,
        'controller_tail_elapsed_ns':q-f,'quiescence_call_elapsed_ns':q-s,
        'drain_including_controller_ns':max(0,q-e-h)}
    _require(all(row.get(k)==v for k,v in expected.items()),'Controller/completion elapsed accounting mismatch')
    starts=row['per_rank_controller_tail_start_ns'];ends=row['per_rank_controller_quiescent_ns']
    _require(len(starts)==len(ends)==2 and starts[0]==s and ends[0]==q,'Controller rank-0 clock mismatch')
    for rank,(rs,rf) in enumerate(zip(starts,ends)):
        _require(type(rs) is int and type(rf) is int and row['per_rank_gpu_complete_ns'][rank][-1]<=rs<=rf,'Invalid per-rank controller tail')
    # Other rank host-return timestamps may exceed rank0 q after the same barrier.
    # They need not precede rank0 observation_end; GPU completion is ACK-enclosed.
    values={}
    for label,summary in m['summary_ns'].items():
        label='queue' if label=='queue_wait' else label
        for stat,value in summary.items():values[f'{label}_{stat}_ns']=value
    values.update(expected)
    values.update(drain_ns=m['drain_ns'],backlog_at_horizon=m['backlog_at_horizon'],
        waiting_jobs_at_horizon=m['waiting_jobs_at_horizon'],throughput_jobs_per_second=m['completed_per_evaluation_second'],
        by_horizon_throughput_jobs_per_second=m['completed_by_horizon_per_second'],
        completed_by_horizon=m['completed_by_horizon'],horizon_ns=h,
        coordinator_busy_fraction=sum(m['vectors_ns']['service'])/(end-e))
    return values


def paired_values(control,treatment):
    """Zero-denominator descriptive ratios stay null; absolute effects remain."""
    _require(set(control)==set(treatment),'Mismatched paired metric schema')
    logs={k:math.log(treatment[k])-math.log(control[k]) for k in control if control[k]>0 and treatment[k]>0}
    _require(all(k in logs for k in ('response_mean_ns','response_p95_ns','service_mean_ns','completed_elapsed_ns')),'Nonpositive primary timing endpoint')
    return dict(control=control,treatment=treatment,log_ratios=logs,
        ratios={k:treatment[k]/v if v>0 else None for k,v in control.items()},
        differences={k:treatment[k]-v for k,v in control.items()})


def _paired_identity(row):
    meta=row.get('input_meta',{});pre=row.get('preflight',{})
    return dict(input={k:meta.get(k) for k in ('block_id','capture_sha256','indices_sha256','source_prompt_ids','source_records','warmup_jobs','measured_jobs','batch_tokens')},
        routes={k:pre.get(k) for k in ('logical_routing_sha256','float32_combining_weights_sha256')},
        gate=row.get('per_rank_state',[{}])[0].get('start',{}).get('gate_weight_sha256'))


def validate_async_lifecycle(state):
    """Allow genuine pending-work deferral without selecting faster stock runs."""
    events=state['async_events'];recorded=state['recorded_step_indices'];deferred=state['deferred_boundary_steps']
    _require(recorded==sorted(set(recorded)) and all(type(x) is int and 0<=x<4096 for x in recorded),'Invalid recorded native steps')
    _require(deferred==sorted(set(deferred)) and all(type(x) is int and 0<=x<4096 for x in deferred),'Invalid deferred native boundaries')
    _require(bool(events) and events[0]['launch_step']==749,'Missing native startup opportunity')
    due=749;previous_map=state['source_map'];expected_deferred=[]
    for event in events:
        step=event['launch_step'];commit=event['commit_step']
        _require(type(step) is int and type(commit) is int and due<=step<=commit<=4095,'Native launch/commit ordering')
        expected_deferred.extend(range(due,step))
        _require(event['source_map']==previous_map,'Native async source/commit map chain')
        selected=[x for x in recorded if x<=step][-1000:]
        _require(event['sampled_step_indices']==selected and event['window_valid_steps']==len(selected),'Native sampled window indices')
        _require(event['launch_counter_before']==2999+(step-due),'Native launch counter')
        if event['commit_in_completion_tail']:_require(commit==4095,'Tail commit step')
        previous_map=event['committed_map'];due=step+3000
    expected_deferred.extend(range(due,4096))
    _require(deferred==expected_deferred,'Missing/extra native deferral evidence')
    _require(state['native_counter']==4095-events[-1]['launch_step'],'Final native counter')
    _require(state['final_map']==previous_map,'Final native committed map')
    _require(sum(x['sent_bytes_local'] for x in events)==state['migration_sent_bytes_local'],'Native sender payload accounting')
    _require(state['native_valid_steps']==len(recorded),'Recorded native step count')


def _row_issues(row,plan,group,case):
    issues=[]
    checks={'schema':'stock_comparison_trial_v1','phase':plan['phase'],
        'plan_sha256':plan['content_sha256'],'source_sha256':plan['source_sha256'],
        'dataset':group['dataset'],'block_id':group['block_id'],'content_group_id':group['block_id'],
        'n_jobs':4096,'status':'complete','evidence_eligible':plan['phase']=='final'}
    for key,value in checks.items():
        if row.get(key)!=value:issues.append(key)
    hw=plan['hardware']
    if str(row.get('allocation_id'))!=str(hw['allocation_id']) or str(row.get('hostname','')).split('.')[0]!=hw['hostname']:issues.append('hardware_allocation')
    if sorted(row.get('gpu_uuids',[]))!=sorted(hw['gpu_uuids']):issues.append('GPU_identity')
    clock=row.get('clock',{})
    if clock.get('monotonic') is not True or clock.get('adjustable') is not False or 'CLOCK_MONOTONIC' not in clock.get('implementation',''):issues.append('shared_monotonic_clock')
    meta=row.get('input_meta',{})
    for key,value in {'block_id':group['block_id'],'capture_sha256':group['capture_sha256'],
        'warmup_jobs':16,'measured_jobs':4096,'batch_tokens':128,'distinct_job_combinations':4112}.items():
        if meta.get(key)!=value:issues.append('input_'+key)
    if not isinstance(meta.get('indices_sha256'),str) or len(meta['indices_sha256'])!=64:issues.append('input_indices_identity')
    if row.get('record_id')!=digest({k:v for k,v in row.items() if k!='record_id'}):issues.append('record_content_hash')
    if case is None:
        if row.get('arm')!='stockoff' or row.get('calendar') is not None or row.get('arrival_offsets_ns')!=[0]*4096 or row.get('horizon_ns')!=1 or row.get('arm_order')!=['stockoff']:issues.append('saturated_calibration')
        try:
            if row.get('calibration_capacity')!=calibration_capacity(row['coordinator_start_ns'],row['coordinator_finish_ns'],expected_jobs=4096):issues.append('calibration_capacity')
        except (KeyError,ValueError,TypeError) as exc:issues.append('calibration_capacity:'+str(exc))
    else:
        cal=case['calendar']
        for key,value in {'calendar':cal,'arrival_type':cal['arrival_type'],'arrival_date':cal['arrival_date'],
            'rho':cal['rho'],'arrival_offsets_ns':cal['arrival_offsets_ns'],'horizon_ns':cal['horizon_ns'],'arm_order':case['arm_order']}.items():
            if row.get(key)!=value:issues.append(key)
    issues.extend(native_issues(row))
    runtime=row.get('runtime_metadata',{})
    if runtime.get('native_vllm_commit')!=plan['vllm_commit'] or runtime.get('model_revision')!=plan['checkpoint_revision'] or runtime.get('world_size')!=2 or runtime.get('native_output_reduction') is not True:issues.append('pinned_native_runtime')
    if row.get('source_weights',{}).get('source_revision')!=plan['checkpoint_revision'] or row.get('source_weights',{}).get('layer')!=0:issues.append('logical_checkpoint_weights')
    return issues


def native_issues(row):
    """Validate genuine native configuration, numerical scope and completed state."""
    issues=[];arm=row['arm'];pre=row.get('preflight',{});post=row.get('postflight',{})
    for name,check,count in (('preflight',pre,4112),('postflight',post,16)):
        expected=dict(correctness_pass=True,full_batch_inputs_checked=count,atol=.02,rtol=.03)
        if any(check.get(k)!=v for k,v in expected.items()):issues.append('correctness_'+name)
        error=check.get('full_batch_max_abs_difference')
        if not isinstance(error,(int,float)) or not math.isfinite(error) or error<0:issues.append('numerical_error_'+name)
    if any(pre.get(k) is not True for k in ('logical_ids_equal','combining_weights_equal','rank_routing_agreement','input_unchanged')):issues.append('preflight_logical_identity')
    for key in ('logical_routing_sha256','float32_combining_weights_sha256'):
        if not isinstance(pre.get(key),str) or len(pre[key])!=64:issues.append('preflight_'+key)
    if pre.get('fp32_job_indices')!=[16,765,3765,4111] or pre.get('fp32_tokens_per_job')!=16:issues.append('fp32_sample_scope')
    if post.get('raw_indices')!=np.linspace(16,4111,16,dtype=np.int64).tolist():issues.append('postflight_sample_scope')
    peers=row.get('per_rank_state',[])
    if len(peers)!=2:return issues+['missing_rank_native_state']
    expected=dict(schema_version=1,arm=arm,enable_eplb=arm!='stockoff',physical_slots=128 if arm=='stockoff' else 160,
        local_slots=64 if arm=='stockoff' else 80,window_steps=16 if arm=='ours' else 1000,
        rebalance_interval=64 if arm=='ours' else 3000,initial_counter=0 if arm=='ours' else 2250 if arm=='stockon32' else None,
        is_async=arm=='stockon32',logical_router_computation_included=True,native_output_reduction=True,
        stock_router_unmodified=arm!='ours',stock_policy_unmodified=arm!='ours',custom_all_reduce_disabled=False,
        completion_disposition='normal_commit_no_discard',pending_result=False,rebalanced=False)
    for rank,peer in enumerate(peers):
        start,end,audit=(peer.get(k,{}) for k in ('start','end','audit'))
        for scope,state in (('start',start),('end',end),('audit',audit)):
            if any(state.get(k)!=v for k,v in expected.items()):issues.append(f'rank{rank}_{scope}_native_configuration')
            slots=expected['physical_slots'];mapping=state.get('final_map',[])
            if len(mapping)!=slots or any(type(x) is not int or not 0<=x<128 for x in mapping) or set(mapping)!=set(range(128)):issues.append(f'rank{rank}_{scope}_legal_map')
            if not isinstance(state.get('gate_weight_sha256'),str) or len(state['gate_weight_sha256'])!=64:issues.append(f'rank{rank}_{scope}_gate_identity')
            if arm!='stockon32' and mapping!=state.get('source_map'):issues.append(f'rank{rank}_{scope}_fixed_map_changed')
        if any(audit.get(k) is not True for k in ('resident_state_verified','rank_map_agreement','normal_completion_verified')):issues.append(f'rank{rank}_weight_or_completion_audit')
        if start.get('native_counter')!=expected['initial_counter'] or start.get('controller_step_calls')!=0 or start.get('async_events')!=[] or start.get('mapper_calls')!=0:issues.append(f'rank{rank}_fresh_start')
        if end.get('controller_step_calls')!=(0 if arm=='stockoff' else 4096):issues.append(f'rank{rank}_controller_step_count')
        if end.get('mapper_calls')!=(4096 if arm=='ours' else 0):issues.append(f'rank{rank}_mapper_call_count')
        if any(end.get(k)!=audit.get(k) for k in ('final_map','native_counter','migration_sent_bytes_local','async_events','mapper_calls')):issues.append(f'rank{rank}_posttiming_state_changed')
        if arm=='stockon32':
            events=end.get('async_events',[])
            if end.get('async_worker_alive') is not True:issues.append(f'rank{rank}_async_completion')
            try:validate_async_lifecycle(end)
            except (ValueError,KeyError,TypeError) as exc:issues.append(f'rank{rank}_async_lifecycle:'+str(exc))
            for event in events:
                window=event.get('window_valid_steps',0)
                counts=np.asarray(event.get('current_counts',[]))
                if counts.size!=128 or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or int(counts.sum())!=window*128*8*2:issues.append(f'rank{rank}_native_window_counts')
                mapping=event.get('committed_map',[])
                if len(mapping)!=160 or set(mapping)!=set(range(128)):issues.append(f'rank{rank}_committed_map')
                if not isinstance(event.get('sent_bytes_local'),int) or event['sent_bytes_local']<0:issues.append(f'rank{rank}_migration_payload')
        elif end.get('async_events')!=[] or end.get('migration_sent_bytes_local')!=0:issues.append(f'rank{rank}_unexpected_migration')
        if arm=='ours':
            decisions=end.get('decisions',[]);guard=end.get('guard',{})
            if [d.get('step') for d in decisions]!=list(range(63,4096,64)) or end.get('terminal_proposals_suppressed')!=0:issues.append(f'rank{rank}_all64_ours_boundaries')
            if any(d.get('allowed') is not False or d.get('current_map')!=d.get('committed_map') or sum(d.get('current_counts',[]))!=16*128*8*2 for d in decisions):issues.append(f'rank{rank}_ours_skip_window')
            required=dict(mode='fast',diagnostic_timing=False,pending_pair=False,original_movement_calls=0,original_commit_calls=0,fallback_calls=0)
            required.update({k:64 for k in ('movement_calls','commit_calls','agreement_calls','local_unchanged_calls','unanimously_unchanged_calls','movement_skips','commit_skips')})
            if any(guard.get(k)!=v for k,v in required.items()):issues.append(f'rank{rank}_shortcut_counts')
            if pre.get('online_assignment_inputs_checked')!=4112 or not pre.get('physical_assignment_digest'):issues.append(f'rank{rank}_online_cpu_oracle')
        if start.get('gate_weight_sha256')!=row.get('input_meta',{}).get('gate_weight_sha256'):issues.append(f'rank{rank}_capture_gate_identity')
    for scope in ('start','end','audit'):
        for field in ('source_map','final_map','gate_weight_sha256','native_counter'):
            if peers[0].get(scope,{}).get(field)!=peers[1].get(scope,{}).get(field):issues.append('rank_disagreement_'+scope+'_'+field)
    return issues


def analyze(plan,records):
    groups,cases=validate_plan(plan);phase=plan['phase']
    expected={(g,None,None,'stockoff') for g in groups} if phase=='calibration' else {(*key,arm) for key in cases for arm in ARMS}
    index={};metrics={};errors=[];summaries=[];actual_orders={k:[] for k in cases}
    for row in records:
        key=tuple(row.get(k) for k in KEY_FIELDS)
        if key not in expected or key in index:
            errors.append(dict(key=list(key),issues=['unexpected_or_duplicate_row']));continue
        index[key]=row
        if phase!='calibration':actual_orders[key[:3]].append(row['arm'])
        case=None if phase=='calibration' else cases[key[:3]]
        issues=_row_issues(row,plan,groups[key[0]],case)
        try:values=clock_metrics(row)
        except (KeyError,ValueError,TypeError,IndexError,AssertionError) as exc:issues.append('clock:'+str(exc));values=None
        if issues:errors.append(dict(key=list(key),issues=issues));continue
        metrics[key]=values
        summaries.append({**dict(zip(KEY_FIELDS,key)),'dataset':row['dataset'],'arrival_date':row.get('arrival_date'),'record_id':row.get('record_id'),'metrics':values,
            'native_launch_steps':[e['launch_step'] for e in row['per_rank_state'][0]['end']['async_events']],
            'native_deferred_boundary_steps':row['per_rank_state'][0]['end'].get('deferred_boundary_steps',[]),
            'migration_sender_bytes':sum(p['end']['migration_sent_bytes_local'] for p in row['per_rank_state'])})
    missing=expected-set(index);pairs=[];positions={a:[0,0,0] for a in ARMS}
    if phase!='calibration':
        for key,case in cases.items():
            keys={a:(*key,a) for a in ARMS}
            if any(k not in metrics for k in keys.values()):continue
            rows={a:index[k] for a,k in keys.items()}
            if actual_orders[key]!=case['arm_order']:errors.append(dict(key=list(key),issues=['actual_arm_order']))
            for pos,arm in enumerate(case['arm_order']):positions[arm][pos]+=1
            # Stock-enabled physical trajectories can legitimately depend on async
            # execution. Only authentic inputs and logical computation are paired.
            if len({digest(_paired_identity(r)) for r in rows.values()})!=1:errors.append(dict(key=list(key),issues=['paired_input_routes_gate']))
            for contrast,(control,treatment) in CONTRASTS.items():
                pairs.append(dict(content_group_id=key[0],dataset=groups[key[0]]['dataset'],arrival_type=key[1],rho=key[2],
                    arrival_date=case['calendar']['arrival_date'],contrast=contrast,control_arm=control,treatment_arm=treatment,
                    **paired_values(metrics[keys[control]],metrics[keys[treatment]])))
        for gid in groups:
            members=[r for k,r in index.items() if k[0]==gid]
            if len({digest(_paired_identity(r)) for r in members})!=1:errors.append(dict(group=gid,issues=['cross_case_input_routes_gate']))
    valid=not errors and not missing
    result=dict(schema='stock_comparison_analysis_v1',phase=phase,valid=valid,plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
        observed_records=len(records),expected_records=len(expected),valid_records=len(metrics),missing_keys=[list(k) for k in sorted(missing,key=str)],errors=errors,
        matched_triplets=len(pairs)//3,paired_comparisons=len(pairs),pairs_per_contrast=dict(Counter(p['contrast'] for p in pairs)),
        trial_summaries=summaries,paired_cases=pairs,arm_position_counts=positions,confirmatory=None,
        scope='Layer jobs; uncertainty conditional on fixed selected arrivals/content construction and one hardware allocation.',
        limitations=['Controller tail is post-output extent, not exclusive controller cost; response ends at normal job ACK.',
            'Coordinator busy fraction is host service occupancy, not GPU utilization.',
            'Zero-denominator descriptive queue/drain ratios are null; absolute differences remain.',
            'External source-row, bank/input and actual native-route auditors remain required.'])
    if valid and phase=='final':result['confirmatory']=infer_endpoints(pairs,[{'id':g,'dataset':b['dataset']} for g,b in groups.items()])
    if valid and phase=='calibration':result['calibration']={r['content_group_id']:r['calibration_capacity'] for r in records}
    return result


def read_records(path):
    path=Path(path)
    paths=sorted((path/'trials').glob('*.jsonl*')) if path.is_dir() else [path]
    _require(bool(paths),'No trial files found')
    rows=[];sources=[]
    for p in paths:
        raw=p.read_bytes();payload=gzip.decompress(raw) if p.suffix=='.gz' else raw
        rows.extend(json.loads(line) for line in payload.decode().splitlines() if line.strip())
        sources.append(dict(path=str(p),stored_bytes_sha256=digest(raw),uncompressed_bytes_sha256=digest(payload)))
    return rows,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();rows,sources=read_records(args.records)
    result=analyze(json.loads(args.plan.read_text()),rows);result['record_files']=sources
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('phase','valid','expected_records','observed_records','paired_comparisons')}))
    if not result['valid']:raise SystemExit(1)

if __name__=='__main__':main()
