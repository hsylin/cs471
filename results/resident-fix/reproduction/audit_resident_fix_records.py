#!/usr/bin/env python3
"""Independent resident-fix clocks, exact assignments and static-state audit.

No production imports. Saved actual-stock routes drive a separate NumPy oracle;
this does not rerun GPU kernels or turn sampled numerical checks into a proof.
"""
import argparse
from collections import Counter
import importlib.util
import itertools
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_stock_raw',HERE/'stock_raw_primitives.py')
stock=importlib.util.module_from_spec(spec);spec.loader.exec_module(stock)
sha,require,read_rows=stock.sha,stock.require,stock.read_rows
ARMS=('stockoff','current','fix')
OLD='9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4'


def greedy_oracle(ids):
    """Scalar LPT over active shared experts, under exact active-count caps."""
    require(ids.shape==(128,8) and np.issubdtype(ids.dtype,np.integer) and np.all((ids>=0)&(ids<128)),'Logical batch geometry/IDs')
    histogram=np.bincount(ids.ravel(),minlength=128)
    forced=(range(32,80),range(80,128))
    active=[sum(histogram[e]>0 for e in block) for block in forced]
    load=[sum(int(histogram[e]) for e in block) for block in forced]
    shared=[e for e in range(32) if histogram[e]>0]
    cap=max(active[0],active[1],(active[0]+active[1]+len(shared)+1)//2)
    remaining=[cap-x for x in active];homes=np.arange(128,dtype=np.int64)
    for expert in sorted(shared,key=lambda e:(-int(histogram[e]),e)):
        eligible=[r for r in (0,1) if remaining[r]>0]
        rank=sorted(eligible,key=lambda r:(load[r],r))[0]
        homes[expert]=expert if rank==0 else 128+expert
        load[rank]+=int(histogram[expert]);remaining[rank]-=1
    physical=homes[ids];counts=np.bincount(physical.ravel(),minlength=160)
    return dict(physical_ids=physical,physical_counts=counts,logical_counts=histogram,
        per_rank_tokens=np.sum(counts.reshape(2,80),axis=1),per_rank_active_copies=np.count_nonzero(counts.reshape(2,80),axis=1),
        physical_sha256=sha(np.ascontiguousarray(physical,dtype='<i8').tobytes(order='C')))


def check_assignments(inputs,gid,meta,bank=None):
    artifact=stock.input_artifacts(inputs,gid,meta)
    with np.load(inputs/(gid+'.npz'),allow_pickle=False) as z:indices=z['indices']
    require(meta['schema']=='resident_fix_hidden_jobs_v1' and meta['seed']==471703,'Fresh composition schema/seed')
    seen=set()
    for raw,row in enumerate(indices):
        for retry in range(1024):
            key=int(sha([471703,gid,128,raw,retry])[:16],16)
            expected=np.random.Generator(np.random.PCG64(key)).permutation(meta['source_records'])[:128].astype(np.int64)
            identity=np.sort(expected).tobytes()
            if identity not in seen:break
        require(np.array_equal(row,expected),'Exact independently regenerated input composition')
        seen.add(identity)
    if bank is not None:require(sha(bank.read_bytes())==meta['capture_sha256'],'Actual captured bank file hash')
    paths={arm:inputs/(gid+'-'+arm+'-resident-diagnostics.json') for arm in ('current','fix')}
    present=[arm for arm,p in paths.items() if p.exists()]
    if not present:return artifact,dict(prepared_batches=4112,physical_oracle_batches=0,assignment_digest=None)
    require(len(present)==2,'Both current/fix diagnostic artifacts required')
    diagnostics={a:json.loads(p.read_text()) for a,p in paths.items()}
    require(all(len(d)==4112 for d in diagnostics.values()),'Every prepared assignment must be checked')
    with np.load(inputs/(gid+'-stock-routes.npz'),allow_pickle=False) as z:ids=z['logical_ids'];weights=z['combining_weights']
    hashes=[];physical_steps=[]
    for raw,x in enumerate(ids):
        oracle=greedy_oracle(x);hashes.append(oracle['physical_sha256']);physical_steps.append(oracle['physical_counts'])
        for arm in ('current','fix'):
            d=diagnostics[arm][raw]
            require(d['physical_sha256']==oracle['physical_sha256'],'Exact physical-ID bytes '+arm)
            for name in ('physical_counts','per_rank_tokens','per_rank_active_copies'):
                require(np.array_equal(d[name],oracle[name]),'CPU oracle '+name+' '+arm)
            require(d['logical_sha256']==sha(np.ascontiguousarray(x,dtype='<i8').tobytes()),'Exact logical-ID bytes')
            require(d['weights_sha256']==sha(np.ascontiguousarray(weights[raw],dtype='<f4').tobytes()),'Exact float32 combining-weight bytes')
    artifact['greedy_physical_counts']=np.stack(physical_steps[16:])
    return artifact,dict(prepared_batches=4112,physical_oracle_batches=4112,assignment_digest=sha(hashes))


def static_state(row,selector):
    pre=row['preflight'];post=row['postflight'];mapping=list(range(128))+list(range(32))
    require(pre['correctness_pass'] and pre['full_batch_inputs_checked']==4112 and pre['input_unchanged'],'Full prepared input preflight')
    require(pre['online_assignment_inputs_checked']==4112 and pre['fp32_job_indices']==[16,765,3765,4111] and pre['fp32_tokens_per_job']==16,'Assignment/FP32 check coverage')
    require(post['correctness_pass'] and post['raw_indices']==np.linspace(16,4111,16,dtype=np.int64).tolist(),'Postflight coverage')
    peers=row['per_rank_state'];require(len(peers)==2,'Both rank states')
    for rank,p in enumerate(peers):
        for scope in ('start','end','audit'):
            s=p[scope];calls=0 if scope=='start' else 4096
            expected=dict(arm='fix',enable_eplb=True,replicas=32,physical_slots=160,local_slots=80,
                static_resident_mode=True,upstream_eplb_disabled=False,controller_active=False,selector_kind=selector,
                source_map=mapping,final_map=mapping,static_step_calls=calls,mapper_calls=calls,controller_step_calls=0,
                native_counter=0,native_valid_steps=0,recorded_step_indices=[],deferred_boundary_steps=[],decisions=[],async_events=[],
                physical_window_next_write=0,terminal_physical_window_totals_local=[0]*160,pending_result=False,rebalanced=False,
                migration_sent_bytes_local=0,migration_sent_messages_local=0,guard=None,terminal_proposals_suppressed=0,
                completion_disposition='normal_commit_no_discard',logical_router_computation_included=True,native_output_reduction=True,
                stock_router_unmodified=False,stock_policy_unmodified=False)
            require(all(s.get(k)==v for k,v in expected.items()),'Static contract '+scope+' rank'+str(rank))
            require(s['gate_weight_sha256']==row['input_meta']['gate_weight_sha256'],'Authenticated gate identity')
        require(all(p['audit'][k] is True for k in ('resident_state_verified','rank_map_agreement','normal_completion_verified','static_state_unchanged','observation_and_migration_inactive')),'Static full-weight/map audit')
        require(p['shortcut_guard'] is None,'No hidden current-controller guard')
        if selector=='fused':
            context=p['audit']['fused_stream_context']
            require(context['device']==f'cuda:{rank}' and type(context['stream_handle']) is int and context['stream_handle']>=0
                and context['sequential_single_stream_required'] is True and context['reused_output_storage'] is True,'Fused stream/storage context')
    if selector=='fused':
        proof=row['runtime_metadata']['fused_adversarial']
        require(proof['passed'] and proof['cases_checked']==9 and proof['gpu_invocations_checked']==18 and proof['reference_count']==2 and proof['rank_agreement'],'Bounded GPU adversarial proof')
    return dict(static_trials=1,static_rank_states=2)


def audit(plan,rows,inputs,repo,calibration_rows=None,calibration_files=None,banks_dir=None):
    require(plan['schema']=='resident_fix_plan_v1' and plan['content_sha256']==sha(stock.unsealed(plan)),'Frozen plan seal')
    require(plan['old_closure_sha256']==OLD and plan['source_sha256']==sha(plan['source_files']),'Execution identities')
    for path,expected in plan['source_files'].items():require(sha((repo/path).read_bytes())==expected,'Changed frozen source '+path)
    require(sha((repo/'docs/resident-fix/protocol-v1.md').read_bytes())==plan['protocol_sha256'],'Protocol changed')
    chosen=(repo/'configs/resident-fix-selected-v1.json').read_bytes()
    require(sha(chosen)==plan['selected_fix_sha256'] and json.loads(chosen)==plan['selected_fix'] and sha(plan['selected_fix'])==plan['selected_fix_content_sha256'],'Selected implementation freeze')
    require(plan['arms']==list(ARMS) and plan['jobs']==4096 and plan['batch_tokens']==128,'Scope')
    groups={b['block_id']:b for b in plan['banks']};n=8 if plan['split']=='final' else 2;phase=plan['phase']
    require(len(groups)==len(plan['banks'])==2*n+1 and Counter(g['dataset'] for g in groups.values())==Counter({'gsm8k':n,'wildchat':n,'mbpp':1}),'Main/sentinel groups')
    cases={}
    for c in plan['cases']:
        stock.calendar_check(c);key=(c['content_group_id'],c['arrival_type'],c['rho'])
        require(key not in cases,'Duplicate case');cases[key]=c
    expected={(g,None,None,'stockoff') for g in groups} if phase=='calibration' else {(*k,a) for k in cases for a in ARMS}
    require(len(expected)==len(groups)*(1 if phase=='calibration' else 12),'Complete plan matrix')
    if phase!='calibration':
        require(calibration_rows is not None,'Actual baseline calibration required')
        by={r['content_group_id']:r for r in calibration_rows};require(len(by)==len(calibration_rows)==len(groups) and set(by)==set(groups),'Calibration matrix')
        for gid,r in by.items():
            require(r['record_id']==sha({k:v for k,v in r.items() if k!='record_id'}) and r['arm']=='stockoff' and r['phase']=='calibration','Baseline-only calibration identity')
            stock.raw_clock_check(r);elapsed=r['coordinator_finish_ns'][-1]-r['coordinator_start_ns'][0]
            p=plan['calibration_provenance'][gid]
            require(p['raw_record_sha256']==sha(r) and p['record_id']==r['record_id'] and p['calibration_elapsed_ns']==elapsed and r['calibration_capacity']['calibration_elapsed_ns']==elapsed,'Calibration raw provenance/capacity')
            require(p['source_sha256']==r['source_sha256']==plan['source_sha256'] and str(r['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration same source/allocation')
            for c in cases.values():
                if c['block_id']==gid:require(c['calendar']['stockoff_calibration_elapsed_ns']==elapsed,'Calendar baseline rate')
        require({k:v['original_bytes_sha256'] for k,v in calibration_files.items()}==plan['calibration_files'],'Exact calibration bytes')
    seen={};artifacts={};diagnostics={};input_identities={};counts=Counter();errors=[];orders={k:[] for k in cases}
    for r in rows:
        key=tuple(r.get(k) for k in ('content_group_id','arrival_type','rho','arm'))
        try:
            require(key in expected and key not in seen,'Unexpected/duplicate row');seen[key]=r;gid=key[0];bank=groups[gid]
            require(r['record_id']==sha({k:v for k,v in r.items() if k!='record_id'}),'Raw record hash')
            require(r['schema']=='resident_fix_trial_v1' and r['phase']==phase and r['status']=='complete' and r['evidence_eligible']==(phase=='final'),'Row phase/disposition')
            require(r['dataset']==bank['dataset'] and r['block_id']==gid and r['inference_role']==('sentinel_descriptive' if bank['dataset']=='mbpp' else 'main'),'Dataset/role identity')
            require(r['plan_sha256']==plan['content_sha256'] and r['source_sha256']==plan['source_sha256'] and r['old_closure_sha256']==OLD,'Frozen row identity')
            hw=plan['hardware'];require(str(r['allocation_id'])==str(hw['allocation_id']) and r['hostname'].split('.')[0]==hw['hostname'] and sorted(r['gpu_uuids'])==sorted(hw['gpu_uuids']),'Physical allocation')
            c=r['clock'];require(c['monotonic'] and not c['adjustable'] and 'CLOCK_MONOTONIC' in c['implementation'],'Shared clock')
            if phase=='calibration':require(r['calendar'] is None and r['arrival_offsets_ns']==[0]*4096 and r['horizon_ns']==1,'Saturated input')
            else:
                case=cases[key[:3]];cal=case['calendar'];orders[key[:3]].append(r['arm'])
                require(r['calendar']==cal and r['arrival_offsets_ns']==cal['arrival_offsets_ns'] and r['horizon_ns']==cal['horizon_ns'] and r['arm_order']==case['arm_order'],'Frozen arrival calendar/order')
            rt=r['runtime_metadata']
            require(rt['study_id']=='resident_fix_v1' and rt['selected_fix']==plan['selected_fix'] and rt['native_vllm_commit']==plan['vllm_commit']
                and rt['model_revision']==plan['checkpoint_revision'] and rt['world_size']==2 and rt['native_output_reduction'] is True,'Native execution identity')
            require(rt['tf32_allowed'] is False and rt['float32_matmul_precision']=='highest'
                and rt['precision_per_rank']==[dict(allow_tf32=False,matmul_precision='highest')]*2,'Both-rank dense FP32 precision')
            require(r['source_weights']['source_revision']==plan['checkpoint_revision'] and r['source_weights']['layer']==0,'Expert checkpoint identity')
            for label,count in (('preflight',4112),('postflight',16)):
                v=r[label];error=v['full_batch_max_abs_difference']
                require(v['correctness_pass'] is True and v['full_batch_inputs_checked']==count and v['atol']==.02 and v['rtol']==.03
                    and type(error) in (int,float) and math.isfinite(error) and error>=0,'Numerical scope/tolerance '+label)
            require(all(r['preflight'][k] is True for k in ('logical_ids_equal','combining_weights_equal','rank_routing_agreement','input_unchanged')),'Actual logical-router equality')
            stock.raw_clock_check(r);counts['raw_clock_records']+=1
            meta=r['input_meta'];identity=sha({k:meta[k] for k in ('capture_sha256','indices_sha256','gate_weight_sha256','source_prompt_ids','source_records','index_artifact_sha256')})
            if gid in input_identities:require(input_identities[gid]==identity,'Cross-case/arm authentic input identity')
            input_identities[gid]=identity
            if gid not in artifacts:
                artifacts[gid],diagnostics[gid]=check_assignments(inputs,gid,r['input_meta'],None if banks_dir is None else banks_dir/bank['capture_path'])
            require(r['input_meta']['capture_sha256']==bank['capture_sha256'],'Captured bank identity')
            for field in ('logical_routing_sha256','float32_combining_weights_sha256'):require(r['preflight'][field]==artifacts[gid][field],'Authenticated stock route output')
            if r['arm']!='stockoff':require(diagnostics[gid]['assignment_digest'] is not None and r['preflight']['physical_assignment_digest']==diagnostics[gid]['assignment_digest'],'Full exact assignment digest')
            if r['arm']=='fix':counts.update(static_state(r,plan['selected_fix']['selector']))
            else:
                counts.update(stock.native_check({**r,'arm':'ours' if r['arm']=='current' else 'stockoff'},artifacts[gid]))
                if r['arm']=='current':
                    for peer in r['per_rank_state']:
                        state=peer['end'];require(state['recorded_step_indices']==[i for i in range(4096) if i%64>=48],'Exact recorded current steps')
                        for decision in state['decisions']:
                            step=decision['step'];expected_physical=artifacts[gid]['greedy_physical_counts'][step-15:step+1].sum(0)
                            require(np.array_equal(decision['physical_window_totals_local'],expected_physical),'Exact greedy physical-slot observation window')
                        require(np.array_equal(state['terminal_physical_window_totals_local'],artifacts[gid]['greedy_physical_counts'][-16:].sum(0)),'Exact terminal greedy physical window')
        except (ValueError,KeyError,TypeError,IndexError,AssertionError) as exc:errors.append(dict(key=list(key),error=str(exc)))
    missing=expected-set(seen)
    for key,order in orders.items():
        if order!=cases[key]['arm_order']:errors.append(dict(key=list(key),error='Actual arm execution order'))
    return dict(schema='resident_fix_independent_record_audit_v1',valid=not errors and not missing,phase=phase,
        expected_records=len(expected),observed_records=len(rows),groups_with_actual_routes=len(artifacts),
        prepared_batches_reconstructed=sum(d['prepared_batches'] for d in diagnostics.values()),
        exact_greedy_assignment_batches=sum(d['physical_oracle_batches'] for d in diagnostics.values()),
        **dict(counts),errors=errors,missing=[list(k) for k in sorted(missing,key=str)],plan_sha256=plan['content_sha256'],
        source_sha256=plan['source_sha256'],old_closure_sha256=OLD,
        limitations=['Independent CPU implementation within same team; no production analysis/inference imports.',
            'Actual stored stock routes and every 4,112 prepared greedy assignment are reconstructed; GPU kernel arithmetic is not rerun.',
            'Full-batch preflight, 64 FP32 positions/group, 16 whole postflight batches/trial and resident-weight checks retain their recorded coverage limits.',
            'No performance selection or additional inference.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','records','inputs-dir','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--banks-dir',type=Path)
    p.add_argument('--repo-root',type=Path,default=HERE.parents[2])
    a=p.parse_args();rows,files=read_rows(a.records);cal,calfiles=read_rows(a.calibration) if a.calibration else (None,None)
    try:result=audit(json.loads(a.plan.read_text()),rows,a.inputs_dir,a.repo_root,cal,calfiles,a.banks_dir)
    except (ValueError,KeyError,OSError,TypeError) as exc:result=dict(schema='resident_fix_independent_record_audit_v1',valid=False,errors=[dict(error=str(exc))])
    result.update(record_files=files,auditor_sha256=sha(Path(__file__).read_bytes()),independent_primitives_sha256=sha((HERE/'stock_raw_primitives.py').read_bytes()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('errors','missing','record_files','limitations')}))
    if not result['valid']:raise SystemExit(1)
if __name__=='__main__':main()
