#!/usr/bin/env python3
"""Independent raw-clock, calendar, stock-route and native-state audit.

Uses only the standard library and NumPy, not production analyzer/runtime code.
The input directory is the runner's authenticated saved inputs, not HF route tapes.
"""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

ARMS=('stockoff','stockon32','ours')
DATASETS=('gsm8k','mbpp','wildchat')


def sha(value):
    if not isinstance(value,bytes):value=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def tensor_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(sha([a.dtype.str,list(a.shape)]).encode());h.update(a.tobytes());return h.hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def unsealed(value):return {k:v for k,v in value.items() if k!='content_sha256'}


def read_rows(path):
    path=Path(path);files=sorted((path/'trials').glob('*.jsonl*')) if path.is_dir() else [path]
    require(bool(files),'No records')
    rows=[];identities={}
    for file in files:
        stored=file.read_bytes();raw=gzip.decompress(stored) if file.suffix=='.gz' else stored
        identities[file.name.removesuffix('.gz')]=dict(stored_sha256=sha(stored),original_bytes_sha256=sha(raw))
        rows.extend(json.loads(x) for x in raw.decode().splitlines() if x.strip())
    return rows,identities


def calendar_check(case):
    c=case['calendar'];s=case['source_block'];n=4096
    require(c['content_sha256']==sha(unsealed(c)),'Calendar hash')
    require(s['jobs']==n and len(s['original_offsets_ns'])==n,'Source calendar length')
    require(c['source_block_id']==s['id'] and c['source_rows_sha256']==s['source_rows_sha256'],'Source calendar identity')
    factor=Fraction(c['stockoff_calibration_elapsed_ns'],1)/Fraction(str(c['rho']))/s['original_horizon_ns']
    require(c['arrival_offsets_ns']==[round(int(x)*factor) for x in s['original_offsets_ns']],'Calendar scaled offsets')
    require(c['horizon_ns']==round(s['original_horizon_ns']*factor),'Calendar scaled horizon')
    require(c['arrival_type']==s['trace'] and c['arrival_date']==s['source_date_utc'],'Calendar service/date')
    require(c['n_jobs']==n and c['rho'] in (.5,.85),'Calendar scope')


def raw_clock_check(r):
    require(r.get('per_job_completion_scope')=='current_inference_stream_after_native_step','Per-job inference-stream completion contract')
    n=4096;e=r['epoch_monotonic_ns'];h=r['horizon_ns'];end=r['observation_end_ns']
    names=('arrival_offsets_ns','coordinator_admit_ns','coordinator_start_ns','coordinator_finish_ns')
    raw=[r[name] for name in names]
    require(all(len(x)==n and all(type(v) is int for v in x) for x in raw),'Raw integer timestamps/count')
    off,admit,start,finish=(np.asarray(x,dtype=np.int64) for x in raw)
    require(off[0]==0 and np.all(np.diff(off)>=0) and off[-1]<h,'Calendar order/horizon')
    arrivals=e+off
    require(np.all(admit>=arrivals) and np.all(start>=admit) and np.all(finish>=start),'FCFS timestamp order')
    require(np.all(admit[1:]>=finish[:-1]),'One-job-in-flight/FCFS admission')
    require(end>=max(e+h,int(finish[-1])),'Full offered horizon and completion retained')
    rs=np.asarray(r['per_rank_start_ns']);rf=np.asarray(r['per_rank_gpu_complete_ns'])
    require(rs.shape==rf.shape==(2,n) and np.issubdtype(rs.dtype,np.integer) and np.issubdtype(rf.dtype,np.integer),'Rank clock arrays')
    require(np.all(rs>=start) and np.all(rf>=rs) and np.all(rf<=finish),'Rank GPU envelope')
    q=r['controller_quiescent_ns'];tail=r['controller_quiescence_start_ns']
    require(finish[-1]<=tail<=q<=end,'Completion/quiescence ordering')
    expected={'completed_elapsed_ns':end-e,'job_completed_elapsed_ns':int(finish[-1])-e,
        'controller_tail_elapsed_ns':q-int(finish[-1]),'quiescence_call_elapsed_ns':q-tail,
        'drain_including_controller_ns':max(0,q-e-h)}
    require(all(r[k]==v for k,v in expected.items()),'Completion/tail accounting')
    require(r['per_rank_controller_tail_start_ns'][0]==tail and r['per_rank_controller_quiescent_ns'][0]==q,'Rank0 tail timestamps')
    for rank in (0,1):require(rf[rank,-1]<=r['per_rank_controller_tail_start_ns'][rank]<=r['per_rank_controller_quiescent_ns'][rank],'Rank tail order')
    vectors={'response':finish-arrivals,'queue_wait':start-arrivals,'service':finish-start,
        'admission_delay':admit-arrivals,'dispatch_delay':start-admit}
    m=r['queue_metrics'];require(m['complete'] and m['completed_jobs']==n and m['unfinished_jobs']==0,'All jobs accounted')
    flat={}
    for name,v in vectors.items():
        require(m['vectors_ns'][name]==v.tolist(),'Stored vector '+name)
        summary={'mean':float(np.mean(v)),'p50':float(np.quantile(v,.5,method='linear')),
            'p95':float(np.quantile(v,.95,method='linear')),'p99':float(np.quantile(v,.99,method='linear')),'max':int(v.max())}
        require(m['summary_ns'][name]==summary,'Stored summary '+name)
        for key,value in summary.items():flat[('queue' if name=='queue_wait' else name)+'_'+key+'_ns']=value
    require(m['queue_occupancy']['integral_job_ns']==int(vectors['queue_wait'].sum()),'Queue occupancy integral')
    require(m['system_occupancy']['integral_job_ns']==int(vectors['response'].sum()),'System occupancy integral')
    by_h=int(np.count_nonzero(finish<=e+h));drain=max(0,int(finish[-1])-e-h)
    require(m['completed_by_horizon']==by_h and m['backlog_at_horizon']==n-by_h and m['drain_ns']==drain,'Horizon completion/backlog/drain')
    flat.update(expected,drain_ns=drain,throughput_jobs_per_second=n*1e9/(end-e))
    require(m['completed_per_evaluation_second']==flat['throughput_jobs_per_second'],'Completed throughput denominator')
    return flat


def input_artifacts(root,gid,meta):
    path=root/(gid+'.npz');routes=root/(gid+'-stock-routes.npz')
    require(sha(path.read_bytes())==meta['index_artifact_sha256'],'Index artifact bytes')
    with np.load(path,allow_pickle=False) as z:
        indices=z['indices'];stored=json.loads(str(z['metadata']))
    require(indices.shape==(4112,128) and indices.dtype==np.dtype('int64'),'Input job matrix dtype/shape')
    require(tensor_sha(indices)==meta['indices_sha256']==stored['indices_sha256'],'Actual input matrix digest')
    require(np.all(indices>=0) and np.all(indices<meta['source_records']),'Input record bounds')
    canonical=np.sort(indices,axis=1)
    require(np.all(np.diff(canonical,axis=1)>0) and len({x.tobytes() for x in canonical})==4112,'Distinct records/job and all job combinations')
    with np.load(routes,allow_pickle=False) as z:
        ids=z['logical_ids'];weights=z['combining_weights']
        require(ids.shape==weights.shape==(4112,128,8) and ids.dtype==np.dtype('int32') and weights.dtype==np.dtype('float32'),'Authentic stock router array geometry/dtype')
        ih,wh=tensor_sha(ids),tensor_sha(weights)
        require(ih==str(z['logical_routing_sha256']) and wh==str(z['float32_combining_weights_sha256']),'Authentic stock route bytes digest')
    require(np.all((ids>=0)&(ids<128)) and np.all(np.diff(np.sort(ids,axis=-1),axis=-1)>0),'Logical top-k IDs unique and legal')
    require(np.all(np.isfinite(weights)) and np.all(weights>=0) and np.allclose(weights.sum(-1),1,atol=2e-6,rtol=0),'Stock combining weights finite/normalized')
    # The selected stock router, not historical HF IDs, supplies this oracle.
    per_step=np.stack([np.bincount(x.ravel(),minlength=128) for x in ids[16:]])
    return dict(logical_routing_sha256=ih,float32_combining_weights_sha256=wh,per_step=per_step,logical_ids=ids[16:])


def native_sampled_physical_counts(logical_ids,selected,initial,events):
    """Reproduce native token-index mapping under each job's committed map.

    A commit in state.step follows that job's forward. Historical physical ring
    entries stay physical when the inverse map changes; they are not relabeled.
    """
    segments={}
    for step in selected:
        mapping=initial
        for event in events:
            if event['commit_step']<step:mapping=event['committed_map']
        segments.setdefault(tuple(mapping),[]).append(step)
    total=np.zeros(160,np.int64)
    for key,steps in segments.items():
        mapping=np.asarray(key,np.int64);copies=np.bincount(mapping,minlength=128)
        inverse=np.zeros((128,int(copies.max())),np.int64)
        for expert in range(128):inverse[expert,:copies[expert]]=np.flatnonzero(mapping==expert)
        ids=logical_ids[steps]
        hashed=(np.arange(128,dtype=np.int64)*2654435769)&0xffffffff
        physical=inverse[ids,hashed[None,:,None]%copies[ids]]
        total+=np.bincount(physical.ravel(),minlength=160)
    return total


def native_check(row,artifact):
    peers=row['per_rank_state'];arm=row['arm'];require(len(peers)==2,'Both rank states')
    physical=128 if arm=='stockoff' else 160;initial=list(range(128))+([] if arm=='stockoff' else list(range(32)))
    pre=row['preflight'];post=row['postflight']
    for field in ('logical_routing_sha256','float32_combining_weights_sha256'):require(pre[field]==artifact[field],'Preflight actual stock route identity')
    require(pre['correctness_pass'] and pre['full_batch_inputs_checked']==4112 and pre['input_unchanged'],'Whole-job numerical preflight')
    require(pre['fp32_job_indices']==[16,765,3765,4111] and pre['fp32_tokens_per_job']==16,'FP32 coverage')
    require(post['correctness_pass'] and post['raw_indices']==np.linspace(16,4111,16,dtype=np.int64).tolist(),'Postflight sample coverage')
    windows=0;decisions=0
    for rank,p in enumerate(peers):
        start,end,audit=p['start'],p['end'],p['audit']
        for snap in (start,end,audit):
            require(snap['source_map']==initial and len(snap['final_map'])==physical and set(snap['final_map'])==set(range(128)),'Legal source/final map')
            require(snap['physical_slots']==physical and snap['local_slots']==physical//2,'Resource geometry')
            require(snap['logical_router_computation_included'] and snap['native_output_reduction'],'Stock operation/reduction included')
            require(not snap['pending_result'] and not snap['rebalanced'] and snap['completion_disposition']=='normal_commit_no_discard','Normal complete state/no discarded work')
            if arm!='ours':require(snap['stock_policy_unmodified'] and snap['stock_router_unmodified'],'Unmodified stock policy/router')
            if arm!='stockon32':require(snap['final_map']==initial,'Fixed map changed')
        require(start['controller_step_calls']==0 and start['async_events']==[] and start['mapper_calls']==0,'Fresh measured state')
        require(end['controller_step_calls']==(0 if arm=='stockoff' else 4096),'Controller invocation count')
        require(end['mapper_calls']==(4096 if arm=='ours' else 0),'Charged mapper invocations')
        require(audit['resident_state_verified'] and audit['rank_map_agreement'] and audit['normal_completion_verified'],'Physical expert weight/normal state verification')
        require(audit['final_map']==end['final_map'],'State changed during excluded audit')
        if arm=='stockon32':
            require(end['is_async'] and end['async_worker_alive'] and end['rebalance_interval']==3000 and end['window_steps']==1000,'Native default controller settings')
            events=end['async_events'];recorded=end['recorded_step_indices'];deferred=[];due=749;previous=initial
            require(recorded==sorted(set(recorded)) and all(type(x) is int and 0<=x<4096 for x in recorded),'Actual native sampled steps')
            require(bool(events) and events[0]['launch_step']==749,'Native startup opportunity')
            for event in events:
                step=event['launch_step'];selected=[x for x in recorded if x<=step][-1000:]
                require(due<=step<=event['commit_step']<=4095,'Normal async launch/commit sequence')
                require(event['launch_counter_before']==2999+step-due,'Native counter at delayed launch')
                deferred.extend(range(due,step));due=step+3000
                require(event['sampled_step_indices']==selected and event['window_valid_steps']==len(selected),'Exact sampled native indices')
                require(event['source_map']==previous,'Native source/committed-map chain')
                observed=np.asarray(event['current_counts']).reshape(-1)
                physical_counts=native_sampled_physical_counts(artifact['logical_ids'],selected,initial,events)
                expected=np.bincount(event['source_map'],weights=physical_counts,minlength=128).astype(np.int64)*2
                require(np.array_equal(observed,expected),'Exact native logical observation window')
                require(len(event['source_map'])==len(event['committed_map'])==160 and set(event['committed_map'])==set(range(128)),'Native committed map coverage')
                previous=event['committed_map']
                windows+=1
            deferred.extend(range(due,4096))
            require(end['deferred_boundary_steps']==deferred,'Pending-work native deferral evidence')
            require(end['native_counter']==4095-events[-1]['launch_step'] and end['native_valid_steps']==len(recorded),'Native final observation/counter state')
            terminal=native_sampled_physical_counts(artifact['logical_ids'],recorded[-1000:],initial,events)
            require(np.array_equal(end['terminal_physical_window_totals_local'],terminal),'Native terminal physical window')
            require(events[-1]['committed_map']==end['final_map'],'Last committed map/final map')
            require(sum(event['sent_bytes_local'] for event in events)==end['migration_sent_bytes_local'],'Actual sender payload accounting')
        elif arm=='ours':
            ds=end['decisions'];require([d['step'] for d in ds]==list(range(63,4096,64)),'All64 charged controller boundaries')
            require(end['terminal_proposals_suppressed']==0,'No suppressed final proposal')
            for d in ds:
                step=d['step'];expected=artifact['per_step'][step-15:step+1].sum(0)*2
                require(np.array_equal(d['current_counts'],expected),'Ours exact logical count window')
                require(d['allowed'] is False and d['current_map']==d['committed_map']==initial,'Ours forced-SKIP map state')
                counts=np.asarray(d['physical_window_totals_local'])
                require(counts.shape==(160,) and np.all(counts>=0) and int(counts.sum())==16*128*8,'Local physical observation conservation')
                collapsed=np.bincount(initial,weights=counts,minlength=128).astype(np.int64)*2
                require(np.array_equal(collapsed,expected),'Physical slot window collapses to logical routes')
                windows+=1;decisions+=1
            guard=end['guard']
            require(not guard['pending_pair'] and not guard['diagnostic_timing'] and guard['mode']=='fast','Normal shortcut guard')
            for k in ('movement_calls','commit_calls','agreement_calls','local_unchanged_calls','unanimously_unchanged_calls','movement_skips','commit_skips'):require(guard[k]==64,'Exact64 shortcut '+k)
            for k in ('original_movement_calls','original_commit_calls','fallback_calls'):require(guard[k]==0,'Unexpected shortcut fallback '+k)
            require(end['migration_sent_bytes_local']==0,'Fixed arm migrated weights')
        else:require(end['async_events']==[] and end['decisions']==[] and end['migration_sent_bytes_local']==0,'Disabled EPLB has control work')
    for scope in ('start','end','audit'):
        for k in ('source_map','final_map','gate_weight_sha256','native_counter'):require(peers[0][scope][k]==peers[1][scope][k],'Rank state agreement '+k)
    return dict(rank_windows_reconstructed=windows,rank_forced_skip_decisions=decisions)


def audit(plan,rows,inputs,*,repo_root,calibration_rows=None,calibration_files=None):
    require(plan['content_sha256']==sha(unsealed(plan)),'Plan seal')
    require(plan['source_sha256']==sha(plan['source_files']),'Execution source manifest seal')
    for path,expected in plan['source_files'].items():require(sha((repo_root/path).read_bytes())==expected,'Changed frozen source '+path)
    require(plan['arms']==list(ARMS) and plan['jobs']==4096 and plan['batch_tokens']==128,'Study scope')
    phase=plan['phase'];groups={b['block_id']:b for b in plan['banks']};cases={}
    for case in plan['cases']:
        calendar_check(case);cal=case['calendar'];key=(case['block_id'],cal['arrival_type'],cal['rho'])
        require(key not in cases,'Duplicate case');cases[key]=case
    expected={(g,None,None,'stockoff') for g in groups} if phase=='calibration' else {(*k,a) for k in cases for a in ARMS}
    require(len(expected)==(len(groups) if phase=='calibration' else len(groups)*12),'Expected matrix')
    if phase!='calibration':
        require(calibration_rows is not None,'Treatment audit requires actual calibration records')
        calibration={r['content_group_id']:r for r in calibration_rows}
        require(set(calibration)==set(groups) and len(calibration_rows)==len(groups),'Calibration matrix')
        for gid,r in calibration.items():
            require(r['record_id']==sha({k:v for k,v in r.items() if k!='record_id'}),'Calibration row hash')
            require(r['phase']=='calibration' and r['arm']=='stockoff','Independent baseline-only calibration')
            raw_clock_check(r)
            elapsed=r['coordinator_finish_ns'][-1]-r['coordinator_start_ns'][0]
            require(r['calibration_capacity']['calibration_elapsed_ns']==elapsed,'Saturated capacity denominator')
            provenance=plan['calibration_provenance'][gid]
            require(provenance['raw_record_sha256']==sha(r) and provenance['record_id']==r['record_id'] and provenance['calibration_elapsed_ns']==elapsed,'Frozen calibration provenance')
            require(str(provenance['allocation_id'])==str(r['allocation_id'])==str(plan['hardware']['allocation_id']),'Calibration physical allocation')
            for case in cases.values():
                if case['block_id']==gid:require(case['calendar']['stockoff_calibration_elapsed_ns']==elapsed,'Frozen rate calibration')
        require({k:v['original_bytes_sha256'] for k,v in calibration_files.items()}==plan['calibration_files'],'Calibration original bytes')
    seen={};actual_orders={k:[] for k in cases};errors=[];flat={};artifacts={};counts=Counter()
    for row in rows:
        key=(row.get('content_group_id'),row.get('arrival_type'),row.get('rho'),row.get('arm'))
        try:
            require(key in expected and key not in seen,'Unexpected/duplicate row');seen[key]=row
            gid=key[0];bank=groups[gid];hw=plan['hardware']
            require(row['record_id']==sha({k:v for k,v in row.items() if k!='record_id'}),'Whole raw row hash')
            require(row['schema']=='stock_comparison_trial_v1' and row['phase']==phase and row['status']=='complete','Record schema/phase/status')
            require(row['evidence_eligible']==(phase=='final'),'Evidence phase')
            require(row['source_sha256']==plan['source_sha256'] and row['plan_sha256']==plan['content_sha256'],'Row frozen source/plan')
            require(row['dataset']==bank['dataset'] and row['block_id']==gid,'Content identity')
            require(row['hostname'].split('.')[0]==hw['hostname'] and str(row['allocation_id'])==str(hw['allocation_id']) and sorted(row['gpu_uuids'])==sorted(hw['gpu_uuids']),'Physical hardware')
            c=row['clock'];require(c['monotonic'] and not c['adjustable'] and 'CLOCK_MONOTONIC' in c['implementation'],'Shared monotonic clock')
            if phase=='calibration':require(row['calendar'] is None and row['arrival_offsets_ns']==[0]*4096 and row['horizon_ns']==1,'Saturated calendar')
            else:
                case=cases[key[:3]];cal=case['calendar'];actual_orders[key[:3]].append(key[-1])
                require(row['calendar']==cal and row['arrival_offsets_ns']==cal['arrival_offsets_ns'] and row['horizon_ns']==cal['horizon_ns'] and row['arm_order']==case['arm_order'],'Frozen calendar/order')
            flat[key]=raw_clock_check(row)
            if gid not in artifacts:artifacts[gid]=input_artifacts(inputs,gid,row['input_meta'])
            require(row['input_meta']['capture_sha256']==bank['capture_sha256'],'Captured hidden bank identity')
            counts.update(native_check(row,artifacts[gid]))
        except (ValueError,KeyError,TypeError,AssertionError,IndexError) as exc:errors.append(dict(key=list(key),error=str(exc)))
    missing=expected-set(seen)
    for key,order in actual_orders.items():
        if order!=cases[key]['arm_order']:errors.append(dict(key=list(key),error='Actual arm execution order'))
    return dict(schema='independent_stock_record_audit_v1',valid=not errors and not missing,expected_records=len(expected),observed_records=len(rows),
        raw_clock_records_checked=len(flat),groups_with_authentic_stock_routes=len(artifacts),**dict(counts),
        errors=errors,missing=[list(x) for x in sorted(missing,key=str)],plan_sha256=plan['content_sha256'],
        limitations=['Authenticates saved native route tensors and reconstructs counts; does not independently rerun CUDA/router kernels.',
            'Numerical correctness is all4112 batches against stockoff, sampled64tokens against FP32, and16 whole postflight batches; full-slot weight checks are native assertions.',
            'Async controller tail is post-output extent, not exclusive controller cost; per-rank host-return timestamps may straddle rank0 observation end.',
            'No new statistical inference or performance selection.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','inputs-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();plan=json.loads(args.plan.read_text());rows,files=read_rows(args.records)
    calrows,calfiles=read_rows(args.calibration) if args.calibration else (None,None)
    try:result=audit(plan,rows,args.inputs_dir,repo_root=args.repo_root,calibration_rows=calrows,calibration_files=calfiles)
    except (ValueError,KeyError,OSError,TypeError) as exc:result=dict(schema='independent_stock_record_audit_v1',valid=False,errors=[dict(error=str(exc))])
    result.update(record_files=files,auditor_sha256=sha(Path(__file__).read_bytes()))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('errors','missing','record_files','limitations')}))
    if not result['valid']:raise SystemExit(1)
if __name__=='__main__':main()
