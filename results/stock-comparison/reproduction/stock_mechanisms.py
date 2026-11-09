#!/usr/bin/env python3
"""Descriptive stock-comparator mechanisms; outside the frozen execution source.

No new inference, policy selection, GPU runs, or runtime predictions. Counts are
structural workload measures; event neighborhoods are temporal associations.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ARMS=('stockoff','stockon32','ours')
BOUNDARIES=np.arange(63,4096,64)


def digest(value):
    if not isinstance(value,bytes):value=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def tensor_digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(digest([a.dtype.str,list(a.shape)]).encode());h.update(a.tobytes());return h.hexdigest()


def read_rows(path):
    path=Path(path);files=sorted((path/'trials').glob('*.jsonl*')) if path.is_dir() else [path]
    rows=[];sources=[]
    for f in files:
        raw=f.read_bytes();raw=gzip.decompress(raw) if f.suffix=='.gz' else raw
        rows.extend(json.loads(s) for s in raw.decode().splitlines() if s.strip());sources.append(dict(path=str(f),uncompressed_sha256=digest(raw)))
    if not rows:raise ValueError('No records')
    return rows,sources


def scalar_summary(values):
    a=np.asarray(values,dtype=float)
    return dict(n=int(a.size),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),max=float(a.max()),sum=float(a.sum())) if a.size else None


def geometric(values):
    values=list(values)
    return math.exp(statistics.mean(math.log(x) for x in values)) if values and all(x>0 for x in values) else None


def native_counts(ids,mapping):
    """Exact pinned token-index mapping, including arbitrary legal native maps."""
    mapping=np.asarray(mapping,np.int64);slots=len(mapping)
    replicas=np.bincount(mapping,minlength=128)
    if np.any(replicas==0):raise ValueError('Lost logical expert')
    inverse=np.zeros((128,int(replicas.max())),np.int64)
    for e in range(128):inverse[e,:replicas[e]]=np.flatnonzero(mapping==e)
    hashed=(np.arange(128,dtype=np.int64)*2654435769)&0xffffffff
    physical=inverse[ids,hashed[None,:,None]%replicas[ids]]
    offsets=np.arange(len(ids),dtype=np.int64)[:,None,None]*slots
    return np.bincount((physical+offsets).ravel(),minlength=len(ids)*slots).reshape(len(ids),slots)


def native_trajectory_counts(ids,state):
    events=state['async_events'];initial=state['source_map'];groups={}
    for step in range(len(ids)):
        mapping=initial
        for event in events:
            if event['commit_step']<step:mapping=event['committed_map']
        groups.setdefault(tuple(mapping),[]).append(step)
    output=np.zeros((len(ids),len(initial)),np.int64)
    for mapping,steps in groups.items():output[steps]=native_counts(ids[steps],mapping)
    return output


def work_counts(counts,tile_m=64):
    slots=counts.shape[1]
    if slots not in (128,160) or not np.all(counts.sum(1)==1024):raise ValueError('Physical route conservation/geometry')
    rank=counts.reshape(len(counts),2,slots//2)
    active=(rank>0).sum(2);tokens=rank.sum(2);tiles=((rank+tile_m-1)//tile_m).sum(2)
    return dict(active_copies_rank0=active[:,0],active_copies_rank1=active[:,1],
        active_copies_sum=active.sum(1),active_copies_maxrank=active.max(1),
        tokens_rank0=tokens[:,0],tokens_rank1=tokens[:,1],tokens_maxrank=tokens.max(1),
        nonempty_expert_tiles_M64_rank0=tiles[:,0],nonempty_expert_tiles_M64_rank1=tiles[:,1],
        nonempty_expert_tiles_M64_sum=tiles.sum(1),nonempty_expert_tiles_M64_maxrank=tiles.max(1),
        padded_expert_rows_M64_sum=tiles.sum(1)*tile_m,padded_expert_rows_M64_maxrank=tiles.max(1)*tile_m)


def load_structural(inputs,gid,rows):
    with np.load(inputs/(gid+'-stock-routes.npz'),allow_pickle=False) as z:
        ids=z['logical_ids'];weights=z['combining_weights']
        ih,wh=tensor_digest(ids),tensor_digest(weights)
        if ih!=str(z['logical_routing_sha256']) or wh!=str(z['float32_combining_weights_sha256']):raise ValueError('Stock route artifact digest')
    if ids.shape!=(4112,128,8):raise ValueError('Unexpected route geometry')
    for r in rows:
        if r['preflight']['logical_routing_sha256']!=ih or r['preflight']['float32_combining_weights_sha256']!=wh:raise ValueError('Cross-arm route identity changed')
    diagnostics=json.loads((inputs/(gid+'-resident-diagnostics.json')).read_text())
    if len(diagnostics)!=4112:raise ValueError('Incomplete resident diagnostics')
    identity=digest([d['physical_sha256'] for d in diagnostics])
    for r in rows:
        if r['arm']=='ours' and r['preflight']['physical_assignment_digest']!=identity:raise ValueError('Resident diagnostic identity')
    counts=np.asarray([d['physical_counts'] for d in diagnostics],np.int64)
    mapping=list(range(128))+list(range(32))
    for raw,(d,c) in enumerate(zip(diagnostics,counts)):
        collapsed=np.bincount(mapping,weights=c,minlength=128).astype(np.int64)
        if not np.array_equal(collapsed,np.bincount(ids[raw].ravel(),minlength=128)):raise ValueError('Resident logical route conservation')
        if not np.array_equal(c.reshape(2,80).sum(1),d['per_rank_tokens']):raise ValueError('Resident owner token counts')
    measured=ids[16:]
    return measured,{'stockoff':native_counts(measured,list(range(128))),'ours':counts[16:]}


def temporal_record(r):
    if r.get('per_job_completion_scope')!='current_inference_stream_after_native_step':raise ValueError('Wrong completion scope')
    if r['record_id']!=digest({k:v for k,v in r.items() if k!='record_id'}):raise ValueError('Raw record hash')
    a=r['epoch_monotonic_ns']+np.asarray(r['arrival_offsets_ns'],np.int64)
    s=np.asarray(r['coordinator_start_ns'],np.int64);f=np.asarray(r['coordinator_finish_ns'],np.int64)
    admit=np.asarray(r['coordinator_admit_ns'],np.int64)
    if len(s)!=4096 or np.any(s<a) or np.any(f<s) or np.any(s[1:]<f[:-1]):raise ValueError('Invalid FCFS clocks')
    service=f-s;queue=s-a;response=f-a
    ordinary=np.ones(4096,bool);ordinary[BOUNDARIES]=False
    labels={name:np.zeros(4096,bool) for name in ('launch_exact','commit_in_job_exact','launch_pm1','commit_in_job_pm1')}
    state=r['per_rank_state'][0]['end'];events=state['async_events']
    for event in events:
        for label,step in [('launch',event['launch_step'])]+([] if event['commit_in_completion_tail'] else [('commit_in_job',event['commit_step'])]):
            labels[label+'_exact'][step]=True
            labels[label+'_pm1'][max(0,step-1):min(4096,step+2)]=True
    union=labels['launch_pm1']|labels['commit_in_job_pm1']
    labels['outside_launch_commit_pm1']=~union
    record={k:r[k] for k in ('content_group_id','dataset','arrival_type','arrival_date','rho','arm','record_id')}
    record.update(response_ns=scalar_summary(response),queue_ns=scalar_summary(queue),service_ns=scalar_summary(service),
        admission_delay_ns=scalar_summary(admit-a),dispatch_delay_ns=scalar_summary(s-admit),
        fixed64_boundaries_service_ns=scalar_summary(service[BOUNDARIES]),ordinary_service_ns=scalar_summary(service[ordinary]),
        native_event_neighborhoods={name:scalar_summary(service[mask]) for name,mask in labels.items()},
        native_events=[{k:e[k] for k in ('launch_step','commit_step','commit_in_completion_tail','sent_bytes_local')} for e in events],
        boundary_service_share=float(service[BOUNDARIES].sum()/service.sum()),
        completed_elapsed_ns=r['completed_elapsed_ns'],job_completed_elapsed_ns=r['job_completed_elapsed_ns'],
        controller_tail_elapsed_ns=r['controller_tail_elapsed_ns'],quiescence_call_elapsed_ns=r['quiescence_call_elapsed_ns'],
        drain_ns=r['queue_metrics']['drain_ns'],drain_including_controller_ns=r['drain_including_controller_ns'],
        horizon_ns=r['horizon_ns'],throughput_jobs_per_second=4096*1e9/r['completed_elapsed_ns'],
        backlog_at_horizon=r['queue_metrics']['backlog_at_horizon'],completed_by_horizon=r['queue_metrics']['completed_by_horizon'],
        resolved_kernel_config=state['resolved_kernel_config'])
    return record,service


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(rows,inputs):
    groups=defaultdict(list);index={};trials=[];service_vectors={}
    for r in rows:
        if r['phase'] not in ('pilot','final') or r['status']!='complete':raise ValueError('Completed treatment rows required')
        key=(r['content_group_id'],r['arrival_type'],r['rho'],r['arm'])
        if key in index:raise ValueError('Duplicate trial')
        index[key]=r;groups[key[0]].append(r)
        t,service=temporal_record(r);trials.append(t);service_vectors[key]=service
    if len({r['phase'] for r in rows})!=1:raise ValueError('Mixed experiment phases')
    if len(index)!=len(groups)*12:raise ValueError('Complete3arm×2type×2load matrix required')
    for gid in groups:
        if {(r['arrival_type'],r['rho'],r['arm']) for r in groups[gid]}!=set(__import__('itertools').product(('code','conversation'),(.5,.85),ARMS)):raise ValueError('Missing case')
    ti={(t['content_group_id'],t['arrival_type'],t['rho'],t['arm']):t for t in trials}
    structural=[];paired=[]
    for gid,members in sorted(groups.items()):
        ids,fixed=load_structural(inputs,gid,members)
        work={a:work_counts(c) for a,c in fixed.items()}
        comparison={name:dict(stockoff=scalar_summary(work['stockoff'][name]),ours=scalar_summary(work['ours'][name]),
            ratio_of_mean=float(np.mean(work['ours'][name])/np.mean(work['stockoff'][name])),
            steps_equal=int(np.count_nonzero(work['ours'][name]==work['stockoff'][name]))) for name in work['ours']}
        structural.append(dict(content_group_id=gid,dataset=members[0]['dataset'],measured_jobs=4096,
            fixedmap_comparison=comparison,
            total_active_copies_equal_all_jobs=bool(np.array_equal(work['stockoff']['active_copies_sum'],work['ours']['active_copies_sum'])),
            total_M64_tiles_equal_all_jobs=bool(np.array_equal(work['stockoff']['nonempty_expert_tiles_M64_sum'],work['ours']['nonempty_expert_tiles_M64_sum']))))
        for r in members:
            if r['arm']=='stockon32':
                c=native_trajectory_counts(ids,r['per_rank_state'][0]['end'])
                ti[(gid,r['arrival_type'],r['rho'],r['arm'])]['native_dynamic_work_counts']={k:scalar_summary(v) for k,v in work_counts(c).items()}
        for typ,rho in __import__('itertools').product(('code','conversation'),(.5,.85)):
            for control in ('stockoff','stockon32'):
                a,b=ti[gid,typ,rho,control],ti[gid,typ,rho,'ours']
                sa,sb=service_vectors[gid,typ,rho,control],service_vectors[gid,typ,rho,'ours']
                ordinary=np.ones(4096,bool);ordinary[BOUNDARIES]=False
                paired.append(dict(content_group_id=gid,dataset=a['dataset'],arrival_type=typ,rho=rho,contrast='ours_over_'+control,
                    response_mean_ratio=b['response_ns']['mean']/a['response_ns']['mean'],
                    response_p95_ratio=b['response_ns']['p95']/a['response_ns']['p95'],
                    queue_mean_ratio=b['queue_ns']['mean']/a['queue_ns']['mean'] if a['queue_ns']['mean'] else None,
                    service_mean_ratio=b['service_ns']['mean']/a['service_ns']['mean'],
                    completed_elapsed_ratio=b['completed_elapsed_ns']/a['completed_elapsed_ns'],
                    throughput_ratio=b['throughput_jobs_per_second']/a['throughput_jobs_per_second'],
                    service_total_difference_ns=int((sb-sa).sum()),
                    service_difference_at_ours64_positions_ns=int((sb[BOUNDARIES]-sa[BOUNDARIES]).sum()),
                    service_difference_at_other_positions_ns=int((sb[ordinary]-sa[ordinary]).sum()),
                    ours_boundary_service_share=b['boundary_service_share']))
    return dict(schema='stock_descriptive_mechanisms_v1',phase=rows[0]['phase'],trials=trials,paired_cases=paired,structural=structural,
        definitions=dict(boundaries='Measured zero-based63,127,...,4095;64 positions. Ours plans at all64. Other arms are compared at identical job positions.',
            neighborhoods='Native launch or in-job commit at step c and c±1. A completion-tail commit is excluded from per-job commit neighborhoods. Neighborhood categories may overlap; do not sum them.',
            service='Coordinator finish−start, including native step, inference-stream completion and CPU dispatch/ACK; not exclusive kernel/selector time.',
            M64='Sum over resident physical experts of ceil(route_count/64), plus padded rows=64×tiles. Structural nonempty expert tiles, not observed CUDA launch/block count or runtime.',
            decomposition='Paired total service difference equals differences at the64 fixed positions plus all other positions. Arithmetic association only, not a causal ablation.'),
        limitations=['No selector-only or controller-only timing ablation; event timing cannot identify exclusive component cost.',
            'Work-count and max-rank changes do not imply runtime speedup; kernel execution, overlap, routing and coordination remain measured only in raw service.',
            'Stockoff128slots and ours160slots have different permitted residency; both use one physical home per active logical expert.',
            'Counts use the same constructed hidden jobs; tokens/jobs are not independent statistical observations.',
            'No new confidence intervals, p-values, margins, selection or tuning; final confirmatory inference remains the separately frozen analysis.'])


def markdown_report(result):
    lines=['# Descriptive stock-comparator mechanisms','',f"Phase: {result['phase']}. {len(result['trials'])} complete treatment trials. Descriptive only; no additional inference.",'',
        'Service includes native control, routing, inference-stream completion and common coordination. Boundary and launch/commit patterns are temporal associations. Structural counts are not runtime predictions.','',
        '| Arm | Mean response (ms) | Mean service (ms) | Mean queue (ms) | Mean completed extent (s) | Mean output/controller drain (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        ts=[t for t in result['trials'] if t['arm']==arm]
        avg=lambda fn:statistics.mean(fn(t) for t in ts)
        lines.append(f"| {arm} | {avg(lambda t:t['response_ns']['mean'])/1e6:.4f} | {avg(lambda t:t['service_ns']['mean'])/1e6:.4f} | {avg(lambda t:t['queue_ns']['mean'])/1e6:.4f} | {avg(lambda t:t['completed_elapsed_ns'])/1e9:.4f} | {avg(lambda t:t['drain_ns'])/1e6:.4f} / {avg(lambda t:t['drain_including_controller_ns'])/1e6:.4f} |")
    lines+=['','Means above average complete trial outcomes equally; they are not the frozen paired geometric endpoint.','',
        '| Arm | Mean service at64 fixed positions (ms) | At other4032 positions (ms) | Observed kernel M |','|---|---:|---:|---|']
    for arm in ARMS:
        ts=[t for t in result['trials'] if t['arm']==arm]
        observed_m=sorted({t['resolved_kernel_config'].get('BLOCK_SIZE_M') for t in ts},key=str)
        lines.append(f"| {arm} | {statistics.mean(t['fixed64_boundaries_service_ns']['mean'] for t in ts)/1e6:.4f} | {statistics.mean(t['ordinary_service_ns']['mean'] for t in ts)/1e6:.4f} | {observed_m} |")
    lines+=['','For stockoff/stockon32 these are the same job positions, not asserted native control boundaries.','',
        '| Stockon32 temporal neighborhood | Mean per-trial service (ms) | Total sampled jobs |','|---|---:|---:|']
    for label in ('launch_exact','commit_in_job_exact','outside_launch_commit_pm1'):
        parts=[t['native_event_neighborhoods'][label] for t in result['trials'] if t['arm']=='stockon32' and t['native_event_neighborhoods'][label] is not None]
        if parts:lines.append(f"| {label} | {statistics.mean(x['mean'] for x in parts)/1e6:.4f} | {sum(x['n'] for x in parts)} |")
    lines+=['','Event labels may overlap; these associations do not isolate the controller or migration cost.','',
        '| Contrast | Geometric mean response ratio | Geometric service ratio | Geometric completed-extent ratio | Mean service difference at64 positions (ms/trial) | At other4032 positions (ms/trial) |','|---|---:|---:|---:|---:|---:|']
    for control in ('stockoff','stockon32'):
        ps=[p for p in result['paired_cases'] if p['contrast']=='ours_over_'+control]
        lines.append(f"| ours/{control} | {geometric(p['response_mean_ratio'] for p in ps):.5f} | {geometric(p['service_mean_ratio'] for p in ps):.5f} | {geometric(p['completed_elapsed_ratio'] for p in ps):.5f} | {statistics.mean(p['service_difference_at_ours64_positions_ns'] for p in ps)/1e6:.4f} | {statistics.mean(p['service_difference_at_other_positions_ns'] for p in ps)/1e6:.4f} |")
    lines+=['','Negative service differences mean ours consumed less total coordinator service. The two columns partition the difference arithmetically; neither isolates a causal mechanism.','',
        '| Structural measure | Mean ours/stockoff ratio of within-group means |','|---|---:|']
    for name in ('active_copies_sum','active_copies_maxrank','tokens_maxrank','nonempty_expert_tiles_M64_sum','nonempty_expert_tiles_M64_maxrank'):
        lines.append(f"| {name} | {statistics.mean(s['fixedmap_comparison'][name]['ratio_of_mean'] for s in result['structural']):.5f} |")
    lines+=['',''+('Total active-copy and M64 tile counts match on every job in every group; any structural advantage lies in distribution across ranks.' if all(s['total_active_copies_equal_all_jobs'] and s['total_M64_tiles_equal_all_jobs'] for s in result['structural']) else 'See per-group structural data for count differences; no count-to-speedup conversion is made.'),'', 'Limitations:']
    lines+=['- '+x for x in result['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','inputs-dir','validated-summary','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summary=json.loads(a.validated_summary.read_text())
    if summary.get('valid') is not True:raise ValueError('Validated production summary required')
    rows,sources=read_rows(a.records)
    if len(rows)!=summary['observed_records'] or any(r['plan_sha256']!=summary['plan_sha256'] for r in rows):raise ValueError('Summary/raw record matrix mismatch')
    if {r['record_id'] for r in rows}!={t['record_id'] for t in summary['trial_summaries']}:raise ValueError('Validated summary/raw record identities differ')
    result=run(rows,a.inputs_dir);result.update(record_files=sources,validated_summary_sha256=digest(a.validated_summary.read_bytes()),script_sha256=digest(Path(__file__).read_bytes()))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'mechanisms.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output_dir/'mechanisms.md').write_text(markdown_report(result))
    write_csv(a.output_dir/'paired-mechanisms.csv',result['paired_cases'])
    print(json.dumps(dict(phase=result['phase'],trials=len(result['trials']),groups=len(result['structural']),output=str(a.output_dir))))
if __name__=='__main__':main()
