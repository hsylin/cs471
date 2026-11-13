"""Conservative fleet-idle intervals inside paired CPU readiness, CPU-only."""
import argparse,bisect,gzip,json
from pathlib import Path
from full_serving.boundary_critical_analysis import rows
from full_serving.boundary_critical_contract import digest,write_new,interval_union
from full_serving.boundary_critical_attribution import calibration,anchor_box,distribution
from full_serving.boundary_critical_readiness import pair_calls,clock_box

def subtract_activity(a,b,activity):
    if b<=a:return []
    result=[];cursor=a
    for x,y in interval_union(activity):
        if y<=cursor:continue
        if x>=b:break
        if x>cursor:result.append((cursor,min(x,b)))
        cursor=max(cursor,y)
        if cursor>=b:break
    if cursor<b:result.append((cursor,b))
    return result

def definite_common(starts,ends):
    # CPU starts: take latest possible entry; CPU ends: earliest possible return.
    a=max(v[1] for v in starts.values());b=min(v[0] for v in ends.values())
    return (a,b) if b>a else None

def next_execute(span,executes):
    # Same main thread, strictly later native ordinal and no overlap.
    xs=[x for x in executes if x['thread']==span['thread'] and x['start_ordinal']>span['start_ordinal'] and x['start_wall_ns']>=span['end_wall_ns']]
    return min(xs,key=lambda x:x['start_wall_ns']) if xs else None

def analyze(directory,window):
    directory=Path(directory);clocks=calibration(directory);side={r:rows(directory/f'critical-rank{r}.jsonl') for r in range(8)}
    calls={};results={};executions={};cc={};activity=[];inputs={};interiors=[];batch={};trace_stats={}
    for rank,ss in side.items():
        bound=next(x for x in ss if x['kind']=='bound');clock=clocks[bound['hostname'].split('.')[0]];cc[rank]=clock
        spans=[x for x in ss if x['kind']=='host_span' and x['window']==window and not x['end_outside_window']]
        executions[rank]=[x for x in spans if x['name']=='execute' and x['thread']==bound['native_thread_id']]
        ii={x['start_ordinal'] for x in executions[rank]};interiors.append(ii-{min(ii),max(ii)} if len(ii)>2 else set())
        calls[rank]={};results[rank]={}
        for x in spans:
            if x['name']=='EPLB._all_ranks_result_ready' and x['thread']==bound['native_thread_id']:calls[rank].setdefault(x['start_ordinal'],[]).append(x)
        for x in ss:
            if x['kind']=='readiness_result' and x['window']==window:results[rank].setdefault(x['ordinal'],[]).append(x)
        batch[rank]={x['ordinal']:x for x in ss if x['kind']=='actual_batch' and x['window']==window}
        p=directory/f'trace-{window}-rank{rank}.json.gz'
        with gzip.open(p,'rt') as f:events=json.load(f)['traceEvents']
        lo,hi=anchor_box(events,ss,window)
        gpu=[x for x in events if x.get('ph')=='X' and x.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
        # Expand activity through both clock uncertainties; never use only attributed kernels.
        activity.extend((x['ts']+lo-clock['high_us'],x['ts']+x['dur']+hi-clock['low_us']) for x in gpu)
        trace_stats[rank]=dict(all_cuda_activity_events=len(gpu),anchor_width_us=hi-lo,clock_width_us=clock['high_us']-clock['low_us'])
        inputs[p.name]=digest(p);p=directory/f'critical-rank{rank}.jsonl';inputs[p.name]=digest(p)
    common=set.intersection(*interiors)
    for table in (calls,results):
        for rank in range(8):table[rank]={k:v for k,v in table[rank].items() if k in common}
    paired,unknown=pair_calls(calls,results,cc);union=interval_union(activity);starts=[a for a,b in union];ends=[b for a,b in union]
    out=[]
    for pair in paired:
        ordinal=pair['ordinal'];spans={r:calls[r][ordinal][0] for r in range(8)}
        region=definite_common({r:clock_box(s['start_wall_ns'],cc[r]) for r,s in spans.items()}, {r:clock_box(s['end_wall_ns'],cc[r]) for r,s in spans.items()})
        intervals=[]
        if region:
            a,b=region;left=bisect.bisect_left(ends,a);right=bisect.bisect_right(starts,b)
            intervals=subtract_activity(a,b,union[left:right])
        following={r:next_execute(s,executions[r]) for r,s in spans.items()}
        known={r:x for r,x in following.items() if x is not None}
        real={r:batch[r].get(x['start_ordinal'],{}).get('real_tokens') for r,x in known.items()}
        out.append(dict(ordinal=ordinal,all_ready=pair['all_ready'],common_readiness_interval_exists=region is not None,
            conservative_all8_no_observed_cuda_us=sum(b-a for a,b in intervals),
            interval_count=len(intervals),next_same_thread_execute_ranks=len(known),
            next_ordinals={r:x['start_ordinal'] for r,x in known.items()},next_recorded_real_tokens=real,
            next_real_work_observed=any(v is not None and v>0 for v in real.values()),
            entry_pending_missing_ranks=pair['entry_pending_missing_ranks']))
    summaries={}
    for label,items in [('all',out),('ready',[x for x in out if x['all_ready']]),('deferred',[x for x in out if not x['all_ready']])]:
        summaries[label]=dict(calls=len(items),with_definite_common_interval=sum(x['common_readiness_interval_exists'] for x in items),
          with_positive_no_observed_cuda_interval=sum(x['conservative_all8_no_observed_cuda_us']>0 for x in items),
          no_observed_cuda_us=distribution([x['conservative_all8_no_observed_cuda_us'] for x in items]),
          all8_next_same_thread_execute_and_realwork=sum(x['next_same_thread_execute_ranks']==8 and x['next_real_work_observed'] for x in items))
    for phase in ('before','after'):
        p=directory/f'clock-calibration-{phase}.json';inputs[p.name]=digest(p)
    return dict(diagnostic_only=True,window=window,summary=summaries,calls=out,unpaired=unknown,trace_stats=trace_stats,input_hashes=inputs,
        limits=['Reports intervals when all8 main threads are inside paired readiness calls and no captured CUDA kernel/copy/memset intersects, after expanding activity by empirical clock envelopes.',
        'Serial same-thread execution prevents next execute submission before readiness returns. This does not prove removing the check preserves correctness or saves the measured interval.',
        'Unrecorded CUDA activity and between-sample clock jumps remain possible. No complete hardware dependency graph is claimed.',
        'False all_ready defers commit, not necessarily forward. Pending snapshots precede native flag reads.',
        'Profiler/coordinator perturb scheduling; these intervals do not quantify unprofiled B4/OFF residual or add to host/kernel sums.',
        'Next recorded actual batches establish some subsequent work, not identical work/routes across controls.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--window',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_new(a.output,analyze(a.directory,a.window))
