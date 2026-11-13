"""CPU-only dependency candidates from private traces. No removable-time claims."""
import argparse,bisect,collections,gzip,json
from pathlib import Path
from full_serving.boundary_critical_analysis import rows,category
from full_serving.boundary_critical_contract import interval_union,write_new,digest
from full_serving.boundary_critical_attribution import anchor_box,distribution

class ActivityIndex:
    def __init__(self,intervals):
        self.values=interval_union(intervals);self.starts=[a for a,b in self.values]
    def trailing_unobserved(self,a,b):
        assert a<=b
        j=bisect.bisect_right(self.starts,b)-1
        if j<0:return b-a
        end=self.values[j][1]
        if end>=b:return 0
        return max(0,b-max(a,end))

def api_scopes(events):
    """Exact same-trace CPU containment, separately per host thread."""
    scopes=collections.defaultdict(list);apis=collections.defaultdict(list)
    for e in events:
        if e.get('ph')!='X':continue
        if e.get('cat')=='user_annotation' and e.get('name','').startswith('cs471::'):
            if 'clock_anchor' not in e['name']:scopes[e['tid']].append(e)
        elif e.get('cat') in ('cuda_runtime','cuda_driver'):apis[e['tid']].append(e)
    result={}
    for tid,aa in apis.items():
        ss=sorted(scopes[tid],key=lambda e:e['ts']);active=[];i=0
        for api in sorted(aa,key=lambda e:e['ts']):
            start,end=api['ts'],api['ts']+api['dur']
            while i<len(ss) and ss[i]['ts']<=start:
                active.append(ss[i]);i+=1
            active=[s for s in active if s['ts']+s['dur']>=start]
            containing=[s['name'][7:] for s in active if s['ts']+s['dur']>=end]
            c=api.get('args',{}).get('correlation')
            if c is not None:result[c]=dict(scopes=containing,thread=tid,api_start=start,api_end=end)
    return result

def family(scopes):
    if any(s.startswith('worker.') for s in scopes):return 'worker'
    if any(s.startswith('history.') for s in scopes):return 'logical_history'
    if any(s.startswith('EPLB.') for s in scopes):return 'EPLB_main'
    if 'DP.dispatch' in scopes:return 'DP_dispatch'
    if 'execute' in scopes:return 'forward'
    if any(s.startswith('event.') for s in scopes):return 'event_sync_thread_unspecified'
    return 'outside_cs471_scope'

def sidecar_family(api,spans,low,high):
    # Only definitive full containment through the empirical anchor interval.
    candidates=[s['name'] for s in spans if s['thread']==api['thread']
        and api['api_start']+low>=s['start_wall_ns']/1000
        and api['api_end']+high<=s['end_wall_ns']/1000]
    return family(candidates) if candidates else None

def rank_review(directory,rank,window):
    side=rows(directory/f'critical-rank{rank}.jsonl')
    with gzip.open(directory/f'trace-{window}-rank{rank}.json.gz','rt') as f:events=json.load(f)['traceEvents']
    low,high=anchor_box(events,side,window);offset=(low+high)/2
    spans=[s for s in side if s['kind']=='host_span' and s['window']==window and not s['end_outside_window']]
    iterations={s['start_ordinal']:s for s in spans if s['name']=='execute'}
    if len(iterations)>2:interior=set(iterations)-{min(iterations),max(iterations)}
    else:interior=set()
    api=api_scopes(events);gpu=[e for e in events if e.get('ph')=='X' and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    index=ActivityIndex([(e['ts'],e['ts']+e['dur']) for e in gpu])
    by_family=collections.defaultdict(list);assignment=collections.Counter()
    background=[s for s in spans if s['name'].startswith(('worker.','event.'))]
    for e in gpu:
        a=api.get(e.get('args',{}).get('correlation'))
        if a is None:fam='unassigned_cuda_activity';assignment['no_api_correlation']+=1
        else:
            fam=family(a['scopes']);assignment['exact_trace_api']+=1
            if fam=='outside_cs471_scope':
                sf=sidecar_family(a,background,low,high)
                if sf is not None:fam=sf;assignment['definite_background_sidecar']+=1
        by_family[fam].append((e['ts'],e['ts']+e['dur']))
    cpu_by_name=collections.defaultdict(list)
    for e in events:
        if e.get('cat')=='user_annotation' and e.get('ph')=='X':cpu_by_name[(e['name'],e['tid'])].append(e)
    cpu_by_name={k:sorted(v,key=lambda e:e['ts']) for k,v in cpu_by_name.items()}
    cpu_starts={k:[e['ts'] for e in v] for k,v in cpu_by_name.items()}
    local=[];by_name=collections.defaultdict(list);main=next(x for x in side if x['kind']=='bound')['native_thread_id']
    executes=sorted(iterations.values(),key=lambda s:s['start_wall_ns']);starts=[s['start_wall_ns'] for s in executes]
    for s in spans:
        if s['start_ordinal'] not in interior:continue
        by_name[s['name']].append(s['duration_ns']/1000)
        if s['thread']!=main or s['name'] not in ('EPLB.step','EPLB._all_ranks_result_ready','EPLB._commit_eplb_maps_for_layer','history.update','history.snapshot'):continue
        # Use the trace CPU scope when present, avoiding cross-clock containment for the tail metric.
        key=('cs471::'+s['name'],main);candidates=cpu_by_name.get(key,[])
        target=s['start_wall_ns']/1000-offset;j=bisect.bisect_left(cpu_starts.get(key,[]),target)
        matches=[candidates[k] for k in (j-1,j,j+1) if 0<=k<len(candidates)
            and abs(candidates[k]['ts']-target)<=max(1000,high-low)]
        if len(matches)!=1:continue
        e=matches[0];a,b=e['ts'],e['ts']+e['dur'];j=bisect.bisect_right(starts,s['end_wall_ns'])
        following=executes[j] if j<len(executes) else None
        local.append(dict(ordinal=s['start_ordinal'],scope=s['name'],cpu_us=e['dur'],
            trailing_no_observed_cuda_activity_us=index.trailing_unobserved(a,b),
            next_execute_gap_us=(following['start_wall_ns']-s['end_wall_ns'])/1000 if following else None,
            local_pending_at_entry=s.get('local_pending_at_entry'),pending_layer_at_entry=s.get('pending_layer_at_entry')))
    ready=[s for s in side if s['kind']=='readiness_result' and s['window']==window and s['ordinal'] in interior]
    return dict(rank=rank,anchor_uncertainty_us=(high-low)/2,interior_iterations=sorted(interior),
        cuda_activity_assignment=dict(assignment),observed_cuda_activity_count=len(gpu),
        cuda_union_by_host_scope_us={k:sum(b-a for a,b in interval_union(v)) for k,v in by_family.items()},
        host_scope_durations_us={k:distribution(v) for k,v in by_name.items()},
        readiness_observations=dict(calls=len(ready),all_ready=sum(bool(x['all_ready']) for x in ready),
            local_pending=sum(bool(x['local_pending_at_entry']) for x in ready)),
        local_tail_candidates=local,
        background_spans_observed=sum(s['name'].startswith('worker.') for s in spans))

def analyze(directory):
    directory=Path(directory)
    windows=sorted(p.name[len('trace-'):-len('-rank0.json.gz')] for p in directory.glob('trace-*-rank0.json.gz'))
    result=dict(diagnostic_only=True,windows={},trace_hashes={},sidecar_hashes={f'critical-rank{r}.jsonl':digest(directory/f'critical-rank{r}.jsonl') for r in range(8)},
        limits=['A trailing region without observed CUDA activity is a local CPU-gating candidate, not proven removable time or a speedup.',
        'GPU kernel, memcpy and memset intervals are unioned across streams; unrecorded activity remains unknown.',
        'Scope activity unions are not additive across families, ranks or nested host scopes.',
        'Background waits that began outside a capture window are unobserved, not zero.',
        'No cross-run trajectory or quality equivalence is assumed. No collective-duration bandwidth inference.',
        'These traces alone do not establish a complete inter-rank dependency graph or causal residual B4/OFF decomposition.'])
    for window in windows:
        ranks=[rank_review(directory,r,window) for r in range(8)]
        common=set.intersection(*(set(r['interior_iterations']) for r in ranks))
        tails=collections.defaultdict(list)
        for r in ranks:
            for x in r['local_tail_candidates']:
                if x['ordinal'] in common:tails[x['scope']].append(x)
        result['windows'][window]=dict(common_interior_iterations=len(common),ranks=ranks,
            local_candidates_by_scope={k:dict(observations=len(v),cpu_us=distribution([x['cpu_us'] for x in v]),
                trailing_no_observed_cuda_activity_us=distribution([x['trailing_no_observed_cuda_activity_us'] for x in v]),
                next_execute_gap_us=distribution([x['next_execute_gap_us'] for x in v if x['next_execute_gap_us'] is not None])) for k,v in tails.items()})
        for rank in range(8):
            p=directory/f'trace-{window}-rank{rank}.json.gz';result['trace_hashes'][p.name]=digest(p)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_new(a.output,analyze(a.directory))
