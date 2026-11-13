"""CPU-only attribution review of private traces. Never converts spans to speedup."""
import argparse,bisect,collections,gzip,json,math,statistics
from pathlib import Path
from full_serving.boundary_critical_contract import interval_union,write_new,digest
from full_serving.boundary_critical_analysis import rows,category

def distribution(xs):
    if not xs:return dict(n=0,mean=None,p50=None,p95=None,max=None)
    s=sorted(xs)
    def q(p):
        i=(len(s)-1)*p;a=int(i);b=min(a+1,len(s)-1)
        return s[a]+(i-a)*(s[b]-s[a])
    return dict(n=len(xs),mean=statistics.mean(xs),p50=q(.5),p95=q(.95),max=max(xs))

class IndexedIntervals:
    def __init__(self, intervals):
        self.intervals=intervals
        self.starts=[x[0] for x in intervals];self.ends=[x[1] for x in intervals]
        self.prefix=[0]
        for a,b in intervals:self.prefix.append(self.prefix[-1]+b-a)
    def before(self,t):
        i=bisect.bisect_right(self.ends,t)
        value=self.prefix[i]
        if i<len(self.starts):value+=max(0,t-self.starts[i])
        return value
    def overlap(self,a,b):return self.before(b)-self.before(a)

def cpu_execute_scopes(events):
    return sorted([e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')=='cs471::execute'],key=lambda e:e['ts'])

def overlap(intervals,a,b):
    if isinstance(intervals,IndexedIntervals):return intervals.overlap(a,b)
    return sum(max(0,min(y,b)-max(x,a)) for x,y in intervals)

def calibration(directory):
    boxes=collections.defaultdict(list)
    for phase in ('before','after'):
        data=json.loads((directory/f'clock-calibration-{phase}.json').read_text())['samples']
        for node in ('kh021','kh023'):
            sample=min((r for r in data if r['node']==node),key=lambda r:r['received_ns']-r['sent_ns'])
            boxes[node].append([v/1000 for v in sample['offset_interval_ns']])
    return {k:dict(low_us=min(x[0] for x in v),high_us=max(x[1] for x in v),
                   sampled_bounds=v,between_samples_guaranteed=False) for k,v in boxes.items()}

def anchor_box(events,sidecar,window):
    anchors={e['name']:e for e in events if e.get('ph')=='X' and e.get('name','').startswith('cs471::clock_anchor::')}
    bounds=[]
    for r in sidecar:
        if r['kind']!='clock_anchor' or r['window']!=window:continue
        e=anchors[r['name']];start=e['ts'];end=start+e['dur']
        a,m,b=(r[k]/1000 for k in ('before_ns','middle_ns','after_ns'))
        low=max(a-start,m-end);high=min(m-start,b-end)
        assert low<=high+2,'CPU/trace clock anchor inconsistent beyond2us rounding'
        bounds.append((low,high))
    assert len(bounds)>=2
    # Conservative observed envelope rather than pretending one exact offset.
    return min(x[0] for x in bounds),max(x[1] for x in bounds)

def correlate_api_ordinals(events, scopes):
    """Only CUDA API calls wholly inside a CPU execute scope on the same thread."""
    by_thread=collections.defaultdict(list)
    for scope in scopes:by_thread[scope[3]].append(scope)
    starts={}
    for tid,ss in by_thread.items():ss.sort();starts[tid]=[x[0] for x in ss]
    correlations={}
    for e in events:
        if e.get('cat') not in ('cuda_runtime','cuda_driver') or e.get('ph')!='X':continue
        tid=e.get('tid');ss=by_thread.get(tid,[])
        j=bisect.bisect_right(starts.get(tid,[]),e['ts'])-1
        if j<0 or e['ts']+e['dur']>ss[j][1]+2:continue
        c=e.get('args',{}).get('correlation')
        if c is not None:correlations[c]=ss[j][2]
    return correlations

def load_rank(directory,rank,window,clock):
    side=rows(directory/f'critical-rank{rank}.jsonl')
    bound=next(r for r in side if r['kind']=='bound');node=bound['hostname'].split('.')[0]
    with gzip.open(directory/f'trace-{window}-rank{rank}.json.gz','rt') as f:trace=json.load(f)
    events=trace['traceEvents'];low,high=anchor_box(events,side,window);offset=(low+high)/2
    clockbox=clock[node];remote=(clockbox['low_us']+clockbox['high_us'])/2
    uncertainty=(high-low+clockbox['high_us']-clockbox['low_us'])/2
    cpu=cpu_execute_scopes(events)
    host=sorted([r for r in side if r['kind']=='host_span' and r['name']=='execute' and r['window']==window],key=lambda r:r['start_wall_ns'])
    starts=[r['start_wall_ns']/1000 for r in host];used=set();scope=[]
    for e in cpu:
        t=e['ts']+offset;j=bisect.bisect_left(starts,t)
        candidates=[k for k in (j-1,j,j+1) if 0<=k<len(host) and k not in used]
        if not candidates:continue
        k=min(candidates,key=lambda k:abs(starts[k]-t))
        if abs(starts[k]-t)>max(1000,high-low):continue
        used.add(k);r=host[k]
        if r['end_outside_window']:continue
        scope.append((e['ts'],e['ts']+e['dur'],r['start_ordinal'],e['tid']))
    correlations=correlate_api_ordinals(events,scope)
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    by=collections.defaultdict(list)
    for e in kernels:
        c=e.get('args',{}).get('correlation');ordinal=correlations.get(c)
        if ordinal is not None:
            by[ordinal].append(dict(name=e['name'],category=category(e['name']),
                start=e['ts']+offset-remote,end=e['ts']+e['dur']+offset-remote,
                stream=e.get('args',{}).get('stream'),correlation=c))
    batches={r['ordinal']:r for r in side if r['kind']=='actual_batch' and r['window']==window}
    all_union=interval_union([(e['ts']+offset,e['ts']+e['dur']+offset) for e in kernels])
    all_union=IndexedIntervals(all_union)
    cpu_overlap=collections.defaultdict(list)
    for r in side:
        if r['kind']!='host_span' or r['window']!=window or r['end_outside_window']:continue
        a,b=r['start_wall_ns']/1000,r['end_wall_ns']/1000
        cpu_overlap[r['name']].append(max(0,b-a-overlap(all_union,a,b)))
    return dict(node=node,uncertainty_us=uncertainty,anchor_width_us=high-low,by=by,batches=batches,
        assigned_kernel_fraction=sum(len(v) for v in by.values())/len(kernels) if kernels else 0,
        scope_count=len(scope),host_count=len(host),
        cpu_no_kernel_overlap_us={k:distribution(v) for k,v in cpu_overlap.items()},
        kernel_count=len(kernels),clockbox=clockbox,
        iterations={e['ordinal']:e for e in side if e['kind']=='iteration' and e['window']==window})

def resolved_latest(values, uncertainties):
    """Resolve only if one rank is later than every other rank's upper bound."""
    if not values:return None
    last=max(values,key=values.get)
    return last if all(values[last]-uncertainties[last]>v+uncertainties[r] for r,v in values.items() if r!=last) else None

def analyze(directory):
    directory=Path(directory);clock=calibration(directory)
    names=sorted({p.name.split('-rank')[0][len('trace-'):] for p in directory.glob('trace-*-rank0.json.gz')})
    result=dict(diagnostic_only=True,windows={},clock_calibration=clock,
        limits=['CPU time with no observed CUDA kernel overlap is not automatically removable critical-path time.',
        'NCCL kernel index pairing is admitted only for matching full name sequences and counts; it is not transport sequence-ID proof.',
        'Rank ordering uses empirical clock bounds; clock jumps between calibration samples remain unexcluded.',
        'Graph kernel assignment uses CUDA API correlation; unassigned activity remains unknown.',
        'Kernel name categories remain provisional; no FLOPs/KV/bandwidth or profiler speedup claim.'])
    for name in names:
        rr={rank:load_rank(directory,rank,name,clock) for rank in range(8)}
        common=set.intersection(*(set(r['by']) for r in rr.values()))
        before_boundary_exclusion=len(common)
        for r in rr.values():
            if r['by']:common-=set((min(r['by']),max(r['by'])))
        allfinish=[];spread=[];collective_pairs=[];finish_ranks=collections.Counter();late_ranks=collections.Counter()
        categories=collections.defaultdict(list);unpaired=0;class_counts=collections.Counter();work_by_class=collections.defaultdict(list)
        for ordinal in sorted(common):
            batches=[]
            for rank,r in rr.items():
                batch=r['batches'].get(ordinal)
                if batch is None:
                    if r['iterations'].get(ordinal,{}).get('zero_actual_work'):
                        batch=dict(real_tokens=0,prefill_tokens=0,decode_tokens=0,attention_pairs_cpu_upper_proxy=0)
                    else:break
                batches.append(batch)
            if len(batches)==8:
                pref=sum(b['prefill_tokens'] for b in batches);dec=sum(b['decode_tokens'] for b in batches)
                cls='mixed' if pref and dec else 'prefill_only' if pref else 'decode_only' if dec else 'all_empty'
                vals=[b['real_tokens'] for b in batches];att=[b['attention_pairs_cpu_upper_proxy'] for b in batches]
                class_counts[cls]+=1
                if sum(vals):work_by_class[cls].append(dict(token_max_mean=max(vals)/(sum(vals)/8),attention_proxy_max_mean=max(att)/(sum(att)/8) if sum(att) else None))
            else:class_counts['unknown']+=1
            grouped={rank:sorted(r['by'][ordinal],key=lambda e:e['start']) for rank,r in rr.items()}
            ends={rank:max(e['end'] for e in es) for rank,es in grouped.items()}
            starts={rank:min(e['start'] for e in es) for rank,es in grouped.items()}
            uncertainties={r:v['uncertainty_us'] for r,v in rr.items()}
            resolved=resolved_latest(ends,uncertainties)
            if resolved is not None:finish_ranks[resolved]+=1
            allfinish.append(max(ends.values())-min(starts.values()));spread.append(max(ends.values())-min(ends.values()))
            for rank,es in grouped.items():
                for cat in set(e['category'] for e in es):
                    union=interval_union([(e['start'],e['end']) for e in es if e['category']==cat])
                    categories[cat].append(sum(b-a for a,b in union))
            nccl={rank:[e for e in es if e['category']=='collective'] for rank,es in grouped.items()}
            seq=[[e['name'] for e in es] for es in nccl.values()]
            if not seq[0] or any(s!=seq[0] for s in seq[1:]):unpaired+=1;continue
            for i in range(len(seq[0])):
                entries={rank:es[i] for rank,es in nccl.items()}
                first=min(e['start'] for e in entries.values());last_start=max(e['start'] for e in entries.values())
                end=max(e['end'] for e in entries.values())
                resolved=resolved_latest({r:e['start'] for r,e in entries.items()},uncertainties)
                if resolved is not None:late_ranks[resolved]+=1
                collective_pairs.append(dict(start_spread_us=last_start-first,tail_after_latest_start_us=end-last_start,
                    span_us=end-first,latest_rank_resolved=resolved is not None))
        result['windows'][name]=dict(all_rank_correlated_iterations=len(common),before_boundary_exclusion=before_boundary_exclusion,
            boundary_exclusion='First and last correlated ordinal of every rank excluded from cross-rank summaries.',
            rank_details={rank:{k:v for k,v in r.items() if k not in ('by','batches','iterations')} for rank,r in rr.items()},
            sampled_classes=dict(class_counts),work_by_class={k:{field:distribution([x[field] for x in vals if x[field] is not None]) for field in ('token_max_mean','attention_proxy_max_mean')} for k,vals in work_by_class.items()},
            global_correlated_kernel_span_us=distribution(allfinish),rank_kernel_finish_spread_us=distribution(spread),
            resolved_last_finishing_rank_counts=dict(finish_ranks),resolved_late_collective_rank_counts=dict(late_ranks),
            per_rank_iteration_category_union_us={k:distribution(v) for k,v in categories.items()},
            matching_collective_pairs=len(collective_pairs),unpaired_iterations=unpaired,
            collective_start_spread_us=distribution([x['start_spread_us'] for x in collective_pairs]),
            collective_tail_after_latest_start_us=distribution([x['tail_after_latest_start_us'] for x in collective_pairs]),
            collective_pairing_basis='Ordered identical kernel name sequences within CUDA-correlated native DP ordinal; diagnostic hypothesis only.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    write_new(a.output,analyze(a.directory))
