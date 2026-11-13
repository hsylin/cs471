"""Inspect same-stream predecessors of hypothesized cross-rank collectives."""
import argparse,collections,json
from pathlib import Path
from full_serving.boundary_critical_attribution import load_rank,calibration,resolved_latest,distribution
from full_serving.boundary_critical_contract import digest,write_new

def predecessor(events,target):
    stream=target.get('stream')
    if stream is None:return None
    xs=[x for x in events if x is not target and x.get('stream')==stream and x['end']<=target['start']]
    return max(xs,key=lambda x:x['end']) if xs else None

def overlap_other_arrivals(previous,entries,rank,uncertainties):
    if previous is None or previous['category']=='collective':return False
    return previous['end']-uncertainties[rank]>max(e['start']+uncertainties[r] for r,e in entries.items() if r!=rank)

def work_at(rr,ordinal):
    batches={}
    for rank,r in rr.items():
        b=r['batches'].get(ordinal)
        if b is None:
            if r['iterations'].get(ordinal,{}).get('zero_actual_work'):b=dict(real_tokens=0,prefill_tokens=0,decode_tokens=0,attention_pairs_cpu_upper_proxy=0)
            else:return None
        batches[rank]=b
    return batches

def analyze(directory,windows):
    directory=Path(directory);clock=calibration(directory);out=dict(diagnostic_only=True,windows={},input_hashes={},limits=[
        'NCCL pairing uses ordered identical full kernel names and counts, not verified transport sequence IDs.',
        'Same-stream predecessor order is observed. Overlap with peers collective starts supports a late-compute candidate, not quantified network wait or removable critical path.',
        'Work association is descriptive: token/attention proxy maxima do not prove kernel cost, FLOPs or the cause of lateness.',
        'Only forward-associated kernels with same-thread CPU CUDA-API correlation are included; unassigned/background activities remain outside this predecessor analysis.',
        'Empirical clock envelopes do not exclude between-sample clock jumps. Profiler/coordinator perturb serving; no extrapolation to unprofiled B4/OFF residual.'])
    for window in windows:
        rr={r:load_rank(directory,r,window,clock) for r in range(8)};unc={r:x['uncertainty_us'] for r,x in rr.items()}
        common=set.intersection(*(set(r['by']) for r in rr.values()))
        for r in rr.values():
            if r['by']:common-={min(r['by']),max(r['by'])}
        counts=collections.Counter();byclass=collections.defaultdict(collections.Counter);predcats=collections.Counter();gaps=[];examples=[];iteration_rows=[]
        for ordinal in sorted(common):
            batches=work_at(rr,ordinal)
            if batches is None:cls='unknown';max_tokens=set();max_att=set()
            else:
                pre=sum(b['prefill_tokens'] for b in batches.values());dec=sum(b['decode_tokens'] for b in batches.values());cls='mixed' if pre and dec else 'prefill_only' if pre else 'decode_only' if dec else 'all_empty'
                tokens={r:b['real_tokens'] for r,b in batches.items()};att={r:b['attention_pairs_cpu_upper_proxy'] for r,b in batches.items()}
                max_tokens={r for r,v in tokens.items() if v==max(tokens.values())};max_att={r for r,v in att.items() if v==max(att.values())}
            grouped={r:sorted(x['by'][ordinal],key=lambda e:e['start']) for r,x in rr.items()}
            ends={r:max(e['end'] for e in es) for r,es in grouped.items()};last=resolved_latest(ends,unc)
            iteration_rows.append(dict(ordinal=ordinal,work_class=cls,resolved_last_finish_rank=last,
                last_rank_max_tokens=last in max_tokens if last is not None and batches else None,
                last_rank_max_attention_proxy=last in max_att if last is not None and batches else None))
            nccl={r:[e for e in es if e['category']=='collective'] for r,es in grouped.items()};seq=[[e['name'] for e in es] for es in nccl.values()]
            if not seq[0] or any(s!=seq[0] for s in seq[1:]):counts['unpaired_iterations']+=1;continue
            for i in range(len(seq[0])):
                counts['hypothesized_collective_pairs']+=1;byclass[cls]['hypothesized_collective_pairs']+=1
                entries={r:es[i] for r,es in nccl.items()};late=resolved_latest({r:e['start'] for r,e in entries.items()},unc)
                if late is None:continue
                counts['resolved_late_pairs']+=1;byclass[cls]['resolved_late_pairs']+=1
                if batches:
                    byclass[cls]['late_rank_has_max_tokens']+=late in max_tokens
                    byclass[cls]['late_rank_has_max_attention_proxy']+=late in max_att
                pred=predecessor(grouped[late],entries[late]);cat=pred['category'] if pred else 'unknown_predecessor';predcats[cat]+=1
                if pred:gaps.append(entries[late]['start']-pred['end'])
                overlap=overlap_other_arrivals(pred,entries,late,unc)
                if overlap:
                    counts['late_noncollective_predecessor_ends_after_all_peer_arrivals']+=1
                    byclass[cls]['late_noncollective_predecessor_ends_after_all_peer_arrivals']+=1
                    if len(examples)<16:examples.append(dict(ordinal=ordinal,work_class=cls,collective_index=i,late_rank=late,predecessor_category=cat,
                        predecessor_duration_us=pred['end']-pred['start'],same_stream_gap_us=entries[late]['start']-pred['end'],
                        late_rank_tokens=batches[late]['real_tokens'] if batches else None,late_rank_attention_proxy=batches[late]['attention_pairs_cpu_upper_proxy'] if batches else None))
        out['windows'][window]=dict(interior_iterations=len(common),counts=dict(counts),by_work_class={k:dict(v) for k,v in byclass.items()},
          resolved_late_predecessor_categories=dict(predcats),predecessor_to_collective_gap_us=distribution(gaps),bounded_examples=examples,
          resolved_finish_iterations=sum(x['resolved_last_finish_rank'] is not None for x in iteration_rows),
          resolved_finish_with_max_tokens=sum(x['last_rank_max_tokens'] is True for x in iteration_rows),
          resolved_finish_with_max_attention_proxy=sum(x['last_rank_max_attention_proxy'] is True for x in iteration_rows))
        for r in range(8):
            p=directory/f'trace-{window}-rank{r}.json.gz';out['input_hashes'][p.name]=digest(p)
    for r in range(8):
        p=directory/f'critical-rank{r}.jsonl';out['input_hashes'][p.name]=digest(p)
    for phase in ('before','after'):
        p=directory/f'clock-calibration-{phase}.json';out['input_hashes'][p.name]=digest(p)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--windows',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_new(a.output,analyze(a.directory,a.windows))
