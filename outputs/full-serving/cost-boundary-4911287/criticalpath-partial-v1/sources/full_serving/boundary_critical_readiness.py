"""All-rank readiness and observed commit review, CPU-only and private."""
import argparse,collections,json
from pathlib import Path
from full_serving.boundary_critical_analysis import rows
from full_serving.boundary_critical_contract import digest,write_new
from full_serving.boundary_critical_attribution import calibration,distribution

def clock_box(wall_ns,clock):
    return (wall_ns/1000-clock['high_us'],wall_ns/1000-clock['low_us'])

def latest_resolved(boxes):
    return next((rank for rank,(a,b) in boxes.items() if all(a>other[1] for r,other in boxes.items() if r!=rank)),None)

def range_bounds(boxes):
    return (max(0,max(a for a,b in boxes.values())-min(b for a,b in boxes.values())),
            max(b for a,b in boxes.values())-min(a for a,b in boxes.values()))

def pair_calls(calls,results,clock_by_rank):
    ordinals=set().union(*(set(table.get(r,{})) for table in (calls,results) for r in range(8)))
    paired=[];unknown=[]
    for ordinal in sorted(ordinals):
        if any(len(calls.get(r,{}).get(ordinal,[]))!=1 or len(results.get(r,{}).get(ordinal,[]))!=1 for r in range(8)):
            unknown.append(dict(ordinal=ordinal,reason='missing_or_multiple_calls'));continue
        spans={r:calls[r][ordinal][0] for r in range(8)};rr={r:results[r][ordinal][0] for r in range(8)}
        values={bool(v['all_ready']) for v in rr.values()}
        if len(values)!=1:
            unknown.append(dict(ordinal=ordinal,reason='readiness_result_disagreement'));continue
        starts={r:clock_box(s['start_wall_ns'],clock_by_rank[r]) for r,s in spans.items()}
        ends={r:clock_box(s['end_wall_ns'],clock_by_rank[r]) for r,s in spans.items()}
        if min(b for a,b in ends.values())<max(a for a,b in starts.values()):
            unknown.append(dict(ordinal=ordinal,reason='return_before_latest_entry_beyond_empirical_clock_bounds'));continue
        spread=range_bounds(starts);tail=(max(0,max(a for a,b in ends.values())-max(b for a,b in starts.values())),
            max(b for a,b in ends.values())-max(a for a,b in starts.values()))
        missing=[r for r,s in spans.items() if not s['local_pending_at_entry']]
        layer_set={s['pending_layer_at_entry'] for s in spans.values() if s['local_pending_at_entry']}
        paired.append(dict(ordinal=ordinal,all_ready=values.pop(),entry_pending_missing_ranks=missing,
            entry_pending_layers=sorted(layer_set),arrival_spread_bounds_us=spread,tail_after_latest_entry_bounds_us=tail,
            latest_entry_rank_resolved=latest_resolved(starts),
            rank_call_duration_us={r:s['duration_ns']/1000 for r,s in spans.items()},
            note='Entry snapshots precede native flag reads; publication races prevent identifying the exact missing native flag.'))
    return paired,unknown

def summarize_pairing(paired,unknown):
    missing=collections.Counter(r for x in paired if not x['all_ready'] for r in x['entry_pending_missing_ranks'])
    latest=collections.Counter(x['latest_entry_rank_resolved'] for x in paired if x['latest_entry_rank_resolved'] is not None)
    return dict(paired_calls=len(paired),unpaired_calls=len(unknown),ready_calls=sum(x['all_ready'] for x in paired),
        deferred_calls=sum(not x['all_ready'] for x in paired),entry_missing_in_deferred_calls=dict(missing),
        resolved_latest_entry_rank_counts=dict(latest),
        publication_race_compatible_calls=sum(x['all_ready'] and bool(x['entry_pending_missing_ranks']) for x in paired),
        arrival_spread_lower_us=distribution([x['arrival_spread_bounds_us'][0] for x in paired]),
        arrival_spread_upper_us=distribution([x['arrival_spread_bounds_us'][1] for x in paired]),
        tail_after_latest_entry_lower_us=distribution([x['tail_after_latest_entry_bounds_us'][0] for x in paired]),
        tail_after_latest_entry_upper_us=distribution([x['tail_after_latest_entry_bounds_us'][1] for x in paired]),
        pooled_rank_call_duration_us=distribution([v for x in paired for v in x['rank_call_duration_us'].values()]))

def commit_summary(events_by_rank):
    cycles=sorted({x.get('cycle_step',x.get('step')) for es in events_by_rank.values() for x in es if x['kind'] in ('proposal','commit_enqueued','commit_completed')})
    out=[]
    for cycle in cycles:
        completed={};queued={};dups={};hashes=collections.defaultdict(set)
        for rank in range(8):
            es=events_by_rank.get(rank,[])
            cc=[x for x in es if x['kind']=='commit_completed' and x['cycle_step']==cycle]
            qq=[x for x in es if x['kind']=='commit_enqueued' and x['cycle_step']==cycle]
            completed[rank]=set(x['layer'] for x in cc);queued[rank]=set(x['layer'] for x in qq)
            dups[rank]=len(cc)-len(completed[rank])
            for x in cc:hashes[x['layer']].add(x['map_sha256'])
        observed=set.intersection(*(completed[r] for r in range(8)))
        out.append(dict(cycle_step=cycle,all_rank_observed_layers=sorted(observed),
            per_rank_observed_layers={r:len(v) for r,v in completed.items()},
            per_rank_queued_without_observed_completion={r:sorted(queued[r]-completed[r]) for r in range(8)},
            duplicate_completions=dups,all_observed_layer_hashes_agree=all(len(v)==1 for v in hashes.values()),
            full_48layer_all_rank_completion=observed==set(range(48))))
    return out

def analyze(directory):
    directory=Path(directory);clock=calibration(directory);side={r:rows(directory/f'critical-rank{r}.jsonl') for r in range(8)}
    clock_by_rank={r:clock[next(x for x in es if x['kind']=='bound')['hostname'].split('.')[0]] for r,es in side.items()}
    windows=sorted({x['window'] for es in side.values() for x in es if x['kind']=='profiler_start_end'})
    report=dict(diagnostic_only=True,windows={},input_hashes={},native_commits=[],
        limits=['A false readiness result defers a map commit; it does not itself mean the serving forward is blocked until migration completes.',
        'Wrapper entry pending snapshots occur before native has_result reads. Treat missing-at-entry ranks as candidates, not exact laggards.',
        'Cross-rank ordering uses empirical network clock envelopes; between-sample clock jumps remain unexcluded.',
        'Arrival spread and post-arrival tail are descriptive inclusive CPU spans, not additive causal taxes or saved time.',
        'Profiler/coordinator perturbations and different token routes/batches prevent direct residual B4/OFF attribution.',
        'Observed commit completions establish completion, not a drain-based inference. Missing observations stay censored.'])
    for window in windows:
        calls={};results={};interiors=[]
        for rank,es in side.items():
            ii={x['start_ordinal'] for x in es if x['kind']=='host_span' and x['name']=='execute' and x['window']==window and not x['end_outside_window']}
            interiors.append(ii-{min(ii),max(ii)} if len(ii)>2 else set())
        common=set.intersection(*interiors)
        for rank,es in side.items():
            calls[rank]=collections.defaultdict(list);results[rank]=collections.defaultdict(list)
            for x in es:
                if x.get('window')!=window:continue
                if x['kind']=='host_span' and x['name']=='EPLB._all_ranks_result_ready' and not x['end_outside_window'] and x['start_ordinal'] in common:
                    calls[rank][x['start_ordinal']].append(x)
                if x['kind']=='readiness_result' and x['ordinal'] in common:results[rank][x['ordinal']].append(x)
        paired,unknown=pair_calls(calls,results,clock_by_rank)
        boundaries={r:{x['kind']:x['wall_ns'] for x in es if x.get('window')==window and x['kind'] in ('profiler_start_end','profiler_stop_begin')} for r,es in side.items()}
        worker={};native_inside={}
        for rank,es in side.items():
            spans=[x for x in es if x['kind']=='host_span' and x['window']==window and x['name'].startswith('worker.') and not x['end_outside_window']]
            worker[rank]={name:distribution([s['duration_ns']/1000 for s in spans if s['name']==name]) for name in sorted({x['name'] for x in spans})}
            p=directory/f'migration-events-rank{rank}.jsonl'
            inside=[]
            if p.exists() and len(boundaries[rank])==2:
                a,b=boundaries[rank]['profiler_start_end'],boundaries[rank]['profiler_stop_begin']
                inside=[x for x in rows(p) if a<=x['wall_ns']<b]
            native_inside[rank]=dict(counts=dict(collections.Counter(x['kind'] for x in inside)),
                proposal_steps=[x['step'] for x in inside if x['kind']=='proposal'],
                observed_commit_cycles=dict(collections.Counter(x['cycle_step'] for x in inside if x['kind']=='commit_completed')))
        report['windows'][window]=dict(common_interior_iterations=len(common),summary=summarize_pairing(paired,unknown),paired=paired,unknown=unknown,
            worker_observed_spans_us=worker,native_events_inside_capture=native_inside)
    migration={}
    for rank in range(8):
        p=directory/f'migration-events-rank{rank}.jsonl'
        if p.exists():migration[rank]=rows(p);report['input_hashes'][p.name]=digest(p)
        p=directory/f'critical-rank{rank}.jsonl';report['input_hashes'][p.name]=digest(p)
    report['native_commits']=commit_summary(migration)
    for phase in ('before','after'):
        p=directory/f'clock-calibration-{phase}.json';report['input_hashes'][p.name]=digest(p)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_new(a.output,analyze(a.directory))
