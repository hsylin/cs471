"""Private trace admission and bounded summaries, no profiler speedup claims."""
import argparse,collections,gzip,json,math,statistics,os
from pathlib import Path
from full_serving.boundary_critical_contract import digest,interval_union,write_new
from full_serving.boundary_sequence_common import arithmetic,verify_arithmetic

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def category(name):
    n=name.lower()
    if 'nccl' in n:return 'collective'
    if any(x in n for x in ('flash_fwd','flashinfer','flashattn','attention','fmha')):return 'attention_candidate'
    if any(x in n for x in ('moe','grouped_gemm','fused_experts')):return 'moe_candidate'
    if any(x in n for x in ('logical_history','logical_hist','scatter_logical')):return 'history_candidate'
    if 'memcpy' in n:return 'copy'
    return 'other_or_unclassified'

def summarize_trace(path):
    with gzip.open(path,'rt') if path.name.endswith('.gz') else path.open() as f:v=json.load(f)
    events=v['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    assert kernels,'No CUDA kernels: graph replay visibility not admitted'
    runtime=[e for e in events if e.get('cat')=='cuda_runtime' and e.get('ph')=='X']
    anchors=[e for e in events if e.get('name','').startswith('cs471::clock_anchor::') and e.get('ph')=='X']
    assert len(anchors)>=2,'Missing trace clock anchors'
    union=interval_union([(e['ts'],e['ts']+e['dur']) for e in kernels])
    cats=collections.defaultdict(list)
    for e in kernels:cats[category(e['name'])].append((e['ts'],e['ts']+e['dur']))
    per_cat={k:dict(count=len(x),union_us=sum(b-a for a,b in interval_union(x))) for k,x in cats.items()}
    return dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size,kernels=len(kernels),
        kernel_union_us=sum(b-a for a,b in union),categories=per_cat,
        graph_launch_calls=sum('GraphLaunch' in e['name'] for e in runtime),
        cuda_sync_calls=sum('Synchronize' in e['name'] for e in runtime),
        cuda_runtime_calls=len(runtime),anchors=[{k:e[k] for k in ('name','ts','dur')} for e in anchors],
        trace_time_base=v.get('baseTimeNanoseconds'),
        frequent_kernel_names=collections.Counter(e['name'] for e in kernels).most_common(12),
        limits='Kernel-name categories are provisional. Union times are overlap-aware activity, NOT causal critical path or bandwidth.')

def ordered_records(values):
    records=sorted([r for r in values if r.get('type')=='request'],key=lambda r:r['request_index'])
    assert [r['request_index'] for r in records]==list(range(len(records))), 'Duplicate or missing request indices'
    assert len({r['request_id'] for r in records})==len(records), 'Duplicate request IDs'
    return records

def verify_request_inputs(records,item,expected,input_file):
    from full_serving.client import canonical_hash
    from full_serving.data import digest as manifest_digest
    assert digest(input_file)==item['requests_sha256']
    inputs={r['id']:r for r in rows(input_file)}
    assert set(inputs)==set(item['request_ids'])
    for r in records:
        messages=inputs[r['request_id']]['messages']
        assert manifest_digest(messages)==expected[r['request_id']]['messages_sha256']
        assert canonical_hash(messages)==r['input_sha256']

def review(root):
    status=json.loads((root/'status.json').read_text());assert status['state']=='completed'
    ds,control=status['dataset'],status['control'];directory=root/ds/control
    phase='probe' if status['pilot'] else 'measured'
    recs=rows(directory/(phase+'.jsonl'));records=ordered_records(recs)
    # The native client has request type in all admitted records.
    assert len(records)==(8 if status['pilot'] else 512),(len(records),[r.get('type') for r in recs[:2]])
    summary=next(r for r in recs if r.get('type')=='run_summary');stats=arithmetic(records);verify_arithmetic(stats,summary)
    lengths=json.loads((root/(ds+'-'+('warmup-01' if status['pilot'] else 'measured')+'-lengths.json')).read_text())
    assert {r['request_id']:r['usage']['completion_tokens'] for r in records}==lengths
    data=json.loads((root/'data-manifest.json').read_text())['datasets'][ds]
    item=data['items']['warmup' if status['pilot'] else 'measured']
    assert [r['request_id'] for r in records]==item['request_ids']
    expected={r['id']:r for r in data['requests']}
    verify_request_inputs(records,item,expected,Path(os.environ['CS471_PRIVATE_INPUTS'])/item['requests_file'])
    if not status['pilot']:
        calendar=json.loads((root/(ds+'-calendar.json')).read_text())
        assert [r['arrival_offset_ns'] for r in records]==calendar['arrival_offsets_ns']
    result=dict(status='passed',source=status['source'],dataset=ds,control=control,pilot=status['pilot'],
        contract_sha256=status['contract_sha256'],requests=len(records),diagnostic_only=True,
        performance_interpretation_prohibited=True,request_arithmetic=stats,ranks={},windows={})
    per_rank={}
    required=['admission','repeat1','repeat2'] if status['pilot'] else ['onset','early','middle','late']
    for rank in range(8):
        events=rows(directory/f'critical-rank{rank}.jsonl');per_rank[rank]=events
        assert all(e['rank']==rank for e in events)
        traces={name:summarize_trace(directory/f'trace-{name}-rank{rank}.json.gz') for name in required}
        if control.startswith('fixed_'):
            bound=next(e for e in events if e['kind']=='map_bound')
            from full_serving.resident_policy import balanced_map
            assert bound['mapping']==[balanced_map()]*48 and bound['native_routing_kernel']
            if status['pilot']:
                assert next(e for e in events if e['kind']=='routing_oracle')['result']['status']=='passed'
                assert next(e for e in events if e['kind']=='full_weight_admission')['result']['full_bf16_replica_hashes_equal']
        if control=='fixed_charged':
            overrides=[e for e in events if e['kind']=='diagnostic_proposal_override']
            assert overrides and all(e['actual_changed_slots']==0 for e in overrides)
        result['ranks'][rank]=dict(traces=traces,graph_replay_observed=any(t['graph_launch_calls']>0 for t in traces.values()))
        assert result['ranks'][rank]['graph_replay_observed'],'CUDA graph replay not observed'
    for name in required:
        bounds=[]
        for rank in range(8):
            br=rows(directory/f'critical-boundary-rank{rank}.jsonl')
            select={x['kind']:x for x in br if x.get('window')==name}
            assert all(k in select for k in ('capture_command','start_agreed','start_complete','stop_agreed','export_complete'))
            start=select['start_agreed']['ordinal'];end=select['stop_agreed']['ordinal']
            assert end-start==64 and select['export_complete']['ordinal']==end
            bounds.append((start,end))
        assert len(set(bounds))==1,'Profiler boundaries were not aligned on all ranks'
        sets=[];by_rank={}
        for rank,events in per_rank.items():
            # Only complete execute calls entirely inside capture, excluding stop/export boundary.
            spans=[e for e in events if e['kind']=='host_span' and e['name']=='execute' and e['window']==name and not e['end_outside_window']]
            by_rank[rank]={e['start_ordinal']:e for e in spans};sets.append(set(by_rank[rank]))
        common=set.intersection(*sets);assert len(common)>=2,(name,'Insufficient all-rank complete iterations',len(common))
        classes=collections.Counter();skews=[]
        for ordinal in sorted(common):
            batches=[]
            for rank,events in per_rank.items():
                values=[e for e in events if e['kind']=='actual_batch' and e['ordinal']==ordinal and e['window']==name]
                if values:batches.append(values[-1])
                else:
                    assert by_rank[rank][ordinal]['dummy'],'Missing actual batch is unknown, not zero'
                    batches.append(dict(real_tokens=0,prefill_tokens=0,decode_tokens=0,attention_pairs_cpu_upper_proxy=0))
            pref=sum(b['prefill_tokens'] for b in batches);dec=sum(b['decode_tokens'] for b in batches)
            classes['mixed' if pref and dec else 'prefill_only' if pref else 'decode_only' if dec else 'all_empty']+=1
            work=[b['real_tokens'] for b in batches]
            if sum(work):skews.append(max(work)/(sum(work)/8))
        result['windows'][name]=dict(all_rank_complete_iterations=len(common),agreed_capture_ordinals=bounds[0],first_ordinal=min(common),last_ordinal=max(common),
            sampled_classes=dict(classes),token_max_over_mean_average=statistics.mean(skews) if skews else None)
    for label in ('before','after'):
        calibration=json.loads((directory/f'clock-calibration-{label}.json').read_text())
        assert len(calibration['samples'])==24
        result['clock_'+label]={node:min((r['received_ns']-r['sent_ns'])/2000 for r in calibration['samples'] if r['node']==node) for node in ('kh021','kh023')}
    result['limits']=['Profile start/stop perturbs scheduling; no response-time benefit inferred.',
        'Same fixed placement and routing rule do not imply identical generated tokens, batches or live demand.',
        'Native DP ordinal alignment requires source-pinned dispatch participation; boundary/missing observations excluded.',
        'Cross-host ordering requires clock-offset intervals and anchor bounds; current summaries do not assert causal rank ordering.',
        'Background CPU spans and kernel activity alone do not prove dependencies; detailed attribution still requires review.']
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    write_new(a.output,review(a.root))
if __name__=='__main__':main()
