"""Complete descriptive analysis; no new confirmatory significance test."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from sequence_study.common import load_plan,ARMS
from stock_comparison.analysis import read_records,clock_metrics,native_issues
from resident_residual.analysis import resident_native_issues
from resident_fastpath.analysis import precision_issues
from stock_comparison.data import digest

METRICS=('response_mean_ns','response_p95_ns','service_mean_ns','queue_mean_ns','completed_elapsed_ns','drain_ns','throughput_jobs_per_second')

def geometric_nonnegative(values):
    if any(v is None for v in values):return None
    if any(v<0 or not math.isfinite(v) for v in values):raise ValueError('Invalid nonnegative ratio')
    if any(v==0 for v in values):return 0.
    return math.exp(statistics.mean(math.log(v) for v in values))

def analyze(plan,rows):
    expected={(t['block_id'],t['mode'],t['rep'],t['arm']):t for t in plan['trials']}
    assert len(expected)==54 and len(rows)==54
    index={};values={};summaries=[]
    for row in rows:
        key=(row['block_id'],row['mode'],row['technical_rep'],row['arm']);assert key in expected and key not in index
        t=expected[key];assert row['record_id']==digest({k:v for k,v in row.items() if k!='record_id'})
        assert row['plan_sha256']==plan['content_sha256'] and row['source_sha256']==plan['source_sha256']
        assert row['schema']=='sequence_study_trial_v1' and row['phase']=='supplement' and row['status']=='complete'
        assert row['evidence_eligible'] is False and row['arm_order']==t['arm_order']
        hw=plan['hardware'];assert row['hostname'].split('.')[0]==hw['hostname'] and sorted(row['gpu_uuids'])==sorted(hw['gpu_uuids']) and str(row['allocation_id'])==str(hw['allocation_id'])
        calendar=plan['calendars'][key[0]]['calendar']
        assert row['calendar']==calendar and row['arrival_offsets_ns']==calendar['arrival_offsets_ns'] and row['horizon_ns']==calendar['horizon_ns']
        assert not precision_issues(row['runtime_metadata'])
        issues=resident_native_issues(row,plan) if key[-1]=='fix' else native_issues(row)
        assert not issues,(key,issues)
        metrics=clock_metrics(row);index[key]=row;values[key]=metrics
        summaries.append(dict(block_id=key[0],dataset=row['dataset'],mode=key[1],technical_rep=key[2],arm=key[3],metrics=metrics))
    assert set(index)==set(expected)
    for b in plan['banks']:
        gid=b['block_id'];subset=[r for k,r in index.items() if k[0]==gid]
        assert len({r['input_meta']['measured_occurrence_multiset_sha256'] for r in subset})==1
        assert len({r['input_meta']['warmup_indices_sha256'] for r in subset})==1
        for mode in {r['mode'] for r in subset}:
            paired=[r for r in subset if r['mode']==mode]
            for field in ('logical_routing_sha256','float32_combining_weights_sha256'):
                assert len({r['preflight'][field] for r in paired})==1
            assert len({r['input_meta']['indices_sha256'] for r in paired})==1
        actual=[(r['mode'],r['technical_rep'],r['arm']) for r in subset]
        planned=[(t['mode'],t['rep'],t['arm']) for t in plan['trials'] if t['block_id']==gid]
        assert actual==planned,'Changed execution order'
    paired=[];break_even=[]
    for gid,mode,rep,arm in index:
        if arm=='stockoff':continue
        off=(gid,mode,rep,'stockoff');key=(gid,mode,rep,arm)
        paired.append(dict(block_id=gid,dataset=index[key]['dataset'],mode=mode,rep=rep,arm=arm,
            ratios={m:values[key][m]/values[off][m] if values[off][m]>0 else None for m in METRICS}))
        on,indexoff=index[key],index[off]
        delta=np.cumsum(np.subtract(on['coordinator_finish_ns'],on['coordinator_start_ns'])-np.subtract(indexoff['coordinator_finish_ns'],indexoff['coordinator_start_ns']))
        persistent=np.flatnonzero(np.maximum.accumulate(delta[::-1])[::-1]<=0)
        break_even.append(dict(block_id=gid,mode=mode,rep=rep,arm=arm,cumulative_service_difference_ns=delta.tolist(),
            terminal_difference_ns=int(delta[-1]),first_job_remaining_nonpositive=None if len(persistent)==0 else int(persistent[0]),
            interpretation='Episode-level cumulative service difference versus off; includes different controller/slot geometry; not isolated migration payback'))
    bundles=[]
    for gid,mode,arm in sorted({(p['block_id'],p['mode'],p['arm']) for p in paired}):
        selected=[p for p in paired if (p['block_id'],p['mode'],p['arm'])==(gid,mode,arm)];assert len(selected)==2
        bundles.append(dict(block_id=gid,dataset=selected[0]['dataset'],mode=mode,arm=arm,
            ratios={m:geometric_nonnegative([p['ratios'][m] for p in selected]) for m in METRICS},
            response_faster_in_both_repetitions=all(p['ratios']['response_mean_ns']<1 for p in selected),
            response_slower_in_both_repetitions=all(p['ratios']['response_mean_ns']>1 for p in selected)))
    interactions=[]
    for b in plan['banks']:
        for arm in ('stockon32','fix'):
            x={p['mode']:p for p in bundles if p['block_id']==b['block_id'] and p['arm']==arm}
            interactions.append(dict(block_id=b['block_id'],arm=arm,contiguous_over_mixed_relative_effect={m:x['contiguous']['ratios'][m]/x['mixed']['ratios'][m] if x['contiguous']['ratios'][m] is not None and x['mixed']['ratios'][m] is not None and x['mixed']['ratios'][m]>0 else None for m in METRICS}))
    return dict(schema='sequence_study_analysis_v1',valid=True,records=54,independent_content_bundles=4,
        no_confirmatory_test=True,trial_summaries=summaries,paired_cases=paired,bundle_effects=bundles,
        ordering_interactions=interactions,episode_break_even=break_even,
        scope='Descriptive constructed-input sensitivity on reused content/calendars; technical reps averaged within bundles; no new fresh-content or long-run amortization claim.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=load_plan(a.plan);rows,files=read_records(a.records);result=analyze(plan,rows);result['record_files']=files
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(valid=True,records=len(rows))))
    from sequence_study.work import reconstruct_work,routing_summary
    from followup.router_tapes import file_sha256
    target=a.records/'work';target.mkdir(exist_ok=False);routes={};summaries=[]
    for b in plan['banks']:
        gid=b['block_id']
        for mode in {r['mode'] for r in rows if r['block_id']==gid}:
            path=a.records/'inputs'/(gid+'-'+mode+'-stock-routes.npz')
            with np.load(path,allow_pickle=False) as z:ids=z['logical_ids'][16:].copy()
            routes[gid,mode]=ids
            summaries.append(dict(block_id=gid,mode=mode,source_file_sha256=file_sha256(path),**routing_summary(ids)))
        assert len({digest(s['total_logical_counts']) for s in summaries if s['block_id']==gid})==1,'Ordering changed aggregate logical routes'
    worksummary=[]
    for r in rows:
        state=r['native_metadata'];w=reconstruct_work(routes[r['block_id'],r['mode']],r['arm'],state['source_map'],state['async_events'])
        name=f"{r['block_id']}-{r['mode']}-{r['technical_rep']}-{r['arm']}.json"
        (target/name).write_text(json.dumps(w)+'\n')
        worksummary.append(dict(block_id=r['block_id'],mode=r['mode'],rep=r['technical_rep'],arm=r['arm'],
            file=name,sha256=file_sha256(target/name),
            mean_rank_tokens=np.mean([x['per_rank_tokens'] for x in w['records']],axis=0).tolist(),
            mean_rank_active_copies=np.mean([x['per_rank_active_copies'] for x in w['records']],axis=0).tolist(),
            mean_rank_padded_rows=np.mean([x['per_rank_padded_rows'] for x in w['records']],axis=0).tolist(),
            mean_max_to_mean_tokens=statistics.mean(x['max_to_mean_token_ratio'] for x in w['records']),
            mean_max_to_mean_padded=statistics.mean(x['max_to_mean_padded_ratio'] for x in w['records'])))
    (target/'summary.json').write_text(json.dumps(dict(routing=summaries,work=worksummary),indent=2)+'\n')

if __name__=='__main__':main()
