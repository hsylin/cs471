"""Render only completed, independently audited sequence sensitivity evidence."""
import argparse,json,math,statistics,html
from pathlib import Path

def table(head,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

def change(value):return 'undefined' if value is None else f'{100*(value-1):+.2f}%'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'results/sequence-study/final-v2'
    data=json.loads((base/'analysis.json').read_text());assert data['valid'] and data['records']==54
    for n in ('independent-execution-audit.json','independent-statistics.json'):assert json.loads((base/n).read_text())['passed']
    assert json.loads((root/'results/sequence-study/reproduction-v2/reproduction.json').read_text())['passed']
    work=json.loads((base/'work/summary.json').read_text());rows=[]
    for x in data['bundle_effects']:
        rows.append([x['block_id'].replace('residual-','').replace('-final-',' '),x['mode'],x['arm'],
            change(x['ratios']['service_mean_ns']),change(x['ratios']['response_mean_ns']),change(x['ratios']['response_p95_ns']),
            change(x['ratios']['completed_elapsed_ns']),x['response_faster_in_both_repetitions']])
    summary=[]
    for mode in ('mixed','contiguous'):
        for arm in ('stockon32','fix'):
            selected=[x for x in data['bundle_effects'] if x['mode']==mode and x['arm']==arm]
            summary.append(dict(mode=mode,arm=arm,bundles=4,
                service_ratio=math.exp(statistics.mean(math.log(x['ratios']['service_mean_ns']) for x in selected)),
                response_ratio=math.exp(statistics.mean(math.log(x['ratios']['response_mean_ns']) for x in selected)),
                faster_both_reps=sum(x['response_faster_in_both_repetitions'] for x in selected),
                response_ratio_range=[min(x['ratios']['response_mean_ns'] for x in selected),max(x['ratios']['response_mean_ns'] for x in selected)]))
    summaries=[[x['mode'],x['arm'],change(x['service_ratio']),change(x['response_ratio']),str(x['faster_both_reps'])+'/4',
        '['+', '.join(change(y) for y in x['response_ratio_range'])+']'] for x in summary]
    absolute=[]
    for mode in ('mixed','contiguous'):
        for arm in ('stockoff','stockon32','fix'):
            selected=[x['metrics'] for x in data['trial_summaries'] if x['mode']==mode and x['arm']==arm]
            absolute.append([mode,arm]+[f'{statistics.mean(x[m] for x in selected)/scale:.3f}' for m,scale in
                [('service_mean_ns',1e3),('response_mean_ns',1e6),('response_p95_ns',1e6),('queue_mean_ns',1e6),
                 ('completed_elapsed_ns',1e9),('drain_ns',1e6),('throughput_jobs_per_second',1)]])
    routing=[]
    for x in work['routing']:
        routing.append([x['block_id'].replace('residual-','').replace('-final-',' '),x['mode'],
            f"{x['mean_distinct_logical_experts_per_job']:.2f}",f"{x['adjacent_job_tv_mean']:.4f}",
            ', '.join(f'{v:.4f}' for v in x['windows']['1000']['consecutive_window_tv'])])
    probes=json.loads((base/'expert-probes.json').read_text());probe_rows=[]
    for label,peers in probes.items():
        for peer in peers:
            probe_rows.append([label,peer['rank'],f"{statistics.mean(peer['expert_path_elapsed_ms'])*1000:.2f}",
                f"{statistics.mean(x['local_assignments'] for x in peer['assignments']):.1f}",
                f"{statistics.mean(x['local_active_copies'] for x in peer['assignments']):.1f}",
                f"{statistics.mean(x['local_padded_assignment_rows'] for x in peer['assignments']):.1f}"])
    body='<h1>Does preserving document order change the result?</h1>'
    body+='<p>The fresh 192-trial experiment remains negative versus EPLB-off: the repaired dispatcher has <b>0.87% higher service time and 7.00% higher mean response</b> at high load (response 95% CI +5.54% to +8.85%). The launcher repair improves the previous implementation but does not establish a speedup over EPLB-off. <a href="../resident-residual/report.html">Full main-study report</a>.</p>'
    body+='<p>This separate 54-trial sensitivity experiment reuses four content bundles and their exact arrival calendars. It changes input construction, without changing the frozen algorithms. All values below compare each method with contemporaneous EPLB-off under the same ordering. Negative percentage changes are faster. Technical repetitions are averaged within each source-disjoint bundle; these four bundles do not support a new confirmatory population claim.</p>'
    body+='<p><b>Preserving document continuity did not reverse the average ranking.</b> The repaired dispatcher remains 0.81% slower in service and 5.13% slower in mean response; no contiguous bundle wins response in both repetitions. The matched mixed condition has a 0.60% service and 1.50% response penalty, with one of four bundles winning response in both repetitions. These are descriptive paired ratios, not new confidence-interval claims.</p>'
    body+='<p>Table labels: <b>stockoff</b> is the native vLLM layer with EPLB disabled; <b>stockon32</b> is native asynchronous EPLB with 32 extra expert copies; <b>fix</b> is our frozen resident dispatcher with its GPU launcher prepared once. The fix selects among existing copies and does not move expert weights during these trials.</p>'
    body+=table(['Input construction','Method','Service change','Mean response change','Faster in both reps','Bundle response range'],summaries)
    body+=table(['Input','Arm','Service µs','Response ms','p95 ms','Queue ms','Completed s','Drain ms','Jobs/s'],absolute)
    body+='<p>Absolute entries average complete runs; p95 is the mean of run-level p95s. All methods have lower absolute mean service with contiguous construction, but EPLB-off retains the best aggregate service and response. Finite-episode throughput is not saturated capacity.</p>'
    body+='<h2>Every bundle and diagnostic condition</h2>'+table(['Bundle','Input','Method','Service','Mean response','p95 response','Completed elapsed','Faster both reps'],rows)
    body+='<p>Completed elapsed includes fixed arrival horizon, full job drain and normal controller commit completion. It may be horizon-dominated even when response differs. Mean and tail response are layer-job latency, not full-request serving latency.</p>'
    body+='<h2>What was controlled</h2><p>Contiguous input traverses captured document prefixes in token order, repeats whole-bank cycles, and truncates to exactly 4,096 × 128 occurrences. Mixed input uses precisely the same occurrence bag with a seeded shuffle and deterministic duplicate-removing cross-batch swaps. No batch repeats a captured token. All arms and modes share common warmup and absolute arrivals. Short source banks repeat every roughly 15–43 batches: this does not create long-lived 1,000-step hotspots.</p><p>Mixed versus contiguous changes batch composition and temporal order together. A six-trial whole-batch shuffle subset preserves batch membership and changes only ordering. It is a limited diagnostic, not a second generalization study. Selection used identifiers and fixed seeds without consulting routing or timing. Original negative results remain archived.</p>'
    body+='<h2>Actual routing and expert work</h2>'+table(['Bundle','Input','Mean active logical experts/job','Adjacent-job TV','Consecutive 1000-job TV'],routing)
    body+=table(['Separate probe','Rank','Expert-path µs','Owned assignments','Active local copies','Useful padded rows'],probe_rows)
    body+='<p>The 64-input probes replay identical mixed inputs on each arm, and native-on under its last committed and restored initial map. CUDA events surround native expert alignment, two GEMMs, activation and local summation; they exclude gate, replica selection and output all-reduce, but can include submission gaps and contention. They are separately instrumented observations, not primary timings, pure kernel residence or migration payback. Full per-job reconstructed rank token/tile work is archived and checked against the probes.</p>'
    body+='<h2>What remains unresolved</h2><p>Native-on can adjust only about twice in these 4,096-job runs, with native readiness deferrals retained. Launch-to-commit time overlaps execution, so it is not exclusive migration cost. Episode-level cumulative service differences are reported, but do not isolate migration-specific payback. Stable severe skew, longer operation, other layers, decode, dynamic batching and token all-to-all remain outside this experiment.</p><p>The same two reserved H100 94 GB GPUs execute one pinned Qwen3-30B-A3B layer 0, BF16/B128/top8. Off uses128 physical slots; resident/native-on use 160 under the maximum 32-replica allowance. This is replicated input plus native output all-reduce and eager execution. No universal EPLB-off superiority or end-to-end serving speedup is claimed.</p>'
    body+='<p>In all 16 native-on trials in the two main supplemental input conditions, terminal cumulative service remained above the paired off run by 449–552 ms. This demonstrates no episode-level recovery of its total extra cost in these runs, not an isolated per-migration payback time. The longest single-document runs contain 101, 103, 512, or 512 tokens across the four bundles (at most four batches); a 1,000-step native observation window averages roughly 23–68 whole-bank cycles. Adjacent-job demand changes can therefore coexist with almost unchanged window-average demand.</p>'
    body+='<p>Native-on retains a 1,000-step observation window and 3,000-step adjustment interval. The harness fixes the initial counter at 2,250, giving its first opportunity after zero-based job 749; native readiness may defer later opportunities. This controls the counter phase and is not untouched full-serving initialization.</p>'
    body+='<h2>Correctness and reproducibility</h2><p>All prepared inputs receive full-output and logical routing checks; resident assignments are checked against the frozen CPU oracle. Rank agreement, checkpoint weights, committed maps, native sampling windows, sender bytes, normal completion, source/input hashes and raw-clock statistics are audited separately. Offline clean-checkout reconstruction is archived. These are same-team independent implementations, not external-laboratory replication.</p><p><a href="https://github.com/hsylin/cs471/tree/experiments/resident-residual-fastpath">Source and evidence in cs471</a>. Executed source commit <code>dd1754e334d40618dc20aebece1b54460fcb47c7</code>; plan v2 and complete reproduction instructions are in <code>docs/sequence-study</code>.</p>'
    a.output.mkdir(parents=True,exist_ok=True)
    css='body{font:17px/1.5 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#18212b}h1{font-size:32px}h2{font-size:23px;margin-top:35px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ccd5df;text-align:left}th{background:#eef3f8}code{overflow-wrap:anywhere}a{color:#185daa}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>cs471 input ordering sensitivity</title><style>'+css+'</style><body>'+body+'</body></html>')
    (a.output/'summary.json').write_text(json.dumps(dict(summary=summary,bundle_effects=data['bundle_effects'],scope=data['scope']),indent=2)+'\n')
    print(a.output/'report.html')

if __name__=='__main__':main()
