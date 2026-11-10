"""Build a concise static report from completed, audited results only."""
import html,json,math,statistics
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.repo_root.resolve();OUT=args.output.resolve()
BASE=ROOT/'results/resident-residual'
A=json.loads((BASE/'final-v1/analysis.json').read_text())
assert A['valid'] and A['observed_records']==192
for phase in ('calibration-pilot','pilot','calibration-final','final'):
 for name in ('independent-execution-audit.json','independent-statistics.json'):
  assert json.loads((BASE/(phase+'-v1')/name).read_text())['passed']
assert json.loads((BASE/'reproduction-v1/reproduction.json').read_text())['passed']
C=A['confirmatory'];P=C['primary'];S=C['secondary_previous']
PAIRS=A['paired_cases'];TRIALS=A['trial_summaries']

def ratio(contrast,metric,rho=.85,dataset=None):
    selected=[p for p in PAIRS if p['contrast']==contrast and p['rho']==rho and (dataset is None or p['dataset']==dataset)]
    by={}
    for p in selected:by.setdefault((p['dataset'],p['content_group_id']),[]).append(p['log_ratios'][metric])
    ds={}
    for (d,g),v in by.items():ds.setdefault(d,[]).append(statistics.mean(v))
    return math.exp(statistics.mean(statistics.mean(v) for v in ds.values()))

def pct(r):return f'{(r-1)*100:+.2f}%'
def esc(x):return html.escape(str(x))

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

metrics=[('response_mean_ns','Mean response'),('response_p95_ns','p95 response'),('service_mean_ns','Mean service'),('queue_mean_ns','Mean queue delay'),('completed_elapsed_ns','Completed elapsed'),('throughput_jobs_per_second','Finite-episode throughput')]
# Metric availability is explicit: do not silently omit a requested endpoint.
metric_names=set(PAIRS[0]['log_ratios'])
for key,label in metrics:
    if key not in metric_names:raise KeyError((key,sorted(metric_names)))
rows=[[label,pct(ratio('previous_over_stockoff',key)),pct(ratio('fix_over_stockoff',key)),pct(ratio('fix_over_previous',key))] for key,label in metrics]
absolute=[]
for arm in ('stockoff','previous','fix'):
 selected=[r['metrics'] for r in TRIALS if r['arm']==arm and r['rho']==.85]
 absolute.append([arm]+[f'{statistics.mean(v[k] for v in selected)/scale:.3f}' for k,scale in [('response_mean_ns',1e6),('response_p95_ns',1e6),('service_mean_ns',1e6),('queue_mean_ns',1e6),('completed_elapsed_ns',1e9),('drain_ns',1e6),('throughput_jobs_per_second',1)]])
guards=[[g['name'],f"{g['ratio']:.4f}",f"{g['upper_confidence_bound']:.4f}",'PASS' if g['passes'] else 'FAIL'] for g in C['guards']]
cellrows=[[r['dataset'],r['arrival_type'],r['rho'],r['contrast'],pct(r['ratios']['response_mean_ns']),pct(r['ratios']['response_p95_ns'])] for r in C['descriptive_cells']]
if P['passes']:primary_text='The registered useful response-gain endpoint passed.'
elif P['ci95'][0]>1:primary_text='Fresh primary response time was worse than EPLB-off; the registered 95% interval lies above parity.'
elif P['ci95'][1]<1:primary_text='A response improvement was measured, but the predeclared 5% useful-gain requirement was not established.'
else:primary_text='A reliable useful response gain over EPLB-off was not established.'
pass_text=primary_text+' '+('All frozen acceptance criteria passed.' if C['overall_advance']=='pass' else 'The overall acceptance criteria did not pass.')
body=f'''<h1>Residual dispatch overhead: diagnosis and fresh tests</h1>
<p class="lead">{esc(pass_text)}</p>
<p>Fresh primary mean-response ratio: <b>{P['ratio']:.4f}</b> (fix / EPLB-off), registered 95% CI <b>[{P['ci95'][0]:.4f}, {P['ci95'][1]:.4f}]</b>. Ratios below 1 favor the fix. The useful-gain requirement was an upper bound below 0.95.</p>
<h2>What caused the penalty, and what changed</h2>
<p>On four previously observed pilot groups, exact physical-ID replay reduced service by about 29.55 µs/job. It preserved actual gate/top-k, combining weights and physical assignments while excluding online selection. The online method was 17.53 µs slower than EPLB-off; replay was 12.02 µs faster. This is a selector-invocation and buffer/cache intervention, not exclusive GPU kernel time or a universal bound on available execution savings.</p>
<p>The implemented fix binds the existing compiled Triton selector once and reuses that launcher. Previous and fix use the same pinned int32/int64 selector GPU binaries. Routing policy, validation, resident placement and native expert kernels remain unchanged. Host submission elapsed fell from approximately 22 to 11.5 µs in an isolated diagnostic that excludes the subsequent completion wait; whole-job development service improved by about 7 µs. Development still lost to EPLB-off in all four groups, and two groups regressed against the previous method. Those results were retained before the fresh study.</p>
<p>A separately tested safe reduction of empty expert-kernel launch positions made service worse by about 6.7 µs after charging its adapter costs. Removing mapper input checks had no consistent benefit. Neither was selected.</p>
<h2>Fresh high-load results</h2>
{table(['Metric','Previous vs EPLB-off','Fix vs EPLB-off','Fix vs previous'],rows)}
{table(['Arm','Response ms','p95 ms','Service ms','Queue ms','Completed s','Drain ms','Jobs/s'],absolute)}
<p>Absolute values are arithmetic means across complete high-load runs; p95 is the mean of run-level p95s, not a pooled-job percentile.</p>
<p>Changes above are geometric paired ratios, equally weighted across content groups and the two datasets. Throughput is finite-episode throughput, not saturated capacity. The secondary fix/previous response result is {esc(S['status'])}; its point comparison is descriptive when the primary gate is closed.</p>
<p><img src="fresh-effects.svg" alt="Per-content-bundle service and response ratios against EPLB-off" style="max-width:100%"></p>
<h2>Workload guards and regressions</h2>
<p>{sum(g['passes'] for g in C['guards'])}/10 registered guards passed. Date sensitivity: {'PASS' if C['calendar_date_robustness']['passes'] else 'FAIL'}. Overall decision: {esc(C['overall_advance'])}.</p>
{table(['Guard','Point ratio','One-sided 99.5% upper','2% guard'],guards)}
{table(['Content','Arrival type','Offered load','Contrast','Mean response change','p95 response change'],cellrows)}
<p>Date-deletion sensitivity versus EPLB-off: {esc(C['calendar_date_robustness']['ratio_ranges']['stockoff'])}; versus previous: {esc(C['calendar_date_robustness']['ratio_ranges']['previous'])}. These are linked-date point sensitivities, not fresh-day confidence intervals.</p>
<h2>What the experiment establishes</h2>
<p>One pinned Qwen3-30B-A3B MoE layer (layer 0, BF16, B128, top-8), the exact reserved two H100 94 GB GPUs on kh014, fixed 160-slot resident placement and a maximum 32-replica allowance. EPLB-off uses 128 slots. Inputs are replicated and outputs use native all-reduce; this does not test token all-to-all or full-model serving.</p>
<p>The launcher repair targets eager submission overhead. Graph-enabled full-model serving may already amortize some of that work; it was not tested here.</p>
<p>There are 192 final treatments: 16 source-disjoint content bundles treated as bootstrap units, each paired across two external arrival blocks, two loads and three contemporaneous arms. The final bundles contain 512 new-to-this-experiment documents (GSM8K training split and WildChat). A separate 128-document pilot, 20 native-only calibrations, 56 reused-input diagnostics and 24 development treatments bring the measured budget to 340 executions. No test-set freshness or independence from model training is claimed.</p>
<p>Each batch is assembled from captured prompt-prefill hidden states; real native routes are computed inside every forward. No autoregressive continuation tokens are generated. Azure timestamps drive layer jobs, not actual original requests; content and arrivals are independently paired. Absolute arrivals remain fixed under overload and every job drains. Setup, capture, conditioning and untimed correctness probes are excluded; online dispatch, native computation, synchronization, coordinator ACKs and completion tails are included in the declared service/episode boundaries.</p>
<p>The 100,000-draw bootstrap resamples complete content bundles within datasets, retaining both arrival types, loads and all arms. It is conditional on this trace material and allocation. The 5% useful-gain criterion, ten 2% regression guards and actual-date sensitivity rule were frozen before fresh performance measurement.</p>
<h2>Correctness, provenance and reproduction</h2>
<p>The capture audit authenticated 640 documents and 57,474 token records. Full prepared-input checks compare all 4,112 batches per group against native EPLB-off, with 64 selected FP32 reference positions and 16 full postflight batches per trial. Separately implemented same-team audits reconstruct physical assignments, routing/combining weights, native geometry, clocks, committed state and registered statistics. These are bounded numerical checks and same-team independent implementations, not an external laboratory replication.</p>
<p>Frozen implementation commit: <code>fbae52ae021d0571bfd2b8f53b66f41303007b54</code>. The 107-file source closure is <code>136259a52849be38f9dc875f6ccef964ba2c92f8cd53ceee2c5e5c51f2e0dce7</code>. Earlier 83 frozen files and prior negative results remain unchanged.</p>
<p><a href="https://github.com/hsylin/cs471/tree/experiments/resident-residual-fastpath">Source, raw records and reproduction procedure in cs471</a>. Run <code>python docs/resident-residual/reproduce.py --output /absolute/fresh/output</code> from a clean checkout for CPU reanalysis. Authentic GPU recapture requires the pinned public source material, checkpoint and matching environment; see the repository's recapture recipe.</p>'''
OUT.mkdir(parents=True,exist_ok=True)
css='''body{font:17px/1.55 system-ui,sans-serif;color:#18212b;max-width:1080px;margin:45px auto;padding:0 25px}h1{font-size:32px;line-height:1.15}h2{font-size:23px;margin-top:35px}.lead{font-size:21px;font-weight:600}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #ccd5df;text-align:left}th{background:#eef3f8}code{font-size:12px;overflow-wrap:anywhere}a{color:#185daa}'''
(OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>cs471 residual overhead experiment</title><style>'+css+'</style><body>'+body+'</body></html>')
(OUT/'key-results.json').write_text(json.dumps(dict(primary=P,secondary=S,acceptance=C['overall_advance'],ratios={key:{c:ratio(c,key) for c in ('previous_over_stockoff','fix_over_stockoff','fix_over_previous')} for key,_ in metrics}),indent=2)+'\n')
print(OUT/'report.html')
