#!/usr/bin/env python3
"""Build final stock-layer reports only after complete evidence and three audits.

This file is outside the frozen execution source. It introduces no inference.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

ARMS=('stockoff','stockon32','ours')
DATASETS=('gsm8k','mbpp','wildchat')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def gm(values):return math.exp(statistics.mean(math.log(v) for v in values))
def fmt(x,digits=4):return f'{x:.{digits}f}'
def change(r):return f'{abs(r-1)*100:.2f}% '+('lower' if r<1 else 'higher' if r>1 else 'unchanged')


def validate_inputs(analysis,mechanism,audits,analysis_sha):
    require(analysis.get('schema')=='stock_comparison_analysis_v1' and analysis.get('phase')=='final' and analysis.get('valid') is True,'Complete valid final analysis required')
    require(all(analysis[k]==216 for k in ('expected_records','observed_records','valid_records')) and analysis['matched_triplets']==72 and analysis['paired_comparisons']==216,'Final216 matrix required')
    require(not analysis['errors'] and not analysis['missing_keys'],'Final analysis contains errors/missing rows')
    confirm=analysis.get('confirmatory');require(confirm is not None and confirm['content_groups']==18,'Final conditional inference missing')
    require(confirm['bootstrap']['samples']==100000 and confirm['bootstrap']['seed']==471612,'Wrong prespecified bootstrap')
    require(len(confirm['guards'])==14,'All14 prespecified guards required')
    require(mechanism.get('schema')=='stock_descriptive_mechanisms_v1' and mechanism.get('phase')=='final','Final descriptive mechanism data required')
    require(mechanism['validated_summary_sha256']==analysis_sha,'Mechanism report uses a different summary')
    require(len(mechanism['trials'])==216 and len(mechanism['structural'])==18 and len(mechanism['paired_cases'])==144,'Complete final mechanism matrix required')
    require({t['record_id'] for t in mechanism['trials']}=={t['record_id'] for t in analysis['trial_summaries']},'Mechanism/raw-analysis record identities differ')
    expected_schemas=('independent_stock_record_audit_v1','independent_stock_lifecycle_audit_v1','stock_independent_statistics_audit_v1')
    for audit,schema in zip(audits,expected_schemas):
        require(audit.get('schema')==schema,'Wrong independent audit type')
        require(audit.get('passed' if schema==expected_schemas[-1] else 'valid') is True,'Independent audit failed')
        require(audit['plan_sha256']==analysis['plan_sha256'],'Audit plan mismatch')
        require(not audit.get('errors') and not audit.get('missing'),'Independent audit has unresolved rows')
        if schema!=expected_schemas[-1]:require(audit['expected_records']==audit['observed_records']==216,'Audit incomplete final matrix')
    require(audits[-1]['compared_analysis_sha256']==analysis_sha,'Statistics audit compared another summary')
    require(audits[-1]['recomputed']['phase']=='final' and audits[-1]['recomputed']['trial_count']==216,'Statistics audit is not final')
    evidence=mechanism.get('runtime_configuration_evidence',[])
    require({e['arm'] for e in evidence}==set(ARMS),'Actual native configuration evidence missing')
    for e in evidence:
        require(e['logical_router_computation_included'] is True,'Native gate/top-k not included')
        require(e['stock_policy_unmodified']==(e['arm']!='ours') and e['stock_router_unmodified']==(e['arm']!='ours'),'Stock/modified treatment identity mismatch')
        require(e['native_experts_class'].endswith('.TritonExperts') and e['resolved_kernel_config']['BLOCK_SIZE_M']==64,'Unexpected actual expert backend; report wording requires review')
        require(e.get('kernel_override_applied') is False,'Native kernel configuration was overridden')
        if e['arm']=='stockon32':require(e.get('communicator')=='torch_gloo' and e.get('communicator_class','').endswith('.TorchDistGlooStagedEplbCommunicator'),'Observed native transport differs; review report wording')
        probe=e['output_reduction_probe']
        require(probe['observed_successful_backends']==['ca_comm'] and not probe['backend_forced'] and not probe['unresolved_native_branch'],'Custom all-reduce evidence differs; review wording')
    return confirm


class Document:
    def __init__(self):self.blocks=[]
    def heading(self,text,level=2):self.blocks.append(('heading',level,text))
    def paragraph(self,text):self.blocks.append(('paragraph',text))
    def table(self,headers,rows):self.blocks.append(('table',headers,rows))
    def markdown(self):
        parts=[]
        for block in self.blocks:
            if block[0]=='heading':parts.append('#'*block[1]+' '+block[2])
            elif block[0]=='paragraph':parts.append(block[1])
            else:
                _,headers,rows=block
                parts.append('\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]))
        return '\n\n'.join(parts)+'\n'
    def html(self):
        parts=[]
        for block in self.blocks:
            if block[0]=='heading':parts.append(f'<h{block[1]}>{html.escape(block[2])}</h{block[1]}>')
            elif block[0]=='paragraph':parts.append('<p>'+html.escape(block[1])+'</p>')
            else:
                _,headers,rows=block
                parts.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
        style='body{font:17px/1.55 Georgia,serif;color:#151515;background:#fff;max-width:1100px;margin:3rem auto;padding:0 1.2rem}h1,h2{font-family:system-ui,sans-serif;line-height:1.2}h1{font-size:2rem}h2{font-size:1.35rem;margin-top:2.1rem}table{border-collapse:collapse;width:100%;font:14px/1.4 system-ui,sans-serif;font-variant-numeric:tabular-nums}th,td{padding:.55rem .65rem;border-bottom:1px solid #bbb;text-align:right}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid #222;border-bottom:2px solid #222}tbody tr:last-child{border-bottom:2px solid #222}.table-scroll{overflow-x:auto}p{max-width:100ch}@media print{body{font-size:11pt;margin:0;max-width:none}table{font-size:9pt}h2{break-after:avoid}.table-scroll{overflow:visible}}'
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stock vLLM layer comparison</title><style>'+style+'</style><main>'+''.join(parts)+'</main></html>\n'


def build_document(a,m,c,audits):
    d=Document();d.heading('Comparison with genuine stock vLLM layer configurations',1)
    p=c['primary'];secondary=c['secondary_stockon32']
    verdict='The prespecified useful-gain claim against both tested stock configurations passed.' if c['acceptance']=='pass' else 'The prespecified useful-gain claim against both tested stock configurations was not established.'
    if p['ci95'][0] > 1:
        verdict += ' The current method has substantially higher high-load response latency than EPLB-disabled stockoff within this frozen, conditional design.'
    d.paragraph(verdict+f" At high offered load, ours/stockoff mean-response ratio was {fmt(p['ratio'])}, a {change(p['ratio'])} paired point estimate; 95% interval [{fmt(p['ci95'][0])}, {fmt(p['ci95'][1])}]. Passing required an upper bound below 0.95 plus all safeguards. A failed safeguard is not automatically proof of a regression beyond 2%.")
    d.heading('What was compared')
    d.paragraph('This is an eager, single Qwen3-30B-A3B MoE layer on the same two H100s, using FCFS jobs and replicated-input TP2+EP. It is not a full engine, full-model generation, chat-request, TTFT or deployment result. All arms include the actual pinned stock gate/top-k and experts. Native untuned Triton resolved to M64; a delegate-only untimed probe observed CustomAllreduce without forcing it. Stockon32 used staged Gloo transport. Only ours changes the policy and physical dispatcher.')
    d.table(['Arm','Resident experts','Controller / mapping'],[
        ['stockoff','128 total; 64/rank','EPLB-default-off within this controlled layer harness; no added forced-SKIP recorder/controller.'],
        ['stockon32','160 total; 80/rank','Stock enabled EPLB; 32 redundant experts; window 1,000 / interval 3,000; native asynchronous proposals and commits.'],
        ['ours','160 total; 80/rank','Resident one-home mapper + unchanged-map shortcut; window 16 / interval 64 forced-SKIP policy; all 64 proposals charged.']])
    d.paragraph('Expert parameters occupy 576 MiB per rank for stockoff and 720 MiB for stockon32/ours, a 25% increase; all arms also have the same 0.5 MiB gate. Enabled EPLB adds buffers. Recorded process peak allocation includes all prepared comparator objects, inputs and references, so it is not standalone-engine memory.')
    d.paragraph('Stockoff uses the EPLB default, not the entire shipped engine configuration. Hardware and maximum replica allowance match; actual residency does not. Stockoff cannot legally retain 32 redundant experts with EPLB disabled. This corrects the earlier comparison against an instrumented forced-SKIP baseline: its positive conditional results remain archived, but they did not establish superiority to stockoff. Neither present comparator represents every tunable vLLM configuration.')
    d.paragraph('These are 4,096-job warm-process, reset-state episodes. Untimed conditioning exercises kernels and asynchronous transport; measured recording history and native counters are then reset. Stockon32 starts with a 750-step partial window, followed typically by one full 1,000-step window at its next native opportunity; observed deferrals and normal completion tails remain included. This is not a lifelong-adaptation or steady-state serving result.')
    d.heading('Prespecified paired inference')
    d.paragraph('The final design contains 18 fresh content groups (six each GSM8K, MBPP and WildChat), two external Azure arrival types, two calibrated loads and three arms: 216 trials of 4,096 B128 layer jobs. A source timestamp releases a constructed fixed-token job; original request lengths and prompt contents are not executed together. Ratios first average paired logs over arrival types within content group, then groups within dataset and datasets equally. The 100,000-draw bootstrap resamples whole content bundles within dataset; jobs are not independent replicates.')
    ci=lambda e:'not tested: hierarchy closed' if e.get('ci95') is None else '['+fmt(e['ci95'][0])+', '+fmt(e['ci95'][1])+']'
    d.table(['High-load mean response','Paired ratio','95% interval','Decision'],[
        ['ours / stockoff',fmt(p['ratio']),ci(p),'5% target established' if p['passes'] else '5% target not established'],
        ['ours / stockon32',fmt(secondary['ratio']),ci(secondary),'5% target established' if secondary.get('passes') is True else '5% target not established' if secondary.get('passes') is False else 'descriptive point only']])
    d.paragraph('The stockon32 secondary is confirmatory only after the stockoff primary, 14 regression guards and date-deletion checks pass. Guards use one-sided percentile upper bounds at 1 − 0.05/14 = 99.6429%, each required ≤ 1.02. They cover each dataset/load mean and aggregate high-load p95 against both controls.')
    guards={(g['contrast'],tuple(g['datasets']),g['rho'],g['metric']):g for g in c['guards']}
    guard_rows=[]
    keys=[(dataset,rho,'response_mean_ns',(dataset,)) for dataset in DATASETS for rho in (.5,.85)]+[('all',.85,'response_p95_ns',DATASETS)]
    for dataset,rho,metric,ds in keys:
        cells=[]
        for control in ('stockoff','stockon32'):
            g=guards['ours_over_'+control,tuple(ds),rho,metric]
            cells.append(f"{fmt(g['ratio'])} / {fmt(g['upper_confidence_bound'])}; "+('pass' if g['passes'] else 'not established'))
        guard_rows.append([f"{dataset}; ρ={rho}; "+('p95' if metric=='response_p95_ns' else 'mean'),*cells])
    d.table(['Guard endpoint','ours/off: point / upper; guard','ours/on32: point / upper; guard'],guard_rows)
    dates=c['calendar_date_robustness'];ranges=dates['ratio_ranges']
    range_text=lambda x:'not estimable' if x is None else fmt(x[0])+'–'+fmt(x[1])
    d.paragraph(f"Deleting each of {dates['date_count']} actual source dates gives high-load point-ratio ranges {range_text(ranges['stockoff'])} versus stockoff and {range_text(ranges['stockon32'])} versus stockon32. The ≤1.02 date-sensitivity safeguard "+('passed.' if dates['passes'] else 'was not established.')+' These are finite-design point checks, not new-day confidence intervals.')
    d.heading('Both loads and completed work')
    paired=[]
    for rho in (.5,.85):
        for control in ('stockoff','stockon32'):
            rows=[x for x in a['paired_cases'] if x['rho']==rho and x['contrast']=='ours_over_'+control]
            paired.append([rho,'ours/'+control,*[fmt(gm(x['ratios'][key] for x in rows)) for key in ('response_mean_ns','response_p95_ns','completed_elapsed_ns')]])
    d.table(['Load ρ','Contrast','Mean-response ratio','p95 ratio','Completed-extent ratio'],paired)
    d.paragraph('The table above gives descriptive paired geometric ratios, without additional intervals or tests. Below are arithmetic means of complete trial summaries; they are different estimands and must not replace the primary paired endpoint.')
    rows=[]
    for rho in (.5,.85):
        for arm in ARMS:
            ts=[t for t in m['trials'] if t['rho']==rho and t['arm']==arm]
            avg=lambda fn:statistics.mean(fn(t) for t in ts)
            rows.append([rho,arm,*[fmt(avg(lambda t,n=n:t[n]['mean'])/1e6,3) for n in ('response_ns','queue_ns','service_ns')],
                fmt(avg(lambda t:t['completed_elapsed_ns'])/1e9,3),fmt(avg(lambda t:t['throughput_jobs_per_second']),1),fmt(avg(lambda t:t['controller_tail_elapsed_ns'])/1e6,3),fmt(avg(lambda t:t['drain_including_controller_ns'])/1e6,3)])
    d.table(['ρ','Arm','Response ms','Queue ms','Service ms','Completed s','Jobs/s','Tail ms','Drain incl. control ms'],rows)
    d.paragraph('Response ends at inference-stream output completion plus CPU ACK. Completed extent includes normal asynchronous controller quiescence and the full offered horizon. Jobs/s is the arithmetic mean of each finite episode’s completed jobs divided by that extent, including horizon/drain; it is not saturated capacity. Tail is post-output wall time, not exclusive controller cost. Device-wide synchronization is absent from each job; native dependencies remain charged.')
    d.heading('Mechanisms and limits')
    work=m['structural'];metrics=('active_copies_sum','active_copies_maxrank','nonempty_expert_tiles_M64_maxrank')
    d.table(['Structural measure','Mean ours/off ratio across group means'],[[name,fmt(statistics.mean(w['fixedmap_comparison'][name]['ratio_of_mean'] for w in work))] for name in metrics])
    equal=all(w['total_active_copies_equal_all_jobs'] and w['total_M64_tiles_equal_all_jobs'] for w in work)
    d.paragraph(('Total active-copy and M64 tile counts match for every compared job; their distribution across ranks can still differ. ' if equal else 'Per-group work counts differ; they are reported without conversion into predicted runtime. ')+
        'Saved actual stock routes authenticate these counts. M64 nonempty expert tiles are structural work measures, not measured CUDA duration. Raw service includes selection, native control, execution and coordination; no ablation isolates their individual costs.')
    ps=[x for x in m['paired_cases'] if x['contrast']=='ours_over_stockoff']
    boundary=statistics.mean(x['service_difference_at_ours64_positions_ns'] for x in ps)/1e6
    ordinary=statistics.mean(x['service_difference_at_other_positions_ns'] for x in ps)/1e6
    life=audits[1]['summary']
    d.paragraph(f"Relative to stockoff, mean summed service difference was {boundary:+.3f} ms/trial at our 64 control positions and {ordinary:+.3f} ms across the other 4,032 positions (negative favors ours). This partitions an observed difference, not causal component attribution. Native stockon32 completed {life['native_cycles']} asynchronous cycles; {life['tail_commit_cycles']} committed in the final tail and {life['trials_with_deferred_boundary']} trials deferred a boundary. Launch/commit neighborhoods are reported separately as temporal associations.")
    d.paragraph('All 216 records passed independent raw-input/timing, native-lifecycle and statistical audits. Per content group, numerical coverage includes all 4,112 prepared batches against stockoff and 64 sampled token positions against dense FP32. Each trial also checks 16 whole postflight batches and every resident expert weight. Uncertainty is conditional on selected content/trace material, reused public weeks, constructed batches and one GPU allocation; it does not generalize to new production days, hardware populations or full-model serving. No final outcomes tuned the method or the 5%/2% margins.')
    d.paragraph('Provenance: frozen plan '+a['plan_sha256']+'; execution source '+a['source_sha256']+'. Complete raw records, frozen protocol, three audits and descriptive mechanism tables accompany this report.')
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','mechanism','record-audit','lifecycle-audit','statistics-audit'):p.add_argument('--'+name,type=Path,required=True)
    location=Path(__file__).resolve().parent
    repo=next((x for x in (location,*location.parents) if (x/'stock_comparison/inference.py').is_file()),None)
    if repo is None and (location/'cs471/stock_comparison/inference.py').is_file():repo=location/'cs471'
    require(repo is not None,'Cannot locate repository; run from its archived reproduction folder')
    default_html=(location.parent/'outputs/stock-comparison/results.html' if location.name=='work' else repo/'results/stock-comparison/report/results.html')
    p.add_argument('--markdown-output',type=Path,default=repo/'docs/stock-comparison/results.md')
    p.add_argument('--html-output',type=Path,default=default_html)
    p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    a,m=read(args.analysis),read(args.mechanism);audits=[read(args.record_audit),read(args.lifecycle_audit),read(args.statistics_audit)]
    confirm=validate_inputs(a,m,audits,sha(args.analysis));doc=build_document(a,m,confirm,audits)
    for path in (args.markdown_output,args.html_output):require(args.overwrite or not path.exists(),'Output exists; deliberate --overwrite required')
    md=doc.markdown();ht=doc.html()
    for path,text in ((args.markdown_output,md),(args.html_output,ht)):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(json.dumps(dict(markdown=str(args.markdown_output),html=str(args.html_output),rough_word_count=len(md.split()),final_trials=216,acceptance=confirm['acceptance'])))
if __name__=='__main__':main()
