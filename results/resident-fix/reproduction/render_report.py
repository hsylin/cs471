#!/usr/bin/env python3
"""Render the complete audited resident-fix result; no production imports.

This presentation tool adds no tests, confidence intervals or outcome selection.
It refuses incomplete artifacts and writes only new report paths unless asked
explicitly to replace an earlier rendering. Never run against partial results.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
MAIN = ('gsm8k', 'wildchat')
DATASETS = (*MAIN, 'mbpp')
ARMS = ('stockoff', 'current', 'fix')
TYPES = ('code', 'conversation')
LOADS = (.5, .85)
CONTRASTS = ('fix_over_stockoff', 'fix_over_current', 'current_over_stockoff')
BRANCH = 'https://github.com/hsylin/cs471'
REF = 'experiments/resident-static-fastpath'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def link(label, path, tree=False):
    return f'[{label}]({BRANCH}/{"tree" if tree else "blob"}/{REF}/{path})'


def check_inputs(repo, results, proof_dir):
    run = results / 'final-v1'
    paths = {'plan': results / 'plans/final-v1.json', 'analysis': run / 'analysis.json',
             'records': run / 'independent-record-audit.json',
             'statistics': run / 'independent-statistics-audit.json',
             'mechanism': run / 'mechanism.json', 'reproduction': proof_dir / 'reproduction-report.json'}
    for name, path in paths.items():
        require(path.is_file(), f'Required complete {name} artifact missing: {path}')
    data = {name: read(path) for name, path in paths.items()}
    p, a, r, s, m, cpu = (data[k] for k in paths)
    require(p['phase'] == 'final' and p['split'] == 'final', 'Final plan required')
    require(p['content_sha256'] == digest({k:v for k,v in p.items() if k != 'content_sha256'}), 'Plan seal differs')
    require(p['source_sha256'] == digest(p['source_files']), 'Source inventory seal differs')
    for name, expected in p['source_files'].items():
        target = (repo / name).resolve()
        require(target.is_relative_to(repo) and file_hash(target) == expected, f'Frozen source changed: {name}')
    require(file_hash(repo / 'docs/resident-fix/protocol-v1.md') == p['protocol_sha256'], 'Protocol changed')
    require(file_hash(repo / 'configs/resident-fix-selected-v1.json') == p['selected_fix_sha256'], 'Selected implementation changed')
    require(read(run / 'provenance/plan.json') == p, 'Executed plan differs')
    groups = {b['block_id']: b['dataset'] for b in p['banks']}
    require(len(groups) == len(p['banks']) == 17 and Counter(groups.values()) == {'gsm8k':8, 'wildchat':8, 'mbpp':1}, 'Final content allocation differs')
    require(a['schema'] == 'resident_fix_analysis_v1' and a['valid'] is True and a['phase'] == 'final', 'Final analysis is invalid')
    require(all(a[k] == 204 for k in ('expected_records', 'observed_records', 'valid_records')), 'Require all 204 final trials')
    require(a['errors'] == [] and a['missing_keys'] == [] and a['matched_triplets'] == 68 and a['paired_comparisons'] == 204, 'Incomplete final comparison matrix')
    require(r['valid'] is True and r['phase'] == 'final' and r['expected_records'] == r['observed_records'] == 204 and not r['errors'] and not r['missing'], 'Independent record audit failed/incomplete')
    require(r['groups_with_actual_routes'] == 17 and r['prepared_batches_reconstructed'] == 17*4112 and r['exact_greedy_assignment_batches'] == 17*4112, 'Independent route audit coverage differs')
    require(s['passed'] is True and s['comparison']['passed'] is True and s['recomputed']['phase'] == 'final' and s['recomputed']['trial_count'] == 204, 'Independent statistics audit failed/incomplete')
    require(s['compared_analysis_sha256'] == file_hash(paths['analysis']), 'Statistics audit refers to another analysis')
    require(m['schema'] == 'resident_fix_descriptive_service_architecture_v1' and m['phase'] == 'final' and m['complete'] is True and m['trials'] == 204, 'Complete final mechanism report required')
    require(m['descriptive_only'] is True and m['confidence_intervals'] is False and m['hypothesis_tests'] is False, 'Mechanism report must remain descriptive')
    for obj in (a, r, s, m):
        require(obj['plan_sha256'] == p['content_sha256'] and obj['source_sha256'] == p['source_sha256'], 'Artifact plan/source identity differs')
    expected = set(itertools.product(groups, TYPES, LOADS, ARMS))
    keys = [tuple(t[k] for k in ('content_group_id','arrival_type','rho','arm')) for t in a['trial_summaries']]
    require(len(keys) == len(set(keys)) == 204 and set(keys) == expected, 'Trial summaries omit or duplicate cells')
    require(all(t['dataset'] == groups[t['content_group_id']] for t in a['trial_summaries']), 'Dataset label differs')
    pk = [tuple(t[k] for k in ('content_group_id','arrival_type','rho','contrast')) for t in a['paired_cases']]
    require(len(pk) == len(set(pk)) == 204 and set(pk) == set(itertools.product(groups,TYPES,LOADS,CONTRASTS)), 'Paired summaries omit or duplicate cells')
    require(len(m['per_trial_service']) == 204 and set(tuple(t[k] for k in ('content_group_id','arrival_type','rho','arm')) for t in m['per_trial_service']) == expected, 'Mechanism service coverage differs')
    require(set(m['architecture_by_group']) == set(groups), 'Mechanism assignment coverage differs')
    require(all(x['measured_batches'] == 4096 and x['current_fix_exact_assignment_identity'] is True for x in m['architecture_by_group'].values()), 'Mechanism assignment identity differs')
    require(cpu['schema'] == 'resident_fix_cpu_reproduction_v1' and cpu['passed'] is True, 'Clean CPU reproduction has not passed')
    phases = {'calibration-pilot':5, 'pilot':60, 'calibration-final':17, 'final':204}
    require(set(cpu['phases']) == set(phases), 'Require the complete four-phase CPU reproduction')
    for phase, count in phases.items():
        item = cpu['phases'][phase]
        require(item['passed'] is True and item['raw_records'] == count and item['archived_analysis_comparison']['passed'] is True and item['statistics_comparison']['passed'] is True, f'CPU reproduction phase failed: {phase}')
        require(item['source_sha256'] == p['source_sha256'], 'CPU reproduction source differs')
    cp = cpu['phases']['final']
    require(cp['plan_sha256'] == p['content_sha256'] and cp['archived_analysis_comparison']['archived_file_sha256'] == file_hash(paths['analysis']), 'Clean CPU proof refers to another final result')
    c = a['confirmatory']
    require(c['schema'] == 'resident_fix_inference_v1' and c['main_content_groups'] == 16 and c['total_treatment_trials'] == 204 and len(c['guards']) == 10 and len(c['mbpp_sentinel']['checks']) == 6, 'Inference allocation or endpoints differ')
    require(c['bootstrap']['samples'] == 100000 and c['bootstrap']['seed'] == 471712 and c['bootstrap']['strata'] == list(MAIN), 'Frozen bootstrap differs')
    require(c['primary']['ratio_limit'] == .95 and all(g['ratio_limit'] == 1.02 and g['one_sided_confidence_level'] == .995 for g in c['guards']), 'Frozen criteria differ')
    require(c['secondary_current']['gate_passed'] or c['secondary_current']['ci95'] is None, 'Gate-closed secondary has an inferential interval')
    require(c['bootstrap']['draw_indices_sha256'] == s['recomputed']['confirmatory']['bootstrap']['draw_indices_sha256'], 'Independent bootstrap draw identity differs')
    # Bind all accepted summaries to the same unchanged raw bytes. Compression
    # may change stored hashes, but never the decompressed record identity.
    pins = {}
    for item in a['record_files']:
        name = Path(item['path']).name.removesuffix('.gz')
        require(name not in pins, 'Duplicate record-file identity')
        choices = [run/'trials'/name, run/'trials'/(name+'.gz')]
        available = [x for x in choices if x.is_file()]
        require(len(available) == 1, 'Missing or duplicated raw trial artifact: '+name)
        target = available[0]
        with (gzip.open(target,'rb') if target.suffix == '.gz' else target.open('rb')) as handle:
            original = hashlib.file_digest(handle, 'sha256').hexdigest()
        require(original == item['uncompressed_bytes_sha256'], 'Raw record bytes changed: '+name)
        pins[name] = (original, file_hash(target))
    require(len(pins) == 17, 'Require all 17 content-group record files')
    require({k.removesuffix('.gz'):v['original_bytes_sha256'] for k,v in r['record_files'].items()} == {k:v[0] for k,v in pins.items()}, 'Record-audit byte identities differ')
    require({v['file'].removesuffix('.gz'):v['uncompressed_sha256'] for v in s['raw_record_files']} == {k:v[0] for k,v in pins.items()}, 'Statistics-audit byte identities differ')
    require(len(m['records_files']) == 17 and all(Path(k).name.removesuffix('.gz') in pins and v in pins[Path(k).name.removesuffix('.gz')] for k,v in m['records_files'].items()), 'Mechanism report raw bytes differ')
    return data, paths, groups


def hierarchical(rows, value, datasets=MAIN):
    """Types within content, contents within dataset, then equal datasets."""
    ds = []
    for dataset in datasets:
        bundles = defaultdict(list)
        for row in rows:
            if row['dataset'] == dataset:
                v = value(row)
                if v is None:
                    return None  # Never drop zero-denominator or missing pairs.
                require(math.isfinite(v), 'Nonfinite descriptive value')
                bundles[row['content_group_id']].append(v)
        require(bool(bundles), 'Missing descriptive dataset')
        ds.append(statistics.mean(statistics.mean(v) for v in bundles.values()))
    return statistics.mean(ds)


def paired_ratio(rows, metric, datasets=MAIN):
    v = hierarchical(rows, lambda r:r['log_ratios'].get(metric), datasets)
    return math.exp(v) if v is not None else None


def absolute_mean(rows, metric):
    return hierarchical(rows, lambda r:r['metrics'][metric])


def number(value, digits=4):
    return 'N/A' if value is None else f'{value:,.{digits}f}'


def result(value):
    return 'pass' if value is True else 'not established' if value is False else 'not tested'


def ci(value):
    return 'not tested' if value is None else '['+', '.join(number(x) for x in value)+']'


def inline(text):
    text = html.escape(str(text))
    text = re.sub(r'\[([^\]]+)\]\((https://[^\s)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)


class Document:
    def __init__(self):
        self.md, self.html = [], []

    def heading(self, title, level=2):
        self.md += ['#'*level+' '+title, '']
        self.html.append(f'<h{level}>{inline(title)}</h{level}>')

    def paragraph(self, text):
        self.md += [text, '']
        self.html.append('<p>'+inline(text)+'</p>')

    def table(self, headers, rows):
        rows = [list(map(str,row)) for row in rows]
        self.md += ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        self.md += ['| '+' | '.join(v.replace('|','\\|') for v in row)+' |' for row in rows]
        self.md.append('')
        self.html.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')

    def standalone(self):
        return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resident-path repair: complete audited results</title><style>
html{color:#000;background:#fff;font:17px/1.55 Georgia,serif}body{max-width:1120px;margin:40px auto;padding:0 24px}h1,h2,h3{font-family:Arial,sans-serif;line-height:1.2}h1{font-size:2rem}h2{font-size:1.4rem;margin-top:2.2em}p{max-width:96ch}a{color:#000;text-decoration:underline}code{font-size:.85em;overflow-wrap:anywhere}.table-wrap{overflow-x:auto;margin:1.2em 0}table{border-collapse:collapse;width:100%;font:13px/1.4 Arial,sans-serif}th,td{border:1px solid #888;padding:7px 9px;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{color:#000;background:#fff;font-weight:bold}td{white-space:nowrap}thead{display:table-header-group}@media print{body{max-width:none;margin:0;padding:0;font-size:10pt}.table-wrap{overflow:visible}table{font-size:8pt}h2{break-after:avoid}tr{break-inside:avoid}a{color:#000}}
</style></head><body>'''+''.join(self.html)+'</body></html>\n'


def build(data, paths, groups):
    p,a,r,s,m,cpu=(data[k] for k in ('plan','analysis','records','statistics','mechanism','reproduction'))
    c=a['confirmatory'];primary=c['primary'];secondary=c['secondary_current'];sentinel=c['mbpp_sentinel']
    doc=Document();doc.heading('Resident-path repair: complete audited results',1)
    verdict = 'The repair met the frozen advance criteria.' if c['overall_advance']=='pass' else 'The repair did not satisfy the frozen advance criteria.'
    doc.paragraph('**'+verdict+'** Main acceptance: `'+c['main_acceptance']+'`; MBPP descriptive sentinel: '+('pass' if sentinel['passes'] else 'point limit exceeded')+'; overall decision: `'+c['overall_advance']+'`. These decisions use every prespecified final case.')
    doc.paragraph(f"At offered load 0.85, the main fix/stockoff mean-response ratio was **{number(primary['ratio'])}**, with conditional 95% interval **{ci(primary['ci95'])}**. The useful-gain criterion requires the interval's upper bound to be strictly below 0.95 (at least 5% lower response). Its result was **{result(primary['passes'])}**. A failure of this target alone is not proof of regression.")
    if primary['ci95'][0] > 1:
        doc.paragraph('Here the entire primary interval is above 1, supporting higher mean response than stockoff within this conditional design. The practical recommendation for this tested configuration remains stock EPLB-off.')
    elif primary['ci95'][1] < 1:
        doc.paragraph('The primary interval is below 1, supporting lower mean response within this conditional design. Whether that improvement meets the useful-gain and regression criteria is assessed separately below.')
    else:
        doc.paragraph('The primary interval includes 1. This experiment does not resolve the direction of the main stockoff response difference under its declared uncertainty model.')
    doc.paragraph(f"The fix/current high-load mean-response point ratio was **{number(secondary['ratio'])}**. "+(f"The hierarchical secondary test was eligible; its 95% interval was {ci(secondary['ci95'])} and its result was {result(secondary['passes'])}." if secondary['gate_passed'] else 'The inferential gate was closed. This point is descriptive; no secondary confidence interval or significance claim is assigned.'))
    doc.paragraph('This is one eager BF16/B128 Qwen3-30B-A3B MoE layer on the same two H100s. Stock gate/top-k, native Triton experts and output all-reduce execute in every arm. Inputs are replicated across ranks. The result concerns completed FCFS layer jobs, not full-model chat latency, generated-token throughput or token all-to-all.')

    doc.heading('Frozen decisions and uncertainty')
    doc.table(['Endpoint','Ratio','95% interval','Required upper bound','Status'],[
        ['Fix / stockoff, high-load mean',number(primary['ratio']),ci(primary['ci95']),'< 0.95',result(primary['passes'])],
        ['Fix / current, gated high-load mean',number(secondary['ratio']),ci(secondary['ci95']),'< 0.95',result(secondary['passes'])]])
    doc.paragraph('Main inference uses 16 whole content bundles: eight GSM8K and eight WildChat. Each bundle retains both arrival types, both loads and all arms. Paired log ratios are averaged across types, groups within dataset, and then the two dataset means equally. The 100,000 PCG64 bootstrap draws use seed 471712. Intervals are conditional on these constructed payloads, assigned public-calendar segments and one allocation. They do not quantify new-day, hardware-population or production-request uncertainty.')
    doc.table(['Guard','Fix/control ratio','One-sided 99.5% UCB','Limit','Status'],[[g['name'].replace('_',' '),number(g['ratio']),number(g['upper_confidence_bound']),'≤ 1.02',result(g['passes'])] for g in c['guards']])
    doc.paragraph('All ten guards belong to the frozen Bonferroni family (0.05/10); the bound must be at most 1.02. The 5% useful-gain and 2% regression margins are carried-forward engineering criteria, separate from numerical tolerances.')
    robustness=c['calendar_date_robustness'];bydate=defaultdict(dict)
    for row in robustness['results']:bydate[row['excluded_date']][row['control']]=row
    doc.table(['Actual date removed','Fix / stockoff','Fix / current','Both point limits ≤ 1.02'],[[day,number(v['stockoff']['ratio']),number(v['current']['ratio']),'pass' if v['stockoff']['passes'] and v['current']['passes'] else 'limit not met'] for day,v in sorted(bydate.items())])
    doc.paragraph('Date deletion removes each actual source date jointly across services, retains surviving types/groups with the frozen weighting, and fails if a dataset disappears. These are point sensitivity checks, not new confidence intervals or a new-day generalization test.')

    doc.heading('Matched main outcomes at both loads')
    doc.paragraph('Loads 0.5 and 0.85 normalize each group’s common absolute arrival rate to its stockoff-only saturation calibration. They are not measured GPU utilization. All arms receive the same immutable calendar, retain original relative arrival structure, and drain every admitted job.')
    metrics=[('response_mean_ns','Response mean'),('response_p95_ns','Trial response p95'),('service_mean_ns','Service mean'),('queue_mean_ns','Queue mean'),('completed_elapsed_ns','Completed extent'),('throughput_jobs_per_second','Finite-episode throughput')]
    rows=[]
    for rho,contrast in itertools.product(LOADS,CONTRASTS):
        subset=[x for x in a['paired_cases'] if x['rho']==rho and x['contrast']==contrast and x['dataset'] in MAIN]
        require(len(subset)==32,'Main descriptive pair count differs')
        rows.append([str(rho),contrast.replace('_over_',' / ')]+[number(paired_ratio(subset,key)) for key,_ in metrics])
    doc.table(['Load','Treatment / control']+[name for _,name in metrics],rows)
    doc.paragraph('These are hierarchically aggregated matched geometric ratios over the 16 main bundles, without additional intervals. Lower latency/time ratios favor the numerator; higher throughput ratios favor it. Finite-episode throughput includes the offered horizon and drain and is not saturated capacity. Queue ratios would be N/A if any paired denominator/log ratio were undefined; no pair would be silently removed.')
    absolute=[('response_mean_ns','Response mean ms',1e6),('response_p95_ns','Mean trial p95 ms',1e6),('service_mean_ns','Service mean ms',1e6),('queue_mean_ns','Queue mean ms',1e6),('completed_elapsed_ns','Extent s',1e9),('drain_including_controller_ns','Drain ms',1e6),('controller_tail_elapsed_ns','Tail ms',1e6),('throughput_jobs_per_second','Jobs/s',1)]
    rows=[]
    for rho,arm in itertools.product(LOADS,ARMS):
        subset=[x for x in a['trial_summaries'] if x['rho']==rho and x['arm']==arm and x['dataset'] in MAIN]
        require(len(subset)==32,'Main descriptive absolute count differs')
        rows.append([str(rho),arm]+[number(absolute_mean(subset,key)/scale,3) for key,_,scale in absolute])
    doc.table(['Load','Arm']+[name for _,name,_ in absolute],rows)
    doc.paragraph('Absolute values are arithmetic means of trial summaries, with the same type/group/dataset hierarchy. “Mean trial p95” averages each completed trial’s p95; it is not a pooled-job p95. Drain is post-horizon extent through normal quiescence; tail is final-output-to-quiescence time. They are not isolated migration cost or GPU utilization.')

    doc.heading('Every workload cell, including regressions')
    rows=[]
    for dataset,typ,rho in itertools.product(DATASETS,TYPES,LOADS):
        values=[]
        for contrast in CONTRASTS:
            subset=[x for x in a['paired_cases'] if (x['dataset'],x['arrival_type'],x['rho'],x['contrast'])==(dataset,typ,rho,contrast)]
            values += [number(paired_ratio(subset,key,(dataset,))) for key in ('response_mean_ns','response_p95_ns')]
        rows.append([dataset+(' (sentinel)' if dataset=='mbpp' else ''),typ,str(rho),'1' if dataset=='mbpp' else '8']+values)
    doc.table(['Dataset','Arrival','Load','Bundles','Fix/off mean','Fix/off p95','Fix/current mean','Fix/current p95','Current/off mean','Current/off p95'],rows)
    doc.paragraph('All cells are descriptive; ratios above 1 remain visible. No subgroup interval or significance claim is added. MBPP cells each use one held-out bundle and remain outside the main bootstrap.')
    doc.heading('MBPP operational sentinel')
    doc.table(['Control','Load','Endpoint','Point ratio','Limit','Status'],[[x['control'],str(x['rho']),x['metric'].replace('_ns','').replace('_',' '),number(x['ratio']),'≤ 1.02','pass' if x['passes'] else 'point limit exceeded'] for x in sentinel['checks']])
    doc.paragraph('These six checks average the two arrival types for one 32-document MBPP bundle. They have no confidence intervals or MBPP-population nonregression guarantee. The sentinel gates overall advance separately and cannot rescue or alter the main inference.')

    doc.heading('What changed, and what the measurements identify')
    doc.paragraph('The 60-run development factorial held the 160-slot placement and exact physical assignments fixed. Its matched package effects were 51.872 µs/job for recording/control and 35.154 µs/job for online selection. They are coordinator-service contrasts between implementation packages, not exclusive kernel/CPU component times. Offline known-home assignments were diagnostic, not an online competitor or strict runtime lower bound.')
    doc.paragraph('The selected repair makes residency explicitly static, fuses histogram/home choice/remapping into one launch, and reuses output storage on one sequential inference stream. It preserves the current method’s exact physical assignments. In a separate 36-trial development check, service ratio was 0.892074 versus current and 1.021542 versus stockoff; every development group still lost to stockoff. This motivated fresh confirmation without establishing a repair advantage over stockoff. The repair changes the controller package and selector implementation together; it is not an isolated fusion ablation.')
    doc.table(['Arm','Resident slots total / rank','Recording and control','Physical assignment'],[
        ['stockoff','128 / 64','Genuine upstream EPLB off','Native logical placement'],
        ['current','160 / 80','16-job observation window; 64-job forced-SKIP controller','Current online home policy'],
        ['fix','160 / 80','Inactive; enabled-layout buffers remain allocated','Exactly the same policy, fused implementation']])
    doc.paragraph('A common maximum allowance of 32 replicas is not equal resident memory. The fix is custom static residency, not supported upstream EPLB-off. Its comparison with stockoff also changes placement and physical geometry; the residual time difference cannot be assigned solely to selector overhead.')
    rows=[]
    for arm in ARMS:
        v=m['group_summaries']['main'][arm]
        require(v['trials']==64 and v['content_groups']==16,'Mechanism main denominator differs')
        rows.append([arm]+[number(v[phase]['mean_service_ms'],6) for phase in ('all','nonrecording_ordinary','recording_ordinary','boundary')])
    doc.table(['Main arm','All jobs ms','Positions 0–47 ms','Positions 48–62 ms','Position 63 ms'],rows)
    doc.paragraph('Actual final coordinator-service means at matched positions modulo 64, averaged over all main groups, loads and arrival types. Each trial contributes 3,072 nonrecording-position jobs, 960 recording-position jobs and 64 boundary-position jobs. The labels describe the current controller; stockoff/fix use the same positions without that controller. Temporal associations are not exclusive causal stage costs.')
    archkeys=[('global_active_copies','Global active copies'),('max_rank_active_copies','Max-rank active copies'),('max_rank_routes','Max-rank routes'),('global_nonempty_m_tiles','Global nonempty M tiles'),('global_padded_rows','Padded rows'),('counted_tile_flops_proxy','Counted-tile GFLOP proxy'),('useful_to_counted_tile_flops_fraction','Useful/proxy fraction')]
    archrows=[]
    main_groups=[g for g,d in groups.items() if d in MAIN]
    for arm in ARMS:
        vals=[]
        for key,_ in archkeys:
            vals.append(statistics.mean(statistics.mean(m['architecture_by_group'][g]['per_arm'][arm][key]['mean'] for g in main_groups if groups[g]==dataset) for dataset in MAIN))
        vals[5]/=1e9
        archrows.append([arm]+[number(v,4) for v in vals])
    doc.table(['Arm']+[label for _,label in archkeys],archrows)
    equal=all(all(m['architecture_by_group'][g]['all_arms_equal_each_batch'].values()) for g in main_groups)
    doc.paragraph(f"Actual-route work-proxy equality on every main measured batch (global active copies, nonempty M tiles, padded rows and counted-tile FLOPs) across all three arms: **{str(equal).lower()}**. Current/fix assignment equality is required independently. These are structural proxies, not measured instructions, SM/tensor-core utilization, HBM traffic or critical-path time. Equal global work does not imply equal launch geometry, cache behavior or latency.")
    cfg=m['observed_kernel_configs'];require(all(cfg[x]==cfg['stockoff'] for x in ARMS),'Unexpected differing configs')
    doc.paragraph('The attested common native configuration is `'+', '.join(f'{k}={v}' for k,v in sorted(cfg['stockoff'].items()))+'`. No tile override was applied in this study.')
    rows=[]
    for arm in ARMS:
        v=m['source_derived_launch_extent'][arm]
        require(v['early_return_programs_included'] is True and v['measured_profiler_launch_counts'] is False,'Launch extent mislabeled')
        rows.append([arm]+[str(v[k]) for k in ('allocated_sorted_id_elements_source_derived','allocated_m_grid_positions_source_derived','gemm1_grid_programs_per_rank_source_derived','gemm2_grid_programs_per_rank_source_derived')])
    doc.table(['Arm','Sorted-ID allocation','M-grid positions','GEMM1 programs/rank','GEMM2 programs/rank'],rows)
    doc.paragraph('Launch extents above are derived from pinned source and actual configuration, including early-return programs; they are not Nsight-observed grid counts or causal measured overhead. They cannot turn a count difference into an attributed runtime cost.')

    doc.heading('Data, correctness and reproducibility')
    doc.paragraph('Final inputs comprise 544 fresh documents: 256 GSM8K, 256 WildChat and 32 MBPP, in 16 main groups plus one descriptive sentinel. There are 204 final trials (192 main, 12 sentinel). Across the full bounded study, 96 development executions, 264 fresh treatment trials and 22 baseline calibrations total 382 executions; capture/preparation/correctness work is separate. Pilot and final captures together contain 704 documents and 62,931 source-position records. Freshness excludes the project’s earlier selected identities/prefixes, not model pretraining or every possible semantic overlap.')
    doc.paragraph('Final source-document and Azure arrival-row intervals are disjoint from earlier selections. The public weeks and services are reused. Payloads and calendars are paired independently, so original production content–arrival dependence and original request lengths are absent. Every trial reuses a bounded bank in 4,096 constructed B128 jobs. Sixteen distinct conditioning-input combinations are cycled for 64 untimed forwards before state reset. Materialization, host-to-device transfer, reference computation and conditioning are excluded from timing.')
    doc.paragraph(f"The complete final independent record audit reconstructed {r['prepared_batches_reconstructed']:,} prepared batches and {r['exact_greedy_assignment_batches']:,} greedy-assignment batches across 17 groups. GPU-recorded numerical coverage is all 4,112 prepared batches/group for current/fix against stockoff, 64 sampled stockoff FP32 token positions/group, and 16 whole postflight batches/trial, with resident map/weight checks and adversarial checks. Tolerances are `atol=0.02, rtol=0.03`; this joint criterion is not a universal absolute-error ceiling or downstream answer-quality result.")
    doc.paragraph('The selected fused path is limited to BF16/B128/top-8, the exact frozen 160-slot map, and sequential reuse on one constructing CUDA stream. Enabled-layout buffers remain allocated. Reported process peaks include comparator layers, prepared inputs and references and are not standalone engine memory. No claim is made for other model layers, batch sizes, models, hardware, all-to-all backends or concurrent serving.')
    doc.paragraph(f"The clean CPU replay passed all four phases (5 pilot calibrations, 60 pilot treatments, 17 final calibrations and 204 final treatments), with exact archived-analysis agreement after path/compression normalization. Independently implemented statistics compared {s['comparison']['scalar_comparisons']:,} scalars; maximum absolute numerical difference was {s['comparison']['maximum_absolute_numeric_difference']:.3g}. This is same-team replay of the same artifacts, not external-laboratory reproduction or an independent GPU replication.")
    doc.paragraph('Earlier negative results remain part of the evidence: '+link('migration/payback evaluation','docs/payback-final-evaluation.md')+', '+link('tile-tuning evaluation','docs/optimization-evaluation.md')+', and '+link('stock-vLLM comparator study','docs/stock-comparison/results.md')+'. This experiment does not retroactively change their designs or conclusions.')
    doc.paragraph('Source and evidence: '+link('branch source','resident_fastpath',True)+'; '+link('frozen protocol','docs/resident-fix/protocol-v1.md')+'; '+link('selected implementation','configs/resident-fix-selected-v1.json')+'; '+link('final raw artifacts and audits','results/resident-fix/final-v1',True)+'; '+link('clean CPU proof','results/resident-fix/clean-cpu-reproduction-v1',True)+'; '+link('reproduction guide','docs/resident-fix/reproduction.md')+'; '+link('data provenance and terms','docs/resident-fix/data-provenance.md')+'.')
    doc.paragraph('Frozen source `'+p['source_sha256']+'`; final plan `'+p['content_sha256']+'`; protocol `'+p['protocol_sha256']+'`; bootstrap draw digest `'+c['bootstrap']['draw_indices_sha256']+'`. The capture ledger retains the archive-only failure after five valid pilot banks; no bank was recaptured or replaced.')
    doc.table(['Required evidence artifact','SHA256'],[[name,file_hash(path)] for name,path in paths.items()])
    return '\n'.join(doc.md), doc.standalone()


def self_test():
    rows=[dict(dataset=d,content_group_id=d+str(g),log_ratios={'x':math.log(v)}) for d,v in [('gsm8k',.5),('wildchat',2.)] for g in range(8) for _ in TYPES]
    require(math.isclose(paired_ratio(rows,'x'),1.), 'Hierarchical geometric aggregation failed')
    rows[0]['log_ratios']={};require(paired_ratio(rows,'x') is None,'Undefined ratios must not be dropped')
    doc=Document();doc.heading('Synthetic <integrity>',1);doc.table(['Value'],[['<unsafe>']])
    require('&lt;unsafe&gt;' in doc.standalone() and '<unsafe>' not in doc.standalone(),'HTML escaping failed')
    print('Basic aggregation and HTML integrity checks passed; no experiment data read.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--results-root',type=Path)
    parser.add_argument('--proof-dir',type=Path)
    parser.add_argument('--markdown-output',type=Path)
    parser.add_argument('--html-output',type=Path)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    repo=args.repo_root.resolve();results=(args.results_root or repo/'results/resident-fix').resolve()
    proof=(args.proof_dir or results/'clean-cpu-reproduction-v1').resolve()
    md=(args.markdown_output or repo/'docs/resident-fix/results.md').resolve()
    output=(args.html_output or repo/'docs/resident-fix/results.html').resolve()
    require(md != output,'Report outputs must differ')
    require(md.suffix.lower()=='.md' and output.suffix.lower()=='.html','Expected Markdown and HTML output suffixes')
    for path in (md,output):require(args.overwrite or not path.exists(),'Report output exists: '+str(path))
    data,paths,groups=check_inputs(repo,results,proof)
    for path in (md,output):
        require(path not in [p.resolve() for p in paths.values()],'Output would overwrite evidence')
        require(not path.is_relative_to(results/'final-v1') and not path.is_relative_to(proof),'Output would modify evidence directory')
        require(path not in [(repo/p).resolve() for p in data['plan']['source_files']],'Output would overwrite frozen source')
        require(path not in [(repo/'docs/resident-fix/protocol-v1.md').resolve(),(repo/'configs/resident-fix-selected-v1.json').resolve()],'Output would overwrite frozen protocol/config')
    markdown,standalone=build(data,paths,groups)
    for path,content in ((md,markdown),(output,standalone)):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print(json.dumps(dict(rendered=True,complete_trials=204,main_groups=16,markdown=str(md),html=str(output),renderer_sha256=file_hash(__file__))))


if __name__=='__main__':main()
