#!/usr/bin/env python3
"""Independent raw-clock statistics audit for the three-arm composition study.

No production analysis/experiment module is imported. Earlier independent
auditors supply timestamp arithmetic and discrete bootstrap enumeration. This
script independently validates three-arm units and performs the frozen PCG64
bootstrap, guard family, and hierarchical gate. It is not a GPU rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('independent_previous_arrival_statistics',
    ROOT/'scripts/verify_arrival_fix_statistics.py')
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
clocks = previous.clocks
require, compare, digest, average = previous.require, previous.compare, previous.digest, previous.average
ARMS = ('native','shortcut','combined')
KEY = ('arrival_block_id','batch_tokens','pattern','rho','repetition','arm')
UNIT = KEY[:-1]
METRICS = ('response_mean_ns','response_p95_ns','service_mean_ns')
CONTRASTS = {'combined_over_shortcut':('combined','shortcut'),
             'combined_over_native':('combined','native'),
             'shortcut_over_native':('shortcut','native')}
SETTINGS = dict(bootstrap_samples=100000,bootstrap_seed=471432,
    bootstrap='stratified_arrival_blocks',primary_ratio_limit=.95,
    guard_ratio_limit=1.02,guard_family_size=3,secondary_ratio_limit=.95)
PAIRED = tuple(f for f in previous.PAIRED if f != 'assignment_sha256')


def bootstrap_draws(strata):
    names,groups = previous.stratum_members(strata)
    rng = np.random.Generator(np.random.PCG64(471432))
    draws = np.empty((100000,8),dtype=np.int64)
    for column,members in enumerate(groups):
        bits = rng.integers(0,2,size=(100000,2))
        draws[:,column*2:column*2+2] = np.asarray(members,dtype=np.int64)[bits]
    return names,groups,draws


def exact_distribution(values,groups,draws,quantile):
    evidence = previous.exact_distribution(values,groups,draws)
    cumulative = 0
    for point in sorted(evidence['support'],key=lambda x:x['ratio']):
        cumulative += point['multiplicity']
        if cumulative/256 >= quantile:
            evidence['exact_requested_upper_quantile'] = point['ratio']
            break
    evidence['requested_upper_probability'] = quantile
    return evidence


def raw_triplets(plan,records):
    """Compute every trial endpoint from clocks before any paired aggregation."""
    plan_hash = digest({k:v for k,v in plan.items() if k != 'freeze'})
    require(plan['phase']=='final' and plan['freeze']['sha256']==plan_hash,'Frozen final plan')
    compare(plan['analysis'],SETTINGS,'Prespecified inference settings')
    require(set(plan['analysis'])==set(SETTINGS),'No additional inference settings')
    matrix = plan['matrix']
    compare(matrix,dict(batches=[128,1024],patterns=['spread','cluster'],rhos=[.5,.85],
        reps=2,arms=list(ARMS)),'Fixed three-arm matrix')
    blocks = {b['id']:b for b in matrix['arrival_blocks']}
    require(len(blocks)==len(matrix['arrival_blocks'])==8 and len({b['seed'] for b in blocks.values()})==8,
        'Eight distinct calendar blocks and seeds')
    strata = {key:(b['block_id'],b['regime']) for key,b in blocks.items()}
    previous.stratum_members(strata)
    hardware = plan['hardware']
    require(len(set(hardware['gpu_uuids']))==2 and bool(hardware['hostname']),'Resolved two-GPU hardware')
    expected = set(itertools.product(sorted(blocks),(128,1024),('spread','cluster'),(.5,.85),(0,1),ARMS))
    require(len(records)==384,'Exactly384 final trials')
    schedules = {}
    expected_order_keys = set()
    for block,batch in itertools.product(blocks,(128,1024)):
        entries = plan['schedules'][f'{block}/B{batch}']
        require(len(entries)==4 and {(s['pattern'],s['rho']) for s in entries}
            == set(itertools.product(('spread','cluster'),(.5,.85))),'Complete frozen calendars')
        for entry in entries:
            require(entry['schedule_sha256']==digest({k:v for k,v in entry.items()
                if k!='schedule_sha256'}),'Calendar checksum')
            schedules[block,batch,entry['pattern'],entry['rho']] = entry
            order_key = f"{block}/B{batch}/{entry['pattern']}/rho{entry['rho']}"
            expected_order_keys.add(order_key)
            orders = plan['arm_orders'][order_key]
            require(len(orders)==2 and set(orders[0])==set(ARMS) and len(orders[0])==3
                and orders[1]==orders[0][::-1],'Frozen three-arm reversed order')
    require(set(plan['arm_orders'])==expected_order_keys,'Exact frozen order inventory')
    index,flat,actual_orders = {},{},defaultdict(list)
    trials = []
    for row in records:
        key = tuple(row.get(f) for f in KEY)
        require(key in expected and key not in index,'Duplicate or unexpected trial')
        index[key] = row
        actual_orders[key[:-1]].append(row['arm'])
        info = blocks[key[0]]
        for field,value in dict(arrival_combo_schema_version=1,phase='final',status='complete',config_hash=plan_hash,
            arrival_combo_source_sha256=plan['arrival_combo_source_sha256'],correctness_pass=True,
            evidence_eligible=True,diagnostic_only=False,normal_completion_verified=True,
            resident_state_verified=True,rank_assignment_agreement=True,guard_symbols_restored=True,
            migration_bytes=0,migration_sent_messages=0,terminal_proposals_suppressed=1,policy='SKIP',
            assignment_selection_cost_included=True,arrival_seed=info['seed'],
            block_id=info['block_id'],regime=info['regime']).items():
            compare(row.get(field),value,'Eligible row/'+field)
        combined = row['arm']=='combined'
        compare(row,dict(dispatch_mode='online' if combined else 'native',
            noop_enabled=row['arm']!='native',online_residency_metadata_verified=combined,
            mapper_calls_measured=256 if combined else 0),'Treatment identity')
        require(row.get('preflight_correctness',{}).get('correctness_pass') is True,'Preflight recorded')
        require(row.get('source_map') is not None and row['source_map']==row.get('final_map'),'Fixed resident map')
        require(sorted(row.get('physical_gpu_uuids',[]))==sorted(hardware['gpu_uuids'])
            and row.get('hostname','').split('.')[0]==hardware['hostname'],'Resolved physical hardware')
        require(sorted(row.get('rank_gpu_uuids',[]))==sorted(hardware['gpu_uuids']),
            'Actual CUDA rank device identities')
        calkey = f"{info['block_id']}/{info['regime']}/B{key[1]}"
        require(bool(row.get('allocation_id')) and str(row['allocation_id'])
            == str(plan['calibration'][calkey]['allocation_id']),'Same allocation as calibration')
        clock = row.get('arrival_clock_contract',{})
        require(clock.get('monotonic') is True and clock.get('adjustable') is False
            and 'CLOCK_MONOTONIC' in clock.get('implementation',''),'Shared monotonic clock')
        schedule = schedules[key[:4]]
        require(row['arrival_schedule']==schedule and row['arrival_offsets_ns']==schedule['arrival_offsets_ns']
            and row['horizon_ns']==schedule['horizon_ns'],'Immutable scheduled arrivals')
        order_key = f'{key[0]}/B{key[1]}/{key[2]}/rho{key[3]}'
        require(row['arm_order']==plan['arm_orders'][order_key][key[4]],'Frozen arm order')
        guards = row.get('noop_guard_ranks')
        if row['arm']=='native':
            require(guards==[],'Unwrapped native baseline')
        else:
            require(isinstance(guards,list) and len(guards)==2,'Both shortcut ranks required')
            required = dict(mode='fast',diagnostic_timing=False,pending_pair=False,movement_calls=4,
                commit_calls=4,agreement_calls=4,local_unchanged_calls=4,unanimously_unchanged_calls=4,
                movement_skips=4,commit_skips=4,original_movement_calls=0,original_commit_calls=0,
                fallback_calls=0,diagnostics=[])
            for guard in guards:compare(guard,required,'Complete fast-path counters')
        flat[key] = previous.flat_metrics(row)
        trials.append({**dict(zip(KEY,key)),'block_id':info['block_id'],'regime':info['regime'],
            'allocation_id':row['allocation_id'],'metrics':flat[key],
            'observed_window_spans':clocks.window_spans(row)})
    require(set(index)==expected,'Complete384-trial matrix')
    allocations = sorted({str(row['allocation_id']) for row in records})
    require(len(allocations)==1,'One conditional allocation')
    triplets = []
    for key in sorted({k[:-1] for k in expected}):
        selected = {arm:index[(*key,arm)] for arm in ARMS}
        for field in PAIRED:
            require(all(field in row for row in selected.values())
                and all(row[field]==selected['native'][field] for row in selected.values()),
                'Common logical/source/placement identity/'+field)
        require(selected['native']['assignment_sha256']==selected['shortcut']['assignment_sha256'],
            'Shortcut must preserve native physical assignment')
        require(actual_orders[key]==selected['native']['arm_order'],'Actual execution order matches freeze')
        metrics = {arm:flat[(*key,arm)] for arm in ARMS}
        require(all(metrics[arm][metric]>0 for arm in ARMS for metric in METRICS),'Positive trial endpoints')
        logs = {contrast:{metric:math.log(metrics[num][metric])-math.log(metrics[den][metric])
            for metric in METRICS} for contrast,(num,den) in CONTRASTS.items()}
        triplets.append({**dict(zip(UNIT,key)),'block_id':selected['native']['block_id'],
            'regime':selected['native']['regime'],'allocation_id':selected['native']['allocation_id'],
            'arms':metrics,'log_ratios':logs})
    for key in {k[:4] for k in expected}:
        require(actual_orders[(*key,0)]==actual_orders[(*key,1)][::-1],'Technical repetitions reverse local order')
    positions = {arm:[0,0,0] for arm in ARMS}
    for order in actual_orders.values():
        for position,arm in enumerate(order):positions[arm][position]+=1
    return dict(plan_sha256=plan_hash,arrival_combo_source_sha256=plan['arrival_combo_source_sha256'],
        trial_summaries=trials,triplets=triplets,block_strata=strata,allocation_ids=allocations,
        actual_arm_position_counts=positions)


def inference(triplets,strata):
    names,groups,draws = bootstrap_draws(strata)
    expected = set(itertools.product(names,(128,1024),('spread','cluster'),(.5,.85),(0,1)))
    index = {}
    for case in triplets:
        key = tuple(case[k] for k in UNIT)
        require(key in expected and key not in index,'Unique matched statistical triplet')
        require(set(case['log_ratios'])==set(CONTRASTS),'All three prespecified contrasts')
        index[key] = case
    require(set(index)==expected,'Complete128-triplet matrix; repetitions/jobs are not blocks')
    exact = {}
    def logs(contrast,rho,metric,batches=(128,1024)):
        return np.asarray([average(average(index[block,batch,pattern,rho,rep]
            ['log_ratios'][contrast][metric] for rep in (0,1))
            for batch,pattern in itertools.product(batches,('spread','cluster')))
            for block in names],dtype=np.float64)
    def endpoint(values,name,contrast,guard=False):
        require(np.isfinite(values).all(),'Finite block log contrasts')
        samples = np.exp(values[draws].mean(axis=1))
        q = 1-.05/3 if guard else .975
        lower,upper = [float(v) for v in np.quantile(samples,[.025,q],method='linear')]
        limit = 1.02 if guard else .95
        passed = upper <= limit if guard else upper < limit
        item = dict(name=name,contrast=contrast,ratio=float(np.exp(values.mean())),
            block_log_ratios=dict(zip(names,map(float,values))),upper_confidence_bound=upper,
            ratio_limit=limit,arrival_blocks=8,passes=bool(passed),
            criterion='upper <= limit' if guard else 'upper < limit',
            status='pass' if passed else 'not_established')
        if guard:
            item.update(one_sided_confidence_level=q,guard_family_size=3,
                confidence='one-sided percentile 98.333333%; Bonferroni for three shortcut-relative guards')
        else:
            item.update(ci95=[lower,upper],confidence='two-sided percentile 95%')
        evidence = exact_distribution(values,groups,draws,q)
        evidence['monte_carlo_minus_exact_requested_upper'] = upper-evidence['exact_requested_upper_quantile']
        exact[name] = evidence
        return item
    primary = endpoint(logs('combined_over_shortcut',.85,'response_mean_ns'),
        'high_load_mean_response','combined_over_shortcut')
    guards = [endpoint(logs('combined_over_shortcut',rho,metric,batches),name,
        'combined_over_shortcut',guard=True) for name,rho,metric,batches in (
        ('low_load_mean_response',.5,'response_mean_ns',(128,1024)),
        ('high_load_p95_response',.85,'response_p95_ns',(128,1024)),
        ('b128_high_load_mean_response',.85,'response_mean_ns',(128,)))]
    gate = primary['passes'] and all(g['passes'] for g in guards)
    native_logs = logs('combined_over_native',.85,'response_mean_ns')
    if gate:
        secondary = endpoint(native_logs,'high_load_mean_response_baseline','combined_over_native')
        secondary.update(gate_passed=True,confirmatory=True)
    else:
        # Deliberately do not construct a secondary sampling distribution when
        # the predeclared gate is closed, even inside this audit artifact.
        secondary = dict(name='high_load_mean_response_baseline',contrast='combined_over_native',
            ratio=math.exp(average(native_logs)),arrival_blocks=8,
            block_log_ratios=dict(zip(names,map(float,native_logs))),ci95=None,
            upper_confidence_bound=None,ratio_limit=.95,gate_passed=False,
            confirmatory=False,passes=None,status='not_tested_gate_closed',
            confidence='point estimate only; hierarchical confirmation gate did not pass')
    return dict(primary=primary,guards=guards,secondary_baseline=secondary,
        incremental_acceptance='pass' if gate else 'not_established',
        acceptance='pass' if gate and secondary['passes'] else 'not_established',
        bootstrap={**SETTINGS,'rng':'numpy.random.default_rng/PCG64','quantile_method':'linear',
            'strata':{block:list(strata[block]) for block in names},
            'resampling':'two arrival blocks within each fixed source/domain stratum; all nested cells/repeats retained',
            'draw_indices_sha256':hashlib.sha256(draws.tobytes()).hexdigest()},
        hierarchy='Test combined/native high-load mean only after incremental primary and all three guards pass. Baseline gains cannot rescue failure to establish incremental dispatcher value.'),exact


def recompute(plan,records):
    raw = raw_triplets(plan,records)
    confirmatory,exact = inference(raw['triplets'],raw['block_strata'])
    pairs = []
    for triplet in raw['triplets']:
        for contrast,(treatment,control) in CONTRASTS.items():
            pairs.append({**{k:triplet[k] for k in UNIT},'contrast':contrast,
                'control_arm':control,'treatment_arm':treatment,'block_id':triplet['block_id'],
                'regime':triplet['regime'],'allocation_id':triplet['allocation_id'],
                'control':triplet['arms'][control],'treatment':triplet['arms'][treatment],
                'log_ratios':triplet['log_ratios'][contrast]})
    cells = []
    for batch,pattern,rho in itertools.product((128,1024),('spread','cluster'),(.5,.85)):
        selected = [t for t in raw['triplets'] if (t['batch_tokens'],t['pattern'],t['rho'])==(batch,pattern,rho)]
        arms = {arm:{metric:average(t['arms'][arm][metric] for t in selected)
            for metric in selected[0]['arms'][arm]} for arm in ARMS}
        contrasts = {name:dict(paired_repetitions=16,ratios={metric:math.exp(average(
            t['log_ratios'][name][metric] for t in selected)) for metric in METRICS}) for name in CONTRASTS}
        cells.append(dict(batch_tokens=batch,pattern=pattern,rho=rho,arrival_blocks=8,
            arms=arms,contrasts=contrasts,
            inference='descriptive only; no cell-level confidence interval or acceptance test'))
    return dict(analysis_schema_version=1,phase='final',valid=True,expected_records=384,
        observed_records=384,valid_records=384,matched_pairs=384,
        pairs_per_contrast={name:128 for name in CONTRASTS},matched_triplets=128,
        missing_keys=[],errors=[],plan_sha256=raw['plan_sha256'],
        arrival_combo_source_sha256=raw['arrival_combo_source_sha256'],
        allocation_ids=raw['allocation_ids'],trial_summaries=raw['trial_summaries'],paired_cases=pairs,
        arm_position_counts=raw['actual_arm_position_counts'],confirmatory=confirmatory,
        descriptive_cells=cells),exact


def verify(plan,records,summary):
    expected,exact = recompute(plan,records)
    lists = ('trial_summaries','paired_cases','descriptive_cells')
    for field,keys in ((lists[0],KEY),(lists[1],(*UNIT,'contrast')),
                      (lists[2],('batch_tokens','pattern','rho'))):
        reported = summary.get(field,[])
        indexed = {tuple(row[k] for k in keys):row for row in reported}
        require(len(indexed)==len(reported)==len(expected[field]),'Unique complete summary/'+field)
        for item in expected[field]:
            key = tuple(item[k] for k in keys)
            require(key in indexed,'Missing summary unit/'+field)
            compare(indexed[key],item,'Independent '+field+'/'+str(key))
    compare(summary,{k:v for k,v in expected.items() if k not in lists},'Independent summary')
    secondary = expected['confirmatory']['secondary_baseline']
    if not secondary['gate_passed']:
        require(set(summary['confirmatory']['secondary_baseline'])==set(secondary),
            'Gate-closed secondary must not add unplanned confidence or testing fields')
    return dict(valid=True,expected_records=384,observed_records=len(records),matched_triplets=128,
        matched_pairs=384,pairs_per_contrast={name:128 for name in CONTRASTS},
        independent_calendar_blocks=8,fixed_payload_strata=4,jobs_per_trial=256,
        confirmatory=expected['confirmatory'],exact_weighted_distributions=exact,
        arm_position_counts=expected['arm_position_counts'],allocation_ids=expected['allocation_ids'],
        plan_sha256=expected['plan_sha256'],hardware=plan['hardware'],
        checks=['integer raw-clock FCFS envelopes; all256jobs in each384trial',
            'all384 trial endpoints and384 contrast pairs reproduced independently',
            'frozen and actual three-arm order, reversed repeats and shared calendar/state',
            'distinct dispatch identity allowed only for combined; native/shortcut assignment match',
            'eight whole calendar blocks stratified within four fixed payload states',
            '100000 PCG64 draws seed471432; linear percentiles and exact81 weighted outcomes',
            'primary combined/shortcut upper95% strict<.95; three98.333333% guards<=1.02',
            'secondary combined/native inference only after the incremental gate passes'],
        limitations=['Recorded numerical/route flags are checked; no GPU execution is reproduced.',
            'Use the separate execution auditor for NPZ routes, weights, calendar generation and source provenance.',
            'Inference is conditional on four reused payload strata, native calibration and one allocation.',
            'Exact weighted quantiles supplement the frozen Monte Carlo rule; they do not replace it.',
            'Three arms estimate incremental value; without online-only they do not identify synergy.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','records','summary','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    require(not args.output.exists(),'Use a fresh audit output path')
    opener = gzip.open if args.records.suffix=='.gz' else open
    with opener(args.records,'rt') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    plan,summary = json.loads(args.plan.read_text()),json.loads(args.summary.read_text())
    provenance = dict(verifier_sha256=clocks.file_hash(__file__),
        independent_helpers_sha256={name:clocks.file_hash(ROOT/name) for name in
            ('scripts/verify_arrival_fix_statistics.py','scripts/verify_instructor_arrivals.py')},
        records_file_sha256=clocks.file_hash(args.records),summary_file_sha256=clocks.file_hash(args.summary),
        plan_file_sha256=clocks.file_hash(args.plan),numpy_version=np.__version__)
    try:
        provenance.update(previous.records_identity(args.records,summary))
        if 'analysis_file_sha256' in plan:
            require(plan['analysis_file_sha256']==clocks.file_hash(ROOT/'arrival_combo/analysis.py'),
                'Frozen analysis file identity; read only, never imported')
        if 'analysis_file_sha256' in summary:
            require(summary['analysis_file_sha256']==plan['analysis_file_sha256'],'Summary analysis pin')
        result = verify(plan,records,summary)
    except (AssertionError,KeyError,TypeError,ValueError,IndexError) as exc:
        result = dict(valid=False,error=str(exc),error_type=type(exc).__name__)
    result.update(provenance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('valid','error','observed_records','matched_triplets') if k in result}))
    if not result['valid']:raise SystemExit(1)


if __name__=='__main__':
    main()
