#!/usr/bin/env python3
"""Independent final-only audit of raw arrival clocks and conditional inference.

No production experiment/analysis module is imported. The earlier independent
timestamp auditor supplies integer-clock arithmetic. Statistical units, paired
aggregation, PCG64 sampling, and the exact 81-count distribution are implemented
here. This verifies recorded evidence, not GPU numerical execution or raw routes.
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
_spec = importlib.util.spec_from_file_location(
    'arrival_statistics_independent_clocks', ROOT/'scripts/verify_instructor_arrivals.py')
clocks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clocks)
require, compare, digest = clocks.require, clocks.compare, clocks.digest
KEY = ('arrival_block_id', 'batch_tokens', 'pattern', 'rho', 'repetition', 'arm')
SETTINGS = dict(bootstrap_samples=100000, bootstrap_seed=471420,
    bootstrap='stratified_arrival_blocks', primary_ratio_limit=.95, guard_ratio_limit=1.02)
METRICS = ('response_mean_ns', 'response_p95_ns', 'service_mean_ns')
PAIRED = ('block_id', 'regime', 'arrival_seed', 'capture_sha256', 'replay_sha256',
    'source_prompt_ids', 'source_weights_sha256', 'source_weight_rank_sha256',
    'source_map', 'final_map', 'resolved_kernel_config', 'logical_routing_sha256',
    'decision_trajectory_sha256', 'arrival_clock_contract', 'warmup_raw_indices',
    'conditioning_raw_indices', 'measured_raw_indices', 'assignment_sha256')


def average(values):
    values = list(values)
    require(bool(values), 'Empty mean')
    return math.fsum(values)/len(values)


def stratum_members(block_strata):
    names = sorted(block_strata)
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[tuple(block_strata[name])].append(index)
    require(len(names) == 8 and len(groups) == 4
            and all(len(members) == 2 for members in groups.values()),
            'Exactly two calendar blocks in each of four fixed payload strata')
    return names, [groups[key] for key in sorted(groups)]


def bootstrap_draws(block_strata):
    """Explicit PCG64 with the pinned draw order; no jobs/repeats are resampled."""
    names, groups = stratum_members(block_strata)
    generator = np.random.Generator(np.random.PCG64(471420))
    draws = np.empty((100000, 8), dtype=np.int64)
    for column, members in enumerate(groups):
        choices = generator.integers(low=0, high=2, size=(100000, 2))
        draws[:, 2*column:2*column+2] = np.asarray(members, dtype=np.int64)[choices]
    return names, groups, draws


def exact_distribution(values, groups, draws):
    """81 count combinations, multiplicities summing to 256 ordered outcomes.

    Exact quantiles use the inverse CDF of this discrete distribution. They are
    supplemental: the frozen decision uses linear quantiles of 100000 PCG draws.
    Equal-valued combinations remain separate because their probabilities add.
    """
    observed = Counter(tuple(int(x) for x in row) for row in np.column_stack([
        (draws[:, 2*j:2*j+2] == members[0]).sum(axis=1)
        for j, members in enumerate(groups)]))
    support = []
    for counts in itertools.product(range(3), repeat=4):
        multiplicity = math.prod(math.comb(2, k) for k in counts)
        value = math.exp(math.fsum(k*values[a]+(2-k)*values[b]
            for k, (a, b) in zip(counts, groups))/8)
        support.append(dict(first_block_counts=list(counts), ratio=value,
            multiplicity=multiplicity, probability=multiplicity/256,
            monte_carlo_draws=observed[counts]))
    require(len(support) == 81 and sum(x['multiplicity'] for x in support) == 256,
            'Exact combinatorial distribution')
    require(sum(x['monte_carlo_draws'] for x in support) == 100000,
            'All draws accounted for by whole-block count vectors')
    def inverse_cdf(q):
        cumulative = 0
        for point in sorted(support, key=lambda x:x['ratio']):
            cumulative += point['multiplicity']
            if cumulative/256 >= q:
                return point['ratio']
        raise AssertionError('Exact CDF incomplete')
    return dict(count_combinations=81, ordered_equiprobable_outcomes=256,
        exact_percentile_025=inverse_cdf(.025), exact_percentile_975=inverse_cdf(.975),
        quantile_definition='smallest ratio with exact cumulative probability >= q',
        used_for_frozen_acceptance=False, support=support)


def inference(pairs, block_strata):
    names, groups, draws = bootstrap_draws(block_strata)
    expected = set(itertools.product(names, (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1)))
    indexed = {}
    for pair in pairs:
        key = tuple(pair[field] for field in KEY[:-1])
        require(key in expected and key not in indexed, 'Duplicate/unexpected paired unit')
        indexed[key] = pair
    require(set(indexed) == expected, 'Complete 128-pair matrix; no repeat/job pseudoreplication')
    endpoints, exact = [], {}
    for name, rho, metric in (('high_load_mean_response', .85, 'response_mean_ns'),
            ('low_load_mean_response', .5, 'response_mean_ns'),
            ('high_load_p95_response', .85, 'response_p95_ns')):
        values = np.asarray([average(average(indexed[block, batch, pattern, rho, rep]
            ['log_ratios'][metric] for rep in (0, 1))
            for batch, pattern in itertools.product((128, 1024), ('spread', 'cluster')))
            for block in names], dtype=np.float64)
        require(np.isfinite(values).all(), 'Finite block log ratios')
        sampled = np.exp(values[draws].mean(axis=1))
        lower, upper = [float(x) for x in np.quantile(sampled, [.025, .975], method='linear')]
        primary = len(endpoints) == 0
        limit = .95 if primary else 1.02
        passes = upper < limit if primary else upper <= limit
        item = dict(name=name, ratio=float(np.exp(values.mean())), arrival_blocks=8,
            block_log_ratios=dict(zip(names, map(float, values))),
            upper_confidence_bound=upper, ratio_limit=limit, passes=bool(passes),
            criterion='upper < limit' if primary else 'upper <= limit',
            status='pass' if passes else 'not_established')
        if primary:
            item.update(ci95=[lower, upper], confidence='two-sided percentile 95%')
        else:
            item.update(one_sided_confidence_level=.975, guard_family_size=2,
                confidence='one-sided percentile 97.5%; Bonferroni for two guards')
        endpoints.append(item)
        exact[name] = exact_distribution(values, groups, draws)
        exact[name]['monte_carlo_minus_exact_quantiles'] = [
            lower-exact[name]['exact_percentile_025'], upper-exact[name]['exact_percentile_975']]
    result = dict(primary=endpoints[0], guards=endpoints[1:],
        acceptance='pass' if all(e['passes'] for e in endpoints) else 'not_established',
        bootstrap={**SETTINGS, 'rng':'numpy.random.default_rng/PCG64', 'quantile_method':'linear',
            'strata':{name:list(block_strata[name]) for name in names},
            'draw_indices_sha256':hashlib.sha256(draws.tobytes()).hexdigest()})
    return result, exact


def flat_metrics(row):
    require(row['n_jobs'] == 256, 'Exactly 256 jobs per final trial')
    metrics = clocks.timing_metrics(row)
    compare(row['queue_metrics'], metrics, 'Stored metrics versus independent raw-clock arithmetic')
    compare(row['observed_window_spans'], clocks.window_spans(row), 'Raw observation spans')
    flat = {category+'_'+name+'_ns':value for category, entry in metrics['summary_ns'].items()
            for name, value in entry.items()}
    for name in ('backlog_at_horizon', 'waiting_jobs_at_horizon', 'drain_ns',
            'completed_by_horizon_per_second', 'completed_per_evaluation_second',
            'required_observation_extent_ns', 'observation_overshoot_ns'):
        flat[name] = metrics[name]
    elapsed = metrics['evaluation_end_ns']-row['epoch_monotonic_ns']
    flat.update(observed_elapsed_seconds=elapsed/1e9,
        completion_elapsed_seconds=(row['coordinator_finish_ns'][-1]-row['epoch_monotonic_ns'])/1e9,
        coordinator_server_busy_fraction=sum(metrics['vectors_ns']['service'])/elapsed)
    return flat


def recompute(plan, records):
    """Validate the complete recorded matrix before constructing any inference."""
    plan_hash = digest({k:v for k,v in plan.items() if k != 'freeze'})
    require(plan['phase'] == 'final' and plan['freeze']['sha256'] == plan_hash, 'Frozen final plan')
    compare(plan['analysis'], SETTINGS, 'Prespecified analysis settings')
    require(set(plan['analysis']) == set(SETTINGS), 'No extra inference settings')
    matrix = plan['matrix']
    compare(matrix, dict(batches=[128, 1024], patterns=['spread', 'cluster'],
        rhos=[.5, .85], reps=2, arms=['native', 'fast']), 'Fixed matrix')
    blocks = {b['id']:b for b in matrix['arrival_blocks']}
    require(len(blocks) == len(matrix['arrival_blocks']) == 8
            and len({b['seed'] for b in blocks.values()}) == 8, 'Eight unique calendar IDs and seeds')
    strata = {k:(b['block_id'], b['regime']) for k,b in blocks.items()}
    stratum_members(strata)
    hardware = plan['hardware']
    require(len(set(hardware['gpu_uuids'])) == 2 and bool(hardware['hostname']), 'Resolved two-GPU hardware plan')
    schedules = {}
    for block, batch in itertools.product(blocks, (128, 1024)):
        entries = plan['schedules'][f'{block}/B{batch}']
        require(len(entries) == 4 and {(x['pattern'],x['rho']) for x in entries}
                == set(itertools.product(('spread','cluster'),(.5,.85))), 'Complete frozen calendars')
        for entry in entries:
            require(entry['schedule_sha256'] == digest({k:v for k,v in entry.items()
                if k != 'schedule_sha256'}), 'Frozen calendar checksum')
            schedules[block,batch,entry['pattern'],entry['rho']] = entry
    expected = set(itertools.product(sorted(blocks), (128,1024), ('spread','cluster'), (.5,.85), (0,1), ('native','fast')))
    require(len(records) == 256, 'Exactly 256 final records')
    index, flat, orders = {}, {}, defaultdict(list)
    trials = []
    for row in records:
        key = tuple(row.get(f) for f in KEY)
        require(key in expected and key not in index, 'Duplicate/unexpected final record')
        index[key] = row
        orders[key[:-1]].append(row['arm'])
        for field,value in dict(status='complete', phase='final', config_hash=plan_hash,
            arrival_fix_source_sha256=plan['arrival_fix_source_sha256'], correctness_pass=True,
            evidence_eligible=True, diagnostic_only=False, guard_symbols_restored=True,
            normal_completion_verified=True, resident_state_verified=True, rank_assignment_agreement=True,
            migration_bytes=0, migration_sent_messages=0, terminal_proposals_suppressed=1, policy='SKIP').items():
            compare(row.get(field), value, 'Eligible final row/'+field)
        info = blocks[key[0]]
        for field,value in dict(arrival_seed=info['seed'], block_id=info['block_id'], regime=info['regime']).items():
            compare(row.get(field), value, 'Calendar block/'+field)
        require(row.get('preflight_correctness',{}).get('correctness_pass') is True, 'Preflight correctness recorded')
        require(row.get('source_map') is not None and row['source_map'] == row.get('final_map'), 'Fixed recorded map')
        require(sorted(row.get('physical_gpu_uuids',[])) == sorted(hardware['gpu_uuids'])
                and str(row.get('hostname','')).split('.')[0] == hardware['hostname'], 'Resolved physical hardware')
        calkey = f"{info['block_id']}/{info['regime']}/B{key[1]}"
        require(bool(row.get('allocation_id')) and str(row['allocation_id'])
                == str(plan['calibration'][calkey]['allocation_id']), 'Same allocation as native calibration')
        clock = row.get('arrival_clock_contract',{})
        require(clock.get('monotonic') is True and clock.get('adjustable') is False
                and 'CLOCK_MONOTONIC' in clock.get('implementation',''), 'Common monotonic clock')
        schedule = schedules[key[:4]]
        require(row['arrival_schedule'] == schedule and row['arrival_offsets_ns'] == schedule['arrival_offsets_ns']
                and row['horizon_ns'] == schedule['horizon_ns'], 'Unshifted frozen arrivals')
        guards = row.get('noop_guard_ranks')
        if row['arm'] == 'native':
            require(guards == [], 'Native baseline not wrapped')
        else:
            require(isinstance(guards,list) and len(guards) == 2, 'Two rank guard records')
            required = dict(mode='fast', diagnostic_timing=False, pending_pair=False,
                movement_calls=4, commit_calls=4, agreement_calls=4, local_unchanged_calls=4,
                unanimously_unchanged_calls=4, movement_skips=4, commit_skips=4,
                original_movement_calls=0, original_commit_calls=0, fallback_calls=0, diagnostics=[])
            for guard in guards:
                compare(guard, required, 'Fast-path completion counters')
        flat[key] = flat_metrics(row)
        trials.append({**dict(zip(KEY,key)), 'block_id':info['block_id'], 'regime':info['regime'],
            'allocation_id':row['allocation_id'], 'metrics':flat[key], 'observed_window_spans':clocks.window_spans(row)})
    require(set(index) == expected, 'Complete final matrix')
    allocations = sorted({str(r['allocation_id']) for r in records})
    require(len(allocations) == 1, 'One conditional allocation')
    pairs = []
    for key in sorted({x[:-1] for x in expected}):
        native, fast = index[(*key,'native')], index[(*key,'fast')]
        for field in PAIRED:
            require(field in native and field in fast and native[field] == fast[field], 'Paired state/'+field)
        require(native['arm_order'] == fast['arm_order'] == orders[key]
                and set(orders[key]) == {'native','fast'}, 'Actual recorded arm order')
        n, f = flat[(*key,'native')], flat[(*key,'fast')]
        require(all(n[m] > 0 and f[m] > 0 for m in METRICS), 'Positive paired endpoints')
        pairs.append({**dict(zip(KEY[:-1],key)), 'block_id':native['block_id'], 'regime':native['regime'],
            'allocation_id':native['allocation_id'], 'native':n, 'fast':f,
            'log_ratios':{m:math.log(f[m])-math.log(n[m]) for m in METRICS}})
    for key in {x[:4] for x in expected}:
        require(orders[(*key,0)] == orders[(*key,1)][::-1], 'Reversed technical-repeat arm order')
    confirmatory, exact = inference(pairs, strata)
    cells = []
    for batch,pattern,rho in itertools.product((128,1024),('spread','cluster'),(.5,.85)):
        selected = [p for p in pairs if (p['batch_tokens'],p['pattern'],p['rho']) == (batch,pattern,rho)]
        cells.append(dict(batch_tokens=batch, pattern=pattern, rho=rho, paired_repetitions=16, arrival_blocks=8,
            ratios={m:math.exp(average(p['log_ratios'][m] for p in selected)) for m in METRICS},
            native={m:average(p['native'][m] for p in selected) for m in selected[0]['native']},
            fast={m:average(p['fast'][m] for p in selected) for m in selected[0]['fast']}))
    return dict(analysis_schema_version=1, phase='final', valid=True,
        expected_records=256, observed_records=256, valid_records=256,
        matched_pairs=128, missing_keys=[], errors=[], plan_sha256=plan_hash,
        arrival_fix_source_sha256=plan['arrival_fix_source_sha256'], allocation_ids=allocations,
        trial_summaries=trials, paired_cases=pairs, confirmatory=confirmatory, descriptive_cells=cells), exact


def verify(plan, records, summary):
    expected, exact = recompute(plan, records)
    # Trial list order may be serialized differently; identities must be unique.
    for field, keys in (('trial_summaries',KEY), ('paired_cases',KEY[:-1]),
                        ('descriptive_cells',('batch_tokens','pattern','rho'))):
        reported = summary.get(field,[])
        indexed = {tuple(x[k] for k in keys):x for x in reported}
        require(len(indexed) == len(reported) == len(expected[field]), 'Summary unique complete '+field)
        for item in expected[field]:
            key = tuple(item[k] for k in keys)
            require(key in indexed, 'Summary missing '+field)
            compare(indexed[key], item, 'Recomputed '+field+'/'+str(key))
    compare(summary, {k:v for k,v in expected.items()
        if k not in ('trial_summaries','paired_cases','descriptive_cells')}, 'Recomputed summary')
    return dict(valid=True, expected_records=256, observed_records=len(records), matched_pairs=128,
        independent_calendar_blocks=8, fixed_payload_strata=4, jobs_per_trial=256,
        technical_repetitions_per_cell=2, confirmatory=expected['confirmatory'],
        exact_weighted_distributions=exact, allocation_ids=expected['allocation_ids'],
        plan_sha256=expected['plan_sha256'], hardware=plan['hardware'],
        checks=['integer raw-clock FCFS envelopes and all 256 jobs per trial',
            'stored metrics and summary endpoints reconstructed from clocks',
            'complete paired matrix, reversed repeats, matched state and frozen arrivals',
            'hardware and calibration allocation read from resolved frozen plan',
            'paired logs, nested repeat/cell means and eight whole-block units',
            'exact pinned 100000 PCG64 draws and linear percentile intervals',
            'primary strict .95 upper bound and both inclusive 1.02 guard bounds',
            'all 81 weighted count outcomes independently enumerated'],
        limitations=['Recorded correctness/route flags are checked, not re-executed on GPUs.',
            'Use the separate execution audit for NPZ route reconstruction, weights, and source provenance.',
            'Inference is conditional on four reused payload strata, frozen calibration and one allocation.',
            'Exact 81-outcome quantiles are supplemental, not a replacement acceptance rule.'])


def records_identity(path, summary):
    """Accept a server raw-JSONL hash through byte-preserving gzip archival."""
    path = Path(path)
    stored = clocks.file_hash(path)
    if path.suffix == '.gz':
        raw = hashlib.sha256()
        with gzip.open(path, 'rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024*1024), b''):
                raw.update(chunk)
        decompressed = raw.hexdigest()
    else:
        decompressed = stored
    if 'records_file_sha256' in summary:
        require(summary['records_file_sha256'] in (stored, decompressed),
                'Summary raw-file identity (archive or exact decompressed bytes)')
    return dict(records_file_sha256=stored, records_decompressed_sha256=decompressed,
        summary_records_file_sha256=summary.get('records_file_sha256'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('plan','records','summary','output'):
        parser.add_argument('--'+field, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a fresh audit output path')
    opener = gzip.open if args.records.suffix == '.gz' else open
    with opener(args.records,'rt') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    plan, summary = json.loads(args.plan.read_text()), json.loads(args.summary.read_text())
    provenance = dict(verifier_sha256=clocks.file_hash(__file__),
        independent_timing_helper_sha256=clocks.file_hash(ROOT/'scripts/verify_instructor_arrivals.py'),
        records_file_sha256=clocks.file_hash(args.records), summary_file_sha256=clocks.file_hash(args.summary),
        plan_file_sha256=clocks.file_hash(args.plan), numpy_version=np.__version__)
    try:
        provenance.update(records_identity(args.records, summary))
        if 'analysis_file_sha256' in plan:
            require(plan['analysis_file_sha256'] == clocks.file_hash(ROOT/'arrival_fix/analysis.py'),
                    'Analysis file differs from frozen pin (read only, never imported)')
        if 'analysis_file_sha256' in summary:
            require(summary['analysis_file_sha256'] == plan['analysis_file_sha256'], 'Summary analysis pin')
        result = verify(plan, records, summary)
    except (AssertionError, KeyError, TypeError, ValueError, IndexError) as exc:
        result = dict(valid=False, error=str(exc), error_type=type(exc).__name__)
    result.update(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('valid','error','observed_records','matched_pairs')}))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
