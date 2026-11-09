#!/usr/bin/env python3
"""Render validated three-arm arrival results; never refit confirmation tests."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

ARMS = ('native', 'shortcut', 'combined')
CONTRASTS = ('combined_over_shortcut', 'combined_over_native', 'shortcut_over_native')
KEY = ('arrival_block_id', 'batch_tokens', 'pattern', 'rho', 'repetition', 'arm')
CELL_KEY = ('batch_tokens', 'pattern', 'rho')
CELL_ORDER = list(itertools.product((128, 1024), ('spread', 'cluster'), (.5, .85)))
RATIO_METRICS = ('response_mean_ns', 'response_p95_ns', 'service_mean_ns')
TABLE_METRICS = (
    ('response_mean_ns', 'response_mean_ms', 1e-6),
    ('response_p50_ns', 'response_p50_ms', 1e-6),
    ('response_p95_ns', 'response_p95_ms', 1e-6),
    ('response_p99_ns', 'response_p99_ms', 1e-6),
    ('queue_wait_mean_ns', 'queue_mean_ms', 1e-6),
    ('queue_wait_p95_ns', 'queue_p95_ms', 1e-6),
    ('service_mean_ns', 'service_mean_ms', 1e-6),
    ('service_p95_ns', 'service_p95_ms', 1e-6),
    ('observed_elapsed_seconds', 'observed_elapsed_seconds', 1.),
    ('completion_elapsed_seconds', 'last_completion_elapsed_seconds', 1.),
    ('drain_ns', 'drain_ms', 1e-6),
    ('completed_by_horizon_per_second', 'horizon_jobs_per_second', 1.),
    ('completed_per_evaluation_second', 'observation_jobs_per_second', 1.),
    ('backlog_at_horizon', 'backlog_at_horizon_jobs', 1.),
    ('coordinator_server_busy_fraction', 'coordinator_busy_fraction', 1.),
)
ARCHITECTURE_FIELDS = ('per_rank_tokens', 'per_rank_active_copies', 'per_rank_m_tiles', 'per_rank_padded_rows')


def finite(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('missing, nonfinite or invalid timing/ratio value')
    return value


def prepare_summary(summary):
    if (summary.get('valid') is not True or summary.get('phase') != 'final'
            or any(summary.get(k) != 384 for k in ('expected_records', 'observed_records', 'valid_records', 'matched_pairs'))
            or summary.get('matched_triplets') != 128 or len(summary.get('allocation_ids', [])) != 1
            or summary.get('pairs_per_contrast') != dict.fromkeys(CONTRASTS, 128)):
        raise ValueError('report requires the complete validated 384-trial, three-arm final summary')
    inference = summary.get('confirmatory', {})
    if (inference.get('primary', {}).get('contrast') != 'combined_over_shortcut'
            or inference.get('primary', {}).get('name') != 'high_load_mean_response'
            or len(inference.get('guards', [])) != 3
            or {g['name'] for g in inference['guards']} !=
               {'low_load_mean_response', 'high_load_p95_response', 'b128_high_load_mean_response'}
            or inference.get('secondary_baseline', {}).get('contrast') != 'combined_over_native'):
        raise ValueError('the primary, three guards and hierarchical secondary are required')
    confirmation_rows(summary)
    blocks = sorted({p['arrival_block_id'] for p in summary['paired_cases']})
    contexts = set(itertools.product(blocks, (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1)))
    expected_pairs = {(*key, contrast) for key in contexts for contrast in CONTRASTS}
    expected_trials = {(*key, arm) for key in contexts for arm in ARMS}
    trials = [tuple(row[k] for k in KEY) for row in summary['trial_summaries']]
    if len(blocks) != 8 or len(trials) != 384 or set(trials) != expected_trials:
        raise ValueError('exactly eight calendar blocks and 384 unique trial summaries required')
    pair_index = {}
    for pair in summary['paired_cases']:
        key = tuple(pair[k] for k in KEY[:-1]) + (pair['contrast'],)
        if key not in expected_pairs or key in pair_index:
            raise ValueError('unexpected or duplicate paired reporting unit')
        pair_index[key] = pair
    if set(pair_index) != expected_pairs:
        raise ValueError('missing paired reporting units')
    seeds = []
    for batch, pattern, rho in CELL_ORDER:
        for block, contrast in itertools.product(blocks, CONTRASTS):
            pairs = [pair_index[block, batch, pattern, rho, rep, contrast] for rep in (0, 1)]
            result = dict(arrival_block_id=block, block_id=pairs[0]['block_id'], regime=pairs[0]['regime'],
                          batch_tokens=batch, pattern=pattern, rho=rho, contrast=contrast)
            for metric in RATIO_METRICS:
                logs = [pair['log_ratios'][metric] for pair in pairs]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in logs):
                    raise ValueError('nonfinite paired log ratio')
                result[metric + '_ratio'] = finite(math.exp(statistics.mean(logs)), positive=True)
            seeds.append(result)
    cells, seen = [], set()
    for cell in summary['descriptive_cells']:
        key = tuple(cell[k] for k in CELL_KEY)
        if (key not in CELL_ORDER or key in seen or cell.get('arrival_blocks') != 8
                or set(cell.get('arms', {})) != set(ARMS) or set(cell.get('contrasts', {})) != set(CONTRASTS)):
            raise ValueError('unexpected, duplicated or incomplete descriptive cell')
        seen.add(key)
        result = dict(zip(CELL_KEY, key))
        for source, column, scale in TABLE_METRICS:
            for arm in ARMS:
                result[arm + '_' + column] = finite(cell['arms'][arm][source]) * scale
        for contrast in CONTRASTS:
            if cell['contrasts'][contrast].get('paired_repetitions') != 16:
                raise ValueError('each descriptive contrast requires sixteen paired trials')
            for metric in RATIO_METRICS:
                point = finite(cell['contrasts'][contrast]['ratios'][metric], positive=True)
                values = [r[metric + '_ratio'] for r in seeds
                          if tuple(r[k] for k in CELL_KEY) == key and r['contrast'] == contrast]
                if not math.isclose(point, math.exp(statistics.mean(map(math.log, values))), rel_tol=1e-12):
                    raise ValueError('cell ratio differs from the eight paired calendar-block points')
                result[metric + '_' + contrast] = point
        cells.append(result)
    if seen != set(CELL_ORDER):
        raise ValueError('all eight descriptive cells required')
    cells.sort(key=lambda row: CELL_ORDER.index(tuple(row[k] for k in CELL_KEY)))
    return cells, seeds


def confirmation_rows(summary):
    inference = summary['confirmatory']
    gate = inference['primary']['passes'] and all(g['passes'] for g in inference['guards'])
    secondary = inference['secondary_baseline']
    if (gate != secondary.get('gate_passed') or gate != secondary.get('confirmatory')
            or inference['incremental_acceptance'] != ('pass' if gate else 'not_established')):
        raise ValueError('secondary confirmation does not respect the hierarchical gate')
    if not gate and any(secondary.get(k) is not None for k in ('ci95', 'upper_confidence_bound', 'passes')):
        raise ValueError('closed secondary gate must not have an interval or confirmatory decision')
    rows = []
    for endpoint in [inference['primary'], *inference['guards'], secondary]:
        tested = endpoint is not secondary or gate
        interval = endpoint.get('ci95')
        rows.append(dict(endpoint=endpoint['name'], contrast=endpoint['contrast'],
            ratio=finite(endpoint['ratio'], positive=True),
            lower_ci95=finite(interval[0], positive=True) if interval else None,
            upper_bound=finite(endpoint['upper_confidence_bound'], positive=True) if tested else None,
            confidence=endpoint['confidence'], requirement=endpoint.get('criterion', 'not tested; gate closed'),
            limit=endpoint['ratio_limit'], status=endpoint['status'], confirmatory_tested=tested))
    return rows


def raw_descriptives(summary, records):
    """Timestamp service summaries and routing counters, with no causal isolation."""
    expected = {tuple(row[k] for k in KEY): row for row in summary['trial_summaries']}
    seen, arrays = set(), {(b, a): [] for b, a in itertools.product((128, 1024), ARMS)}
    architecture = {key: [] for key in arrays}
    present_architecture = []
    for row in records:
        key = tuple(row[k] for k in KEY)
        if (key not in expected or key in seen or row.get('config_hash') != summary['plan_sha256']
                or row.get('arrival_combo_source_sha256') != summary['arrival_combo_source_sha256']
                or row.get('phase') != 'final' or row.get('status') != 'complete'
                or row.get('correctness_pass') is not True or row.get('evidence_eligible') is not True):
            raise ValueError('raw timing rows do not match the validated final summary and source')
        seen.add(key)
        starts, finishes = row['coordinator_start_ns'], row['coordinator_finish_ns']
        if (len(starts) != 256 or len(finishes) != 256
                or any(type(t) is not int for t in starts + finishes)
                or any(f <= s for s, f in zip(starts, finishes))):
            raise ValueError('every raw trial requires 256 positive service intervals')
        values = np.asarray(finishes, dtype=np.int64)-np.asarray(starts, dtype=np.int64)
        if not math.isclose(float(values.mean()), expected[key]['metrics']['service_mean_ns'], rel_tol=1e-12):
            raise ValueError('raw service intervals differ from the analyzed trial summary')
        arrays[row['batch_tokens'], row['arm']].append(values)
        metrics = row.get('assignment_metrics')
        present_architecture.append(metrics is not None)
        if metrics is not None:
            if len(metrics) != 256:
                raise ValueError('partial architecture observations')
            for metric in metrics:
                if any(len(metric.get(field, [])) != 2
                       or any(type(v) is not int or v < 0 for v in metric[field]) for field in ARCHITECTURE_FIELDS):
                    raise ValueError('invalid two-rank architecture counters')
                architecture[row['batch_tokens'], row['arm']].append(metric)
    if seen != set(expected) or len(seen) != 384 or any(len(v) != 64 for v in arrays.values()):
        raise ValueError('missing raw per-step service observations')
    if any(present_architecture) and not all(present_architecture):
        raise ValueError('architecture counters are present for only part of the experiment')
    steps, boundary_rows, architecture_rows = [], [], []
    ordinary = [i for i in range(256) if i not in (63, 127, 191, 255)]
    for batch in (128, 1024):
        ms = {arm: np.asarray(arrays[batch, arm], dtype=np.int64)/1e6 for arm in ARMS}
        for step in range(256):
            item = dict(batch_tokens=batch, step=step, trials_per_arm=64)
            for arm in ARMS:
                item[arm+'_mean_service_ms'] = float(ms[arm][:, step].mean())
                item[arm+'_median_service_ms'] = float(np.median(ms[arm][:, step]))
            steps.append(item)
        for arm in ARMS:
            for group, indices in (('ordinary', ordinary), ('nonterminal_boundary', [63, 127, 191]), ('terminal_boundary', [255])):
                values = ms[arm][:, indices]
                boundary_rows.append(dict(batch_tokens=batch, arm=arm, group=group, trials=64,
                    observations=int(values.size), mean_service_ms=float(values.mean()),
                    median_service_ms=float(np.median(values)),
                    mean_trial_group_total_ms=float(values.sum(axis=1).mean())))
            if all(present_architecture):
                observed = architecture[batch, arm]
                item = dict(batch_tokens=batch, arm=arm, trials=64, observed_jobs=len(observed))
                for field in ARCHITECTURE_FIELDS:
                    counts = np.asarray([r[field] for r in observed], dtype=np.int64)
                    for rank in (0, 1): item[field+f'_rank{rank}_mean'] = float(counts[:, rank].mean())
                    item[field+'_max_rank_mean'] = float(counts.max(axis=1).mean())
                    item[field+'_sum_rank_mean'] = float(counts.sum(axis=1).mean())
                architecture_rows.append(item)
    return steps, boundary_rows, architecture_rows


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}


def _save(fig, folder, name):
    for suffix in ('png', 'pdf'): fig.savefig(folder/f'{name}.{suffix}', dpi=300, bbox_inches='tight')


def plot_seed_ratios(cells, seeds, output_dir, *, qa_only=False):
    plt = _plotting()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.7), sharex=True)
        if qa_only: fig.suptitle('SYNTHETIC CPU FIXTURE: LAYOUT QA ONLY', fontsize=11)
        for column, contrast in enumerate(CONTRASTS[:2]):
            for row, (metric, title) in enumerate(zip(RATIO_METRICS[:2], ('Mean response', 'Within-trial p95 response'))):
                axis = axes[row, column]
                for i, cell in enumerate(cells):
                    selected = [r for r in seeds if tuple(r[k] for k in CELL_KEY) == tuple(cell[k] for k in CELL_KEY)
                                and r['contrast'] == contrast]
                    axis.scatter(i+np.linspace(-.18, .18, 8), [r[metric+'_ratio'] for r in selected],
                                 s=19, facecolors='white', edgecolors='black', linewidth=.7, zorder=3)
                    point = cell[metric+'_'+contrast]
                    axis.plot([i-.24, i+.24], [point, point], color='black', linewidth=1.6, zorder=4)
                axis.axhline(1., color='.4', linestyle='--', linewidth=.8)
                axis.set_title(title+' — '+contrast.replace('_over_', ' / '))
                axis.set_ylabel('Paired ratio')
                axis.grid(axis='y', color='.9', linewidth=.5); axis.set_axisbelow(True)
                axis.set_ylim(bottom=0)  # Upper bound is automatic: regressions above one stay visible.
            axes[-1, column].set_xticks(range(8), [f"{r['batch_tokens']}\n{r['pattern']}\n{r['rho']:.2f}" for r in cells])
            axes[-1, column].set_xlabel('Batch tokens / arrival shape / offered load ρ')
        fig.text(.5, .015, 'Open circles: eight calendar blocks; each averages two repeat log ratios. Bar: geometric mean. Dashed line: equal performance.\n'
                 'All cell ratios are descriptive, without cell intervals. Four reused payload strata; one allocation. Ratios above one indicate slower response.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.06, right=.99, top=.90 if qa_only else .95, bottom=.20, hspace=.28, wspace=.2)
        _save(fig, output_dir, 'arrival-combo-seed-ratios'); plt.close(fig)


def plot_step_service(rows, output_dir, *, qa_only=False):
    plt = _plotting()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        if qa_only: fig.suptitle('SYNTHETIC CPU FIXTURE: LAYOUT QA ONLY', fontsize=11)
        for bi, batch in enumerate((128, 1024)):
            selected = [r for r in rows if r['batch_tokens'] == batch]
            for ci, scale in enumerate(('linear', 'log')):
                axis = axes[bi, ci]
                for arm, style, marker in zip(ARMS, ('-', '--', '-.'), ('o', 's', '^')):
                    axis.plot([r['step'] for r in selected], [r[arm+'_mean_service_ms'] for r in selected],
                        color='black', linestyle=style, linewidth=1., label=arm, marker=marker,
                        markevery=[63, 127, 191, 255], markersize=4, markeredgecolor='black',
                        markerfacecolor='black' if arm == 'native' else 'white')
                for step in (63, 127, 191, 255): axis.axvline(step, color='.75', linestyle=':', linewidth=.6)
                axis.set_yscale(scale)
                if scale == 'linear': axis.set_ylim(bottom=0)
                axis.set_title(f'B = {batch}; {scale} scale'); axis.set_xticks([0, 63, 127, 191, 255])
                axis.set_xlabel('Job index'); axis.set_ylabel('Mean charged service (ms)')
                axis.grid(axis='y', color='.9', linewidth=.5); axis.legend(frameon=False)
        fig.text(.5, .015, 'Each point averages 64 trials per arm and batch across both loads and shapes, eight calendar blocks and two repeats.\n'
                 'Command-to-ACK intervals include host control, GPU work, collectives and synchronization; these are not exclusive CPU or GPU times.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.08, right=.98, top=.90 if qa_only else .95, bottom=.14, hspace=.37, wspace=.27)
        _save(fig, output_dir, 'arrival-combo-step-service'); plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def report(summary_path, output_dir, records_path=None):
    summary_path, output_dir = Path(summary_path), Path(output_dir)
    summary = json.loads(summary_path.read_text())
    cells, seeds = prepare_summary(summary)
    confirmations = confirmation_rows(summary)
    steps, boundaries, architecture, raw_hash = [], [], [], None
    if records_path is not None:
        records_path = Path(records_path); payload = records_path.read_bytes()
        plain = gzip.decompress(payload) if records_path.suffix == '.gz' else payload
        raw_hash = hashlib.sha256(payload).hexdigest()
        if summary.get('records_file_sha256') not in {raw_hash, hashlib.sha256(plain).hexdigest()}:
            raise ValueError('raw file checksum differs from analyzed source; refusing mixed-evidence figures')
        steps, boundaries, architecture = raw_descriptives(summary, [json.loads(line) for line in plain.decode().splitlines() if line.strip()])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('cells', cells), ('seed-ratios', seeds), ('confirmation', confirmations),
                       ('step-service', steps), ('boundary-service', boundaries), ('architecture', architecture)):
        if rows: write_csv(output_dir/f'arrival-combo-{name}.csv', rows)
    plot_seed_ratios(cells, seeds, output_dir)
    if steps: plot_step_service(steps, output_dir)
    lines = ['# Three-arm arrival results', '',
        f"Incremental dispatcher acceptance: **{summary['confirmatory']['incremental_acceptance']}**. Hierarchical overall acceptance: **{summary['confirmatory']['acceptance']}**.", '',
        'Native uses the native dispatcher and SKIP policy. Shortcut adds only the unchanged-map shortcut. Combined adds the frozen online resident dispatcher to shortcut. The primary contrast is combined/shortcut; a combined/native gain alone cannot establish incremental dispatcher value.', '',
        '| Endpoint | Contrast | Ratio | Interval or upper bound | Requirement | Status |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in confirmations:
        if not row['confirmatory_tested']:
            interval, requirement = 'Not tested: hierarchical gate closed; point only', 'Not tested'
        else:
            interval = (f"95% CI [{row['lower_ci95']:.4f}, {row['upper_bound']:.4f}]" if row['lower_ci95'] is not None
                        else f"One-sided 98.333333% upper {row['upper_bound']:.4f}")
            requirement = ('Upper < ' if row['lower_ci95'] is not None else 'Upper ≤ ')+str(row['limit'])
        lines.append(f"| {row['endpoint']} | {row['contrast']} | {row['ratio']:.4f} | {interval} | {requirement} | {row['status']} |")
    lines += ['', 'The three shortcut-relative guards cover low-load mean response, high-load p95 response and B128 high-load mean response. They use Bonferroni-corrected one-sided bounds. A nonpassing guard does not automatically prove a regression. The 5% usefulness and 2% guard margins are carried-forward engineering criteria, not fitted to these outcomes or numerical tolerances.', '',
        'Inference resamples two calendar blocks within each of four fixed payload strata with the frozen 100,000-draw bootstrap. Only 81 stratum-count combinations exist. Jobs and technical repeats do not increase the eight-block sample size. The claim is conditional on reused payloads, native capacity calibration and one allocation; it does not describe an unseen-document or hardware population.', '',
        f"Exact first/middle/last arm-position counts: {json.dumps(summary['arm_position_counts'], sort_keys=True)}. Reversal balances pairwise precedence; position counts are only near-balanced.", '',
        'N/S/C means native/shortcut/combined. Absolute values below are arithmetic averages of trial summaries; p95 values are averages of within-trial p95s, not pooled percentiles. The CSV additionally includes p50/p99 response, queue/service p95, throughput, backlog and coordinator busy fraction.', '',
        '| B | Shape | ρ | Mean response N/S/C (ms) | p95 response N/S/C (ms) | Mean queue N/S/C (ms) | Mean service N/S/C (ms) | Observed elapsed N/S/C (s) | Last completion N/S/C (s) | Drain N/S/C (ms) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in cells:
        values = [' / '.join(f"{row[arm+'_'+metric]:.4f}" for arm in ARMS) for metric in
            ('response_mean_ms', 'response_p95_ms', 'queue_mean_ms', 'service_mean_ms',
             'observed_elapsed_seconds', 'last_completion_elapsed_seconds', 'drain_ms')]
        lines.append(f"| {row['batch_tokens']} | {row['pattern']} | {row['rho']:.2f} | "+' | '.join(values)+' |')
    lines += ['', '![Descriptive paired calendar-block response ratios](arrival-combo-seed-ratios.png)', '',
        'The eight cells and all three contrasts are descriptive. Their ratios average paired logs, so they need not equal ratios of displayed absolute arithmetic means. Each point retains one calendar block after collapsing its two repeats. All three contrasts appear in the CSV; the figure separates incremental value from the native baseline.', '',
        'Response starts at the immutable scheduled release. Observed elapsed includes the offered horizon, drain and observed end overshoot; last completion is relative to the origin. These are fixed-token layer jobs, not full-model user requests.']
    if steps:
        lines += ['', '![Charged per-step service](arrival-combo-step-service.png)', '',
            'Service is measured directly from coordinator command start to completion ACK. Vertical guides mark native control boundaries at jobs 63/127/191 and terminal job 255. Boundary and ordinary-job CSVs are descriptive associations, not isolated component costs or additive CPU/GPU profiles. Each per-step mean/median pools 64 trials per arm and batch.']
    if architecture:
        lines += ['', 'The architecture CSV summarizes recorded per-rank token assignments, activated physical copies, M tiles and padded rows. It reports means of each rank, the per-job rank maximum and rank sum across all 16,384 observed jobs per arm/batch. These counters motivate mechanisms but are not runtime bounds; correlated job observations receive no inferential test.']
    elif records_path is not None:
        lines += ['', 'Architecture counters were absent from every provided raw record; no architecture values were imputed.']
    (output_dir/'arrival-combo-results.md').write_text('\n'.join(lines)+'\n')
    provenance = dict(summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), raw_file_sha256=raw_hash,
        plan_sha256=summary['plan_sha256'], arrival_combo_source_sha256=summary['arrival_combo_source_sha256'],
        allocation_ids=summary['allocation_ids'], cells=8, arrival_blocks=8, trials=384, contrasts=3,
        step_plots_included=bool(steps), architecture_included=bool(architecture),
        interpretation='Only frozen confirmation endpoints are inferential; cell, step and architecture summaries are descriptive.')
    (output_dir/'report-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.summary, args.output_dir, args.records)))


if __name__ == '__main__': main()
