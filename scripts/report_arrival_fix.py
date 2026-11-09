#!/usr/bin/env python3
"""Render frozen arrival-fix inference and descriptive timing, without refitting."""
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


def finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('missing, nonfinite or invalid timing/ratio value')
    return value


def prepare_summary(summary):
    if (summary.get('valid') is not True or summary.get('phase') != 'final'
            or any(summary.get(k) != 256 for k in ('expected_records', 'observed_records', 'valid_records'))
            or summary.get('matched_pairs') != 128 or len(summary.get('allocation_ids', [])) != 1):
        raise ValueError('report requires the complete validated 256-trial final summary')
    inference = summary.get('confirmatory')
    if (not inference or inference['primary']['name'] != 'high_load_mean_response'
            or len(inference['guards']) != 2
            or {g['name'] for g in inference['guards']} != {'low_load_mean_response', 'high_load_p95_response'}):
        raise ValueError('the three prespecified confirmation endpoints are required')
    pair_index = {}
    blocks = sorted({p['arrival_block_id'] for p in summary['paired_cases']})
    expected_pairs = set(itertools.product(blocks, (128, 1024), ('spread', 'cluster'), (.5, .85), (0, 1)))
    if len(blocks) != 8:
        raise ValueError('exactly eight arrival blocks required')
    for pair in summary['paired_cases']:
        key = tuple(pair[k] for k in KEY[:-1])
        if key not in expected_pairs or key in pair_index:
            raise ValueError('unexpected or duplicate paired reporting unit')
        pair_index[key] = pair
    if set(pair_index) != expected_pairs:
        raise ValueError('missing paired reporting units')
    seeds = []
    for batch, pattern, rho in CELL_ORDER:
        for block in blocks:
            pairs = [pair_index[block, batch, pattern, rho, rep] for rep in (0, 1)]
            result = dict(arrival_block_id=block, block_id=pairs[0]['block_id'], regime=pairs[0]['regime'],
                          batch_tokens=batch, pattern=pattern, rho=rho)
            for metric in RATIO_METRICS:
                logs = [pair['log_ratios'][metric] for pair in pairs]
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logs):
                    raise ValueError('nonfinite paired log ratio')
                result[metric + '_fast_over_native'] = finite(math.exp(statistics.mean(logs)), positive=True)
            seeds.append(result)
    cells, seen = [], set()
    for cell in summary['descriptive_cells']:
        key = tuple(cell[k] for k in CELL_KEY)
        if key not in CELL_ORDER or key in seen or cell.get('arrival_blocks') != 8 or cell.get('paired_repetitions') != 16:
            raise ValueError('unexpected, duplicated or incomplete descriptive cell')
        seen.add(key)
        result = dict(zip(CELL_KEY, key))
        for source, column, scale in TABLE_METRICS:
            for arm in ('native', 'fast'):
                result[arm + '_' + column] = finite(cell[arm][source]) * scale
        for metric in RATIO_METRICS:
            point = finite(cell['ratios'][metric], positive=True)
            selected = [seed[metric + '_fast_over_native'] for seed in seeds if tuple(seed[k] for k in CELL_KEY) == key]
            if not math.isclose(point, math.exp(statistics.mean(math.log(value) for value in selected)), rel_tol=1e-12):
                raise ValueError('cell ratio differs from the eight paired seed points')
            result[metric + '_fast_over_native'] = point
        cells.append(result)
    if seen != set(CELL_ORDER):
        raise ValueError('all eight descriptive cells required')
    cells.sort(key=lambda row: CELL_ORDER.index(tuple(row[k] for k in CELL_KEY)))
    return cells, seeds


def confirmation_rows(summary):
    inference = summary['confirmatory']
    rows = []
    for endpoint in [inference['primary'], *inference['guards']]:
        rows.append(dict(endpoint=endpoint['name'], fast_over_native=finite(endpoint['ratio'], positive=True),
            lower_ci95=endpoint.get('ci95', [None])[0], upper_bound=finite(endpoint['upper_confidence_bound'], positive=True),
            confidence=endpoint['confidence'], requirement=endpoint['criterion'], limit=endpoint['ratio_limit'],
            status=endpoint['status']))
    return rows


def raw_step_rows(summary, records):
    """Descriptive mean/median of charged command-to-ACK service at each step."""
    expected = {tuple(row[k] for k in KEY) for row in summary['trial_summaries']}
    seen, arrays = set(), {(batch, arm): [] for batch, arm in itertools.product((128, 1024), ('native', 'fast'))}
    for row in records:
        key = tuple(row[k] for k in KEY)
        if key not in expected or key in seen or row['config_hash'] != summary['plan_sha256'] or row['phase'] != 'final':
            raise ValueError('raw timing rows do not match the validated final summary')
        seen.add(key)
        starts, finishes = row['coordinator_start_ns'], row['coordinator_finish_ns']
        if len(starts) != 256 or len(finishes) != 256:
            raise ValueError('per-step plot requires every one of the 256 service intervals')
        if any(type(t) is not int for t in starts + finishes) or any(f <= s for s, f in zip(starts, finishes)):
            raise ValueError('invalid raw service interval')
        arrays[row['batch_tokens'], row['arm']].append(np.asarray(finishes, dtype=np.int64)-np.asarray(starts, dtype=np.int64))
    if seen != expected or len(seen) != 256 or any(len(value) != 64 for value in arrays.values()):
        raise ValueError('missing raw per-step service observations')
    result = []
    for batch in (128, 1024):
        values = {arm: np.asarray(arrays[batch, arm], dtype=np.int64) / 1e6 for arm in ('native', 'fast')}
        for step in range(256):
            row = dict(batch_tokens=batch, step=step, trials_per_arm=64)
            for arm in ('native', 'fast'):
                row[arm + '_mean_service_ms'] = float(np.mean(values[arm][:, step]))
                row[arm + '_median_service_ms'] = float(np.median(values[arm][:, step]))
            result.append(row)
    return result


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


STYLE = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}


def _save(fig, folder, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / f'{name}.{suffix}', dpi=300, bbox_inches='tight')


def plot_seed_ratios(cells, seeds, output_dir, *, qa_only=False):
    plt = _plotting()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
        if qa_only:
            fig.suptitle('SYNTHETIC CPU FIXTURE: LAYOUT QA ONLY', fontsize=10)
        for axis, metric, title in zip(axes, ('response_mean_ns', 'response_p95_ns'), ('Mean response', 'Within-trial p95 response')):
            for i, cell in enumerate(cells):
                selected = [row for row in seeds if tuple(row[k] for k in CELL_KEY) == tuple(cell[k] for k in CELL_KEY)]
                values = [row[metric + '_fast_over_native'] for row in selected]
                axis.scatter(i + np.linspace(-.18, .18, 8), values, s=22, facecolors='white', edgecolors='black', linewidth=.7, zorder=3)
                point = cell[metric + '_fast_over_native']
                axis.plot([i-.24, i+.24], [point, point], color='black', linewidth=1.6, zorder=4)
            axis.axhline(1., color='.4', linestyle='--', linewidth=.8)
            axis.set_ylabel('Fast / native ratio')
            axis.set_title(title)
            axis.grid(axis='y', color='.9', linewidth=.5)
            axis.set_axisbelow(True)
            axis.set_ylim(bottom=0)
        axes[-1].set_xticks(range(8), [f"B={row['batch_tokens']}\n{row['pattern']}\nρ={row['rho']:.2f}" for row in cells])
        fig.text(.5, .015, 'Open circles: eight calendar blocks; each collapses two repeat log ratios. Bar: geometric mean.\n'
                 'Cell comparisons are descriptive, without cell confidence intervals. Four reused payload strata; one allocation.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.09, right=.98, top=.90 if qa_only else .95, bottom=.19, hspace=.3)
        _save(fig, output_dir, 'arrival-fix-seed-ratios'); plt.close(fig)


def plot_step_service(rows, output_dir, *, qa_only=False):
    plt = _plotting()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
        if qa_only:
            fig.suptitle('SYNTHETIC CPU FIXTURE: LAYOUT QA ONLY', fontsize=10)
        for bi, batch in enumerate((128, 1024)):
            selected = [row for row in rows if row['batch_tokens'] == batch]
            for ci, scale in enumerate(('linear', 'log')):
                axis = axes[bi, ci]
                for arm, linestyle in (('native', '-'), ('fast', '--')):
                    axis.plot([r['step'] for r in selected], [r[arm + '_mean_service_ms'] for r in selected],
                              color='black', linestyle=linestyle, linewidth=1., label=arm,
                              marker='o' if arm == 'native' else 's', markevery=[63, 127, 191, 255],
                              markersize=4, markeredgecolor='black',
                              markerfacecolor='black' if arm == 'native' else 'white')
                for step in (63, 127, 191, 255):
                    axis.axvline(step, color='.75', linestyle=':', linewidth=.6)
                axis.set_yscale(scale)
                if scale == 'linear': axis.set_ylim(bottom=0)
                axis.set_title(f'B = {batch}; {scale} scale')
                axis.set_xticks([0, 63, 127, 191, 255])
                axis.set_xlabel('Job index')
                axis.set_ylabel('Mean charged service (ms)')
                axis.grid(axis='y', color='.9', linewidth=.5)
                axis.legend(frameon=False)
        fig.text(.5, .015, 'Each point averages 64 trials per arm and batch across both loads and shapes, eight calendar blocks and two repeats.\n'
            'Command-to-ACK service combines host control, GPU work, collectives and synchronization; it is not exclusive CPU/GPU time.',
            ha='center', fontsize=8)
        fig.subplots_adjust(left=.08, right=.98, top=.90 if qa_only else .95, bottom=.14, hspace=.37, wspace=.27)
        _save(fig, output_dir, 'arrival-fix-step-service'); plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def report(summary_path, output_dir, records_path=None):
    summary_path, output_dir = Path(summary_path), Path(output_dir)
    summary = json.loads(summary_path.read_text())
    cells, seeds = prepare_summary(summary)
    confirmations = confirmation_rows(summary)
    steps, raw_hash = None, None
    if records_path is not None:
        records_path = Path(records_path)
        payload = records_path.read_bytes()
        plain = gzip.decompress(payload) if records_path.suffix == '.gz' else payload
        raw_hash = hashlib.sha256(payload).hexdigest()
        if summary.get('records_file_sha256') not in {raw_hash, hashlib.sha256(plain).hexdigest()}:
            raise ValueError('raw file checksum differs from the analyzed source; refusing a mixed-evidence plot')
        steps = raw_step_rows(summary, [json.loads(line) for line in plain.decode().splitlines() if line.strip()])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir/'arrival-fix-cells.csv', cells)
    write_csv(output_dir/'arrival-fix-seed-ratios.csv', seeds)
    write_csv(output_dir/'arrival-fix-confirmation.csv', confirmations)
    plot_seed_ratios(cells, seeds, output_dir)
    if steps is not None:
        write_csv(output_dir/'arrival-fix-step-service.csv', steps)
        plot_step_service(steps, output_dir)
    lines = ['# Arrival-fix results', '', 'Prespecified acceptance: **'+summary['confirmatory']['acceptance']+'**.', '',
        'These are the three frozen confirmation tests. The eight B/pattern/load cells below are descriptive and receive no additional confidence intervals or acceptance tests.', '',
        '| Endpoint | Fast/native | Interval or upper bound | Requirement | Status |', '| --- | --- | --- | --- | --- |']
    for row in confirmations:
        interval = (f"95% CI [{row['lower_ci95']:.4f}, {row['upper_bound']:.4f}]" if row['lower_ci95'] is not None
                    else f"One-sided 97.5% upper {row['upper_bound']:.4f}")
        limit = ('< ' if row['endpoint']=='high_load_mean_response' else '≤ ') + str(row['limit'])
        lines.append(f"| {row['endpoint']} | {row['fast_over_native']:.4f} | {interval} | Upper {limit} | {row['status']} |")
    lines += ['', 'Primary: high-load mean response. Guards: low-load mean response and high-load p95 response, with Bonferroni correction for the two one-sided guards. A guard that is not established is not automatically proof of a regression.', '',
        'Inference resamples two calendar blocks within each of four fixed payload strata, using the frozen 100,000-draw bootstrap. It is conditional on reused payloads, native capacity calibration and one allocation. There are only eight independent arrival blocks and 81 distinct stratum-count combinations; jobs and technical repeats do not increase that sample size.', '',
        'N/F denotes native/fast. Absolute values are arithmetic averages of trial summaries; trial p95 values are not pooled percentiles. The full CSV also includes response p50/p99, queue/service p95, horizon and observation throughput, backlog and coordinator busy fraction.', '',
        '| B | Pattern | ρ | Mean response N/F (ms) | Trial p95 response N/F (ms) | Mean queue N/F (ms) | Mean service N/F (ms) | Observed elapsed N/F (s) | Last completion N/F (s) | Drain N/F (ms) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in cells:
        values = [' / '.join(f"{row[arm+'_'+metric]:.4f}" for arm in ('native','fast')) for metric in
            ('response_mean_ms','response_p95_ms','queue_mean_ms','service_mean_ms','observed_elapsed_seconds','last_completion_elapsed_seconds','drain_ms')]
        lines.append(f"| {row['batch_tokens']} | {row['pattern']} | {row['rho']:.2f} | "+' | '.join(values)+' |')
    lines += ['', '![Descriptive paired calendar-block ratios](arrival-fix-seed-ratios.png)', '',
        'Response starts at the immutable scheduled release. Service includes the charged CPU command/ACK, native control, GPU work, collectives and synchronization. Observed elapsed includes the offered horizon, any drain and observed end overshoot; last completion is measured from the origin. These layer jobs are not full-model user requests.']
    if steps is not None:
        lines += ['', '![Charged per-step service](arrival-fix-step-service.png)', '',
            'The service plot uses per-step means across 64 trials for each arm and B, including both arrival shapes and loads. Its CSV also gives per-step medians. Vertical guides mark native boundaries at jobs 63/127/191 and the terminal boundary at 255. These host-clock intervals do not isolate individual CPU routines or GPU kernels.']
    (output_dir/'arrival-fix-results.md').write_text('\n'.join(lines)+'\n')
    provenance = dict(summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), raw_file_sha256=raw_hash,
        plan_sha256=summary['plan_sha256'], allocation_ids=summary['allocation_ids'],
        cells=8, arrival_blocks=8, trials=256, step_plots_included=steps is not None,
        interpretation='Only frozen confirmation tests are inferential; cell and step plots are descriptive.')
    (output_dir/'report-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.summary, args.output_dir, args.records)))


if __name__ == '__main__':
    main()
