#!/usr/bin/env python3
"""Render the complete descriptive arrival summary; never rerun or tune analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


METRICS = (
    ('response_mean_ns', 'response_mean_ms', 'Mean response (ms)', 1e-6),
    ('queue_wait_mean_ns', 'queue_mean_ms', 'Mean queue (ms)', 1e-6),
    ('service_mean_ns', 'service_mean_ms', 'Mean service (ms)', 1e-6),
    ('response_p95_ns', 'response_p95_ms', 'Trial response p95 (ms)', 1e-6),
    ('backlog_at_horizon', 'backlog_at_horizon_jobs', 'Backlog at H (jobs)', 1.),
    ('drain_ns', 'drain_ms', 'Drain (ms)', 1e-6),
    ('coordinator_server_busy_fraction', 'coordinator_busy_fraction', 'Server busy fraction', 1.),
)
FACTORS = ('block_id', 'regime', 'batch_tokens', 'pattern', 'rho')


def table_rows(summary):
    """Refuse partial/invalid summaries instead of silently plotting survivors."""
    if (summary.get('valid') is not True or summary.get('phase') != 'evaluation'
            or any(summary.get(name) != 128 for name in ('expected_records', 'observed_records', 'valid_records'))
            or len(summary.get('allocation_ids', [])) != 1):
        raise ValueError('a validated complete 128-trial, one-allocation evaluation summary is required')
    cells = summary.get('repeat_averaged_cases', [])
    blocks = sorted({c['block_id'] for c in cells})
    if len(blocks) != 2:
        raise ValueError('exactly two development source blocks are required')
    expected = set(itertools.product(blocks, ('domain0', 'domain1'), (128, 1024), ('constant', 'burst8'), (.5, .85)))
    observed, rows = set(), []
    for cell in cells:
        key = tuple(cell[field] for field in FACTORS)
        if key not in expected or key in observed or cell.get('technical_repetitions') != 2:
            raise ValueError(f'unexpected, duplicate or incomplete reporting cell: {key}')
        observed.add(key)
        row = dict(zip(FACTORS, key))
        for source, column, _, scale in METRICS:
            for arm in ('native', 'online'):
                value = cell['contrasts'][source][arm]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'invalid {source}/{arm} in {key}')
                if source == 'coordinator_server_busy_fraction' and value > 1:
                    raise ValueError('coordinator busy fraction exceeds one')
                row[f'{arm}_{column}'] = value * scale
        rows.append(row)
    if observed != expected:
        raise ValueError('all 32 reporting cells must be present')
    return sorted(rows, key=lambda row: (row['block_id'], row['regime'], row['batch_tokens'], row['rho'], row['pattern'] != 'constant'))


def markdown_table(rows):
    headers = ['Source block', 'Domain', 'B', 'Arrivals', 'ρ'] + [label + ' N / O' for _, _, label, _ in METRICS]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        fields = [str(row[field]) for field in FACTORS]
        for _, column, _, _ in METRICS:
            fields.append(f"{row['native_' + column]:.3f} / {row['online_' + column]:.3f}")
        lines.append('| ' + ' | '.join(fields) + ' |')
    return '\n'.join(lines)


def plot_response(rows, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    states = sorted({(row['block_id'], row['regime']) for row in rows})
    indexed = {(row['block_id'], row['regime'], row['batch_tokens'], row['pattern'], row['rho']): row for row in rows}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5), squeeze=False)
        for ri, rho in enumerate((.5, .85)):
            for bi, batch in enumerate((128, 1024)):
                ax = axes[ri, bi]
                for si, (block, domain) in enumerate(states):
                    for pattern, offset, marker in (('constant', -.16, 'o'), ('burst8', .16, 's')):
                        row = indexed[block, domain, batch, pattern, rho]
                        x = si + offset
                        native, online = row['native_response_mean_ms'], row['online_response_mean_ms']
                        ax.plot([x-.035, x+.035], [native, online], color='.5', linewidth=.7, zorder=1)
                        for arm, dx, value in (('native', -.035, native), ('online', .035, online)):
                            ax.plot(x+dx, value, marker=marker, markersize=5.5, linestyle='none',
                                    markeredgecolor='black', markerfacecolor='black' if arm == 'native' else 'white', zorder=2)
                ax.set_title(f'B = {batch}; ρ = {rho:.2f}')
                ax.set_xticks(range(len(states)), [f'{block}\n{domain}' for block, domain in states])
                ax.tick_params(axis='x', labelsize=7.5)
                ax.set_ylabel('Mean response (ms)')
                ax.set_ylim(bottom=0)
                ax.grid(axis='y', color='.88', linewidth=.5)
                ax.set_axisbelow(True)
        legend = [Line2D([], [], marker=marker, markersize=5.5, linestyle='none', markeredgecolor='black',
                        markerfacecolor='black' if arm == 'native' else 'white', label=f'{pattern}: {arm}')
                  for pattern, marker in (('Constant', 'o'), ('Burst of 8', 's')) for arm in ('native', 'online')]
        fig.legend(handles=legend, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .97))
        fig.text(.5, .015, 'Each point averages two trial means. Two reused development source blocks; one allocation.\n'
                 'Synthetic fixed-batch layer-job arrivals; descriptive comparisons, with no confidence intervals.',
                 ha='center', va='bottom', fontsize=8)
        fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.14, hspace=.42, wspace=.26)
        for suffix in ('png', 'pdf'):
            fig.savefig(output_dir / f'arrival-response-means.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)


def report(summary_path, output_dir):
    summary_path, output_dir = Path(summary_path), Path(output_dir)
    summary = json.loads(summary_path.read_text())
    rows = table_rows(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'arrival-cells.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    note = ('# Supplementary layer-job arrivals\n\n'
        'All 32 prespecified cells are shown. N / O denotes native / online. Each number is the arithmetic mean '
        'of two trial-level summaries; the p95 column averages the two within-trial p95 values and is not a pooled percentile. '
        'Each trial contains 256 jobs. The two reused development source-document pairs and one allocation support a descriptive '
        'comparison, without confidence intervals, p-values, or a new performance acceptance decision.\n\n'
        'Response is completion minus immutable scheduled arrival. Queue delay includes admission delay; service includes '
        'CPU dispatch/ACK and GPU synchronization. Server busy fraction is coordinator service occupancy, not GPU utilization. '
        'Both arms completing every job by the same horizon have equal horizon throughput by construction.\n\n')
    note += markdown_table(rows) + '\n\n![Paired mean response times](arrival-response-means.png)\n\n'
    note += ('The figure shows every cell separately, with filled native and open online markers. Circles denote constant arrivals '
             'and squares denote bursts of eight; short connectors identify a paired comparison. Panel scales may differ. '
             'The raw summary retains both technical repetitions, all quantiles, horizon throughput, window spans, and observation overshoot.\n')
    (output_dir / 'arrival-cells.md').write_text(note)
    plot_response(rows, output_dir)
    provenance = dict(summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=summary.get('plan_sha256'), arrival_source_sha256=summary.get('arrival_source_sha256'),
        allocation_ids=summary['allocation_ids'], cells=32, trials=128,
        source='Validated production analysis summary; no raw observations are fabricated or filled in.')
    (output_dir / 'report-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.summary, args.output_dir)))


if __name__ == '__main__':
    main()
