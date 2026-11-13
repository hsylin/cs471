"""Read-only, post-hoc accounting for the optimization review; no model tuning."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from full_serving.resident_policy import balanced_map


def audit(repo):
    report_path = repo / 'outputs/full-serving/proposal-followup-comparison.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'completed_and_audited'
    root = repo / 'results/full-serving/proposal-followup-v1/run-v1'
    result = {
        'status': 'post_hoc_source_and_record_audit_not_new_performance_measurement',
        'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'limitations': [
            'One group; ratios are descriptive, not estimates of repeatability.',
            'Host planning duration is not exposed critical-path delay and excludes subsequent JSON emission.',
            'Aggregated windows cannot identify sampling noise or within-window temporal ordering.',
            'Recorded received weight payload is not wire traffic or proof of completed migration.',
        ],
        'datasets': {},
        'gate_cycles': {},
    }
    for dataset in report['datasets']:
        arms = dataset['arms']
        off = arms['off']['metrics']
        result['datasets'][dataset['dataset']] = {
            arm: {
                'metrics': data['metrics'],
                'percent_vs_off': {
                    key: 100 * (value / off[key] - 1)
                    for key, value in data['metrics'].items()
                },
            } for arm, data in arms.items()
        }
    for path in sorted(root.rglob('gate-events-rank0.jsonl')):
        cycles = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] != 'decision':
                continue
            layers = row['layers']
            tv = [x['tv'] for x in layers if x['tv'] is not None]
            counts = [(x['previous_assignments'], x['recent_assignments']) for x in layers]
            cycles.append({
                'step': row['step'],
                'host_plan_and_gate_ms': row['host_plan_and_gate_ns'] / 1e6,
                'reasons': dict(Counter(x['reason'] for x in layers)),
                'tv_range': [min(tv), max(tv)] if tv else None,
                'layer0_previous_recent_assignments': list(counts[0]),
                'recent_over_previous_total_range': [
                    min(b / a for a, b in counts if a),
                    max(b / a for a, b in counts if a),
                ] if any(a for a, _ in counts) else None,
                'candidate_changed_slots_range': [
                    min(x['changed_physical_slots'] for x in layers),
                    max(x['changed_physical_slots'] for x in layers),
                ],
            })
        result['gate_cycles'][str(path.relative_to(root))] = cycles
    mapping = balanced_map()
    copies = Counter(mapping)
    edges = set()
    for expert, count in copies.items():
        if count > 1:
            ranks = tuple(sorted({slot // 20 for slot, value in enumerate(mapping) if value == expert}))
            edges.add(ranks)
    assert len(mapping) == 160 and len(copies) == 128
    assert Counter(copies.values()) == {1: 96, 2: 32}
    assert edges == {(0, 1), (2, 3), (4, 5), (6, 7)}
    result['resident_placement'] = {
        'logical_experts': 128, 'physical_slots': 160,
        'single_copy_experts': 96, 'duplicated_experts': 32,
        'rank_pair_edges': sorted(map(list, edges)),
        'limit': 'Neither current dispatcher can move work between these four components.',
    }
    result['gate_storage'] = {
        'logical_history_bytes_per_rank': 2000 * 48 * 129 * 8,
        'two_reduced_histograms_bytes': 2 * 48 * 128 * 8,
        'interpretation': 'Source-derived storage sizes, not measured bandwidth or latency.',
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.repo), indent=2) + '\n')


if __name__ == '__main__':
    main()
