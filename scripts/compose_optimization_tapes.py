#!/usr/bin/env python3
"""Compose fresh whole-policy replays without changing captured banks.

Uses raw544 = warm32 + measured512, phase120/112, duration16,
one transient (repeat1024), and seed10471. No performance is inspected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from followup.router_tapes import compose_router_tape, file_sha256, load_router_bank
from optimization.input_sources import validate_source_manifest


def compose_directory(banks_dir: Path, output_dir: Path, split: str = 'all',
                      sources: Path = ROOT/'optimization/corpus/sources.json') -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Use a new empty tape directory; existing inputs are never overwritten')
    if output_dir.resolve() == banks_dir.resolve():
        raise ValueError('Replay output must differ from the bank directory')
    manifest = json.loads(sources.read_text())
    validate_source_manifest(manifest, sources.parent)
    expected = {b['block_id']: b for b in manifest['blocks'] if split == 'all' or b['split'] == split}
    banks = []
    for path in sorted(banks_dir.glob('*.npz')):
        bank = load_router_bank(path)
        if split != 'all' and bank.metadata['split'] != split:
            continue
        if bank.metadata['block_id'] not in expected:
            raise ValueError('Captured bank is not in the fresh source inventory')
        block = expected[bank.metadata['block_id']]
        if (bank.metadata.get('study_id') != 'kernel_optimization_v1'
                or bank.metadata['source_manifest_sha256'] != file_sha256(sources)
                or bank.metadata['source_prompt_ids'] != [s['source_id'] for s in block['sources']]
                or bank.metadata['split'] != block['split']):
            raise ValueError('Fresh capture provenance differs from source manifest')
        banks.append(bank)
    if len(banks) != len(expected) or {b.metadata['block_id'] for b in banks} != set(expected):
        raise ValueError('Missing or duplicate captured source blocks')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bank in banks:
        for batch in (128, 1024):
            for regime in ('stable', 'transient', 'persistent', 'switching'):
                path = output_dir/f"{bank.metadata['block_id']}-{regime}-b{batch}.npz"
                tape = compose_router_tape(bank.path, path, regime=regime, batch_tokens=batch,
                    num_steps=544, warmup_steps=32, phase_steps=112 if regime == 'switching' else 120,
                    duration_steps=16, repeat_steps=1024, seed=10471)
                schedule = tape.metadata['domain_schedule']
                if regime == 'transient' and (sum(schedule) != 16 or schedule[152:168] != [1]*16):
                    raise AssertionError('Transient differs from the declared single event')
                rows.append({'block_id': bank.metadata['block_id'], 'split': bank.metadata['split'],
                    'regime': regime, 'batch_tokens': batch, 'path': str(path),
                    'sha256': tape.sha256, 'capture_sha256': bank.sha256})
        if file_sha256(bank.path) != bank.sha256:
            raise AssertionError('Capture bank changed during composition')
        print(json.dumps({'event': 'block_composed', 'block_id': bank.metadata['block_id'], 'replays': 8}), flush=True)
    result = {'schema': 'qwen_optimization_replay_index_v1',
        'source_manifest_sha256': file_sha256(sources),
        'composition_script_sha256': file_sha256(__file__),
        'router_tapes_module_sha256': file_sha256(ROOT/'followup/router_tapes.py'),
        'contract': {'raw_steps': 544, 'warmup_steps': 32, 'measured_steps': 512,
            'phase_steps': 120, 'switching_phase_steps': 112, 'duration_steps': 16,
            'repeat_steps': 1024, 'seed': 10471, 'natural_temporal_demand': False},
        'capture_provenance': 'Original bank contents and capture-script hashes preserved', 'rows': rows}
    (output_dir/'replay-index.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=('pilot', 'final', 'all'), default='all')
    p.add_argument('--sources', type=Path, default=ROOT/'optimization/corpus/sources.json')
    args = p.parse_args()
    result = compose_directory(args.banks_dir, args.output_dir, args.split, args.sources)
    print(json.dumps({'event': 'composition_complete', 'replays': len(result['rows'])}), flush=True)


if __name__ == '__main__':
    main()
