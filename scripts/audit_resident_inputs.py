#!/usr/bin/env python3
"""Independently audit fresh capture inputs without inspecting routing distributions.

Checks artifact/content hashes, exact capture source reuse, checkpoint provenance,
source isolation, routing-array validity, and deterministic replay indices. It
never computes expert-load histograms, runtime outcomes, or tuning preferences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resident_dispatch.corpus.validation import validate_source_manifest, validate_captured_provenance

MODEL = 'Qwen/Qwen3-30B-A3B'
REVISION = 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
MODELING = 'b2666af64b7b31a02a152e9187ed2c5671262543b61510d0359c26fa491105ca'
MASKING = '50a737f63d8c778a5597fa34ac139049af921f44958205e3e1c29fe2bae77254'
REGIMES = ('stable', 'transient', 'persistent', 'switching')
BANK_ARRAYS = {'hidden_states_bf16', 'expert_ids', 'combining_weights', 'token_ids',
               'prompt_index', 'token_position', 'domain_ids'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz_contents(path, expected_hash):
    assert file_hash(path) == expected_hash, f'Artifact file checksum mismatch: {path}'
    with np.load(path, allow_pickle=False) as saved:
        assert len(saved.files) == len(set(saved.files)), 'Repeated NPZ member'
        metadata = json.loads(str(saved['metadata']))
        arrays = {name: saved[name].copy() for name in saved.files
                  if name not in ('metadata', 'content_sha256')}
        h = hashlib.sha256(canonical(metadata).encode())
        for name, array in sorted(arrays.items()):
            a = np.ascontiguousarray(array)
            h.update(canonical([name, a.dtype.str, list(a.shape)]).encode())
            h.update(a.tobytes())
        assert h.hexdigest() == str(saved['content_sha256']), f'Content checksum mismatch: {path}'
    return metadata, arrays


def validate_bank_arrays(bank):
    """Array-level checks only; never construct an expert-demand histogram."""
    assert set(bank) == BANK_ARRAYS, 'Unexpected bank array schema'
    hidden = bank['hidden_states_bf16']
    assert hidden.shape == (4096, 2048) and hidden.dtype == np.uint16
    assert np.all((hidden & 0x7f80) != 0x7f80), 'Nonfinite BF16 hidden value'
    ids, weights = bank['expert_ids'], bank['combining_weights']
    assert ids.shape == (4096, 8) and ids.dtype == np.int64
    assert np.all((ids >= 0) & (ids < 128)), 'Logical expert outside checkpoint'
    assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0), 'Repeated expert within a token'
    assert weights.shape == (4096, 8) and weights.dtype == np.float32
    assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
    assert np.all(np.abs(weights.sum(axis=1)-1) <= .004)
    assert np.all((weights.view(np.uint32) & 0xffff) == 0), 'Combining values do not retain exact BF16 rounding'
    assert bank['domain_ids'].dtype == np.int8 and np.array_equal(bank['domain_ids'], np.repeat([0, 1], 2048))
    assert bank['prompt_index'].dtype == np.int32 and np.array_equal(bank['prompt_index'], bank['domain_ids'])
    assert bank['token_position'].dtype == np.int32 and np.array_equal(bank['token_position'], np.tile(np.arange(2048), 2))
    assert bank['token_ids'].shape == (4096,) and bank['token_ids'].dtype == np.int64
    assert np.all(bank['token_ids'] >= 0)


def expected_positions(batch, seed=20471):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(2048, size=batch, replace=False) for _ in range(544)])


def expected_domains(regime):
    domains = np.zeros(544, np.int64)
    if regime == 'transient':
        domains[152:168] = 1
    elif regime == 'persistent':
        domains[152:] = 1
    elif regime == 'switching':
        for start in range(144, 544, 32):
            domains[start:start+16] = 1
    else:
        assert regime == 'stable'
    return domains


def verify_replay_arrays(tape, arrays, batch, positions):
    assert set(arrays) == {'indices'}, 'Unexpected replay tensor'
    domains = expected_domains(tape['regime'])
    assert np.array_equal(domains, tape['domain_schedule'])
    indices = arrays['indices']
    assert indices.dtype == np.int64 and indices.shape == (544, batch)
    assert np.array_equal(indices, positions + 2048*domains[:, None]), 'Replay indices differ from independently regenerated schedule'
    uses = np.bincount(indices.ravel(), minlength=4096)  # Captured-record reuse, never expert load.
    assert np.array_equal(uses, tape['token_use_counts'])
    assert tape['unique_captured_records_used'] == int((uses > 0).sum())
    assert tape['total_replayed_tokens'] == indices.size
    return indices


def cleaned_source(source, root):
    path = root/source['file']
    assert file_hash(path) == source['sha256']
    text = path.read_text(encoding='utf-8-sig')
    if source['cleaning'] == 'content_between_gutenberg_start_end_markers':
        boundaries = [re.search(r'\*\*\* '+name+r' OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', text)
                      for name in ('START', 'END')]
        assert all(boundaries) and boundaries[0].end() < boundaries[1].start()
        text = text[boundaries[0].end():boundaries[1].start()].strip()
    else:
        assert source['cleaning'] == 'none'
    digest = hashlib.sha256(text.encode()).hexdigest()
    if 'cleaned_text_sha256' in source:
        assert digest == source['cleaned_text_sha256']
    return digest


def audit(capture_dir, source_path, previous_bank=None):
    index_path = capture_dir/'capture-index.json'
    manifest = json.loads(source_path.read_text())
    source_audit = validate_source_manifest(manifest, source_path.parent)
    assert source_audit['source_documents'] == 16 and source_audit['prior_documents_excluded'] == 40
    prior_paths = [ROOT/name for name in ('followup/corpus/sources.json', 'optimization/corpus/sources.json')]
    assert manifest['excluded_source_manifests'] == [
        {'path': str(p.relative_to(ROOT)), 'sha256': file_hash(p)} for p in prior_paths]
    inventory = {b['block_id']: b for b in manifest['blocks']}
    assert len(inventory) == len(manifest['blocks']) == 8
    assert set(inventory) == {f'rd-final-{i:02d}' for i in range(1, 9)}
    assert all(b['split'] == 'final' for b in inventory.values())
    paths = [capture_dir/'banks'/f'{block}.npz' for block in inventory]
    paths += [capture_dir/'tapes'/f'{block}-{regime}-b{batch}.npz'
              for block in inventory for regime in REGIMES for batch in (128, 1024)]
    missing = [str(p) for p in [index_path, *paths] if not p.exists()]
    if missing:
        return {'status': 'incomplete', 'missing_files': missing, 'expected_bank_files': 8,
                'expected_replay_files': 64, 'routing_distributions_inspected': False,
                'runtime_or_tuning_outcomes_inspected': False}
    index = json.loads(index_path.read_text())
    assert len({r['block_id'] for r in index}) == len(index), 'Repeated capture-index block'
    assert {r['block_id'] for r in index} <= set(inventory), 'Unexpected capture-index block'
    if len(index) != 8:
        return {'status': 'incomplete', 'missing_files': [], 'capture_index_blocks': len(index),
                'expected_bank_files': 8, 'expected_replay_files': 64,
                'routing_distributions_inspected': False, 'runtime_or_tuning_outcomes_inspected': False}
    assert set((capture_dir/'banks').glob('*.npz')) == set(paths[:8]), 'Unexpected bank artifacts'
    assert set((capture_dir/'tapes').glob('*.npz')) == set(paths[8:]), 'Unexpected replay artifacts'
    old_sources = [(s, p.parent) for p in prior_paths
                   for b in json.loads(p.read_text())['blocks'] for s in b['sources']]
    new_sources = [s for b in inventory.values() for s in b['sources']]
    assert len(new_sources) == 16 and len(old_sources) == 40
    for field in ('source_id', 'url', 'sha256'):
        new = [s[field] for s in new_sources]
        assert len(set(new)) == 16 and not set(new) & {s[field] for s, _ in old_sources}
    old_cleaned = {cleaned_source(s, parent) for s, parent in old_sources}
    new_cleaned = {cleaned_source(s, source_path.parent) for s in new_sources}
    assert len(new_cleaned) == 16 and not new_cleaned & old_cleaned
    assert file_hash(source_path.parent/'raw/cpython_LICENSE') == manifest['cpython_license_sha256']
    for source in new_sources:
        if source['domain'] == 'prose':
            path = source_path.parent/source['metadata_file']
            assert file_hash(path) == source['metadata_sha256']
            assert 'Public domain in the USA' in path.read_text()
    capture_script = ROOT/'scripts/capture_resident_tapes.py'
    predecessor = ROOT/'scripts/capture_optimization_tapes.py'
    old_code, new_code = predecessor.read_text(), capture_script.read_text()
    for start, stop in [('    config_path, index_path =', '    index_file ='),
                        ('        for domain, source in enumerate', '        arrays =')]:
        assert old_code[old_code.index(start):old_code.index(stop)] == new_code[new_code.index(start):new_code.index(stop)]
    model_fields = ('source_model', 'source_revision', 'layer', 'num_experts', 'top_k', 'hidden_size',
                    'source_shards', 'source_tensor_sha256', 'config_sha256', 'index_sha256',
                    'tokenizer_files', 'transformers_version', 'modeling_sha256', 'masking_sha256')
    previous_metadata = None
    if previous_bank:
        with np.load(previous_bank, allow_pickle=False) as saved:
            previous_metadata = json.loads(str(saved['metadata']))
    common_model = None
    artifacts = []
    # Seed-only position draws are independent of observed expert routing.
    positions = {}
    for batch in (128, 1024):
        positions[batch] = expected_positions(batch)
    for entry in index:
        block = inventory[entry['block_id']]
        bank_path = capture_dir/'banks'/f"{block['block_id']}.npz"
        assert Path(entry['bank_file']).name == bank_path.name
        metadata, bank = npz_contents(bank_path, entry['capture_sha256'])
        validate_captured_provenance(metadata, manifest, source_path)
        assert entry['correctness_pass'] is True and entry['tokens'] == 4096 and entry['split'] == block['split']
        expected = {'schema': 'qwen_router_bank_v1', 'block_id': block['block_id'],
            'split': block['split'], 'source_model': MODEL, 'source_revision': REVISION,
            'layer': 0, 'num_experts': 128, 'top_k': 8, 'hidden_size': 2048,
            'source_manifest_sha256': file_hash(source_path), 'study_id': 'resident_dispatch_v1',
            'origin': 'teacher_forced_causal_prefill_layer0', 'native_serving_trace': False,
            'source_prompt_ids': [s['source_id'] for s in block['sources']],
            'attention_backend': 'eager', 'special_tokens_added': False, 'chat_template': None,
            'correctness_pass': True, 'transformers_version': '5.17.0',
            'modeling_sha256': MODELING, 'masking_sha256': MASKING,
            'capture_script_sha256': file_hash(capture_script),
            'capture_computation_predecessor_sha256': file_hash(predecessor),
            'source_validation_module_sha256': file_hash(ROOT/'resident_dispatch/corpus/validation.py'),
            'source_text_module_sha256': file_hash(ROOT/'optimization/input_sources.py'),
            'router_tapes_module_sha256': file_hash(ROOT/'followup/router_tapes.py'),
            'excluded_source_manifests': manifest['excluded_source_manifests']}
        assert all(metadata.get(k) == v for k, v in expected.items()), block['block_id']
        assert metadata['source_disjointness_audit'] == source_audit
        assert metadata['source_disjointness_audit']['overlap_by_id_url_raw_or_cleaned_content'] is False
        current_model = {field: metadata[field] for field in model_fields}
        if common_model is None:
            common_model = current_model
        assert current_model == common_model
        if previous_metadata:
            assert current_model == {field: previous_metadata[field] for field in model_fields}
        validate_bank_arrays(bank)
        assert len(metadata['prompts']) == 2
        for domain, (prompt, source) in enumerate(zip(metadata['prompts'], block['sources'])):
            assert all(prompt[field] == value for field, value in source.items())
            assert prompt['captured_tokens'] == 2048 and prompt['total_source_tokens'] >= 2048
            assert prompt['token_offset'] == prompt['position_start'] == 0
            assert all(prompt[flag] is True for flag in ('native_boundary_exact', 'native_router_exact',
                                                       'manual_router_exact', 'bf16_roundtrip_exact'))
            assert prompt['token_ids_sha256'] == hashlib.sha256(bank['token_ids'][domain*2048:(domain+1)*2048].tobytes()).hexdigest()
        artifacts.append({'kind': 'bank', 'file': str(bank_path.relative_to(capture_dir)),
                          'sha256': entry['capture_sha256'], 'block_id': block['block_id'], 'split': block['split']})
        replay_entries = {Path(row['file']).name: row for row in entry['replays']}
        assert len(replay_entries) == len(entry['replays']) == 8
        for batch in (128, 1024):
            paired_indices = {}
            for regime in REGIMES:
                path = capture_dir/'tapes'/f"{block['block_id']}-{regime}-b{batch}.npz"
                tape, data = npz_contents(path, replay_entries[path.name]['sha256'])
                phase = 112 if regime == 'switching' else 120
                declared = {'schema': 'qwen_router_replay_v1', 'block_id': block['block_id'], 'split': block['split'],
                    'regime': regime, 'batch_tokens': batch, 'source_model': MODEL, 'source_revision': REVISION,
                    'layer': 0, 'source_prompt_ids': expected['source_prompt_ids'], 'capture_sha256': entry['capture_sha256'],
                    'num_steps': 544, 'warmup_steps': 32, 'phase_steps': phase, 'duration_steps': 16,
                    'repeat_steps': 1024, 'seed': 20471, 'origin': 'constructed_repeated_prefill_replay',
                    'natural_temporal_demand': False}
                assert all(tape[k] == v for k, v in declared.items())
                assert (path.parent/tape['bank_file']).resolve() == bank_path.resolve()
                indices = verify_replay_arrays(tape, data, batch, positions[batch])
                paired_indices[regime] = indices
                artifacts.append({'kind': 'replay', 'file': str(path.relative_to(capture_dir)),
                    'sha256': replay_entries[path.name]['sha256'], 'block_id': block['block_id'],
                    'split': block['split'], 'regime': regime, 'batch_tokens': batch})
            assert np.array_equal(paired_indices['transient'][:168], paired_indices['persistent'][:168])
            assert np.all(paired_indices['transient'][168:] < 2048)
            assert np.all(paired_indices['persistent'][168:] >= 2048)
            assert all(np.array_equal(indices[:144], paired_indices['stable'][:144]) for indices in paired_indices.values())
    return {'status': 'verified', 'banks': 8, 'replays': 64, 'captured_token_records': 32768,
        'source_document_pairs': {'pilot': 0, 'final': 8}, 'source_documents': 16,
        'prior_documents_excluded': 40, 'source_manifest_sha256': file_hash(source_path),
        'audit_script_sha256': file_hash(__file__),
        'capture_index_sha256': file_hash(index_path), 'all_artifact_and_content_hashes_verified': True,
        'source_ids_urls_raw_and_cleaned_hashes_distinct': True,
        'capture_checkpoint_loading_and_router_computation_exact_predecessor': True,
        'model_checkpoint_tokenizer_provenance_identical_across_banks': True,
        'model_provenance_matches_previous_capture': previous_bank is not None,
        'previous_bank_sha256': file_hash(previous_bank) if previous_bank is not None else None,
        'captured_hidden_and_routing_array_validity_pass': True,
        'all_native_capture_correctness_attestations_pass': True,
        'all_replay_indices_independently_regenerated': True,
        'current_trace_contract': {'raw_steps': 544, 'warmup_steps': 32, 'measured_steps': 512,
            'phase_steps': 120, 'switching_phase_steps': 112, 'duration_steps': 16, 'repeat_steps': 1024, 'seed': 20471},
        'same_observed_transient_persistent_history': True,
        'routing_distributions_inspected': False, 'runtime_or_tuning_outcomes_inspected': False,
        'limits': ['This is a same-team CPU audit, not external-lab reproduction. It verifies actual artifacts and recorded GPU capture attestations; it does not independently rerun pretrained attention/router computation.',
                   'Valid unique top-k IDs and rounded combining weights do not independently prove their learned-router origin; pinned capture computation and its exactness attestations supply that evidence.',
                   'Document disjointness concerns document identities and whole-content hashes, not absence of shared language, comments, authors, or project structure.',
                   'Constructed repeated-prefill replay and fixed document blocks are not natural serving traffic or random production samples.'],
        'artifacts': artifacts}


def main():
    if not __debug__:
        raise SystemExit('Run without -O: this audit requires assertions enabled.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--sources', type=Path, default=ROOT/'resident_dispatch/corpus/sources.json')
    parser.add_argument('--previous-bank', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    for suffix in ('.json', '.md'):
        if args.output_prefix.with_suffix(suffix).exists():
            raise FileExistsError('Use a fresh audit output prefix')
    result = audit(args.capture_dir, args.sources, args.previous_bank)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] == 'verified':
        text = ('Resident-dispatch input audit: verified all 8 banks and 64 replay tapes.\n\n'
                'All file/content hashes, source exclusions, capture provenance, and array-validity checks passed. '
                'The capture checkpoint loading and native router computation match the frozen predecessor exactly. '
                'All replay indices were independently regenerated from seed 20471; raw 544/warm 32/phase 120/switch 112/duration 16/repeat 1024 match. '
                'Transient/persistent histories match through raw167 and then diverge as declared.\n\n'
                'No expert-demand distributions, runtime results, or tuning choices were inspected.\n\n'
                + '\n'.join('- '+item for item in result['limits'])+'\n')
    else:
        text = ('Audit incomplete: '+str(len(result['missing_files']))+' expected input files are missing; '
                'capture-index blocks: '+str(result.get('capture_index_blocks', 'not checked'))+'.\n')
    args.output_prefix.with_suffix('.md').write_text(text)
    summary = {k: v for k, v in result.items() if k not in ('artifacts', 'missing_files')}
    if 'missing_files' in result:
        summary['missing_file_count'] = len(result['missing_files'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
