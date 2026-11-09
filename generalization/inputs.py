"""Authenticated new-content inputs for fixed-token layer-service jobs.

Each job samples complete captured records. The grouping is constructed: it is
not an autoregressive request, a natural batch, or a new independent document.
The seed and indices do not depend on treatment, calendar, or repetition.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from followup.router_tapes import load_router_bank
from optimization.common import array_hash, canonical_hash

INPUT_SEED = 471503
BATCH_TOKENS = 128
WARMUP_JOBS = 16
MEASURED_JOBS = 256
TOTAL_JOBS = WARMUP_JOBS + MEASURED_JOBS


def _positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'Invalid {name}')


def build_job_indices(record_count, block_id, batch_tokens=BATCH_TOKENS, *, seed=INPUT_SEED):
    """Return 272 distinct unordered record combinations, ordered within jobs.

    A fresh PCG64 permutation is derived independently for every raw job. A
    deterministic retry rejects the exceedingly unlikely repeated combination.
    Reusing a captured record in different jobs is allowed and fully counted.
    """
    _positive_int(record_count, 'captured record count')
    _positive_int(batch_tokens, 'batch tokens')
    _positive_int(seed, 'input seed', minimum=0)
    if not isinstance(block_id, str) or not block_id:
        raise ValueError('A source block identity is required')
    if batch_tokens != BATCH_TOKENS:
        raise ValueError('This frozen study requires B128')
    if record_count < batch_tokens or math.comb(record_count, batch_tokens) < TOTAL_JOBS:
        raise ValueError('Insufficient distinct captured-record combinations')
    indices, seen, retries = [], set(), 0
    for raw in range(TOTAL_JOBS):
        for attempt in range(1024):
            key = [seed, block_id, batch_tokens, raw, attempt]
            job_seed = int(canonical_hash(key)[:16], 16)
            rng = np.random.Generator(np.random.PCG64(job_seed))
            picked = rng.permutation(record_count)[:batch_tokens].astype(np.int64)
            identity = np.sort(picked).tobytes()
            if identity not in seen:
                seen.add(identity)
                indices.append(picked)
                retries += attempt
                break
        else:
            raise ValueError('Unable to construct distinct combinations within fixed retry bound')
    result = np.stack(indices)
    reuse = reuse_summary(result, record_count)
    metadata = {
        'schema': 'generalization_layer_jobs_v1', 'block_id': block_id,
        'seed': seed, 'batch_tokens': batch_tokens, 'jobs': TOTAL_JOBS,
        'warmup_range_half_open': [0, WARMUP_JOBS],
        'measured_range_half_open': [WARMUP_JOBS, TOTAL_JOBS],
        'rng': 'NumPy PCG64; independent full-bank permutation per raw job',
        'seed_derivation': 'uint64 first16hex SHA256(canonical JSON [seed,block_id,B,raw_index,retry])',
        'combination_collision_retries': retries,
        'selection_independent_of_arm_calendar_repetition': True,
        'natural_request_or_batch_trace': False,
        'sampling': '128 distinct captured records per job; records may recur across jobs; 272 distinct unordered combinations',
        'indices_sha256': array_hash(result), 'job_reuse': reuse,
    }
    return result, metadata


def reuse_summary(indices, record_count):
    """Count source reuse; distinct job combinations are not distinct tokens."""
    a = np.asarray(indices)
    if a.dtype != np.int64 or a.shape != (TOTAL_JOBS, BATCH_TOKENS):
        raise ValueError('Require int64 indices[272,128]')
    if (a < 0).any() or (a >= record_count).any():
        raise ValueError('Index outside captured bank')
    if (np.diff(np.sort(a, axis=1), axis=1) == 0).any():
        raise ValueError('A captured record occurs twice within a job')
    unique = len({row.tobytes() for row in np.sort(a, axis=1)})
    measured_unique = len({row.tobytes() for row in np.sort(a[WARMUP_JOBS:], axis=1)})
    if unique != TOTAL_JOBS or measured_unique != MEASURED_JOBS:
        raise ValueError('Repeated job combination')
    def describe(x):
        counts = np.bincount(x.ravel(), minlength=record_count)
        return {'token_record_occurrences': int(x.size),
                'distinct_captured_records_used': int(np.count_nonzero(counts)),
                'unused_captured_records': int(np.count_nonzero(counts == 0)),
                'minimum_uses_including_unused': int(counts.min()),
                'maximum_uses': int(counts.max()),
                'record_uses_sha256': array_hash(counts)}
    return {'bank_records': record_count, 'unique_all_job_combinations': unique,
            'unique_measured_job_combinations': measured_unique,
            'all_jobs': describe(a), 'warmup_jobs': describe(a[:WARMUP_JOBS]),
            'measured_jobs': describe(a[WARMUP_JOBS:]),
            'independent_unit_warning': 'Resampled jobs share captured tokens/documents; source block, calendar block and technical repetition are distinct units.'}


def validate_input_bank(bank, entry, weights_config, plan, group):
    """Check checkpoint, selected content and every captured record's identity."""
    m = bank.metadata
    if bank.sha256 != entry.get('capture_sha256'):
        raise ValueError('Bank file differs from frozen capture SHA')
    if plan.get('input_seed') != INPUT_SEED:
        raise ValueError('Missing or changed frozen input seed')
    if m.get('selection_content_sha256') != plan.get('content_selection_sha256') or not plan.get('content_selection_sha256'):
        raise ValueError('Bank belongs to a different frozen content selection')
    if m.get('study_id') != 'generalization_v1' or m.get('correctness_pass') is not True:
        raise ValueError('Missing generalization capture/correctness attestation')
    if m.get('native_serving_trace') is not False or m.get('timing_evidence') is not False:
        raise ValueError('Captured records must be untimed constructed-input evidence')
    if m.get('origin') != 'prompt_causal_prefill_layer0':
        raise ValueError('Expected authentic causal prefill layer0 inputs')
    if group.get('regime') != 'newcontent' or group.get('batch_tokens') != BATCH_TOKENS:
        raise ValueError('Wrong new-content input group')
    expected = {'block_id': group['block_id'], 'dataset': group['dataset'],
                **{k: weights_config[k] for k in ('source_model', 'source_revision', 'layer', 'num_experts', 'top_k', 'hidden_size')}}
    if entry.get('block_id') != group['block_id']:
        raise ValueError('Capture entry differs from source block')
    for k, v in expected.items():
        if m.get(k) != v:
            raise ValueError(f'Bank/checkpoint identity differs: {k}')
    if (m['layer'], m['num_experts'], m['top_k'], m['hidden_size']) != (0, 128, 8, 2048):
        raise ValueError('Unsupported model geometry')
    if m.get('split') not in ('pilot', 'final') or ('split' in entry and m['split'] != entry['split']):
        raise ValueError('Capture split differs from frozen inventory')
    # A calibration may use either split, but each row must be pinned by plan.
    if plan.get('phase') == 'pilot' and m['split'] != 'pilot':
        raise ValueError('Pilot cannot inspect held-out final capture')
    if plan.get('phase') == 'final' and m['split'] != 'final':
        raise ValueError('Final cannot reuse pilot capture')
    prompts = m.get('prompts', [])
    names = m.get('source_prompt_ids', [])
    if not prompts or len(prompts) != len(names) or len(names) != len(set(names)):
        raise ValueError('Missing or duplicated source prompt identities')
    if 'source_prompt_ids' in entry and names != entry['source_prompt_ids']:
        raise ValueError('Source documents differ from frozen inventory')
    if set(np.unique(bank.prompt_index)) != set(range(len(prompts))):
        raise ValueError('Invalid/missing prompt index')
    if not np.array_equal(bank.domain_ids, bank.prompt_index % 2):
        raise ValueError('Legacy parity metadata differs from prompt provenance')
    for i, p in enumerate(prompts):
        if p.get('dataset') != group['dataset'] or names[i] != p['dataset'] + ':' + p['source_id']:
            raise ValueError('Captured prompt identity differs from its dataset/source')
        if p.get('prompt_id') != p.get('source_id'):
            raise ValueError('Captured prompt alias differs from source identity')
        if not all(p.get(k) is True for k in ('native_boundary_exact', 'native_router_exact', 'manual_router_exact', 'bf16_roundtrip_exact')):
            raise ValueError('Missing exact native capture attestation')
        mask = bank.prompt_index == i
        n = int(mask.sum())
        if n != p.get('captured_tokens') or not 1 <= n <= 512 or p.get('total_source_tokens', 0) < n:
            raise ValueError('Captured prompt length differs')
        if p.get('position_start') != 0 or p.get('token_offset') != 0 or not np.array_equal(bank.token_position[mask], np.arange(n)):
            raise ValueError('Captured positions are not the actual independent prompt prefix')
        if hashlib.sha256(np.asarray(bank.token_ids[mask], dtype=np.int64).tobytes()).hexdigest() != p.get('token_ids_sha256'):
            raise ValueError('Captured token IDs differ from source attestation')
    if (np.asarray(bank.hidden_states_bf16, dtype=np.uint16) & 0x7f80 == 0x7f80).any():
        raise ValueError('Nonfinite BF16 hidden state')
    return {'source_block': m['block_id'], 'dataset': m['dataset'], 'split': m['split'],
            'source_documents': len(prompts), 'captured_records': len(bank.expert_ids),
            'content_selection_sha256': m['selection_content_sha256'],
            'authentic_record_identity_verified': True}


def prepare(rt, weights, base, args, group, plan):
    """Build all tensors/references/oracles before any measured serving trial."""
    from followup.payback import _materialize
    from optimization.common import agree_config, make_layer
    from optimization.kernel_config import resolve_kernel_config
    from resident_dispatch.assignments import validate_assignment
    from resident_dispatch.common import references_for
    from resident_dispatch.online import verify_adversarial
    from resident_dispatch.pilot import _save_array_artifact
    from resident_dispatch.runner import _oracle, make_online
    from skewbench.runner import snapshot_weight_hash

    entries = [x for x in plan['banks'] if x['block_id'] == group['block_id']]
    if len(entries) != 1:
        raise ValueError('Exactly one frozen capture entry per source block required')
    entry = entries[0]
    bank = load_router_bank(Path(args.banks_dir) / entry['capture_path'])
    attestation = validate_input_bank(bank, entry, weights.config, plan, group)
    indices, input_meta = build_job_indices(len(bank.expert_ids), group['block_id'], group['batch_tokens'], seed=plan['input_seed'])
    input_meta.update(attestation)
    input_meta.update(capture_sha256=bank.sha256, source_prompt_ids=bank.metadata['source_prompt_ids'])
    sequence, warmup = list(range(WARMUP_JOBS, TOTAL_JOBS)), list(range(WARMUP_JOBS))
    conditioning = [i % WARMUP_JOBS for i in range(64)]
    reference_steps, fp32_steps = list(range(TOTAL_JOBS)), [WARMUP_JOBS, TOTAL_JOBS - 1]
    tensors = _materialize(rt, SimpleNamespace(bank=bank, indices=indices))
    logical = np.asarray(bank.expert_ids)[indices]
    layer = make_layer(rt, weights, base, 'SKIP')
    try:
        layer.assert_maps_and_weights()
        mapping = layer.map_snapshot()
        if mapping.tolist() != plan['physical_to_logical_map']:
            raise AssertionError('Resident mapping differs from frozen plan')
        config = resolve_kernel_config(layer, group['batch_tokens'])
        if config != base['default_configs'][str(group['batch_tokens'])]:
            raise AssertionError('Native DEFAULT kernel changed')
        agree_config(rt, config)
        weight_hash, rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
        refs = references_for(rt, layer, tensors, fp32_steps, raw_steps=reference_steps)
        mapper = make_online(rt, layer, group['batch_tokens'])
        adversarial = verify_adversarial(rt, layer, mapper, group['batch_tokens'])
        ls = layer.layer_state
        buf = rt.torch.zeros_like(ls.expert_load_view)
        record = rt.torch.ones((), device=rt.device, dtype=rt.torch.bool)
        real = rt.torch.tensor(group['batch_tokens'], device=rt.device, dtype=rt.torch.int32)
        counts, route_hashes, diagnostics, snapshots = {}, {}, {}, {}
        for arm, fn in [('native', layer.mapping_kernel), ('online', mapper)]:
            raw_counts, raw_diagnostics, sample = [], [], []
            digester = hashlib.sha256()
            for raw, ids_np in enumerate(logical):
                expected = _oracle(ids_np, mapping, config['BLOCK_SIZE_M'], arm)
                buf.zero_()
                actual = fn(tensors[1][raw], buf, ls.logical_to_physical_map,
                            ls.logical_replica_count, record, real).cpu().numpy()
                if not np.array_equal(actual, expected):
                    raise AssertionError(f'{arm} GPU assignment differs from CPU oracle at input {raw}')
                diag = validate_assignment(ids_np, actual, mapping, block_size_m=config['BLOCK_SIZE_M'])
                if not np.array_equal(buf.cpu().numpy(), diag['physical_counts']):
                    raise AssertionError('Actual physical counter differs from assignment')
                digester.update(np.asarray(actual, dtype=np.int64).tobytes())
                raw_counts.append(diag['physical_counts'])
                raw_diagnostics.append({k: diag[k] for k in ('per_rank_tokens', 'per_rank_active_copies', 'per_rank_m_tiles', 'per_rank_padded_rows', 'minimum_possible_max_rank_active_copies')})
                if raw in fp32_steps:
                    sample.append(np.asarray(actual, dtype=np.uint8))
            counts[arm] = np.asarray(raw_counts, dtype=np.int32)
            diagnostics[arm], snapshots[arm] = raw_diagnostics, np.stack(sample)
            route_hashes[arm] = digester.hexdigest()
        agree_config(rt, route_hashes)
        replay_hash = array_hash(indices)
        name = f"{group['block_id']}-newcontent-b{group['batch_tokens']}"
        metadata = dict(input_meta, replay_sha256=replay_hash,
                        logical_routing_sha256=array_hash(logical),
                        full_physical_route_sha256=route_hashes,
                        sample_raw_indices=fp32_steps,
                        reference_raw_indices=reference_steps,
                        route_scope='Every one of 272 GPU assignments checked for both arms; per-step counts and source-record indices saved; full routes reconstructible from pinned CPU oracles')
        artifact = _save_array_artifact(rt, Path(args.output).parent/'inputs'/f'{name}.npz',
            indices=indices, physical_to_logical_map=mapping,
            selected_prompt_index=bank.prompt_index[indices],
            selected_token_position=bank.token_position[indices], selected_token_ids=bank.token_ids[indices],
            all_record_use_counts=np.bincount(indices.ravel(), minlength=len(bank.expert_ids)),
            measured_record_use_counts=np.bincount(indices[WARMUP_JOBS:].ravel(), minlength=len(bank.expert_ids)),
            native_physical_counts=counts['native'], online_physical_counts=counts['online'],
            native_sample_physical_ids=snapshots['native'], online_sample_physical_ids=snapshots['online'],
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        layer.assert_maps_and_weights()
    finally:
        layer.close()
    return dict(tensors=tensors, logical=logical, references=refs, reference_steps=reference_steps,
        fp32_steps=fp32_steps, sequence=sequence, warmup=warmup, conditioning=conditioning,
        mapping=mapping, counts=counts, diagnostics=diagnostics, route_hashes=route_hashes,
        artifact=artifact, capture_sha256=bank.sha256, replay_sha256=replay_hash,
        source_prompt_ids=bank.metadata['source_prompt_ids'], source_weights_sha256=weight_hash,
        source_weight_rank_sha256=rank_hashes, resolved_kernel_config=config,
        adversarial_checks=adversarial, selector_diagnostic=None,
        job_reuse=input_meta['job_reuse'], newinputmeta=input_meta)
