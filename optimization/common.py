"""Identity and data contracts shared by the new optimization runners."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket

from followup.payback import followup_fingerprint, validate_execution_metadata
from optimization.kernel_config import verify_kernel_sources
from skewbench.native import NativeLayer


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'optimization').glob('*.py'))
    paths += [root / 'scripts' / name for name in
              ('capture_optimization_tapes.py', 'compose_optimization_tapes.py')]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths) if p.exists()}
    return canonical_hash(files), files


def manifest_hash(manifest):
    return canonical_hash({k:v for k,v in manifest.items() if k != 'freeze'})


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('optimization_schema_version') != 1:
        raise ValueError('Wrong optimization schema')
    if manifest.get('freeze', {}).get('sha256') != manifest_hash(manifest):
        raise ValueError('Missing or modified frozen manifest')
    return manifest


def runtime_metadata(rt, weights, manifest):
    legacy = rt.metadata['harness_sha256']
    followup, _ = followup_fingerprint()
    optimized, files = source_fingerprint()
    rt.metadata.update(dtype='bfloat16', legacy_harness_sha256=legacy,
        followup_sha256=followup, optimization_sha256=optimized,
        optimization_file_sha256=files,
        allocation_id=os.environ.get('SLURM_JOB_ID', 'unallocated'),
        hostname=socket.gethostname(),
        process_run_id=os.environ.get('TORCHELASTIC_RUN_ID', str(os.getpid())),
        kernel_source_attestation=verify_kernel_sources())
    validate_execution_metadata(manifest, rt.metadata, weights.config)
    for field in ('optimization_sha256', 'legacy_harness_sha256', 'followup_sha256'):
        if manifest['environment'].get(field) != rt.metadata[field]:
            raise ValueError(f'Unpinned or different execution source: {field}')
    if weights.config['weights_sha256'] != manifest['environment']['weights_sha256']:
        raise ValueError('Unpinned expert checkpoint')
    return dict(rt.metadata)


def make_layer(rt, weights, manifest, policy):
    p = manifest['policy']
    return NativeLayer(rt, weights, method=policy, replicas=p['replica_budget'],
        window_steps=manifest['trace']['window_steps'],
        rebalance_interval=p['rebalance_interval'], epsilon=p['epsilon'],
        tau=p['tau'], cooldown_steps=p['cooldown_steps'], seed=10471)


def array_hash(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def agree_config(rt, config):
    values = [None]*rt.world_size
    rt.torch.distributed.all_gather_object(values, config, group=rt.ep_group.cpu_group)
    if any(value != values[0] for value in values):
        raise AssertionError('Ranks disagree on the actual kernel configuration')
    return True


def check_tape(manifest, entry, tape):
    if tape.sha256 != entry['replay_sha256'] or tape.bank.sha256 != entry['capture_sha256']:
        raise ValueError('Tape/capture SHA differs from frozen inventory')
    if not tape.bank.metadata.get('correctness_pass'):
        raise ValueError('Capture correctness missing')
    for name in ('block_id', 'regime', 'batch_tokens'):
        if tape.metadata[name] != entry[name]:
            raise ValueError(f'Tape differs from declared {name}')
    t = manifest['trace']
    for key, field in [('num_steps','raw_steps'),('warmup_steps','warmup_steps'),('seed','seed')]:
        if tape.metadata[key] != t[field]:
            raise ValueError(f'Tape geometry mismatch: {key}')
    case = next(x for x in t['cases'] if x['regime'] == entry['regime'])
    for key in ('phase_steps','duration_steps','repeat_steps'):
        if tape.metadata[key] != case[key]:
            raise ValueError(f'Tape schedule mismatch: {key}')
    block = next(x for x in manifest['blocks'] if x['block_id'] == entry['block_id'])
    if tape.bank.metadata['split'] != manifest['split'] or tape.metadata['source_prompt_ids'] != block['source_document_ids']:
        raise ValueError('Capture split or source documents differ')


def build_references(rt, layer, tensors, raw_steps):
    """Untimed full-batch default outputs plus sixteen FP32 reference tokens."""
    from optimization.kernel_config import kernel_override
    hidden, ids, combine = tensors
    batch = hidden.shape[1]
    positions = [i * (batch - 1) // 15 for i in range(16)]
    selected = rt.torch.tensor(positions, device=rt.device, dtype=rt.torch.long)
    references = []
    with kernel_override(None):
        for raw in sorted(set(raw_steps)):
            actual = layer.forward(hidden[raw], ids[raw], combine[raw], observe=False).clone()
            expected = layer.reference(hidden[raw][selected], ids[raw][selected], combine[raw][selected])
            rt.torch.testing.assert_close(actual[selected].float(), expected, atol=.02, rtol=.03)
            references.append({'raw_step':raw, 'positions':positions,
                               'default_output':actual, 'fp32_output':expected})
    rt.sync()
    return references


def check_references(rt, layer, tensors, references):
    hidden, ids, combine = tensors
    checks = []
    for reference in references:
        raw = reference['raw_step']
        output = layer.forward(hidden[raw], ids[raw], combine[raw], observe=False)
        rt.torch.testing.assert_close(output.float(), reference['default_output'].float(), atol=.02, rtol=.03)
        positions = rt.torch.tensor(reference['positions'], device=rt.device, dtype=rt.torch.long)
        rt.torch.testing.assert_close(output[positions].float(), reference['fp32_output'], atol=.02, rtol=.03)
        error = float((output[positions].float()-reference['fp32_output']).abs().max().item())
        delta = float((output.float()-reference['default_output'].float()).abs().max().item())
        checks.append({'raw_step':raw, 'reference_token_positions':reference['positions'],
            'reference_tokens':16, 'native_forward_batch_tokens':int(hidden.shape[1]),
            'full_batch_default_comparison':True,
            'fp32_max_abs_error':max(rt.gather_scalars(error)),
            'full_batch_default_max_abs_difference':max(rt.gather_scalars(delta))})
    return {'correctness_pass':True, 'correctness_checks':checks,
            'correctness_max_abs_error':max(x['fp32_max_abs_error'] for x in checks),
            'correctness_atol':.02, 'correctness_rtol':.03}
