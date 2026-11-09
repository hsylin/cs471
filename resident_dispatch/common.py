"""Source, environment and numerical contracts for resident dispatch."""
from pathlib import Path
import hashlib
import json

from optimization.common import (canonical_hash, array_hash, make_layer,
                                 manifest_hash, runtime_metadata as old_metadata)


def execution_fingerprint():
    root = Path(__file__).resolve().parents[1]
    names = ['resident_dispatch/' + name for name in
             ('__init__.py', 'common.py', 'assignments.py', 'gpu.py', 'pilot.py')]
    # Added only if an online implementation is justified by the frozen pilot.
    names += [name for name in ('resident_dispatch/online.py', 'resident_dispatch/runner.py')
              if (root / name).exists()]
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in names}
    return canonical_hash(files), files


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if m.get('resident_dispatch_schema_version') != 1:
        raise ValueError('Wrong resident-dispatch schema')
    if m.get('freeze', {}).get('sha256') != manifest_hash(m):
        raise ValueError('Missing or changed frozen manifest')
    return m


def runtime_metadata(rt, weights, manifest):
    old_metadata(rt, weights, manifest)
    digest, files = execution_fingerprint()
    if digest != manifest['environment']['resident_dispatch_sha256']:
        raise ValueError('Changed resident-dispatch execution sources')
    rt.metadata.update(resident_dispatch_sha256=digest,
                       resident_dispatch_file_sha256=files)
    if rt.world_size != 2 or manifest['policy']['replica_budget'] != 32:
        raise ValueError('Study requires two ranks and32 extra physical slots')
    return rt.metadata


def numerical_check(rt, layer, tensors, references, raw_indices, select_raw):
    """Full-batch comparison for every unique input, FP32 at selected endpoints."""
    errors = []
    deltas = []
    for raw in raw_indices:
        select_raw(raw)
        output = layer.forward(*(t[raw] for t in tensors), observe=False)
        reference = references[raw]
        rt.torch.testing.assert_close(output.float(), reference['native'].float(),
                                      atol=.02, rtol=.03)
        deltas.append((output.float() - reference['native'].float()).abs().max())
        if 'fp32' in reference:
            selected = reference['positions']
            rt.torch.testing.assert_close(output[selected].float(), reference['fp32'],
                                          atol=.02, rtol=.03)
            errors.append((output[selected].float() - reference['fp32']).abs().max())
    rt.sync()
    err = float(rt.torch.stack(errors).max().item()) if errors else 0.
    delta = float(rt.torch.stack(deltas).max().item())
    return dict(correctness_pass=True, correctness_atol=.02, correctness_rtol=.03,
                correctness_max_abs_error=max(rt.gather_scalars(err)),
                full_batch_max_abs_difference=max(rt.gather_scalars(delta)),
                full_batch_inputs_checked=len(raw_indices))


def references_for(rt, layer, tensors, fp32_steps):
    refs = {}
    positions = rt.torch.tensor([i * (tensors[0].shape[1] - 1) // 15 for i in range(16)],
                               device=rt.device, dtype=rt.torch.long)
    for raw in range(len(tensors[0])):
        args = [t[raw] for t in tensors]
        output = layer.forward(*args, observe=False).clone()
        refs[raw] = {'native': output}
        if raw in fp32_steps:
            fp32 = layer.reference(*(t[positions] for t in args))
            rt.torch.testing.assert_close(output[positions].float(), fp32, atol=.02, rtol=.03)
            refs[raw].update(fp32=fp32, positions=positions)
    rt.sync()
    return refs
