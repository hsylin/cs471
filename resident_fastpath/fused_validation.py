"""Untimed adversarial proof for the static one-launch selector.

No timing evidence or expert-arithmetic claim is produced. The caller separately
checks actual full-layer outputs on captured inputs before any measurement.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from resident_dispatch.online import adversarial_inputs, cpu_greedy_assignment
from resident_fastpath.fused import IMPLEMENTATION_ID, POLICY_ID, cpu_reference


FIELDS=('physical_ids','homes','logical_counts','summary')


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<i8').tobytes()).hexdigest()


def expected_fields(ids,mapping,real_tokens):
    """Require agreement of separately expressed old and new CPU references."""
    new=cpu_reference(ids,mapping,real_tokens=real_tokens)
    old=cpu_greedy_assignment(ids,mapping,real_tokens=real_tokens)
    legacy={field:getattr(old,field) for field in FIELDS[:-1]}
    legacy['summary']=np.asarray(old.diagnostics['summary'],dtype=np.int64)
    for field in FIELDS:
        if not np.array_equal(new[field],legacy[field]):
            raise AssertionError('CPU policy specifications disagree: '+field)
    return {field:new[field] for field in FIELDS}


def check_snapshot(snapshot,expected):
    """Validate all kernel outputs, not just the final routed IDs."""
    for field in FIELDS:
        actual=np.asarray(snapshot[field])
        if (not np.issubdtype(actual.dtype,np.integer)
                or actual.shape!=expected[field].shape
                or not np.array_equal(actual,expected[field])):
            raise AssertionError('Fused GPU output differs from both CPU references: '+field)
    return {field+'_sha256':array_digest(snapshot[field]) for field in FIELDS}


def verify_adversarial_fused(rt,mapper):
    """Check nine cases × two integer dtypes, with replicated-rank agreement.

    Passed only explicitly supplied isolated real-token scalars. The mapper's
    internal full-batch scalar must remain128 throughout. Readback and all
    verification collectives occur outside the primary measured region.
    """
    if rt.world_size!=2 or mapper.batch_tokens!=128 or mapper.top_k!=8:
        raise ValueError('Fused proof requires the declared two-rank B128/top8 setup')
    torch=rt.torch
    before_context=mapper.verify_context()
    before_real=mapper.real_tokens.detach().cpu().numpy().copy()
    if before_real.size!=1 or int(before_real.reshape(-1)[0])!=128:
        raise AssertionError('Fused default real-token scalar was not128')
    start_calls=mapper.calls
    evidence=[]
    for case in adversarial_inputs(128):
        for dtype_name,np_dtype,torch_dtype in (
                ('int32',np.int32,torch.int32),('int64',np.int64,torch.int64)):
            host_ids=case['logical_ids'].astype(np_dtype,copy=True)
            expected=expected_fields(host_ids,mapper.physical_to_logical_map,case['real_tokens'])
            ids=torch.from_numpy(host_ids).to(rt.device,torch_dtype)
            real=torch.tensor(case['real_tokens'],device=rt.device,dtype=torch.int32)
            output=mapper(ids,real)
            if output.dtype!=torch_dtype or tuple(output.shape)!=(128,8):
                raise AssertionError('Fused output changed dtype or shape')
            snapshot=mapper.snapshot_diagnostics()
            hashes=check_snapshot(snapshot,expected)
            if not np.array_equal(ids.cpu().numpy(),host_ids):
                raise AssertionError('Fused kernel mutated logical input IDs')
            if int(real.cpu().item())!=case['real_tokens']:
                raise AssertionError('Fused kernel mutated supplied real-token scalar')
            evidence.append(dict(case=case['name'],dtype=dtype_name,
                real_tokens=int(case['real_tokens']),input_ids_sha256=array_digest(host_ids),**hashes))
    after_context=mapper.verify_context()
    after_real=mapper.real_tokens.detach().cpu().numpy()
    if before_context!=after_context or not np.array_equal(before_real,after_real):
        raise AssertionError('Fused proof changed stream context or internal real-token scalar')
    calls=mapper.calls-start_calls
    if calls!=18:
        raise AssertionError('Fused adversarial invocation count differs')
    peers=rt.gather_objects(evidence)
    if len(peers)!=2 or any(peer!=evidence for peer in peers):
        raise AssertionError('Ranks disagree on fused adversarial outputs')
    return dict(schema='resident_fused_adversarial_proof_v1',passed=True,
        policy_id=POLICY_ID,implementation_id=IMPLEMENTATION_ID,
        cases_checked=9,dtypes_checked=['int32','int64'],gpu_invocations_checked=calls,
        reference_count=2,rank_agreement=True,stream_context_unchanged=True,
        internal_real_tokens_unchanged=True,internal_real_tokens=128,
        input_ids_unchanged=True,external_real_tokens_unchanged=True,
        field_hash_encoding='little-endian int64; C-order; no header',
        scope='Physical IDs, homes, logical counts and summary; no expert arithmetic or performance claim',
        cases=evidence,evidence_sha256=hashlib.sha256(json.dumps(evidence,sort_keys=True,
            separators=(',',':')).encode()).hexdigest())
