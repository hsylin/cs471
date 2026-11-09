"""Untimed real-weight proof of the scoped unchanged-map adapter.

The proof uses a fresh layer and restores its original placement before returning.
It exercises the intercepted eplb_state module symbols directly, including two
real cross-rank weight migrations. It is not a primary performance observation.
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

from arrival_fix.noop import unchanged_map_fast_path


def _check_guard_rows(rows):
    if len(rows) != 2:
        raise AssertionError('GPU proof requires both rank guard reports')
    expected = dict(movement_calls=4, commit_calls=4, agreement_calls=4,
        local_unchanged_calls=2, unanimously_unchanged_calls=2,
        movement_skips=2, commit_skips=2, original_movement_calls=2,
        original_commit_calls=2, fallback_calls=2)
    for row in rows:
        if row['mode'] != 'fast' or row['diagnostic_timing'] or row['pending_pair'] or row['diagnostics']:
            raise AssertionError('Unexpected proof mode or incomplete guarded pair')
        for key, value in expected.items():
            if row[key] != value:
                raise AssertionError(f'Wrong GPU proof guard count: {key}={row[key]}')
    if rows[0] != rows[1]:
        raise AssertionError('Ranks disagree on GPU proof movement/commit trajectory')


def run_gpu_proof(rt, weights, base, cell):
    """Test unchanged, changed, restore, unchanged; return identical rank reports.

    Call after the existing input/reference preparation and before any primary
    timer. The source and restored weights are hashed; every intermediate map,
    inverse map, replica count and resident tensor is checked against checkpoint
    identity. Full-batch/native and sampled-FP32 checks use cell.fp32_steps.
    """
    import numpy as np
    from vllm.distributed.eplb import eplb_state

    from optimization.common import array_hash, make_layer
    from resident_dispatch.common import numerical_check
    from skewbench.runner import snapshot_weight_hash

    if rt.world_size != 2:
        raise ValueError('The GPU proof requires two ranks')
    layer = make_layer(rt, weights, base, 'SKIP')
    try:
        layer.assert_maps_and_weights()
        original = layer.map_snapshot()
        if original.shape != (160,) or not np.array_equal(original, cell['mapping']):
            raise AssertionError('Unexpected proof source placement')
        source_hash, source_rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
        if source_hash != cell['source_weights_sha256']:
            raise AssertionError('Fresh proof layer differs from prepared source weights')
        changed = original.copy()
        changed[0], changed[80] = changed[80], changed[0]
        if np.array_equal(changed, original):
            raise AssertionError('The declared cross-rank swap did not change placement')
        # Both maps are newly allocated host tensors; each same object is passed
        # to its movement and commit calls so pending-pair guards are exercised.
        steps = [('unchanged_before', original, True),
                 ('changed_swap_0_80', changed, False),
                 ('restore_original', original, False),
                 ('unchanged_after_restore', original, True)]
        step_reports = []
        native_symbols = (eplb_state.rearrange_expert_weights_inplace, eplb_state._commit_eplb_maps)
        with unchanged_map_fast_path(layer) as guard:
            for label, expected_map, expected_skip in steps:
                candidate = rt.torch.as_tensor(expected_map.copy(), dtype=rt.torch.long, device='cpu').reshape(1, -1)
                before_bytes, before_messages = layer.counter.sent_bytes, layer.counter.sent_messages
                before_skips = guard.counters['movement_skips']
                # Do not use NativeLayer.force_migration: it imports movement
                # from rebalance_execute and would bypass the intercepted symbol.
                eplb_state.rearrange_expert_weights_inplace(
                    layer.ms.physical_to_logical_map, candidate,
                    layer.ms.model.expert_weights, layer.ms.expert_buffer,
                    rt.ep_group.device_group, layer.ms.communicator,
                    is_profile=False, rank_mapping=None)
                eplb_state._commit_eplb_maps(layer.ms, candidate)
                rt.sync()
                if guard.pending is not None or guard.counters['movement_skips']-before_skips != int(expected_skip):
                    raise AssertionError('GPU proof used the wrong movement/commit branch')
                if not np.array_equal(layer.map_snapshot(), expected_map):
                    raise AssertionError('Native proof commit differs from its requested map')
                layer.assert_maps_and_weights()
                numerical = numerical_check(rt, layer, cell['tensors'], cell['references'],
                                            cell['fp32_steps'], lambda _: None)
                payload = dict(rank=rt.rank,
                    bytes=int(layer.counter.sent_bytes-before_bytes),
                    messages=int(layer.counter.sent_messages-before_messages))
                payload_ranks = [None]*rt.world_size
                rt.torch.distributed.all_gather_object(payload_ranks, payload, group=rt.ep_group.cpu_group)
                actual_bytes = sum(item['bytes'] for item in payload_ranks)
                actual_messages = sum(item['messages'] for item in payload_ranks)
                if expected_skip and (actual_bytes or actual_messages):
                    raise AssertionError('Unchanged proof branch transferred weights')
                if not expected_skip and (actual_bytes <= 0 or actual_messages <= 0):
                    raise AssertionError('Changed proof branch did not perform actual transport')
                if layer.ms.pending_result is not None or layer.ms.rebalanced or layer.state.is_async:
                    raise AssertionError('GPU proof left asynchronous or pending work')
                step_reports.append(dict(name=label, expected_skip=expected_skip,
                    map_sha256=array_hash(expected_map), exact_checkpoint_weights_verified=True,
                    inverse_map_and_replica_count_verified=True,
                    transfer_payload_by_rank=payload_ranks, actual_transfer_bytes=actual_bytes,
                    actual_transfer_messages=actual_messages, **numerical))
            local_guard = guard.snapshot()
        if (eplb_state.rearrange_expert_weights_inplace, eplb_state._commit_eplb_maps) != native_symbols:
            raise AssertionError('GPU proof did not restore native module symbols')
        guard_ranks = [None]*rt.world_size
        rt.torch.distributed.all_gather_object(guard_ranks, local_guard, group=rt.ep_group.cpu_group)
        _check_guard_rows(guard_ranks)
        layer.assert_maps_and_weights()
        if not np.array_equal(layer.map_snapshot(), original):
            raise AssertionError('Proof must restore placement before returning')
        restored_hash, restored_rank_hashes = snapshot_weight_hash(rt, (layer.w13, layer.w2))
        if restored_hash != source_hash or restored_rank_hashes != source_rank_hashes:
            raise AssertionError('Restored proof weights differ from original source')
        layer.close()
        report = dict(gpu_noop_proof_schema_version=1, passed=True, diagnostic_only=True,
            evidence_eligible=False, batch_tokens=int(cell['tensors'][0].shape[1]),
            native_vllm_commit=rt.metadata['native_vllm_commit'],
            capture_sha256=cell['capture_sha256'], replay_sha256=cell['replay_sha256'],
            source_prompt_ids=cell['source_prompt_ids'], assignment_artifact=cell['artifact'],
            logical_routing_sha256=array_hash(cell['logical']),
            resolved_kernel_config=cell['resolved_kernel_config'],
            source_map_sha256=array_hash(original), changed_map_sha256=array_hash(changed),
            source_weights_sha256=source_hash, restored_weights_sha256=restored_hash,
            source_weight_hashes_by_rank=source_rank_hashes,
            restored_weight_hashes_by_rank=restored_rank_hashes,
            source_restore_verified=True, native_symbols_restored=True,
            no_pending_async_work=True, guard_counters_by_rank=guard_ranks,
            total_transfer_bytes=sum(s['actual_transfer_bytes'] for s in step_reports),
            numerical_raw_indices=list(cell['fp32_steps']), steps=step_reports,
            proof_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Untimed fresh-layer unchanged skip, cross-rank swap0/80 changed fallback, normal restore, unchanged skip; no primary latency or mixed-rank GPU corruption test.')
    finally:
        # On failure the fresh layer is discarded and the caller must abort;
        # do not start recovery collectives after a potentially one-rank error.
        del layer
        gc.collect()
    rt.sync()
    rt.barrier()
    return report
