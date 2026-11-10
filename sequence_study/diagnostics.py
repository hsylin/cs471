"""Separate, instrumented expert-path observations; never primary timings.

The selected instance's native ``TritonExperts.apply`` is delegated unchanged.
Its interval includes alignment, both expert GEMMs, activation and local top-k
summation. It excludes routing and the later output collective. CUDA-event
elapsed time can include submission gaps and contention; it is not pure kernel
residence or a migration-payback estimate.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import numpy as np


def _hash(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


@contextmanager
def _replace_apply(instance, replacement):
    """Restore the exact instance attribute state, including inherited methods."""
    absent = object()
    saved = instance.__dict__.get('apply', absent)
    instance.apply = replacement
    try:
        yield
    finally:
        if saved is absent:
            del instance.apply
        else:
            instance.apply = saved


def _assignment_metrics(ids, weights, expert_map, physical_map, local_slots, block_m):
    """CPU reconstruction from actual native expert-call arguments."""
    ids = np.asarray(ids)
    weights = np.asarray(weights)
    expert_map = np.asarray(expert_map)
    physical_map = np.asarray(physical_map)
    if ids.shape != (128, 8) or weights.shape != ids.shape:
        raise AssertionError('Diagnostic requires actual B128/top8 arguments')
    if (not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any()
            or (ids >= len(physical_map)).any() or not np.isfinite(weights).all()):
        raise AssertionError('Invalid actual physical assignments/weights')
    if expert_map.shape != physical_map.shape or block_m <= 0:
        raise AssertionError('Unsupported expert-map or tile geometry')
    owned = expert_map >= 0
    if not np.array_equal(np.sort(expert_map[owned]), np.arange(local_slots)):
        raise AssertionError('Local expert-map coverage is not one slot per index')
    global_counts = np.bincount(ids.ravel(), minlength=len(physical_map))
    local_ids = expert_map[ids]
    local_counts = np.bincount(local_ids[local_ids >= 0], minlength=local_slots)
    logical = physical_map[ids]
    if ((logical < 0).any() or (logical >= 128).any()
            or np.any(np.diff(np.sort(logical, axis=1), axis=1) == 0)):
        raise AssertionError('Lost logical top-k identity or unique logical experts')
    tokens = int(local_counts.sum())
    padded = int((((local_counts + block_m - 1) // block_m) * block_m).sum())
    return dict(physical_ids=ids.astype(np.int64).tolist(),
        physical_ids_sha256=_hash(ids, '<i8'), logical_ids_sha256=_hash(logical, '<i8'),
        combining_weights_sha256=_hash(weights, '<f4'),
        physical_counts=global_counts.tolist(), local_expert_map=expert_map.tolist(),
        local_counts=local_counts.tolist(), local_assignments=tokens,
        local_active_copies=int(np.count_nonzero(local_counts)),
        local_nonempty_m_tiles=padded // block_m, local_padded_assignment_rows=padded,
        useful_two_gemm_flops_proxy=tokens * 6 * 2048 * 768,
        padded_two_gemm_flops_proxy=padded * 6 * 2048 * 768,
        useful_assignment_fraction=None if padded == 0 else tokens / padded,
        proxy_scope='Counted useful/nonempty padded rows, not hardware FLOP counters or allocated launch extent')


def _quiescent(layer):
    if layer.ms is not None and (layer.ms.rebalanced or layer.ms.pending_result is not None):
        raise AssertionError('Diagnostic requires prior normal native completion')
    if getattr(layer, '_active', False):
        raise AssertionError('Diagnostic must run outside another execution scope')


def diagnostic(rt, layer, hidden, reference, raw_indices, label):
    """Return two gathered rank records for one completed, fixed-map state.

    Exactly 64 selected inputs are replayed twice: an untimed argument-capture
    pass and a separately instrumented event pass. No ``step`` or controller
    method is called. The caller owns initial/final-map selection and must call
    this only after primary timing and normal native quiescence.
    """
    torch = rt.torch
    indices = [int(x) for x in raw_indices]
    if (len(indices) != 64 or len(set(indices)) != 64
            or min(indices) < 0 or max(indices) >= len(hidden)
            or tuple(reference.shape) != tuple(hidden.shape)):
        raise ValueError('Exactly64 distinct valid raw indices and matching references required')
    _quiescent(layer)
    experts = layer.routed.quant_method.moe_kernel.fused_experts
    if (type(experts).__name__ != 'TritonExperts'
            or type(experts).__module__ != 'vllm.model_executor.layers.fused_moe.experts.triton_moe'):
        raise AssertionError('Only the reviewed native TritonExperts path is supported')
    source_sha = hashlib.sha256(Path(inspect.getsourcefile(type(experts))).read_bytes()).hexdigest()
    if source_sha != 'ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab':
        raise AssertionError('Native expert implementation changed')
    original = experts.apply
    signature = inspect.signature(original)  # once, outside either forward loop
    required = {'topk_ids', 'topk_weights', 'expert_map', 'global_num_experts'}
    if not required.issubset(signature.parameters):
        raise AssertionError('Native expert signature changed')
    config = layer.resolved_kernel_config()
    block_m = int(config['BLOCK_SIZE_M'])
    rt.sync(); rt.barrier()
    start = layer.audit()
    mapping = np.asarray(start['final_map'], dtype=np.int64)
    if len(mapping) not in (128, 160) or layer.slots * 2 != len(mapping):
        raise AssertionError('Unsupported physical residency')
    # Verify observation buffers as well as public counters remain untouched.
    watched = []
    if layer.state is not None:
        for name in ('expert_load_pass', 'expert_load_window', 'last_expert_load'):
            tensor = getattr(layer.ms, name)
            watched.append((name, tensor, tensor.detach().clone()))
        watched.append(('should_record_tensor', layer.state.should_record_tensor,
                        layer.state.should_record_tensor.detach().clone()))
    captured = []

    def capture(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        if bound['expert_map'] is None or int(bound['global_num_experts']) != len(mapping):
            raise AssertionError('Expected explicit native EP map/global slots')
        captured.append(tuple(bound[k].detach().clone()
                              for k in ('topk_ids', 'topk_weights', 'expert_map')))
        return original(*args, **kwargs)

    maximum = 0.
    input_hash = hashlib.sha256()
    with layer.measured(), _replace_apply(experts, capture):
        for raw in indices:
            before = len(captured)
            actual = layer.forward(hidden[raw], observe=False)
            if len(captured) != before + 1:
                raise AssertionError('Expected exactly one expert call per input')
            torch.testing.assert_close(actual.float(), reference[raw].float(), atol=.02, rtol=.03)
            maximum = max(maximum, float((actual.float() - reference[raw].float()).abs().max()))
            input_hash.update(hidden[raw].detach().view(torch.uint16).cpu().numpy().astype('<u2', copy=False).tobytes())
    rt.sync()
    assignments = []
    for raw, values in zip(indices, captured):
        ids, weights, expert_map = (x.cpu().numpy() for x in values)
        assignments.append(dict(raw_index=raw, **_assignment_metrics(
            ids, weights, expert_map, mapping, layer.slots, block_m)))
    del captured

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in indices]
    stream = torch.cuda.current_stream(rt.device)
    for begin, end in events:
        begin.record(stream); end.record(stream)
    rt.sync(); rt.barrier()  # initialize event storage outside the observed pass
    event_index = 0

    def observed(*args, **kwargs):
        nonlocal event_index
        if event_index >= len(events):
            raise AssertionError('Too many native expert invocations')
        begin, end = events[event_index]
        event_index += 1
        begin.record(stream)
        result = original(*args, **kwargs)
        end.record(stream)
        return result

    outputs = []
    with layer.measured(), _replace_apply(experts, observed):
        for position, raw in enumerate(indices):
            if event_index != position:
                raise AssertionError('Expert event/input association changed')
            # Keep each result even if a native backend aliases its workspace.
            # This clone is outside the expert-event interval.
            outputs.append(layer.forward(hidden[raw], observe=False).clone())
        if event_index != len(indices):
            raise AssertionError('Missing native expert invocation')
    rt.sync()
    elapsed = [float(begin.elapsed_time(end)) for begin, end in events]
    if not all(np.isfinite(x) and x > 0 for x in elapsed):
        raise AssertionError('Invalid CUDA event intervals')
    for raw, actual in zip(indices, outputs):
        torch.testing.assert_close(actual.float(), reference[raw].float(), atol=.02, rtol=.03)
        maximum = max(maximum, float((actual.float() - reference[raw].float()).abs().max()))
    del outputs
    if torch.cuda.current_stream(rt.device).cuda_stream != stream.cuda_stream:
        raise AssertionError('Diagnostic changed the inference stream')
    for name, actual, expected in watched:
        if not torch.equal(actual, expected):
            raise AssertionError('Observation buffer changed: ' + name)
    _quiescent(layer)
    end = layer.audit()
    unchanged = ('final_map', 'controller_step_calls', 'native_counter', 'native_valid_steps',
        'async_events', 'recorded_step_indices', 'deferred_boundary_steps', 'decisions',
        'terminal_physical_window_totals_local', 'physical_window_next_write',
        'migration_sent_bytes_local', 'migration_sent_messages_local')
    if any(start[k] != end[k] for k in unchanged):
        raise AssertionError('Diagnostic changed controller/history/placement state')
    local = dict(schema='sequence_expert_path_diagnostic_v1', label=str(label), arm=layer.arm,
        rank=rt.rank, raw_indices=indices, selected_hidden_bf16_sha256=input_hash.hexdigest(),
        map=mapping.tolist(), map_sha256=_hash(mapping, '<i8'), kernel_config=config,
        native_experts_source_sha256=source_sha,
        argument_capture_pass_separate=True, instrumented_forward_calls=64,
        capture_forward_calls=64, numerical_full_batch_checks=128, correctness_pass=True,
        max_abs_difference=maximum, atol=.02, rtol=.03, events_preinitialized=True,
        expert_apply_restored=('apply' not in experts.__dict__ or experts.apply == original),
        no_controller_steps=True, observation_buffers_unchanged=True,
        expert_path_elapsed_ms=elapsed, assignments=assignments,
        start_state=start, end_state=end,
        timing_scope='CUDA-event interval around native TritonExperts.apply: alignment, two GEMMs, activation, local moe_sum; excludes gate/topk/mapping/output collective',
        limitations=['Separate instrumented diagnostic, not primary service/response timing.',
            'GPU stream elapsed can include host submission gaps and contention; not pure GEMM kernel residence.',
            'No migration payback or causal before/after-commit claim.',
            'Global physical assignments repeat on both input-replicated ranks; sum only local ownership counts.'])
    peers = rt.gather_objects(local)
    if len(peers) != 2 or any(p['raw_indices'] != indices or p['map'] != local['map']
                             or p['selected_hidden_bf16_sha256'] != local['selected_hidden_bf16_sha256'] for p in peers):
        raise AssertionError('Diagnostic rank input/map disagreement')
    for a, b in zip(peers[0]['assignments'], peers[1]['assignments']):
        if any(a[k] != b[k] for k in ('physical_ids_sha256', 'logical_ids_sha256', 'combining_weights_sha256')):
            raise AssertionError('Diagnostic actual rank route disagreement')
        if a['local_assignments'] + b['local_assignments'] != 1024:
            raise AssertionError('Diagnostic global token conservation failed')
    return peers
