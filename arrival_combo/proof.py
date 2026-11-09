"""Untimed composition proof for fixed-residency dispatch and no-op commits.

The existing mapper and shortcut are reused without modification. Physical
assignment counters are compared against their own arm's prepared CPU oracle;
only reconstructed logical demand and the resulting planner/map trajectory must
match across arms. This proof is not a performance measurement or a test of
online dispatch after residency changes, which that mapper does not support.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
from pathlib import Path

import numpy as np


ARMS = ('native', 'shortcut', 'combined')
TRAJECTORY_FIELDS = ('step', 'allowed', 'reason', 'current_counts', 'current_map',
                     'proposed_map', 'committed_map')


def routing_arm(arm):
    if arm not in ARMS:
        raise ValueError('Unknown composition proof arm')
    return 'online' if arm == 'combined' else 'native'


def check_guard_reports(arm, rows, calls):
    """Guard accounting is independent of the choice of physical dispatcher."""
    routing_arm(arm)
    if arm == 'native':
        if rows != []:
            raise AssertionError('The native proof baseline must not be wrapped')
        return
    expected = dict(mode='fast', diagnostic_timing=False, pending_pair=False,
        movement_calls=calls, commit_calls=calls, agreement_calls=calls,
        local_unchanged_calls=calls, unanimously_unchanged_calls=calls,
        movement_skips=calls, commit_skips=calls, original_movement_calls=0,
        original_commit_calls=0, fallback_calls=0, diagnostics=[])
    if len(rows) != 2 or rows[0] != rows[1]:
        raise AssertionError('Both proof ranks must agree on shortcut completion')
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise AssertionError('Wrong composition proof guard field: '+field)


def validate_arm_evidence(arm, decisions, terminal_window, cell, sequence, *, world_size=2):
    """CPU validation of separately recorded physical and logical histories."""
    if world_size != 2 or len(sequence) != 128:
        raise ValueError('Composition proof requires two ranks and128 steps')
    route = routing_arm(arm)
    mapping = np.asarray(cell['mapping'])
    if mapping.shape != (160,):
        raise ValueError('Composition proof requires160 fixed physical slots')
    if [d['step'] for d in decisions] != [63]:
        raise AssertionError('One nonterminal proposal at63 required')
    for decision in decisions:
        raw = sequence[decision['step']-15:decision['step']+1]
        expected_physical = np.asarray(cell['counts'][route])[raw]
        actual = np.asarray(decision['physical_slot_assignments_window'])
        expected_logical = np.bincount(np.asarray(cell['logical'])[raw].ravel(), minlength=128)*world_size
        if not np.array_equal(actual, expected_physical):
            raise AssertionError('Arm-specific physical observation mismatch')
        recovered = np.zeros(128, dtype=np.int64)
        np.add.at(recovered, mapping, actual.sum(axis=0))
        if not np.array_equal(recovered*world_size, expected_logical):
            raise AssertionError('Physical observations do not conserve logical demand')
        if not np.array_equal(decision['current_counts'], expected_logical):
            raise AssertionError('Native planner did not receive the exact logical window')
        if (decision['allowed'] or not decision.get('commit_completed')
                or decision.get('transfer_sent_bytes') != 0
                or not np.array_equal(decision['current_map'], mapping)
                or not np.array_equal(decision['committed_map'], mapping)):
            raise AssertionError('SKIP placement or normal commit contract changed')
    if not np.array_equal(terminal_window, np.asarray(cell['counts'][route])[sequence[-16:]]):
        raise AssertionError('Terminal arm-specific physical observation mismatch')
    return [{key:decision[key] for key in TRAJECTORY_FIELDS} for decision in decisions]


def check_common_trajectory(arm_trajectories):
    if set(arm_trajectories) != set(ARMS):
        raise AssertionError('All three composition proof arms required')
    reference = arm_trajectories['native']
    if any(value != reference for value in arm_trajectories.values()):
        raise AssertionError('Logical/candidate/commit trajectory differs across arms')
    return reference


def run_gpu_proof(rt, weights, base, cell):
    """Run all three fresh arms before primary timing; never tune from results.

   128 actual native steps include one normal proposal and one terminal no-op.
    The proof checks every prepared unique full-batch output before and after the
    sequence; existing selected FP32 references retain their original scope.
    Changed-map fallback is covered by arrival_fix's separate migration proof,
    not by this fixed-map-only composition test.
    """
    from vllm.distributed.eplb import eplb_state

    from arrival_fix.noop import unchanged_map_fast_path
    from arrival_study.runner import serve
    from optimization.common import array_hash, canonical_hash, make_layer
    from resident_dispatch.common import numerical_check
    from resident_dispatch.online import verify_adversarial
    from resident_dispatch.runner import make_online, reset_observation
    from skewbench.runner import finalize_decision_log, snapshot_weight_hash

    if (rt.world_size != 2 or base['trace']['window_steps'] != 16
            or base['policy']['rebalance_interval'] != 64 or base['policy']['replica_budget'] != 32):
        raise ValueError('Unsupported composition proof geometry')
    sequence = list(cell['sequence'][:128])
    if len(sequence) != 128:
        raise ValueError('Prepared replay is too short for composition proof')
    batch = int(cell['tensors'][0].shape[1])
    if batch not in (128,1024):
        raise ValueError('Unsupported composition proof batch')
    native_symbols = (eplb_state.rearrange_expert_weights_inplace, eplb_state._commit_eplb_maps)
    arms, trajectories = [], {}
    for arm in ARMS:
        layer = make_layer(rt, weights, base, 'SKIP')
        mapper = None
        try:
            layer.assert_maps_and_weights()
            if not np.array_equal(layer.map_snapshot(), cell['mapping']):
                raise AssertionError('Composition proof source placement changed')
            adversarial = None
            if arm == 'combined':
                mapper = make_online(rt, layer, batch)
                adversarial = verify_adversarial(rt, layer, mapper, batch)
                layer.mapping_kernel = mapper
            before = numerical_check(rt, layer, cell['tensors'], cell['references'],
                cell['reference_steps'], lambda raw:None)
            reset_observation(layer)
            mapper_before = mapper.calls if mapper is not None else 0
            context = unchanged_map_fast_path(layer) if arm != 'native' else nullcontext(None)
            with context as guard:
                # Saturated command/ACK serving is exercised only as an untimed
                # correctness workload. Its returned clocks are not evidence.
                serve(rt, layer, cell['tensors'], sequence, [0]*128, 1, terminal=True)
                local_guard = guard.snapshot() if guard is not None else None
            if (eplb_state.rearrange_expert_weights_inplace, eplb_state._commit_eplb_maps) != native_symbols:
                raise AssertionError('Composition proof did not restore native symbols')
            calls = mapper.calls-mapper_before if mapper is not None else 0
            if calls != (128 if arm == 'combined' else 0):
                raise AssertionError('Unexpected online mapper invocation count')
            decisions = finalize_decision_log(rt, layer)
            terminal = layer.ms.expert_load_window[:,0,:].cpu().numpy().copy()
            trajectories[arm] = validate_arm_evidence(arm, decisions, terminal, cell,
                sequence, world_size=rt.world_size)
            layer.assert_maps_and_weights()
            final_map = layer.map_snapshot()
            if (not np.array_equal(final_map, cell['mapping']) or layer.counter.sent_bytes
                    or layer.counter.sent_messages or layer.terminal_proposals_suppressed != 1):
                raise AssertionError('Fixed residency, zero payload, or terminal contract changed')
            if mapper is not None:
                ls = layer.layer_state
                mapper.verify_metadata(ls.logical_to_physical_map, ls.logical_replica_count,
                    layer.ms.physical_to_logical_map[0])
            after = numerical_check(rt, layer, cell['tensors'], cell['references'],
                cell['reference_steps'], lambda raw:None)
            final_hash, final_rank_hashes = snapshot_weight_hash(rt, (layer.w13,layer.w2))
            if (final_hash != cell['source_weights_sha256']
                    or final_rank_hashes != cell['source_weight_rank_sha256']):
                raise AssertionError('Composition proof changed checkpoint weights')
            peers = [None]*rt.world_size
            rt.torch.distributed.all_gather_object(peers, dict(guard=local_guard,
                mapper_calls_measured=calls, sent_bytes=int(layer.counter.sent_bytes),
                sent_messages=int(layer.counter.sent_messages)), group=rt.ep_group.cpu_group)
            if any(peer != peers[0] for peer in peers):
                raise AssertionError('Ranks disagree on composition proof completion')
            guard_rows = [peer['guard'] for peer in peers] if arm != 'native' else []
            check_guard_reports(arm, guard_rows, calls=2)
            layer.close()
            arms.append(dict(arm=arm, routing_arm=routing_arm(arm), passed=True,
                correctness_pass=True, normal_completion_verified=True,
                exact_checkpoint_weights_verified=True, inverse_map_and_replica_count_verified=True,
                native_symbols_restored=True, online_residency_metadata_verified=mapper is not None,
                source_map=cell['mapping'].tolist(), final_map=final_map.tolist(),
                source_weights_sha256=cell['source_weights_sha256'], final_weights_sha256=final_hash,
                source_weight_rank_sha256=cell['source_weight_rank_sha256'],
                final_weight_rank_sha256=final_rank_hashes, assignment_sha256=cell['route_hashes'][routing_arm(arm)],
                mapper_calls_measured=calls, measured_raw_indices=sequence,
                guard_counters_by_rank=guard_rows, completion_by_rank=peers,
                migration_bytes=0, migration_sent_messages=0, terminal_proposals_suppressed=1,
                no_pending_async_work=True, decision_log=decisions,
                terminal_physical_window=terminal.tolist(), adversarial_checks=adversarial,
                preflight_correctness=before, postflight_correctness=after))
        finally:
            del layer, mapper
            gc.collect()
        rt.sync()
        rt.barrier()
    common = check_common_trajectory(trajectories)
    return dict(gpu_combo_proof_schema_version=1, passed=True, diagnostic_only=True,
        evidence_eligible=False, correctness_pass=True, batch_tokens=batch,
        native_vllm_commit=rt.metadata['native_vllm_commit'],
        capture_sha256=cell['capture_sha256'], replay_sha256=cell['replay_sha256'],
        source_prompt_ids=cell['source_prompt_ids'], assignment_artifact=cell['artifact'],
        logical_routing_sha256=array_hash(cell['logical']), resolved_kernel_config=cell['resolved_kernel_config'],
        source_map_sha256=array_hash(cell['mapping']), source_weights_sha256=cell['source_weights_sha256'],
        logical_candidate_trajectory_sha256=canonical_hash(common),
        common_logical_candidate_trajectory_verified=True,
        arm_specific_physical_windows_verified=True, native_symbols_restored=True,
        reference_raw_indices=list(cell['reference_steps']), fp32_raw_indices=list(cell['fp32_steps']),
        proof_steps_per_arm=128, expected_decision_steps=[63], expected_noop_pairs_per_shortcut_arm=2,
        arms=arms, proof_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Untimed three-arm fixed-map composition proof; arm-specific physical histories, common logical/planner/map trajectory, real native terminal completion and numerical outputs. No changed-residency online support, no primary timing evidence.')
