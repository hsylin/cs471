"""Scoped fast path for an unchanged map in pinned synchronous vLLM EPLB.

The upstream planner, observation windows, count collectives, policy and closing
timing event remain unchanged. Only the movement/commit pair is skipped after a
charged CPU map comparison and unanimous Gloo MIN agreement. No installed or
archived vLLM source is edited. Importing this module does not import Torch/vLLM.
"""
from __future__ import annotations

import importlib
import inspect
import time


PINNED_COMMIT = 'db9527a46873454610df6dbedf79a36d6bf1a7f6'
MOVEMENT_PARAMETERS = (
    'old_global_expert_indices', 'new_global_expert_indices', 'expert_weights',
    'expert_buffer', 'ep_group', 'communicator', 'is_profile', 'rank_mapping',
)
COMMIT_PARAMETERS = ('model_state', 'new_physical_to_logical_map')
_ACTIVE = None


def _signature(tensor):
    """Cheap storage/version guard; never reads device tensor contents."""
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), str(tensor.dtype), str(tensor.device), tensor._version)


class UnchangedMapFastPath:
    """One owner layer, one normal movement call followed by one normal commit.

    ``mode='observe'`` is diagnostic only: it performs the same checks/agreement,
    but always calls both original functions. Stage durations, when requested,
    are CPU function wall intervals (including any natural blocking), not GPU
    kernel durations. The primary baseline must run without this context.
    """

    def __init__(self, layer, *, mode='fast', diagnostic_timing=False, module=None):
        if mode not in ('fast', 'observe'):
            raise ValueError('mode must be fast or observe')
        if mode == 'observe' and not diagnostic_timing:
            raise ValueError('observe mode is diagnostic only; enable diagnostic_timing')
        self.layer, self.torch = layer, layer.torch
        self.mode, self.diagnostic_timing = mode, bool(diagnostic_timing)
        self.module = module
        self.pending = None
        self.active = False
        self.diagnostics = []
        self.counters = {}
        self.reset_counters()

    def reset_counters(self):
        if self.pending is not None:
            raise RuntimeError('Cannot reset an incomplete movement/commit pair')
        self.counters = dict(movement_calls=0, commit_calls=0, agreement_calls=0,
            local_unchanged_calls=0, unanimously_unchanged_calls=0,
            movement_skips=0, commit_skips=0, original_movement_calls=0,
            original_commit_calls=0, fallback_calls=0)
        self.diagnostics.clear()

    def snapshot(self):
        return dict(mode=self.mode, diagnostic_timing=self.diagnostic_timing,
                    pending_pair=self.pending is not None, **self.counters,
                    diagnostics=[dict(row) for row in self.diagnostics])

    def _validate_owner(self):
        layer = self.layer
        if layer.rt.world_size != 2:
            raise ValueError('This reviewed adapter requires exactly two ranks')
        if layer.rt.metadata.get('native_vllm_commit') != PINNED_COMMIT:
            raise ValueError('Unpinned native runtime')
        if layer.state.is_async or layer.ms.pending_result is not None or layer.ms.rebalanced:
            raise ValueError('Only synchronous EPLB without pending work is supported')
        if len(layer.state.model_states) != 1 or next(iter(layer.state.model_states.values())) is not layer.ms:
            raise ValueError('This scoped adapter requires one owner model state')

    def __enter__(self):
        global _ACTIVE
        if self.active or _ACTIVE is not None:
            raise RuntimeError('Nested or concurrent EPLB adapters are unsupported')
        self._validate_owner()
        module = self.module or importlib.import_module('vllm.distributed.eplb.eplb_state')
        self.module = module
        self.original_movement = module.rearrange_expert_weights_inplace
        self.original_commit = module._commit_eplb_maps
        if tuple(inspect.signature(self.original_movement).parameters) != MOVEMENT_PARAMETERS:
            raise ValueError('Pinned movement function signature changed')
        if tuple(inspect.signature(self.original_commit).parameters) != COMMIT_PARAMETERS:
            raise ValueError('Pinned commit function signature changed')
        # One preallocated CPU scalar; fill and Gloo reduction remain charged.
        self.flag = self.torch.zeros(1, dtype=self.torch.int64, device='cpu')
        self.installed_movement = self._movement
        self.installed_commit = self._commit
        module.rearrange_expert_weights_inplace = self.installed_movement
        module._commit_eplb_maps = self.installed_commit
        self.active = True
        _ACTIVE = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _ACTIVE
        incomplete = self.pending is not None
        modified = (self.module.rearrange_expert_weights_inplace is not self.installed_movement
                    or self.module._commit_eplb_maps is not self.installed_commit)
        self.module.rearrange_expert_weights_inplace = self.original_movement
        self.module._commit_eplb_maps = self.original_commit
        self.pending = None
        self.active = False
        _ACTIVE = None
        if exc_type is None:
            if modified:
                raise RuntimeError('Another component replaced the scoped native functions')
            if incomplete:
                raise RuntimeError('Movement did not receive its matching native map commit')
        return False

    def _time_start(self):
        return time.perf_counter() if self.diagnostic_timing else None

    def _elapsed(self, start):
        return time.perf_counter() - start if start is not None else None

    def _movement(self, old_global_expert_indices, new_global_expert_indices,
                  expert_weights, expert_buffer, ep_group, communicator,
                  is_profile=False, rank_mapping=None):
        owner = self.layer.ms
        # Other models are unaffected by this scoped, process-local interception.
        if old_global_expert_indices is not owner.physical_to_logical_map:
            return self.original_movement(old_global_expert_indices, new_global_expert_indices,
                expert_weights, expert_buffer, ep_group, communicator, is_profile, rank_mapping)
        self._validate_owner()
        if self.pending is not None:
            raise RuntimeError('Second movement before matching map commit')
        if is_profile or rank_mapping is not None:
            raise ValueError('Profile/resizing/rank-remapping paths are unsupported')
        if (expert_weights is not owner.model.expert_weights or expert_buffer is not owner.expert_buffer
                or communicator is not owner.communicator or ep_group is not self.layer.rt.ep_group.device_group):
            raise ValueError('Movement arguments do not belong to the scoped layer')
        if (new_global_expert_indices.device.type != 'cpu'
                or tuple(new_global_expert_indices.shape) != tuple(old_global_expert_indices.shape)
                or new_global_expert_indices.dtype != old_global_expert_indices.dtype):
            raise ValueError('Normal synchronous host map contract changed')
        started = self._time_start()
        signatures = tuple(_signature(t) for t in (owner.physical_to_logical_map,
            owner.logical_to_physical_map, owner.logical_replica_count))
        new_signature = _signature(new_global_expert_indices)
        equality_started = self._time_start()
        # This additional map readback and equality check are inside primary time.
        local_unchanged = bool(self.torch.equal(old_global_expert_indices.cpu(), new_global_expert_indices))
        equality_seconds = self._elapsed(equality_started)
        self.flag.fill_(int(local_unchanged))
        agreement_started = self._time_start()
        self.torch.distributed.all_reduce(self.flag, op=self.torch.distributed.ReduceOp.MIN,
                                         group=self.layer.rt.ep_group.cpu_group)
        agreed = int(self.flag.item())
        agreement_seconds = self._elapsed(agreement_started)
        if agreed not in (0, 1) or (agreed == 1 and not local_unchanged):
            raise RuntimeError('Invalid unanimous unchanged-map agreement')
        skip = self.mode == 'fast' and agreed == 1
        self.counters['movement_calls'] += 1
        self.counters['agreement_calls'] += 1
        self.counters['local_unchanged_calls'] += int(local_unchanged)
        self.counters['unanimously_unchanged_calls'] += agreed
        self.counters['fallback_calls'] += int(not agreed)
        self.pending = dict(model=owner, new_map=new_global_expert_indices,
            new_signature=new_signature, owner_signatures=signatures, skip=skip,
            local_unchanged=local_unchanged, unanimous_unchanged=bool(agreed),
            pair_index=self.counters['movement_calls'])
        original_seconds = 0.0 if self.diagnostic_timing else None
        if skip:
            self.counters['movement_skips'] += 1
        else:
            original_started = self._time_start()
            self.original_movement(old_global_expert_indices, new_global_expert_indices,
                expert_weights, expert_buffer, ep_group, communicator, is_profile, rank_mapping)
            original_seconds = self._elapsed(original_started)
            self.counters['original_movement_calls'] += 1
        if self.diagnostic_timing:
            self.diagnostics.append(dict(stage='movement', pair_index=self.pending['pair_index'],
                local_unchanged=local_unchanged, unanimous_unchanged=bool(agreed), skipped=skip,
                equality_host_wall_seconds=equality_seconds, agreement_host_wall_seconds=agreement_seconds,
                original_function_host_wall_seconds=original_seconds,
                wrapper_host_wall_seconds=self._elapsed(started),
                timing_scope='CPU function wall interval, including natural blocking; no added CUDA synchronization'))

    def _commit(self, model_state, new_physical_to_logical_map):
        if model_state is not self.layer.ms:
            return self.original_commit(model_state, new_physical_to_logical_map)
        self._validate_owner()
        pending = self.pending
        if pending is None:
            raise RuntimeError('Native commit without a matching movement decision')
        if (new_physical_to_logical_map is not pending['new_map']
                or _signature(new_physical_to_logical_map) != pending['new_signature']):
            raise RuntimeError('Candidate map identity/storage/version changed before commit')
        current = tuple(_signature(t) for t in (model_state.physical_to_logical_map,
            model_state.logical_to_physical_map, model_state.logical_replica_count))
        if current != pending['owner_signatures']:
            raise RuntimeError('Committed map state changed between movement and commit')
        started = self._time_start()
        original_seconds = 0.0 if self.diagnostic_timing else None
        if pending['skip']:
            self.counters['commit_skips'] += 1
        else:
            original_started = self._time_start()
            self.original_commit(model_state, new_physical_to_logical_map)
            original_seconds = self._elapsed(original_started)
            self.counters['original_commit_calls'] += 1
        self.counters['commit_calls'] += 1
        if self.diagnostic_timing:
            self.diagnostics.append(dict(stage='commit', pair_index=pending['pair_index'],
                local_unchanged=pending['local_unchanged'],
                unanimous_unchanged=pending['unanimous_unchanged'], skipped=pending['skip'],
                original_function_host_wall_seconds=original_seconds,
                wrapper_host_wall_seconds=self._elapsed(started),
                timing_scope='CPU function wall interval, including natural blocking; no added CUDA synchronization'))
        self.pending = None


def unchanged_map_fast_path(layer, *, mode='fast', diagnostic_timing=False, module=None):
    """Return a scoped adapter; omit it entirely for the primary native baseline."""
    return UnchangedMapFastPath(layer, mode=mode, diagnostic_timing=diagnostic_timing, module=module)
