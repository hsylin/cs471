"""Explicit custom static-residency mode, not upstream EPLB-disabled mode.

The native enabled geometry constructs160 slots. Fixed maps and checkpoint
weights remain resident, but observation, planning and migration are inactive.
All normal stock gate/top-k, experts and native output reduction still execute.
"""
from contextlib import contextmanager
from types import MethodType

import numpy as np

from stock_comparison.native import StockLayer, _hash_array


class StaticResidentLayer(StockLayer):
    def __init__(self, rt, weights, *, selector='three_kernel'):
        self.selector_kind = 'three_kernel'
        super().__init__(rt, weights, 'ours')
        del rt.layers['ours']
        rt.layers['fix'] = self
        self.arm = 'fix'
        self.spec = {**self.spec, 'arm': 'fix'}
        if selector == 'fused':
            from resident_fastpath.fused import FusedMapper
            self.mapper = FusedMapper(self.torch, self.initial_map, 128)
            self.mapper.verify_metadata(self.layer_state.logical_to_physical_map,
                self.layer_state.logical_replica_count, self.ms.physical_to_logical_map[0])
            self.mapping_kernel = self.mapper
            self.selector_kind = 'fused'
            owner = self
            def apply_mapping(router, ids):
                return owner.mapper(ids)
            self.router._apply_eplb_mapping = MethodType(apply_mapping, self.router)
        elif selector != 'three_kernel':
            raise ValueError('Unknown static selector')
        self.reset_trial()

    @contextmanager
    def measured(self):
        if self._active:
            raise RuntimeError('Nested execution scope')
        self._active = True
        self.guard = None
        try:
            yield self
        finally:
            self._active = False

    activation_context = measured
    active = measured

    def reset_trial(self):
        super().reset_trial()
        # Setup only: the selected experiment has fixed B128 and fixed residency.
        # No per-forward flag/length/history writes remain in this custom mode.
        self.state.should_record_tensor.fill_(False)
        for t in self.ms.num_unpadded_tokens_tensors:
            t.fill_(self.batch_tokens)
        self.rt.sync()

    def forward(self, hidden, *, observe=True):
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        if tuple(hidden.shape) != (self.batch_tokens,2048) or hidden.dtype != self.dtype:
            raise ValueError('Static resident forward requires BF16[128,2048]')
        with self.torch.no_grad(), set_current_vllm_config(self.config), set_forward_context(None,self.config,num_tokens=len(hidden)):
            output = self.block(hidden)
        if observe:
            self.observation_steps += 1
        return output

    def step(self, step_index):
        if not self._active:
            raise RuntimeError('Use measured execution scope')
        self.step_index = int(step_index)
        self.measured_step_calls += 1

    def verify_online_assignment(self, hidden):
        if self.selector_kind != 'fused':
            return super().verify_online_assignment(hidden)
        from resident_dispatch.online import cpu_greedy_assignment
        ids,weights = self.routing(hidden)
        expected = cpu_greedy_assignment(ids.cpu().numpy(), self.initial_map)
        actual = self.mapper(ids).cpu().numpy()
        if not np.array_equal(actual, expected.physical_ids):
            raise AssertionError('Fused physical assignment differs from frozen CPU oracle')
        signature = _hash_array(actual,'<i8')
        if len(set(self.rt.gather_objects(signature))) != 1:
            raise AssertionError('Ranks disagree on fused physical assignment')
        return dict(physical_sha256=signature,
            logical_sha256=_hash_array(ids.cpu().numpy(),'<i8'),
            weights_sha256=_hash_array(weights.cpu().numpy(),'<f4'),
            **expected.diagnostics)

    def snapshot(self):
        result = super().snapshot()
        result.update(static_resident_mode=True, selector_kind=self.selector_kind,
            controller_active=False, controller_step_calls=0,
            static_step_calls=self.measured_step_calls,
            upstream_eplb_disabled=False,
            mode_description='Custom fixed160-slot mode; native gate/experts/reduction; no recording/planning/migration')
        return result

    def audit(self):
        result = super().audit()
        if (self.recorded_steps or self.decisions or self.async_events
            or self.state.expert_rearrangement_step != 0
            or self.state.expert_load_window_step != 0
            or bool(self.state.should_record_tensor.item())
            or self.torch.count_nonzero(self.ms.expert_load_pass).item()
            or self.torch.count_nonzero(self.ms.expert_load_window).item()
            or self.counter.sent_bytes or self.counter.sent_messages):
            raise AssertionError('Static mode advanced recording/control/transport state')
        result.update(static_state_unchanged=True, observation_and_migration_inactive=True)
        return result


def build_layers(rt, weights, *, selector='three_kernel'):
    off = StockLayer(rt, weights, 'stockoff')
    current = StockLayer(rt, weights, 'ours')
    rt.layers['current'] = rt.layers.pop('ours')
    fixed = StaticResidentLayer(rt, weights, selector=selector)
    return {'stockoff': off, 'current': current, 'fix': fixed}
