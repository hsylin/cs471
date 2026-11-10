"""Mutation tests for independent sequence routing and probe reconstruction."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

import audit_sequence as a


class SequenceExecutionTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(1024, dtype=np.int32).reshape(128, 8) % 128
        self.weights = np.full((128, 8), .125, np.float32)

    def test_single_copy_native_identity(self):
        self.assertTrue(np.array_equal(a.native_physical(self.ids, np.arange(128)), self.ids))

    def test_shared_native_hash_uses_token_not_choice(self):
        ids = np.tile(np.arange(8, dtype=np.int32), (128, 1))
        expected = ids + ((np.arange(128) * 2654435769 & 0xffffffff) % 2)[:, None] * 128
        self.assertTrue(np.array_equal(a.native_physical(ids, a.INITIAL), expected))
        # Odd columns must not choose a different copy from even columns.
        self.assertEqual(len(np.unique(a.native_physical(ids, a.INITIAL)[0] // 128)), 1)

    def test_swapped_map_reconstructs_inverse_positions(self):
        mapping = a.INITIAL.copy(); mapping[[0, 80]] = mapping[[80, 0]]
        actual = a.native_physical(self.ids, mapping)
        for r in range(128):
            for c in range(8):
                eligible = [i for i, x in enumerate(mapping) if x == self.ids[r, c]]
                selected = eligible[((r * 2654435769) & 0xffffffff) % len(eligible)]
                self.assertEqual(actual[r, c], selected)
        self.assertTrue(np.array_equal(mapping[actual], self.ids))

    def test_batched_native_reconstruction_matches_single(self):
        many = np.stack((self.ids, self.ids[::-1], self.ids[:, ::-1]))
        actual = a.native_physical(many, a.INITIAL)
        for i in range(3): self.assertTrue(np.array_equal(actual[i], a.native_physical(many[i], a.INITIAL)))

    def test_illegal_native_map_rejected(self):
        mapping = a.INITIAL.copy(); mapping[mapping == 127] = 0
        with self.assertRaises(ValueError): a.native_physical(self.ids, mapping)

    def test_local_work_conservation_and_tile_geometry(self):
        actual = a.physical_work(self.ids, 128)
        self.assertEqual(actual['per_rank_tokens'], [512, 512])
        self.assertEqual(actual['per_rank_active_copies'], [64, 64])
        self.assertEqual(actual['per_rank_padded_rows'], [4096, 4096])
        self.assertEqual(sum(actual['physical_counts']), 1024)

    def probe(self, physical, mapping, rank):
        slots = len(mapping); width = slots // 2
        counts = np.bincount(physical.ravel(), minlength=slots)
        owned = counts[rank * width:(rank + 1) * width]
        expert_map = np.full(slots, -1, np.int64); expert_map[rank * width:(rank + 1) * width] = np.arange(width)
        padded = int(np.sum((owned + 63) // 64 * 64)); tokens = int(owned.sum())
        return dict(physical_ids=physical.tolist(), physical_ids_sha256=a.raw_hash(physical),
            logical_ids_sha256=a.raw_hash(self.ids), combining_weights_sha256=a.raw_hash(self.weights, '<f4'),
            physical_counts=counts.tolist(), local_expert_map=expert_map.tolist(), local_counts=owned.tolist(),
            local_assignments=tokens, local_active_copies=int(np.count_nonzero(owned)),
            local_nonempty_m_tiles=padded // 64, local_padded_assignment_rows=padded,
            useful_two_gemm_flops_proxy=tokens * 6 * 2048 * 768,
            padded_two_gemm_flops_proxy=padded * 6 * 2048 * 768,
            useful_assignment_fraction=tokens / padded if padded else None)

    def test_probe_reconstructs_actual_arguments_and_proxies(self):
        physical = a.native_physical(self.ids, a.INITIAL)
        for rank in (0, 1):
            a.check_probe_assignment(self.probe(physical, a.INITIAL, rank), physical, self.ids, self.weights, a.INITIAL, rank)

    def test_probe_mutations_rejected(self):
        physical = a.native_physical(self.ids, a.INITIAL); original = self.probe(physical, a.INITIAL, 0)
        mutations = {
            'wrong actual physical ID': lambda r: r['physical_ids'][0].__setitem__(0, 127),
            'wrong float weight digest': lambda r: r.__setitem__('combining_weights_sha256', '0' * 64),
            'global duplicated count': lambda r: r.__setitem__('local_assignments', 1024),
            'active versus allocated slots': lambda r: r.__setitem__('local_active_copies', 80),
            'wrong two GEMM FLOPs': lambda r: r.__setitem__('padded_two_gemm_flops_proxy', r['padded_two_gemm_flops_proxy'] // 2),
            'wrong ownership': lambda r: r['local_expert_map'].__setitem__(80, 0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(original); mutate(changed)
                with self.assertRaises(ValueError): a.check_probe_assignment(changed, physical, self.ids, self.weights, a.INITIAL, 0)

    def identity_fixture(self):
        idx = np.tile(np.arange(128), (18, 1))
        logical = (np.arange(128)[:, None] + np.arange(8)[None]) % 128
        weights = np.full((128, 8), .125, np.float32)
        result = {}
        for mode, indices in [('contiguous', idx), ('mixed', idx[:, ::-1])]:
            result['g', mode] = dict(indices=indices.copy(), ids=logical[indices].astype(np.int32),
                weights=weights[indices].copy(), metadata=dict(source_records=128))
        return result

    def test_per_record_routes_across_reordered_occurrences(self):
        report = a.check_token_identity(self.identity_fixture(), ['g'])
        self.assertTrue(report['g']['float32_combining_weights_bitwise_equal'])

    def test_per_record_weight_mutation_not_hidden_by_aggregate_counts(self):
        cache = self.identity_fixture()
        cache['g', 'mixed']['weights'][17, 0, [0, 1]] = [.124, .126]
        with self.assertRaisesRegex(ValueError, 'combining-weight bytes'):
            a.check_token_identity(cache, ['g'])

    def test_per_record_route_swap_preserving_aggregate_counts_rejected(self):
        cache = self.identity_fixture(); ids = cache['g', 'mixed']['ids']
        ids[17, [0, 1]] = ids[17, [1, 0]]
        with self.assertRaisesRegex(ValueError, 'logical routing changed'):
            a.check_token_identity(cache, ['g'])

    def test_exact_current_frozen_plan_and_source(self):
        plan = json.loads((a.REPO / 'configs/sequence-study-v2.json').read_text())
        a.plan_check(plan, a.REPO)

    def test_order_and_source_mutations_rejected(self):
        original = json.loads((a.REPO / 'configs/sequence-study-v2.json').read_text())
        for field in ('order', 'calendar', 'candidate'):
            with self.subTest(field=field):
                p = copy.deepcopy(original)
                if field == 'order': p['trials'][0], p['trials'][1] = p['trials'][1], p['trials'][0]
                elif field == 'calendar': next(iter(p['calendars'].values()))['calendar']['arrival_offsets_ns'][1] += 1
                else: p['selected_fix']['mapper_input_checks'] = False
                p['content_sha256'] = a.digest(a.stock.unsealed(p))
                with self.assertRaises(ValueError): a.plan_check(p, a.REPO)


if __name__ == '__main__': unittest.main()
