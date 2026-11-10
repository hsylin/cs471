"""CPU-only mutation tests; no production schedule imports."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('sequence_input_primitives',
    Path(__file__).with_name('sequence_input_primitives.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SequenceInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = np.repeat(np.arange(2, dtype=np.int32), 128)
        cls.positions = np.tile(np.arange(128, dtype=np.int32), 2)
        cls.rec = audit.reconstruct(cls.prompts, cls.positions, 'auditor-fixture', jobs=4)

    def check(self, arrays, metadata):
        return audit.check_inputs(arrays, metadata, self.prompts, self.positions,
                                  'auditor-fixture', reconstruction=self.rec)

    def test_cycle_and_exact_bag(self):
        arrays, metadata = self.rec
        for mode in audit.MODES:
            with self.subTest(mode=mode):
                result = self.check(arrays[mode], metadata[mode])
                self.assertTrue(result['passed'])
                self.assertTrue(np.array_equal(np.bincount(arrays[mode][16:].ravel()), np.full(256, 2)))
                self.assertTrue(all(len(set(row)) == 128 for row in arrays[mode]))
        cycle = arrays['contiguous'][16:18].ravel()
        self.assertTrue(np.array_equal(arrays['contiguous'][18:].ravel(), cycle))
        for prompt in (0, 1):
            self.assertTrue(np.array_equal(self.positions[cycle[self.prompts[cycle] == prompt]], np.arange(128)))
        self.assertEqual(metadata['contiguous']['complete_bank_cycles'], 2)
        self.assertEqual(metadata['mixed']['collision_repair']['final_duplicate_occurrences'], 0)
        self.assertGreater(metadata['mixed']['collision_repair']['swaps'], 0)

    def test_full_batch_membership_preserved_only_in_batch_shuffle(self):
        arrays, _ = self.rec
        a = sorted(tuple(row) for row in arrays['contiguous'][16:])
        b = sorted(tuple(row) for row in arrays['batch_shuffle'][16:])
        c = sorted(tuple(row) for row in arrays['mixed'][16:])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_index_swap_rejected_even_when_bag_and_hash_updated(self):
        arrays, metadata = self.rec
        altered = arrays['mixed'].copy()
        altered[16, 0], altered[16, 1] = altered[16, 1], altered[16, 0]
        meta = copy.deepcopy(metadata['mixed'])
        meta['indices_sha256'] = audit.array_hash(altered)
        with self.assertRaisesRegex(AssertionError, 'independent frozen reconstruction'):
            self.check(altered, meta)

    def test_core_metadata_mutations_rejected(self):
        arrays, metadata = self.rec
        for key in ('measured_record_occurrences', 'warmup_indices_sha256', 'source_records',
                    'mixed_may_repeat_captured_record_within_job', 'new_ordering_only'):
            with self.subTest(key=key):
                meta = copy.deepcopy(metadata['mixed'])
                value = meta[key]
                meta[key] = not value if type(value) is bool else value + 1 if type(value) is int else '0'*64
                with self.assertRaisesRegex(AssertionError, 'metadata mismatch'):
                    self.check(arrays['mixed'], meta)
        meta = copy.deepcopy(metadata['mixed'])
        meta['collision_repair']['swaps'] += 1
        with self.assertRaisesRegex(AssertionError, 'metadata mismatch'):
            self.check(arrays['mixed'], meta)

    def test_outer_capture_metadata_allowed(self):
        arrays, metadata = self.rec
        meta = dict(metadata['contiguous'], capture_sha256='outer-file-hash', source_prompt_ids=['a', 'b'])
        self.assertTrue(self.check(arrays['contiguous'], meta)['passed'])

    def test_incomplete_or_duplicate_prefix_rejected(self):
        for bad in (np.r_[self.positions[:-1], 126], np.r_[self.positions[:-1], 129]):
            with self.subTest(last=int(bad[-1])):
                with self.assertRaisesRegex(AssertionError, 'complete unique prefix'):
                    audit.reconstruct(self.prompts, bad, 'auditor-fixture', jobs=4)

    def test_missing_document_label_rejected(self):
        prompts = self.prompts.copy(); prompts[prompts == 1] = 2
        with self.assertRaisesRegex(AssertionError, 'no gaps'):
            audit.reconstruct(prompts, self.positions, 'auditor-fixture', jobs=4)

    def test_wrong_integer_storage_rejected(self):
        arrays, metadata = self.rec
        with self.assertRaisesRegex(AssertionError, 'Saved indices'):
            self.check(arrays['contiguous'].astype(np.int32), metadata['contiguous'])

    def test_record_storage_order_does_not_change_document_token_order(self):
        order = np.arange(256)[::-1]
        p, t = self.prompts[order], self.positions[order]
        arrays, metadata = audit.reconstruct(p, t, 'auditor-fixture', jobs=4)
        for mode in audit.MODES:
            self.assertTrue(audit.check_inputs(arrays[mode], metadata[mode], p, t, 'auditor-fixture')['passed'])
        row = arrays['contiguous'][16]
        self.assertTrue(np.array_equal(t[row], np.arange(128)))


if __name__ == '__main__':
    unittest.main()
