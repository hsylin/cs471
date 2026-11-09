"""Calendar provenance and endpoint checks, independent of GPU performance."""
from collections import Counter
import copy
import unittest

from arrival_fix.calendars import arrival_blocks, frozen_weights, make_calendar, validate_calendar, digest


class CalendarTests(unittest.TestCase):
    def test_balanced_seeds_and_paired_calendar_scaling(self):
        blocks = arrival_blocks()
        self.assertEqual(len({b['seed'] for b in blocks}), 8)
        self.assertEqual(sorted(Counter((b['block_id'], b['regime']) for b in blocks).values()), [2]*4)
        w = frozen_weights(blocks[0]['seed'], 'spread')
        a = make_calendar(w, .5, 256000000)
        b = make_calendar(w, .85, 256000000)
        self.assertEqual(len(a['arrival_offsets_ns']), 256)
        self.assertEqual(a, make_calendar(copy.deepcopy(w), .5, 256000000))
        self.assertEqual(a['horizon_ns'], 512000000)
        self.assertGreater(a['horizon_ns'], b['horizon_ns'])
        self.assertEqual(a['weights_sha256'], b['weights_sha256'])
        self.assertTrue(validate_calendar(a))
        self.assertNotEqual(w['weights_sha256'], frozen_weights(blocks[1]['seed'], 'spread')['weights_sha256'])

    def test_tiny_terminal_gap_retains_every_job_before_horizon(self):
        w = frozen_weights(3, 'cluster')
        w['gamma_draws_hex'] = [float(1).hex()]*255+[float(1e-300).hex()]
        w['weights_sha256'] = digest({k:v for k,v in w.items() if k != 'weights_sha256'})
        c = make_calendar(w, .5, 256000000)
        self.assertEqual(c['arrival_offsets_ns'][-1], c['horizon_ns']-1)
        self.assertEqual(len(c['arrival_offsets_ns']), 256)
        self.assertTrue(validate_calendar(c))

    def test_modified_draw_or_offset_is_rejected(self):
        w = frozen_weights(4, 'cluster')
        c = make_calendar(w, .85, 256000000)
        c['arrival_offsets_ns'][1] += 1
        with self.assertRaises(ValueError): validate_calendar(c)
        w['gamma_draws_hex'][0] = float(2).hex()
        with self.assertRaises(ValueError): make_calendar(w, .85, 256000000)


if __name__ == '__main__':
    unittest.main()
