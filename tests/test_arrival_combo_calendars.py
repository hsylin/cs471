"""Freshness and crossover-order properties for the composition experiment."""
from collections import Counter
import itertools
import unittest

from arrival_combo.calendars import arrival_blocks, frozen_orders, frozen_weights
from arrival_fix.calendars import arrival_blocks as previous_blocks


class ComboCalendarTests(unittest.TestCase):
    def matrix(self):
        return dict(arrival_blocks=arrival_blocks(), batches=[128,1024],
                    patterns=['spread','cluster'], rhos=[.5,.85])

    def test_freshness_and_fixed_payload_strata(self):
        blocks = arrival_blocks()
        self.assertEqual(len({b['seed'] for b in blocks}),8)
        self.assertFalse({b['seed'] for b in blocks} & {b['seed'] for b in previous_blocks()})
        self.assertEqual(sorted(Counter((b['block_id'],b['regime']) for b in blocks).values()),[2]*4)
        self.assertEqual(len({frozen_weights(b['seed'],p)['weights_sha256']
                              for b in blocks for p in ('spread','cluster')}),16)

    def test_pairwise_reversal_and_near_position_balance(self):
        orders = frozen_orders(self.matrix(),471431)
        self.assertEqual(len(orders),64)
        position = [Counter() for _ in range(3)]
        precedence = Counter()
        for first,second in orders.values():
            self.assertEqual(first,second[::-1])
            for order in (first,second):
                self.assertEqual(set(order),{'native','shortcut','combined'})
                for i,arm in enumerate(order): position[i][arm] += 1
                for a,b in itertools.combinations(order,2): precedence[a,b] += 1
        for a,b in itertools.combinations(('native','shortcut','combined'),2):
            self.assertEqual(precedence[a,b],64)
            self.assertEqual(precedence[b,a],64)
        self.assertEqual(sum(sum(v.values()) for v in position),384)
        for counts in position:
            self.assertLessEqual(max(counts.values())-min(counts.values()),2)
        self.assertTrue(any(len(set(counts.values()))>1 for counts in position))
        self.assertEqual(orders,frozen_orders(self.matrix(),471431))
        self.assertNotEqual(orders,frozen_orders(self.matrix(),471432))
