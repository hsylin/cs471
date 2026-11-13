import unittest
from full_serving.boundary_native_record_reproducer import FACTORS,PARENTS,changed_factors
class Isolation(unittest.TestCase):
    def test_all_pairwise_interventions_are_single_factor(self):
        self.assertEqual(changed_factors('map_record_off'),[])
        self.assertEqual(changed_factors('map_record_on'),['recording'])
        self.assertEqual(changed_factors('blocking_output'),['blocking_output'])
        self.assertEqual(changed_factors('eager_record'),['graph'])
        self.assertTrue(all(c['worker'] and c['async_output'] and c['sync_events'] for c in FACTORS.values()))
