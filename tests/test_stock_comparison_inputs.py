import unittest
import numpy as np
from stock_comparison.inputs import build_indices, array_digest, WARMUP
from stock_comparison.common import scale_calendar, validate_calendar

class StockInputTests(unittest.TestCase):
    def test_deterministic_fresh_combinations_and_reported_reuse(self):
        a,ma=build_indices(500,'stock-mbpp-pilot-01',jobs=64)
        b,mb=build_indices(500,'stock-mbpp-pilot-01',jobs=64)
        c,_=build_indices(500,'stock-mbpp-pilot-02',jobs=64)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ma,mb)
        self.assertFalse(np.array_equal(a,c))
        self.assertEqual(a.shape,(80,128))
        self.assertEqual(len({np.sort(row).tobytes() for row in a}),80)
        self.assertTrue(np.all(np.diff(np.sort(a,axis=1),axis=1)>0))
        self.assertEqual(ma['measured_record_occurrences'],64*128)
        uses=np.bincount(a[WARMUP:].ravel(),minlength=500)
        self.assertEqual(ma['maximum_record_uses'],int(uses.max()))
    def test_float_hash_preserves_values_and_dtype(self):
        a=np.array([.1,.2],np.float32);b=np.array([.1,.3],np.float32)
        self.assertNotEqual(array_digest(a),array_digest(b))
        self.assertNotEqual(array_digest(a),array_digest(a.astype(np.float64)))
    def test_input_capacity_rejects_degenerate_bank(self):
        for count in (127,128,False):
            with self.assertRaises(ValueError):build_indices(count,'x',jobs=64)
    def test_absolute_arrivals_preserve_ties_and_authenticate(self):
        source=dict(original_offsets_ns=[0,2,2,6],original_horizon_ns=8,jobs=4,
            id='x',source_rows_sha256='abc',trace='code',source_date_utc='2024-05-10')
        c=scale_calendar(source,800,rho=.5,jobs=4)
        self.assertEqual(c['arrival_offsets_ns'],[0,400,400,1200])
        self.assertEqual(c['horizon_ns'],1600)
        validate_calendar(c,source)
        c['arrival_offsets_ns'][2]+=1
        with self.assertRaises(ValueError):validate_calendar(c,source)
if __name__=='__main__':unittest.main()
