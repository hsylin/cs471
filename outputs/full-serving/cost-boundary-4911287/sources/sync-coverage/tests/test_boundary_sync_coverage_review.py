import copy,unittest
from full_serving.boundary_sync_coverage_review import coverage,ratio
def fixture(offsets=(1,55,105)):
 samples=[];rows=[]
 for i,t in enumerate(offsets):
  samples.append(dict(sync_call=64*(i+1),real_tokens_per_rank=[1,0,0,0,0,0,0,0],prefill_tokens_per_rank=[0]*8,decode_tokens_per_rank=[1,0,0,0,0,0,0,0],dummy_ranks=list(range(1,8)),total_real_tokens=1,real_token_max_over_mean=8,attention_pairs_upper_per_rank=[10,0,0,0,0,0,0,0],padded_tokens_across_dp=[1]*8))
  rows.append(dict(kind="actual_batch",sync_call=64*(i+1),execute_start_wall_ns=int(t*1e9),wall_ns=int((t+.01)*1e9),real_tokens=1,sample_policy="native_dp_sync_ordinal",profile=False))
 return samples,rows,dict(epoch_wall_ns_approx=0,offered_horizon_ns=100_000_000_000,completed_elapsed_ns=110_000_000_000)
class TestCoverage(unittest.TestCase):
 def test_time_and_drain(self):
  r=coverage(*fixture());self.assertEqual(r["active_offered_bins"],2);self.assertEqual(r["drain_samples"]["active"],1);self.assertEqual(r["rank0_last_active_elapsed_s"],105)
 def test_missing_bin_unknown(self):
  r=coverage(*fixture());self.assertIsNone(r["offered_decile_bins"][1]["observed_active_work"])
 def test_all_deciles(self):
  r=coverage(*fixture(range(1,100,10)));self.assertTrue(r["sampled_active_work_in_all_offered_deciles"])
 def test_start_only_not_sustained(self):
  r=coverage(*fixture(range(1,9)));self.assertEqual(r["active_samples"],8);self.assertFalse(r["sampled_active_work_in_all_offered_deciles"])
 def test_zero_work_no_skew(self):
  s,r,t=fixture((1,));s[0].update(real_tokens_per_rank=[0]*8,decode_tokens_per_rank=[0]*8,total_real_tokens=0,real_token_max_over_mean=None,dummy_ranks=list(range(8)))
  r[0].update(kind="explicit_empty_batch",real_tokens=0)
  out=coverage(s,r,t);self.assertEqual(out["active_samples"],0);self.assertEqual(out["offered_decile_bins"][0]["all_empty"],1);self.assertIsNone(out["sampled_work_classes"]["decode_only"]["real_token_max_over_mean"]["mean"])
 def test_exact_horizon_is_drain(self):self.assertEqual(coverage(*fixture((100,)))["drain_samples"]["active"],1)
 def test_missing_rank0_fails(self):
  s,r,t=fixture()
  with self.assertRaises(AssertionError):coverage(s,r[1:],t)
 def test_inconsistent_work_fails(self):
  s,r,t=fixture();s[0]["prefill_tokens_per_rank"][0]=1
  with self.assertRaises(AssertionError):coverage(s,r,t)
 def test_dummy_nonzero_fails(self):
  s,r,t=fixture();s[0]["dummy_ranks"]=[0]
  with self.assertRaises(AssertionError):coverage(s,r,t)
 def test_duplicate_fails(self):
  s,r,t=fixture()
  with self.assertRaises(AssertionError):coverage(s+[s[0]],r,t)
 def test_padding_is_not_work(self):
  s,r,t=fixture((1,));s[0]["padded_tokens_across_dp"]=[1024]*8
  out=coverage(s,r,t)["sampled_work_classes"]["decode_only"]
  self.assertEqual(out["real_token_max_over_mean"]["mean"],8);self.assertEqual(out["padded_tokens_per_real_token"]["mean"],8192)
 def test_work_classes_separated(self):
  s,r,t=fixture((1,2));s[1]["prefill_tokens_per_rank"]=[1,0,0,0,0,0,0,0];s[1]["decode_tokens_per_rank"]=[0]*8
  out=coverage(s,r,t)["sampled_work_classes"];self.assertEqual(out["decode_only"]["active_ranks"]["n"],1);self.assertEqual(out["prefill_only"]["active_ranks"]["n"],1);self.assertEqual(out["mixed"]["active_ranks"]["n"],0)
 def test_ratio_zero(self):self.assertIsNone(ratio([0]*8))
 def test_partial_rank_ratio_rejected(self):
  with self.assertRaises(AssertionError):ratio([1]*7)
if __name__=="__main__":unittest.main()
