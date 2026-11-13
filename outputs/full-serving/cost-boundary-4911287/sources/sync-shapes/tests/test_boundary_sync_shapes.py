import ast,copy
from pathlib import Path
import unittest
from full_serving.boundary_sync_shape_contract import selected,empty_summary,aligned_summary

def row(rank,ordinal=64,dummy=False):
 return dict(sync_call=ordinal,sample_policy="native_dp_sync_ordinal",padded_tokens_across_dp=[8]*8,
             real_tokens=0 if dummy else rank+1,prefill_tokens=0,decode_tokens=0 if dummy else rank+1,
             attention_pairs_cpu_upper_proxy=0 if dummy else 100*(rank+1),profile=False,dummy=dummy,
             no_actual_request_batch=dummy)
class Aligned(unittest.TestCase):
 def test_systematic_global_ordinal(self):
  self.assertTrue(selected(64));self.assertFalse(selected(63));self.assertTrue(selected(8))
 def test_no_profile_or_capture(self):
  self.assertFalse(selected(64,True));self.assertFalse(selected(64,False,True))
 def test_zero_native_work_not_fake_padding(self):
  s=empty_summary(8192,8);self.assertEqual(s["real_tokens"],0);self.assertEqual(s["padded_tokens"],8192)
 def test_missing_rank_is_not_zero(self):
  peers=[{64:row(i)} for i in range(8)];peers[7]={}
  self.assertFalse(aligned_summary(peers)["simultaneous_evidence_available"])
 def test_explicit_empty_in_skew(self):
  peers=[{64:row(i,dummy=i==0)} for i in range(8)]
  result=aligned_summary(peers)["samples"][0]
  self.assertEqual(result["dummy_ranks"],[0]);self.assertEqual(result["total_real_tokens"],35)
  self.assertAlmostEqual(result["real_token_max_over_mean"],8/(35/8))
 def test_dummy_with_real_tokens_rejected(self):
  peers=[{64:row(i)} for i in range(8)];peers[3][64]["dummy"]=True
  with self.assertRaises(AssertionError):aligned_summary(peers)
 def test_profile_row_rejected(self):
  peers=[{64:row(i)} for i in range(8)];peers[2][64]["profile"]=True
  with self.assertRaises(AssertionError):aligned_summary(peers)
 def test_misaligned_vectors_rejected(self):
  peers=[{64:row(i)} for i in range(8)];peers[2][64]["padded_tokens_across_dp"]=[9]*8
  with self.assertRaises(AssertionError):aligned_summary(peers)
 def test_all_zero_not_divided(self):
  r=aligned_summary([{64:row(i,dummy=True)} for i in range(8)])["samples"][0]
  self.assertIsNone(r["real_token_max_over_mean"])
 def test_hook_uses_native_ordinal_not_execute_counter(self):
  s=(Path(__file__).parents[1]/"full_serving/boundary_shapes.py").read_text()
  self.assertIn("selected(sync_calls+1,profile",s)
  self.assertNotIn("calls%64",s)
  self.assertIn("dispatches=dummy or scheduled>0",s)
  self.assertIn('assert record["expected_sync_call"]==sync_calls',s)
 def test_no_gpu_synchronize_or_extra_collective(self):
  s=(Path(__file__).parents[1]/"full_serving/boundary_shapes.py").read_text()
  for node in ast.walk(ast.parse(s)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
    self.assertNotIn(node.func.attr,("synchronize","all_reduce","all_gather","barrier"))
 def test_native_early_return_and_dummy_semantics_pinned(self):
  p=Path("/work/cs-471/skewbench-hlin/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")
  if not p.exists():self.skipTest("Pinned native source only present on cluster")
  s=p.read_text();node=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=="execute_model")
  body=ast.get_source_segment(s,node)
  self.assertLess(body.index("scheduler_output.total_num_scheduled_tokens == 0"),body.index("dispatch_cg_and_sync_dp("))
  self.assertIn("No actual tokens to run. A dummy run for DP or memory profiling.",body)
  self.assertIn("is_padding=not is_profile",body)
if __name__=="__main__":unittest.main()
