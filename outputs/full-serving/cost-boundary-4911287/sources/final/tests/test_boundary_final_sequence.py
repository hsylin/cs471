import copy,json
from pathlib import Path
import tempfile,unittest
from argparse import Namespace
from unittest.mock import patch
from full_serving import boundary_final_sequence as m

def spec():
 return {"treatments":list(m.TREATMENTS),"orders":{"longbench":list(m.TREATMENTS),"numinamath":list(reversed(m.TREATMENTS))},"selected_fix":"exact_unchanged_map_commit_v1"}
def peers(fix=95):
 out={}
 for t,mean in (("off",100),("b4",110),("b4_noop",fix)):
  out[t]={"phases":{"measured":{"arithmetic":{"response_ms":{"mean":mean,"p95":mean+20},"ttft_ms":{"mean":10,"p95":20},
              "requests":512,"failed_requests":0,"output_tokens":75246,"ceiling_hits":0},
        "timing":{"completed_elapsed_ns":300000000000,"offered_horizon_ns":270000000000,"drain_ns":30000000000}}}}
 return out
def cells():
 hardware={"runtime":[{"hostname":host,"devices":[{"uuid":host+str(i),"name":"NVIDIA H100"} for i in range(4)]} for host in ("kh021","kh023")],"checkpoint":{"weights":"same"}}
 result={}
 for ds in m.TOTALS:
  for t,c in peers().items():
   c["runtime_signature"]=copy.deepcopy(hardware);c["phases"]["measured"]["arithmetic"]["output_tokens"]=m.TOTALS[ds]
   result[(ds,t)]=c
 return result
class Tests(unittest.TestCase):
 def test_complete_fixed_order(self):
  self.assertEqual(len(m.plan(spec())),6);self.assertEqual(m.plan(spec())[3],("numinamath","b4_noop"))
 def test_wrong_order_rejected(self):
  s=spec();s["orders"]["longbench"].reverse()
  with self.assertRaises(AssertionError):m.plan(s)
 def test_five_percent_target_boundary(self):
  self.assertTrue(m.comparisons(peers(95))["b4_noop/off"]["useful_mean_gain_at_least5percent"])
  self.assertFalse(m.comparisons(peers(95.01))["b4_noop/off"]["useful_mean_gain_at_least5percent"])
 def test_two_percent_guard_boundary(self):
  self.assertFalse(m.comparisons(peers(102))["b4_noop/off"]["mean_response_regression_exceeds2percent"])
  self.assertTrue(m.comparisons(peers(102.01))["b4_noop/off"]["mean_response_regression_exceeds2percent"])
 def test_no_average_hides_dataset_regression(self):
  c=cells();c[("numinamath","b4_noop")]["phases"]["measured"]["arithmetic"]["response_ms"]["mean"]=104
  result=m.combine(c,spec(),"s","p","a")
  self.assertTrue(result["datasets"]["numinamath"]["comparisons"]["b4_noop/off"]["mean_response_regression_exceeds2percent"])
  self.assertTrue(result["datasets"]["longbench"]["comparisons"]["b4_noop/off"]["useful_mean_gain_at_least5percent"])
 def test_missing_arm_rejected(self):
  c=cells();c.pop(("longbench","off"))
  with self.assertRaises(AssertionError):m.combine(c,spec(),"s","p","a")
 def test_changed_hardware_rejected(self):
  c=cells();c[("longbench","off")]["runtime_signature"]["runtime"][0]["devices"][0]["uuid"]="other"
  with self.assertRaises(AssertionError):m.combine(c,spec(),"s","p","a")
 def test_no_gpu_rerun_existing_dir(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m.subprocess,"run") as run:
   with self.assertRaises(AssertionError):m.execute(Namespace(run=Path(d)))
   run.assert_not_called()
 def test_command_has_only_selected_fix(self):
  c=m.controller_command(Path("/tmp/r"),"longbench","b4_noop",Path("/tmp/d"),Path("/tmp/p"),Path("/tmp/a"),"s"*40)
  self.assertEqual(c[c.index("--map-commit")+1],"noop_maps")
  self.assertEqual(c[c.index("--arm")+1],"b4")
  self.assertEqual(c[c.index("--host-probe")+1],"off")
  self.assertNotIn("--pilot",c);self.assertNotIn("--branch",c)
 def test_failed_predecessor_not_bypassed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"completed.txt").write_text("sequence_exit=1\n")
   with self.assertRaises(AssertionError):m.verify_predecessor(p)
 def test_render_has_six_cells_no_untrusted_markup(self):
  r=m.combine(cells(),spec(),"<script>alert(1)</script>","p","a")
  page=m.render_html(r)
  self.assertNotIn("<script>",page);self.assertIn("&lt;script&gt;",page)
  self.assertIn("尚未發布",page);self.assertIn("p95",page);self.assertIn("排空",page)
  self.assertEqual(page.count("<td>512</td>"),6)
 def test_transition_evidence_required(self):
  from full_serving.boundary_final_contract import EVIDENCE
  self.assertIn("transition_controls",EVIDENCE)
if __name__=="__main__":unittest.main()
