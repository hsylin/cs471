import copy
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from full_serving import boundary_transition_sequence as m
from full_serving.boundary_transition_comparison import completed_prefix

def spec():
 return {"arms":list(m.MODES),"orders":{"longbench":list(m.MODES),"numinamath":list(reversed(m.MODES))}}
def cells():
 hardware={"runtime":[{"hostname":host,"devices":[{"uuid":host+str(i),"name":"NVIDIA H100"} for i in range(4)]} for host in ("kh021","kh023")],"checkpoint":{"weights":"same"}}
 return {(ds,mode):{"runtime_signature":copy.deepcopy(hardware),
     "phases":{"measured":{"arithmetic":{"response_ms":{"mean":100,"p95":150},"ttft_ms":{"mean":10,"p95":20}}}}}
     for ds,mode in m.plan(spec())}
def comparisons():
 return {ds:{"status":"passed","causal_payback_established":False} for ds in m.TOTALS}
class Sequence(unittest.TestCase):
 def test_complete_frozen_order(self):
  self.assertEqual(m.plan(spec()),[("longbench","keep"),("longbench","preinstalled"),("longbench","charged"),
                                  ("numinamath","charged"),("numinamath","preinstalled"),("numinamath","keep")])
 def test_reordered_plan_rejected(self):
  s=spec();s["orders"]["longbench"].reverse()
  with self.assertRaises(AssertionError):m.plan(s)
 def test_controller_targets_and_no_shadow(self):
  c=m.controller_command(Path("/tmp/new"),"longbench","charged",Path("/tmp/admission"))
  self.assertEqual(c[c.index("--reference")+1],str(m.REFERENCE))
  self.assertEqual(c[c.index("--mode")+1],"matched")
  self.assertEqual(c[c.index("--host-probe")+1],"off")
  self.assertNotIn("--pilot",c);self.assertNotIn("--branch",c)
 def test_existing_directory_never_reruns(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m.subprocess,"run") as launch:
   with self.assertRaises(AssertionError):m.execute(Namespace(run=Path(d)))
   launch.assert_not_called()
 def test_incomplete_comparison_rejected(self):
  c=cells();c.pop(("longbench","charged"))
  with self.assertRaises(AssertionError):m.combine(c,comparisons(),spec())
 def test_changed_gpu_rejected(self):
  c=cells();c[("longbench","charged")]["runtime_signature"]["runtime"][0]["devices"][0]["uuid"]="different"
  with self.assertRaises(AssertionError):m.combine(c,comparisons(),spec())
 def test_no_automatic_causal_payback(self):
  c=comparisons();c["longbench"]["causal_payback_established"]=True
  with self.assertRaises(AssertionError):m.combine(cells(),c,spec())
 def test_all_six_summary_limits(self):
  c=m.combine(cells(),comparisons(),spec())
  self.assertEqual(c["sessions"],6);self.assertEqual(c["measured_requests"],3072)
  self.assertFalse(c["causal_payback_established"])
  self.assertIn("warmup",c["datasets"]["longbench"]["contrasts"]["charged/preinstalled"]["scope"])
 def test_failed_predecessor_does_not_launch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"completed.txt").write_text("sequence_exit=1\n")
   with self.assertRaises(AssertionError):m.verify_predecessor(p)
 def test_review_rejects_future_result_selection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m.write_new(p/"admission.json",{"status":"passed"})
   review={"status":"passed","scope":"transition_main_admission","runtime_source":m.SOURCE,
    "protocol_sha256":m.PROTOCOL_SHA,"candidate_sha256":m.CANDIDATE_SHA,
    "admission_sha256":m.sha(p/"admission.json"),"all_three_pilots_reviewed":True,
    "all_rank_graph_weight_metadata_reviewed":True,"charged_target_all48_observed_commits_reviewed":True,
    "pilot_logprob_limitations_reviewed":True,"main_measurements_used_for_admission":True,"causal_payback_established":False}
   with self.assertRaises(AssertionError):m.verify_review(review,p)
   review["main_measurements_used_for_admission"]=False;m.verify_review(review,p)
   review["charged_target_all48_observed_commits_reviewed"]=False
   with self.assertRaises(AssertionError):m.verify_review(review,p)
 def test_failed_request_cannot_be_dropped(self):
  with self.assertRaises(AssertionError):m.arithmetic([{"success":False}])
 def test_numeric_quantile_and_hash_only_output(self):
  r=[{"success":True,"end_ns":v*1000000,"first_content_ns":1000000,"scheduled_ns":0,"usage":{"completion_tokens":8},"private_text":"private"} for v in (10,30)]
  a=m.arithmetic(r);self.assertEqual(a["response_ms"],{"mean":20,"p95":29})
  self.assertNotIn("private",json.dumps(a))
 def test_checkpoint_time_only_excluded(self):
  roles=[{"checkpoint_manifest_sha256":"one"}]*2
  a=m.runtime_signature(roles,{"complete":True,"kind":"full_checkpoint_sha256","checked_at":"a","shards":{"weight":"X"}},"one")
  b=m.runtime_signature([{"checkpoint_manifest_sha256":"two"}]*2,{"complete":True,"kind":"full_checkpoint_sha256","checked_at":"b","shards":{"weight":"X"}},"two")
  self.assertEqual(a,b)
  b["checkpoint"]["shards"]["weight"]="Y";self.assertNotEqual(a,b)
 def test_equal_counts_not_equal_output_prefix(self):
  left=[{"request_id":"x","input_sha256":"same","output_sha256":"A","end_ns":5}]
  right=[dict(left[0],output_sha256="B")]
  timing={"epoch_wall_ns_approx":0,"epoch_monotonic_ns":0}
  out=completed_prefix(left,right,timing,timing,10,10)
  self.assertFalse(out["every_comparable_completed_output_matches"])
 def test_missing_opportunity_is_not_comparable(self):
  r=[{"request_id":"x","input_sha256":"same","output_sha256":"A","end_ns":5}]
  timing={"epoch_wall_ns_approx":0,"epoch_monotonic_ns":0}
  self.assertIsNone(completed_prefix(r,r,timing,timing,None,None)["every_comparable_completed_output_matches"])
 def test_serial_failure_cannot_be_bypassed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"completed.txt").write_text("sequence_exit=1\n")
   with self.assertRaises(AssertionError):m.verify_serial_predecessor(p)
 def test_serial_requires_expected_sync_runtime(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/"completed.txt").write_text("sequence_exit=0\n")
   m.write_new(p/"admission.json",{"status":"passed","sync_shape_probe_passed":True,"source_revision":"wrong"})
   with self.assertRaises(AssertionError):m.verify_serial_predecessor(p)
 def test_serial_is_mandatory_before_any_controller(self):
  import ast
  source=Path(m.__file__).read_text();tree=ast.parse(source)
  node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="execute")
  text=ast.get_source_segment(source,node)
  self.assertLess(text.index("verify_serial_predecessor"),text.index("controller_command"))
 def test_serial_default_tracks_existing_queue(self):
  self.assertIn('default=ROOT/"sync-shapes-run-v1"',Path(m.__file__).read_text())
 def test_new_receipt_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"receipt";m.write_new(p,{"old":True})
   with self.assertRaises(FileExistsError):m.write_new(p,{"new":True})
if __name__=="__main__":unittest.main()
