import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from full_serving import boundary_cost_sequence as m

def spec():
    return {"arms":list(m.CONTROLS),"orders":{"longbench":list(m.CONTROLS),"numinamath":list(reversed(m.CONTROLS))},
            "diagnostic":{"contrasts":{"fixed32_record/fixed32_idle":"record plus native window copies"}}}
def record(index,end,first=1):
    return {"success":True,"request_id":str(index),"scheduled_ns":0,"end_ns":int(end*1e6),
            "first_content_ns":int(first*1e6),"usage":{"completion_tokens":10}}
class Contracts(unittest.TestCase):
    def test_frozen_complete_order(self):
        cells=m.plan(spec())
        self.assertEqual(len(cells),16);self.assertEqual(len(set(cells)),16)
        self.assertEqual(cells[8],("numinamath","b4_observed"))
    def test_reordered_protocol_rejected(self):
        s=spec();s["orders"]["longbench"].reverse()
        with self.assertRaises(AssertionError):m.plan(s)
    def test_failed_request_not_dropped(self):
        rows=[record(0,2),record(1,4)];rows[1]["success"]=False
        with self.assertRaises(AssertionError):m.arithmetic(rows)
    def test_missing_ttft_rejected(self):
        rows=[record(0,2)];rows[0]["first_content_ns"]=None
        with self.assertRaises(AssertionError):m.arithmetic(rows)
    def test_independent_percentile_interpolates(self):
        value=m.arithmetic([record(0,10),record(1,30)])
        self.assertEqual(value["response_ms"],{"mean":20,"p95":29})
    def test_wrong_summary_rejected(self):
        v=m.arithmetic([record(0,10)])
        s={"response_from_schedule_ms":{"mean":9,"p95":10},
           "ttft_from_schedule_ms":{"mean":1,"p95":1}}
        with self.assertRaises(AssertionError):m.verify_arithmetic(v,s)
    def test_no_private_content_in_arithmetic(self):
        r=record(0,10);r["private_output"]="secret";r["token_ids"]=[1,2]
        self.assertNotIn("secret",json.dumps(m.arithmetic([r])))
        self.assertNotIn("token_ids",json.dumps(m.arithmetic([r])))
    def test_common_targets_and_no_profiles(self):
        cmd=m.controller_command(Path("/tmp/newrun"),"longbench","fixed32_idle",Path("/tmp/admission"))
        self.assertEqual(cmd[cmd.index("--reference")+1],str(m.REFERENCE))
        self.assertEqual(cmd[cmd.index("--mode")+1],"matched")
        self.assertEqual(cmd[cmd.index("--host-probe")+1],"off")
        self.assertEqual(cmd[cmd.index("--map-commit")+1],"native")
        self.assertNotIn("--pilot",cmd);self.assertNotIn("--branch",cmd)
    def test_existing_run_never_reexecutes(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as d, patch.object(m.subprocess,"run") as launch:
            with self.assertRaises(AssertionError):m.execute(Namespace(run=Path(d)))
            launch.assert_not_called()
    def test_incomplete_matrix_rejected(self):
        with self.assertRaises(AssertionError):m.combine({},spec())
    def test_ratio_direction_explicit(self):
        self.assertAlmostEqual(m.ratio(110,100)["change_percent"],10)
    def test_failing_predecessor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"completed.txt").write_text("sequence_exit=1\n")
            with self.assertRaises(AssertionError):m.verify_predecessor(p)
    def test_review_cannot_use_main_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m.write_new(p/"admission.json",{"status":"passed"})
            r={"status":"passed","scope":"cost_controls_main_admission","runtime_source":m.SOURCE,
               "protocol_sha256":m.PROTOCOL_SHA,"admission_sha256":m.sha(p/"admission.json"),
               "all_eight_pilots_reviewed":True,"main_measurements_used_for_admission":True}
            with self.assertRaises(AssertionError):m.verify_review(r,p)
            r["main_measurements_used_for_admission"]=False;m.verify_review(r,p)
    def test_tampered_admission_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"completed.txt").write_text("sequence_exit=0\n");hashes={}
            for arm in m.CONTROLS:
                (p/arm).mkdir()
                m.write_new(p/arm/"audit.json",{"status":"passed","pilot":True,"control":arm})
                hashes[arm]=m.sha(p/arm/"audit.json")
            m.write_new(p/"admission.json",{"status":"passed","controls":list(m.CONTROLS),
               "source_revision":m.SOURCE,"protocol_sha256":m.PROTOCOL_SHA,"audit_sha256":hashes})
            m.verify_admission(p)
            (p/m.CONTROLS[0]/"audit.json").write_text("{}")
            with self.assertRaises(AssertionError):m.verify_admission(p)
    def test_checkpoint_timestamp_is_not_weight_identity(self):
        roles=[{"hostname":"a","checkpoint_manifest_sha256":"old"},{"hostname":"b","checkpoint_manifest_sha256":"old"}]
        checkpoint={"complete":True,"kind":"full_checkpoint_sha256","checked_at":"first","shards":{"a":"hash"}}
        first=m.runtime_signature(roles,checkpoint,"old")
        for r in roles:r["checkpoint_manifest_sha256"]="new"
        checkpoint["checked_at"]="second"
        self.assertEqual(first,m.runtime_signature(roles,checkpoint,"new"))
        checkpoint["shards"]["a"]="different"
        self.assertNotEqual(first,m.runtime_signature(roles,checkpoint,"new"))
    def test_unbound_checkpoint_rejected(self):
        with self.assertRaises(AssertionError):
            m.runtime_signature([{"checkpoint_manifest_sha256":"bad"}]*2,
                                {"complete":True,"kind":"full_checkpoint_sha256"},"expected")
    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"receipt.json";m.write_new(p,{"old":1})
            with self.assertRaises(FileExistsError):m.write_new(p,{"new":2})
            self.assertEqual(m.read(p),{"old":1})
if __name__=="__main__":unittest.main()
