import json
import tempfile
import unittest
from pathlib import Path
from full_serving.boundary_final_analysis import arithmetic, noop_evidence

class Tests(unittest.TestCase):
    def emit(self, path, rows):
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))

    def fixture(self, root):
        for rank in range(8):
            self.emit(root/f"gate-events-rank{rank}.jsonl",[
                dict(kind="decision",step=750,rank_agreement=True,digest="shared",
                     old_map=[[0,1],[2,3]],admitted_map=[[0,1],[3,2]])])
            self.emit(root/f"noop-commit-rank{rank}.jsonl",[
                dict(kind="unchanged_proof",cycle_step=750,unchanged_layers=[0],shadow=False,rank=rank),
                dict(kind="map_commit_elided",cycle_step=750,layer=0,rank=rank)])
            self.emit(root/f"migration-events-rank{rank}.jsonl",[
                dict(kind="commit_enqueued",cycle_step=750,layer=0)])

    def test_unobserved_commit_is_censored(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            result=noop_evidence(root,True)
            self.assertTrue(all(x["censored_elided_commits"]==1 and x["observed_elided_commits"]==0 for x in result["ranks"]))

    def test_changed_layer_cannot_be_elided(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            p=root/"noop-commit-rank0.jsonl";rr=[json.loads(x) for x in p.read_text().splitlines()]
            rr[-1]["layer"]=1;self.emit(p,rr)
            with self.assertRaises(AssertionError):noop_evidence(root,True)

    def test_timing_shadow_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            p=root/"noop-commit-rank0.jsonl"
            with p.open("a") as f:f.write(json.dumps(dict(kind="weight_shadow_passed",rank=0))+"\n")
            with self.assertRaises(AssertionError):noop_evidence(root,True)

    def test_off_rejects_candidate_events(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaises(AssertionError):noop_evidence(root,False)

    def test_tail_quantile_and_ttft_from_raw(self):
        rr=[dict(success=True,scheduled_ns=1000000,end_ns=(x+1)*1000000,
                 first_content_ns=1500000,usage=dict(completion_tokens=3)) for x in (1,2,3,100)]
        result=arithmetic(rr)
        self.assertAlmostEqual(result["response_ms"]["mean"],26.5)
        self.assertAlmostEqual(result["response_ms"]["p95"],85.45)
        self.assertEqual(result["ttft_ms"]["mean"],.5)
        self.assertEqual(result["output_tokens"],12)
        rr[0]["success"]=False
        with self.assertRaises(AssertionError):arithmetic(rr)

if __name__=="__main__":unittest.main()
