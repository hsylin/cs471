import unittest
from full_serving.boundary_transition_comparison import completed_prefix
class Tests(unittest.TestCase):
    def fixture(self):
        timing=dict(epoch_wall_ns_approx=1000,epoch_monotonic_ns=100)
        a=[dict(request_id=str(i),input_sha256=str(i),output_sha256=str(i),end_ns=e) for i,e in enumerate([110,150,120])]
        b=[dict(x) for x in a]
        return a,b,timing
    def test_noncontiguous_completed_subset_not_full_prefix(self):
        a,b,t=self.fixture();r=completed_prefix(a,b,t,t,1040,1040)
        self.assertEqual(r["completed_before_both_proposals"],2)
        self.assertEqual(r["contiguous_completed_request_prefix"],1)
    def test_equal_counts_never_imply_equal_outputs(self):
        a,b,t=self.fixture();b[0]["output_sha256"]="different"
        r=completed_prefix(a,b,t,t,1040,1040)
        self.assertFalse(r["every_comparable_completed_output_matches"])
    def test_missing_opportunity_is_unknown(self):
        a,b,t=self.fixture();r=completed_prefix(a,b,t,t,None,1040)
        self.assertEqual(r["completed_before_both_proposals"],0)
        self.assertIsNone(r["every_comparable_completed_output_matches"])
    def test_requires_both_runs_completed_before_decision(self):
        a,b,t=self.fixture();b[0]["end_ns"]=200
        r=completed_prefix(a,b,t,t,1040,1040)
        self.assertEqual(r["completed_before_both_proposals"],1)
        self.assertEqual(r["contiguous_completed_request_prefix"],0)
if __name__=="__main__":unittest.main()
