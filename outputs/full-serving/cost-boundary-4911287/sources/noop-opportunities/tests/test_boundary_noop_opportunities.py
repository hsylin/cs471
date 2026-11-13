import copy,unittest
from full_serving.boundary_noop_opportunities import reconstruct
def fixture():
 old=[[j%128 for j in range(160)] for i in range(48)];new=copy.deepcopy(old);new[3][0]=1
 return [[dict(rank=r,kind="decision",step=3750,rank_agreement=True,old_map=old,admitted_map=new,digest="agreed",wall_ns=100),
          *[dict(rank=r,kind=k,cycle_step=3750,layer=i) for k in ("commit_enqueued","commit_completed") for i in range(48)]] for r in range(8)]
class Tests(unittest.TestCase):
 def test_partial_change(self):
  v=reconstruct(fixture(),50,150)["partitions_by_proposal_time"]["during_measurement"]
  self.assertEqual(v["unchanged_layer_opportunities"],47);self.assertEqual(v["changed_layer_opportunities"],1);self.assertEqual(v["unchanged_all_rank_completed"],47)
 def test_missing_completion_censored(self):
  p=fixture();p[7]=[r for r in p[7] if not(r["kind"]=="commit_completed" and r["layer"]==0)]
  v=reconstruct(p,50,150)["partitions_by_proposal_time"]["during_measurement"];self.assertEqual(v["unchanged_censored_or_unenqueued"],1)
 def test_missing_enqueue_censored(self):
  p=fixture();p[0]=[r for r in p[0] if r["kind"]=="decision" or r["layer"]!=0]
  v=reconstruct(p,50,150)["partitions_by_proposal_time"]["during_measurement"];self.assertEqual(v["unchanged_all_rank_enqueued"],46)
 def test_nonagreed_rejected(self):
  p=fixture();p[2][0]["rank_agreement"]=False
  with self.assertRaises(AssertionError):reconstruct(p,50,150)
 def test_disagreed_maps_rejected(self):
  p=copy.deepcopy(fixture());p[1][0]["old_map"]=copy.deepcopy(p[1][0]["old_map"]);p[1][0]["old_map"][0][0]=7
  with self.assertRaises(AssertionError):reconstruct(p,50,150)
 def test_duplicate_completion_rejected(self):
  p=fixture();p[0].append(p[0][-1])
  with self.assertRaises(AssertionError):reconstruct(p,50,150)
 def test_completion_without_enqueue_rejected(self):
  p=fixture();p[0]=[r for r in p[0] if not(r["kind"]=="commit_enqueued" and r["layer"]==0)]
  with self.assertRaises(AssertionError):reconstruct(p,50,150)
 def test_phase_is_proposal_time(self):
  out=reconstruct(fixture(),101,150)["partitions_by_proposal_time"];self.assertEqual(out["before_measurement"]["unchanged_all_rank_completed"],47);self.assertEqual(out["during_measurement"]["proposal_cycles"],0)
 def test_missing_rank_rejected(self):
  with self.assertRaises(AssertionError):reconstruct(fixture()[:-1],50,150)
 def test_missing_cycle_rejected(self):
  p=fixture();p[7]=[]
  with self.assertRaises(AssertionError):reconstruct(p,50,150)
if __name__=="__main__":unittest.main()
