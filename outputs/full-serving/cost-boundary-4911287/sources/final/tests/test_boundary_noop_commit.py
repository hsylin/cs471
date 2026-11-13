import unittest
from types import SimpleNamespace
from full_serving.boundary_noop_commit import CycleProof,commit_with_proof,invalidate_then_call

class Tensor:
 def __init__(self,v,cpu=True):self.v=v;self.is_cpu=cpu
 def tolist(self):return list(self.v)
class Tests(unittest.TestCase):
 def test_external_map_mutation_invalidates_before_call(self):
  ms=SimpleNamespace(cs471_noop_commit_proof=CycleProof(1,[[0]],[[0]]),pending_result=object())
  st=SimpleNamespace(model_states={"m":ms});pending=ms.pending_result;seen=[]
  def original(state,value,*,flag):
   self.assertFalse(ms.cs471_noop_commit_proof.rows)
   self.assertIs(ms.pending_result,pending);seen.append((value,flag));return pending
  self.assertIs(invalidate_then_call(original,st,7,flag=True),pending)
  self.assertEqual(seen,[(7,True)])
 def test_external_failure_still_invalidates(self):
  ms=SimpleNamespace(cs471_noop_commit_proof=CycleProof(1,[[0]],[[0]]))
  def bad(*a):raise RuntimeError("x")
  with self.assertRaises(RuntimeError):invalidate_then_call(bad,SimpleNamespace(model_states={"m":ms}))
  self.assertFalse(ms.cs471_noop_commit_proof.rows)
 def test_mixed(self):
  p=CycleProof(4,[[0,1],[2,3]],[[0,1],[3,2]])
  self.assertTrue(p.consume(0,[0,1]));self.assertFalse(p.consume(1,[3,2]))
 def test_once(self):
  p=CycleProof(4,[[0]],[[0]])
  self.assertTrue(p.consume(0,[0]));self.assertFalse(p.consume(0,[0]))
 def test_wrong_target_invalidates_layer(self):
  p=CycleProof(4,[[0]],[[0]])
  self.assertFalse(p.consume(0,[1]));self.assertFalse(p.consume(0,[0]))
 def test_full_invalidation(self):
  p=CycleProof(4,[[0]],[[0]]);p.invalidate();self.assertFalse(p.consume(0,[0]))
 def test_snapshot_immutable(self):
  a=[[0,1]];p=CycleProof(4,a,[[0,1]]);a[0][0]=7
  self.assertTrue(p.consume(0,[0,1]))
 def test_reject_accept_reject(self):
  ms=SimpleNamespace();calls=[];events=[]
  def native(ms,new,layer):calls.append(new.tolist());return "native"
  for old,new in [([0,1],[0,1]),([0,1],[1,0]),([1,0],[1,0])]:
   ms.cs471_noop_commit_proof=CycleProof(1,[old],[new])
   commit_with_proof(native,ms,Tensor(new),0,lambda *a,**k:events.append(k))
  self.assertEqual(calls,[[1,0]]);self.assertEqual(len(events),2)
 def test_no_proof_fallback_returns(self):
  self.assertEqual(commit_with_proof(lambda *a:42,SimpleNamespace(),Tensor([0]),0,lambda *a,**k:None),42)
 def test_gpu_source_fallback(self):
  ms=SimpleNamespace(cs471_noop_commit_proof=CycleProof(1,[[0]],[[0]]))
  self.assertEqual(commit_with_proof(lambda *a:42,ms,Tensor([0],False),0,lambda *a,**k:None),42)
  self.assertFalse(ms.cs471_noop_commit_proof.eligible(0,[0]))
 def test_shadow_before_event_and_no_native(self):
  order=[];ms=SimpleNamespace(cs471_noop_commit_proof=CycleProof(1,[[0]],[[0]]))
  commit_with_proof(lambda *a:order.append("native"),ms,Tensor([0]),0,
                    lambda *a,**k:order.append("event"),lambda *a:order.append("shadow"))
  self.assertEqual(order,["shadow","event"])
 def test_failed_shadow_not_reported_success(self):
  ms=SimpleNamespace(cs471_noop_commit_proof=CycleProof(1,[[0]],[[0]]));events=[]
  def shadow(*args):raise AssertionError("bad metadata")
  with self.assertRaises(AssertionError):commit_with_proof(lambda *a:None,ms,Tensor([0]),0,lambda *a,**k:events.append(k),shadow)
  self.assertFalse(events)
 def test_geometry_rejected(self):
  with self.assertRaises(ValueError):CycleProof(1,[[0]],[[0,1]])
 def test_new_cycle_replaces_proof(self):
  old=CycleProof(1,[[0]],[[0]]);old.consume(0,[0])
  new=CycleProof(2,[[1]],[[1]]);self.assertFalse(new.eligible(0,[0]));self.assertTrue(new.consume(0,[1]))
if __name__=="__main__":unittest.main()
