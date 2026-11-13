"""Exercise pinned native retirement/drain code with deterministic delayed events.

No model or CUDA mocking is claimed as GPU equivalence. The AST compiled here is
the actual pinned native implementation; only its external move/map/event calls
are replaced to observe ordering, partial maps and failure propagation.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from full_serving.boundary_noop_commit import CycleProof,commit_with_proof
from test_boundary_noop_commit import Tensor

NATIVE=Path("/work/cs-471/skewbench-hlin/venv/lib/python3.12/site-packages/vllm/distributed/eplb/eplb_state.py")
class Event:
 def __init__(self,fn):self.fn=fn;self.count=0
 def record(self):self.count+=1;self.fn()
class Row(Tensor):
 def numpy(self):return self.tolist()

class NativeRetirementTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import json
  root=Path(__file__).resolve().parents[1]
  protocol=json.loads((root/"results/full-serving/cost-boundary-4911287/host-pair-v1/protocol.json").read_text())
  pins=json.loads((root/protocol["native_pins"]).read_text())
  assert hashlib.sha256(NATIVE.read_bytes()).hexdigest()==pins["distributed/eplb/eplb_state.py"]
  tree=ast.parse(NATIVE.read_text())
  cls.nodes={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ("_move_to_workspace","drain_async")}
 def native(self,name,env):
  # Future annotations avoids importing the native module or initializing CUDA.
  tree=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),self.nodes[name]],type_ignores=[])
  exec(compile(ast.fix_missing_locations(tree),str(NATIVE),"exec"),env)
  return env[name]
 def fixture(self,layer=0,changed=False,move_error=False):
  order=[];target=[1,0] if changed else [0,1]
  ms=NS(model=NS(expert_weights=[["w0"],["w1"]],num_moe_layers=2),expert_buffer=["buffer"],
        rebalanced=True,cs471_noop_commit_proof=CycleProof(4,[[0,1],[0,1]],[[0,1],target] if layer==1 else [target,[0,1]]))
  event=Event(lambda:(self.assertIsNone(ms.pending_result),order.append("consumed")))
  ms.pending_result=NS(layer_idx=layer,new_physical_to_logical_map=Row(target),transfer_metadata=NS(recv_count=0),consumed_event=event)
  def move(**kw):
   self.assertIsNotNone(ms.pending_result);order.append("move")
   self.assertEqual(kw["new_indices"],target)
   if move_error:raise RuntimeError("move failed")
  def commit(model_state,new_physical_to_logical_map,layer):
   return commit_with_proof(lambda *a:order.append("native_map"),model_state,new_physical_to_logical_map,layer,lambda *a,**k:order.append("elided"))
  native=self.native("_move_to_workspace",{"move_from_buffer":move,"_commit_eplb_maps_for_layer":commit})
  return native,ms,event,order
 def test_unchanged_retirement_order_and_exactly_once(self):
  native,ms,event,order=self.fixture();native(ms,0)
  self.assertEqual(order,["move","elided","consumed"]);self.assertTrue(ms.rebalanced);self.assertEqual(event.count,1)
 def test_next_prefetch_can_fill_slot_before_late_shadow_sample(self):
  native,ms,event,order=self.fixture()
  old_result=ms.pending_result
  replacement=NS(layer_idx=1)
  def worker_progress():
   # Pinned worker checks this before advancing to its next layer.
   self.assertIsNone(ms.pending_result)
   ms.pending_result=replacement
  event.fn=worker_progress
  native(ms,0)
  self.assertIs(ms.pending_result,replacement)
  self.assertIsNot(ms.pending_result,old_result)
  self.assertEqual(event.count,1)
 def test_changed_layer_keeps_native_commit(self):
  native,ms,event,order=self.fixture(changed=True);native(ms,0)
  self.assertEqual(order,["move","native_map","consumed"])
 def test_last_layer_clears_rebalanced_before_unblock(self):
  native,ms,event,order=self.fixture(layer=1)
  event.fn=lambda:(self.assertIsNone(ms.pending_result),self.assertFalse(ms.rebalanced),order.append("consumed"))
  native(ms,7);self.assertEqual(order,["move","elided","consumed"])
 def test_failure_does_not_unblock_or_retire(self):
  native,ms,event,order=self.fixture(move_error=True)
  with self.assertRaises(RuntimeError):native(ms,3)
  self.assertIsNotNone(ms.pending_result);self.assertTrue(ms.rebalanced);self.assertEqual(event.count,0)
 def test_pending_delayed_until_native_called(self):
  native,ms,event,order=self.fixture()
  result=ms.pending_result
  for _ in range(3):self.assertIs(ms.pending_result,result);self.assertEqual(event.count,0)
  native(ms,4);self.assertIsNone(ms.pending_result)
 def test_drain_acknowledges_without_elision_or_map_commit(self):
  order=[];ms=NS(rebalanced=True,model=NS(num_moe_layers=2))
  remaining=iter([0,1])
  def next_result():
   try:layer=next(remaining)
   except StopIteration:return
   ms.pending_result=NS(layer_idx=layer,consumed_event=Event(lambda:(self.assertIsNone(ms.pending_result),order.append("ack"),next_result())))
  next_result();delays=[False,False,True,True]
  obj=NS(is_async=True,model_states={"m":ms},_all_ranks_result_ready=lambda _:delays.pop(0))
  fn=self.native("drain_async",{"logger":NS(info=lambda *a:None),"time":NS(sleep=lambda _:order.append("wait"))})
  fn(obj)
  self.assertEqual(order,["wait","wait","ack","ack"]);self.assertFalse(ms.rebalanced);self.assertIsNone(ms.pending_result)
 def test_empty_pending_is_not_commit(self):
  native,ms,event,order=self.fixture();ms.pending_result=None
  with self.assertRaises(AssertionError):native(ms,0)
  self.assertFalse(order);self.assertEqual(event.count,0)
if __name__=="__main__":unittest.main()
