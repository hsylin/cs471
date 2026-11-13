import unittest
from types import SimpleNamespace as NS
import numpy as np
from full_serving.boundary_shapes import summarize_batch,PendingSpans
class ShapesTests(unittest.TestCase):
 def batch(self):
  return NS(num_reqs=2,num_scheduled_tokens=np.array([3,1]),num_computed_tokens_np=np.array([7,19]),
   is_prefilling_np=np.array([True,False]),num_tokens=4,num_tokens_after_padding=8,num_reqs_after_padding=4,num_draft_tokens=0)
 def test_native_batch_phase_and_upper_bound(self):
  d=summarize_batch(self.batch())
  self.assertEqual((d["prefill_tokens"],d["decode_tokens"]),(3,1))
  self.assertEqual(d["attention_pairs_cpu_upper_proxy"],50)
  self.assertEqual(d["sequence_cpu_upper_sum"],30)
  self.assertFalse(d["exact_kv_occupancy"])
 def test_empty_cpu_batch(self):
  b=self.batch();b.num_reqs=b.num_tokens=0
  b.num_scheduled_tokens=b.num_computed_tokens_np=np.array([],dtype=np.int64);b.is_prefilling_np=np.array([],dtype=bool)
  self.assertEqual(summarize_batch(b)["attention_pairs_cpu_upper_proxy"],0)
 def test_never_reads_token_fields(self):
  b=self.batch()
  class Secret:
   def __array__(self,*a,**k):raise AssertionError("private tensor accessed")
  b.input_ids=b.req_ids=b.positions=Secret()
  self.assertEqual(summarize_batch(b)["real_tokens"],4)
 def test_non_numpy_metadata_rejected(self):
  b=self.batch();b.num_computed_tokens_np=[7,19]
  with self.assertRaises(AssertionError):summarize_batch(b)
 def test_speculation_not_silently_relabelled(self):
  b=self.batch();b.num_draft_tokens=1
  with self.assertRaises(AssertionError):summarize_batch(b)
 def test_pending_only_observed_after_query_ready(self):
  ready=[False];seen=[];p=PendingSpans()
  start=NS(elapsed_time=lambda end:2.5);end=NS(query=lambda:ready[0])
  p.add(start,end,{"execute_call":64});p.poll(lambda *a,**k:seen.append(k))
  self.assertEqual(p.censored(),[{"execute_call":64}]);self.assertFalse(seen)
  ready[0]=True;p.poll(lambda *a,**k:seen.append(k))
  self.assertEqual(seen,[{"gpu_stream_ms":2.5,"execute_call":64}]);self.assertFalse(p.censored())
if __name__=="__main__":unittest.main()
