import unittest
import numpy as np
from full_serving.resident_policy import balanced_map,cpu_reference
class StaticHomeTests(unittest.TestCase):
 def test_every_expert_maps_to_canonical_original_copy(self):
  ids=np.arange(128,dtype=np.int32).reshape(16,8)
  out=cpu_reference(ids,mode="static_home")["physical_ids"]
  expected=np.array([[20*(int(e)//16)+int(e)%16 for e in row] for row in ids])
  np.testing.assert_array_equal(out,expected)
  np.testing.assert_array_equal(np.asarray(balanced_map())[out],ids)
 def test_padding_and_empty_ranks(self):
  for rows in (0,1,17,128,1024):
   ids=np.full((rows,8),-1,dtype=np.int64)
   out=cpu_reference(ids,mode="static_home")["physical_ids"]
   np.testing.assert_array_equal(out,ids)
 def test_duplicate_experts_keep_one_home(self):
  ids=np.random.default_rng(471).integers(-1,128,(513,8))
  report=cpu_reference(ids,mode="static_home");out=report["physical_ids"]
  for expert in range(128):
   slots=set(out[ids==expert].tolist())
   self.assertLessEqual(len(slots),1)
  np.testing.assert_array_equal(np.asarray(balanced_map())[out[ids>=0]],ids[ids>=0])
 def test_invalid_ids_rejected(self):
  for expert in (-2,128):
   with self.assertRaises(ValueError):cpu_reference(np.full((1,8),expert),mode="static_home")
 def test_resident_and_hash_still_conserve_all_tokens(self):
  ids=np.random.default_rng(471).integers(-1,128,(17,8))
  for mode in ("resident_v1","native_hash","static_home"):
   out=cpu_reference(ids,mode=mode)["physical_ids"]
   np.testing.assert_array_equal(np.asarray(balanced_map())[out[ids>=0]],ids[ids>=0])
   self.assertEqual(np.count_nonzero(out>=0),np.count_nonzero(ids>=0))
if __name__=="__main__":unittest.main()
