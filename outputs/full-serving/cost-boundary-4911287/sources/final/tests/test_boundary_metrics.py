import unittest
from full_serving.boundary_metrics import parse
class MetricTests(unittest.TestCase):
 def test_only_allowlisted_numeric_fields_leave_parser(self):
  text='vllm:kv_cache_usage_perc{engine="3",model_name="private-model"} 0.5\nvllm:request_output_tokens_sum{engine="3"} 128\nsecret_prompt{engine="3"} text'
  self.assertEqual(parse(text),[{"metric":"kv_cache_usage_perc","engine":3,"value":.5}])
 def test_unknown_engine_not_assumed_a_rank(self):
  self.assertFalse(parse('vllm:num_requests_running{engine="8"} 1'))
 def test_invalid_gauge_fails_closed(self):
  with self.assertRaises(AssertionError):parse('vllm:kv_cache_usage_perc{engine="0"} 3')
if __name__=="__main__":unittest.main()
