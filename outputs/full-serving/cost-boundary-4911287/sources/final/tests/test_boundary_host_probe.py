import unittest
from concurrent.futures import ThreadPoolExecutor
from full_serving.boundary_host_probe import HostLedger
class Tests(unittest.TestCase):
 def test_return_identity(self):
  ticks=iter([10,30]); l=HostLedger(lambda:next(ticks));value=object()
  f=l.wrap("x",lambda a,scale=1:a);self.assertIs(f(value,scale=2),value)
  self.assertEqual(l.take()["x"],dict(calls=1,total_ns=20,max_ns=20,failed_calls=0))
  self.assertEqual(l.take(),{})
 def test_exception_identity(self):
  ticks=iter([0,11]);l=HostLedger(lambda:next(ticks));err=ValueError("sentinel")
  def f():raise err
  try:l.wrap("bad",f)()
  except ValueError as e:self.assertIs(e,err)
  else:self.fail("exception swallowed")
  self.assertEqual(l.take()["bad"]["failed_calls"],1)
 def test_once_and_metadata(self):
  calls=[];l=HostLedger()
  def example(a,*,b):calls.append((a,b));return a+b
  wrapped=l.wrap("x",example);self.assertEqual(wrapped(3,b=4),7)
  self.assertEqual(calls,[(3,4)]);self.assertEqual(wrapped.__name__,"example")
 def test_threads(self):
  l=HostLedger();f=l.wrap("shared",lambda x:x)
  with ThreadPoolExecutor(max_workers=4) as pool:self.assertEqual(list(pool.map(f,range(200))),list(range(200)))
  v=l.take()["shared"];self.assertEqual(v["calls"],200);self.assertEqual(v["failed_calls"],0)
 def test_nested_not_additive(self):
  ticks=iter([0,2,7,10]);l=HostLedger(lambda:next(ticks));inner=l.wrap("inner",lambda:17)
  self.assertEqual(l.wrap("outer",inner)(),17);v=l.take()
  self.assertEqual(v["outer"]["total_ns"],10);self.assertEqual(v["inner"]["total_ns"],5)
if __name__=="__main__":unittest.main()
