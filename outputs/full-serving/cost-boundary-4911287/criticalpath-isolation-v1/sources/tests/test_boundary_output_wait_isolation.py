import unittest,contextlib
from types import SimpleNamespace
from full_serving.boundary_output_wait_isolation import install
class OutputWait(unittest.TestCase):
    def test_record_before_wait_and_consumer_unchanged(self):
        calls=[]
        class Output:
            def __init__(self,value):
                calls.append(('record',value));self.copy_event=SimpleNamespace(synchronize=lambda:calls.append('wait'));self.value=value
            def get_output(self):return self.value
        consumer=Output.get_output
        cap=SimpleNamespace(span=lambda name:contextlib.nullcontext())
        install(Output,lambda:cap);out=Output(7)
        self.assertEqual(calls,[('record',7),'wait']);self.assertIs(Output.get_output,consumer);self.assertEqual(out.get_output(),7)
    def test_no_wait_after_original_init_failure(self):
        class Output:
            def __init__(self):raise ValueError('original')
        install(Output,lambda:self.fail('must not capture after failed init'))
        with self.assertRaisesRegex(ValueError,'original'):Output()
    def test_wait_failure_is_preserved(self):
        def wait():raise RuntimeError('CUDA original failure')
        class Output:
            def __init__(self):self.copy_event=SimpleNamespace(synchronize=wait)
        install(Output,lambda:SimpleNamespace(span=lambda _:contextlib.nullcontext()))
        with self.assertRaisesRegex(RuntimeError,'CUDA original failure'):Output()
