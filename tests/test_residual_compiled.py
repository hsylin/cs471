from types import SimpleNamespace

import pytest

from resident_residual.compiled import PreboundCompiledMapper, _storage


class Device:
    index=0
    def __str__(self):return 'cuda:0'


class Tensor:
    def __init__(self,device,dtype='i32',shape=(128,8),pointer=4096,contiguous=True):
        self.device,self.dtype,self.shape=device,dtype,shape
        self.pointer,self.contiguous=pointer,contiguous
    def data_ptr(self):return self.pointer
    def is_contiguous(self):return self.contiguous
    def numel(self):
        n=1
        for x in self.shape:n*=x
        return n


def mapper():
    m=PreboundCompiledMapper.__new__(PreboundCompiledMapper)
    m.device=Device();m.stream_handle=17;m.calls=0;m.last_physical_ids=None
    m.torch=SimpleNamespace(int32='i32',int64='i64',cuda=SimpleNamespace(
        current_device=lambda:0,current_stream=lambda d:SimpleNamespace(cuda_stream=17)))
    m.real_tokens=Tensor(m.device,shape=(),pointer=8192)
    m._output_i32=Tensor(m.device,pointer=12288)
    m._output_i64=Tensor(m.device,dtype='i64',pointer=16384)
    m.homes=Tensor(m.device,shape=(128,),pointer=20480)
    m.logical_counts=Tensor(m.device,shape=(128,),pointer=24576)
    m.summary=Tensor(m.device,shape=(8,),pointer=28672)
    m._fixed_tensors=(m.real_tokens,m._output_i32,m._output_i64,m.homes,m.logical_counts,m.summary)
    m._fixed_storage=tuple(_storage(t) for t in m._fixed_tensors)
    m.kernel=SimpleNamespace(pre_run_hooks=[],used_global_vals={})
    m._knobs=SimpleNamespace(runtime=SimpleNamespace(add_stages_inspection_hook=None,debug=False),
        compilation=SimpleNamespace(instrumentation_mode=''))
    m.launched=[]
    def run(*args,**kwargs):m.launched.append((args,kwargs))
    m._runners={'i32':run,'i64':run}
    m._kernel_evidence={}
    return m


@pytest.mark.parametrize('dtype',['i32','i64'])
def test_prebound_call_preserves_all_arguments_and_fixed_stream(dtype):
    m=mapper();ids=Tensor(m.device,dtype=dtype)
    result=m(ids)
    args,kwargs=m.launched[0]
    assert args==(ids,m.real_tokens,result,m.homes,m.logical_counts,m.summary,1024,8)
    assert kwargs=={'stream':17}
    assert result is (m._output_i32 if dtype=='i32' else m._output_i64)
    assert m.calls==1 and m.last_physical_ids is result
    # Input storage may change across real jobs; alignment and dtype contract hold.
    m(Tensor(m.device,dtype=dtype,pointer=32768))
    assert m.calls==2


@pytest.mark.parametrize('change',['shape','dtype','contiguous','device','unaligned'])
def test_input_checks_fail_before_launch(change):
    m=mapper();ids=Tensor(m.device)
    if change=='shape':ids.shape=(64,8)
    if change=='dtype':ids.dtype='f32'
    if change=='contiguous':ids.contiguous=False
    if change=='device':ids.device=Device()
    if change=='unaligned':ids.pointer+=4
    with pytest.raises(ValueError):m(ids)
    assert not m.launched and m.calls==0


@pytest.mark.parametrize('change',['size','dtype','device','unaligned'])
def test_external_real_scalar_contract(change):
    m=mapper();real=Tensor(m.device,shape=(),pointer=32768)
    if change=='size':real.shape=(2,)
    if change=='dtype':real.dtype='i64'
    if change=='device':real.device=Device()
    if change=='unaligned':real.pointer+=4
    with pytest.raises(ValueError):m(Tensor(m.device),real)
    assert not m.launched


def test_valid_external_scalar_and_no_input_value_readback():
    m=mapper();real=Tensor(m.device,shape=(),pointer=32768)
    m(Tensor(m.device),real)
    assert m.launched[0][0][1] is real


@pytest.mark.parametrize('change',['storage','stream','device','prehook','globals','stages','debug','instrumentation'])
def test_context_detects_changes_before_or_after_trial(change):
    m=mapper()
    assert m.verify_context()['fixed_argument_storage_verified']
    if change=='storage':m.homes.pointer+=16
    if change=='stream':m.torch.cuda.current_stream=lambda d:SimpleNamespace(cuda_stream=18)
    if change=='device':m.torch.cuda.current_device=lambda:1
    if change=='prehook':m.kernel.pre_run_hooks=[lambda:None]
    if change=='globals':m.kernel.used_global_vals={'x':1}
    if change=='stages':m._knobs.runtime.add_stages_inspection_hook=lambda:None
    if change=='debug':m._knobs.runtime.debug=True
    if change=='instrumentation':m._knobs.compilation.instrumentation_mode='sanitizer'
    with pytest.raises((ValueError,RuntimeError)):m.verify_context()


def test_dynamic_launch_hooks_not_blocked_or_silently_bypassed():
    m=mapper()
    m._knobs.runtime.launch_enter_hook=lambda:None
    m._knobs.runtime.launch_exit_hook=lambda:None
    m.verify_context()  # CompiledKernel's own runner consults these dynamically.
    report=m.compiled_metadata()
    assert report['dynamic_launch_metadata_hooks_preserved']
    assert report['gpu_algorithm_changed'] is False
