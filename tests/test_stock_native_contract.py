"""CPU tests for native facade instrumentation, without importing torch/vLLM."""
from types import SimpleNamespace

import pytest

from stock_comparison.native import AsyncPayloadCounter,StockLayer,arm_spec


class Tensor:
    def __init__(self,n,bytes_=2):self.n,self.bytes_=n,bytes_
    def numel(self):return self.n
    def element_size(self):return self.bytes_


class Transport:
    def __init__(self):self.calls=[]
    def add_send(self,tensors,dst,**kwargs):
        self.calls.append((tuple(tensors),dst,kwargs));return 'native-result'


def test_async_cycle_identity_separates_next_sends_before_previous_accounting():
    first,second=object(),object()
    model=SimpleNamespace(eplb_stats=first)
    transport=Transport();counter=AsyncPayloadCounter(transport,model)
    assert counter.add_send([Tensor(10)],1,layer_idx=0)=='native-result'
    model.eplb_stats=second
    counter.add_send([Tensor(7),Tensor(3,4)],0,layer_idx=0)
    # The next worker has already sent before the main thread logs old commit.
    assert counter.sent_bytes==46
    assert counter.cycle_bytes(id(first))==20
    assert counter.cycle_bytes(id(second))==26
    assert sum(v['bytes'] for v in counter.cycles.values())==counter.sent_bytes
    assert counter.cycles[id(first)]['stats'] is first
    assert transport.calls[0][1:]==(1,{'layer_idx':0})
    counter.reset_counts()
    assert counter.sent_bytes==counter.sent_messages==0 and counter.cycles=={}


def test_async_send_without_cycle_fails_before_transport():
    transport=Transport();counter=AsyncPayloadCounter(transport,SimpleNamespace(eplb_stats=None))
    with pytest.raises(AssertionError,match='cycle identity'):
        counter.add_send([Tensor(1)],1)
    assert transport.calls==[]


class Backend:
    disabled=False
    def all_reduce(self,value):return value


@pytest.mark.parametrize('fail',[False,True])
def test_reduction_probe_delegates_and_restores_instance_lookup(fail):
    backend=Backend()
    communicator=SimpleNamespace(pynccl_comm=backend)
    rt=SimpleNamespace(tp_group=SimpleNamespace(device_communicator=communicator),
                       metadata={},sync=lambda:None)
    layer=object.__new__(StockLayer)
    layer.rt=rt;layer._active=False;layer.ms=None
    class Hidden:
        dtype='bfloat16'
        def __len__(self):return 128
    hidden=Hidden()
    def forward(value,*,observe):
        assert not observe and value is hidden
        assert backend.all_reduce(value) is hidden
        if fail:raise RuntimeError('forward failed')
    layer.forward=forward
    assert 'all_reduce' not in vars(backend)
    if fail:
        with pytest.raises(RuntimeError,match='forward failed'):
            layer.attest_output_reduction(hidden)
        assert 'output_reduction_probe' not in rt.metadata
    else:
        record=layer.attest_output_reduction(hidden)
        assert record['observed_successful_backends']==['pynccl_comm']
        assert not record['backend_forced'] and not record['unresolved_native_branch']
    assert 'all_reduce' not in vars(backend)
    assert backend.all_reduce(hidden) is hidden


def test_effective_arm_contracts_distinguish_default_config_and_disabled_state():
    assert arm_spec('stockoff')['physical_slots']==128
    assert not arm_spec('stockoff')['enable_eplb']
    native=arm_spec('stockon32');ours=arm_spec('ours')
    assert (native['initial_counter'],native['window_steps'],native['rebalance_interval'])==(2250,1000,3000)
    assert native['is_async']
    assert (ours['initial_counter'],ours['window_steps'],ours['rebalance_interval'])==(0,16,64)
    assert not ours['is_async']
    with pytest.raises(ValueError):arm_spec('imitation-stock')
