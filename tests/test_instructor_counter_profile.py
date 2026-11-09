"""The counter range must end before the normal output collective executes."""
from types import SimpleNamespace

import pytest

from arrival_study.counter_profile import capture_experts_only


def fake_layer(fail=False):
    events=[]
    def original(*args,**kwargs):
        events.append('gemms')
        if fail:raise RuntimeError('kernel failure')
        return 7
    cuda=SimpleNamespace(synchronize=lambda:events.append('sync'),
        nvtx=SimpleNamespace(range_push=lambda _:events.append('push'),range_pop=lambda:events.append('pop')),
        profiler=SimpleNamespace(start=lambda:events.append('start'),stop=lambda:events.append('stop')))
    return SimpleNamespace(torch=SimpleNamespace(cuda=cuda),fused_experts=original),events,original


def test_only_experts_are_inside_counter_range():
    layer,events,original=fake_layer()
    with capture_experts_only(layer,'label',lambda *_:events.append('physical_check')) as calls:
        events.append('mapping')
        assert layer.fused_experts()==7
        events.append('allreduce')
    assert events==['mapping','physical_check','sync','push','start','gemms','sync','stop','pop','allreduce']
    assert calls==['label'] and layer.fused_experts is original


def test_failed_expert_call_stops_profiling_and_restores_callable():
    layer,events,original=fake_layer(True)
    with pytest.raises(RuntimeError,match='kernel failure'):
        with capture_experts_only(layer,'label',lambda *_:None):layer.fused_experts()
    assert events[-2:]==['stop','pop']
    assert layer.fused_experts is original
