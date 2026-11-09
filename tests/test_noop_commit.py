"""CPU semantic contracts for the no-op pair, without importing GPU packages."""
from types import SimpleNamespace

import numpy as np
import pytest

from arrival_fix.noop import PINNED_COMMIT, unchanged_map_fast_path


class Tensor:
    def __init__(self, values, device='cpu'):
        self.array=np.asarray(values,dtype=np.int64).copy()
        self.device=SimpleNamespace(type=device)
        self.dtype='int64';self._version=0
    @property
    def shape(self):return self.array.shape
    def stride(self):return tuple(s//self.array.itemsize for s in self.array.strides)
    def data_ptr(self):return self.array.ctypes.data
    def storage_offset(self):return 0
    def cpu(self):return Tensor(self.array)
    def item(self):return self.array.item()
    def fill_(self,value):self.array.fill(value);self._version+=1;return self
    def copy_(self,other):self.array[...]=other.array;self._version+=1;return self


def fixture(peer_equal=1):
    events=[]
    gpu_group=object();cpu_group=object()
    def reduce(flag,op,group):
        assert op=='MIN' and group is cpu_group
        events.append('agreement')
        flag.fill_(min(int(flag.item()),peer_equal))
    torch=SimpleNamespace(int64='int64',zeros=lambda n,**_:Tensor(np.zeros(n,dtype=np.int64)),
        equal=lambda a,b:np.array_equal(a.array,b.array),
        distributed=SimpleNamespace(all_reduce=reduce,ReduceOp=SimpleNamespace(MIN='MIN')))
    physical=Tensor([[0,1,2,3,0,1]],device='cuda')
    logical=Tensor([[[0,4],[1,5],[2,-1],[3,-1]]],device='cuda')
    replicas=Tensor([[2,2,1,1]],device='cuda')
    weights=[Tensor([[10],[20],[30],[40],[10],[20]])]
    model=SimpleNamespace(expert_weights=weights)
    owner=SimpleNamespace(physical_to_logical_map=physical,logical_to_physical_map=logical,
        logical_replica_count=replicas,model=model,expert_buffer=[],communicator=object(),
        pending_result=None,rebalanced=False)
    state=SimpleNamespace(is_async=False,model_states={'owner':owner})
    layer=SimpleNamespace(torch=torch,ms=owner,state=state,
        rt=SimpleNamespace(world_size=2,metadata={'native_vllm_commit':PINNED_COMMIT},
            ep_group=SimpleNamespace(device_group=gpu_group,cpu_group=cpu_group)))
    def move(old_global_expert_indices,new_global_expert_indices,expert_weights,
             expert_buffer,ep_group,communicator,is_profile=False,rank_mapping=None):
        events.append('original_move')
        old=old_global_expert_indices.array[0];new=new_global_expert_indices.array[0]
        source=expert_weights[0].array.copy()
        expert_weights[0].array[:]=np.array([source[np.flatnonzero(old==e)[0]] for e in new])
    def commit(model_state,new_physical_to_logical_map):
        events.append('original_commit')
        ids=new_physical_to_logical_map.array[0]
        inverse=np.full_like(model_state.logical_to_physical_map.array,-1)
        copies=np.zeros_like(model_state.logical_replica_count.array)
        for e in range(4):
            places=np.flatnonzero(ids==e);inverse[0,e,:len(places)]=places;copies[0,e]=len(places)
        model_state.physical_to_logical_map.copy_(new_physical_to_logical_map)
        model_state.logical_to_physical_map.copy_(Tensor(inverse))
        model_state.logical_replica_count.copy_(Tensor(copies))
    module=SimpleNamespace(rearrange_expert_weights_inplace=move,_commit_eplb_maps=commit)
    return layer,module,events


def movement(layer,module,new,**kwargs):
    m=layer.ms
    module.rearrange_expert_weights_inplace(m.physical_to_logical_map,new,m.model.expert_weights,
        m.expert_buffer,layer.rt.ep_group.device_group,m.communicator,**kwargs)


def pair(layer,module,new):
    movement(layer,module,new);module._commit_eplb_maps(layer.ms,new)


def snapshots(layer):
    m=layer.ms
    return [(t.array.copy(),t._version) for t in [m.physical_to_logical_map,
        m.logical_to_physical_map,m.logical_replica_count,*m.model.expert_weights]]


def assert_snapshot_equal(a,b):
    for (x,xv),(y,yv) in zip(a,b):
        assert np.array_equal(x,y) and xv==yv


def test_unanimous_noop_preserves_every_map_count_weight_and_version():
    layer,module,events=fixture();before=snapshots(layer)
    originals=(module.rearrange_expert_weights_inplace,module._commit_eplb_maps)
    with unchanged_map_fast_path(layer,module=module) as guard:
        pair(layer,module,layer.ms.physical_to_logical_map.cpu())
        assert guard.snapshot()['pending_pair'] is False
        assert guard.counters['movement_skips']==guard.counters['commit_skips']==1
        assert guard.diagnostics==[]
    assert events==['agreement']
    assert_snapshot_equal(before,snapshots(layer))
    assert (module.rearrange_expert_weights_inplace,module._commit_eplb_maps)==originals


def test_changed_map_uses_normal_transfer_and_all_map_commits():
    layer,module,events=fixture();new=Tensor([[2,1,0,3,0,1]])
    with unchanged_map_fast_path(layer,module=module) as guard:
        pair(layer,module,new)
        assert guard.counters['fallback_calls']==1 and guard.counters['movement_skips']==0
    assert events==['agreement','original_move','original_commit']
    assert np.array_equal(layer.ms.physical_to_logical_map.array,new.array)
    assert layer.ms.model.expert_weights[0].array[:,0].tolist()==[30,20,10,40,10,20]
    assert layer.ms.logical_to_physical_map.array[0,0].tolist()==[2,4]
    assert layer.ms.logical_replica_count.array.tolist()==[[2,2,1,1]]


@pytest.mark.parametrize('local_equal,peer_equal',[(True,0),(False,1)])
def test_mixed_rank_flags_force_original_path_on_both_rank_views(local_equal,peer_equal):
    layer,module,events=fixture(peer_equal)
    new=layer.ms.physical_to_logical_map.cpu() if local_equal else Tensor([[2,1,0,3,0,1]])
    with unchanged_map_fast_path(layer,module=module) as guard:
        pair(layer,module,new)
        assert guard.counters['unanimously_unchanged_calls']==0
        assert guard.counters['original_movement_calls']==guard.counters['original_commit_calls']==1
    assert events==['agreement','original_move','original_commit']


def test_observe_mode_calls_originals_and_records_host_intervals():
    layer,module,events=fixture();before=snapshots(layer)
    with unchanged_map_fast_path(layer,mode='observe',diagnostic_timing=True,module=module) as guard:
        pair(layer,module,layer.ms.physical_to_logical_map.cpu())
        assert guard.counters['movement_skips']==guard.counters['commit_skips']==0
        assert [d['stage'] for d in guard.diagnostics]==['movement','commit']
        assert all(d['original_function_host_wall_seconds']>=0 for d in guard.diagnostics)
    for (a,_),(b,_) in zip(before,snapshots(layer)):assert np.array_equal(a,b)
    assert events==['agreement','original_move','original_commit']


@pytest.mark.parametrize('mutation',['candidate','inverse','different_candidate'])
def test_pending_pair_rejects_mutated_or_different_map_and_restores_context(mutation):
    layer,module,_=fixture();originals=(module.rearrange_expert_weights_inplace,module._commit_eplb_maps)
    with pytest.raises(RuntimeError,match='changed'):
        with unchanged_map_fast_path(layer,module=module):
            new=layer.ms.physical_to_logical_map.cpu();movement(layer,module,new)
            if mutation=='candidate':new.fill_(0)
            elif mutation=='inverse':layer.ms.logical_to_physical_map.fill_(0)
            else:new=new.cpu()
            module._commit_eplb_maps(layer.ms,new)
    assert (module.rearrange_expert_weights_inplace,module._commit_eplb_maps)==originals


def test_missing_commit_and_nested_context_fail_closed():
    layer,module,_=fixture()
    with pytest.raises(RuntimeError,match='matching native map commit'):
        with unchanged_map_fast_path(layer,module=module):movement(layer,module,layer.ms.physical_to_logical_map.cpu())
    with unchanged_map_fast_path(layer,module=module):
        with pytest.raises(RuntimeError,match='Nested'):
            with unchanged_map_fast_path(layer,module=module):pass


@pytest.mark.parametrize('unsupported',[{'is_profile':True},{'rank_mapping':{0:0,1:1}}])
def test_special_paths_are_rejected(unsupported):
    layer,module,events=fixture()
    with pytest.raises(ValueError,match='unsupported'):
        with unchanged_map_fast_path(layer,module=module):
            movement(layer,module,layer.ms.physical_to_logical_map.cpu(),**unsupported)
    assert events==[]


def test_reset_requires_complete_pair_and_clears_warmup_counts():
    layer,module,_=fixture()
    with unchanged_map_fast_path(layer,module=module) as guard:
        new=layer.ms.physical_to_logical_map.cpu();movement(layer,module,new)
        with pytest.raises(RuntimeError,match='incomplete'):guard.reset_counters()
        module._commit_eplb_maps(layer.ms,new);guard.reset_counters()
        assert all(v==0 for v in guard.counters.values())


def test_async_and_wrong_source_are_rejected_before_installation():
    layer,module,_=fixture();layer.state.is_async=True
    with pytest.raises(ValueError,match='synchronous'):
        with unchanged_map_fast_path(layer,module=module):pass
    layer.state.is_async=False;layer.rt.metadata['native_vllm_commit']='other'
    with pytest.raises(ValueError,match='Unpinned'):
        with unchanged_map_fast_path(layer,module=module):pass
