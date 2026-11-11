"""Pilot-gated packed activation dispatch, preserving native combine and policies."""
import hashlib
import json
import os
from pathlib import Path

_INSTALLED=False


def register():
    global _INSTALLED
    if _INSTALLED:return
    arm=os.environ['CS471_EXPERIMENT_ARM'];base=arm.removesuffix('_packed')
    assert arm.endswith('_packed') and base in ('off','b3_light','b4_light','resident_v2')
    # Invoke the unchanged policy implementation under its original arm name.
    os.environ['CS471_EXPERIMENT_ARM']=base
    try:
        if base=='resident_v2':
            from full_serving.proposal_plugin import register as policy
            policy()
        elif base.endswith('_light'):
            from full_serving.monitoring_plugin import register as policy
            policy()
    finally:os.environ['CS471_EXPERIMENT_ARM']=arm
    import torch
    import vllm
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from full_serving.packed_dispatch import packed_gather
    pins=json.loads(Path(os.environ['CS471_PACKED_NATIVE_PINS']).read_text())
    for name,digest in pins.items():
        assert hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()==digest,name
    original=AgRsAll2AllManager.dispatch
    pilot=os.environ.get('CS471_PACKED_SHADOW')=='1';seen=set();bound=False
    def dispatch(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel=False,extra_tensors=None):
        nonlocal bound
        capturing=torch.cuda.is_current_stream_capturing()
        if bound and not pilot and not capturing:
            return original(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel,extra_tensors)
        group=self._get_comm_group(is_sequence_parallel);sizes=self._get_sizes(hidden_states.shape[0],group)
        assert len(sizes)==8 and sizes[group.rank_in_group]==hidden_states.shape[0]
        # Fixed candidate operating limit, selected before final measurements:
        # only the observed smallest padded decode shape (one row per rank).
        eligible=(sizes==[1]*8 and extra_tensors is None and not is_sequence_parallel)
        if not capturing and not bound:
            output=Path(os.environ['FULL_SERVING_RUN_DIR'])/f'packed-bindings-rank{group.rank_in_group}.json'
            output.write_text(json.dumps({'status':'bound','arm':arm,'rank':group.rank_in_group,'base_arm':base,'rule':'CUDA graph capture and sizes == [1]*8; no extras; no sequence parallelism; native eager fallback','combine':'unchanged native reduce_scatterv','pilot_shadow':pilot,'native_source_sha256':pins,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');bound=True
        if not eligible:return original(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel,extra_tensors)
        if not capturing and (not pilot or tuple(sizes) in seen):
            return original(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel,extra_tensors)
        result=packed_gather(group,[hidden_states,topk_weights,topk_ids],sizes)
        if pilot and not capturing and tuple(sizes) not in seen:
            reference=original(self,hidden_states,topk_weights,topk_ids,is_sequence_parallel,extra_tensors)
            assert all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(result,reference))
            seen.add(tuple(sizes))
            output=Path(os.environ['FULL_SERVING_RUN_DIR'])/f'packed-shadow-rank{group.rank_in_group}.json'
            output.write_text(json.dumps({'status':'passed','rank':group.rank_in_group,'sizes':sizes,'bitwise_equal_hidden_weights_ids':True,'scope':'Representative eager dispatch; arbitrary-bit and graph checks are separate collective replay evidence.'},indent=2)+'\n')
            return reference
        return tuple(result)
    AgRsAll2AllManager.dispatch=dispatch
    _INSTALLED=True
