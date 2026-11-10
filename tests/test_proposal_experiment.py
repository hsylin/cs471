import json
from full_serving.proposal_experiment import ARMS,server_arguments,clean


def test_native_arms_and_replica_budget():
    for arm in ARMS:
        a=server_arguments(arm,'snapshot','127.0.0.1','primary')
        assert a[:2]==['serve','snapshot']
        assert a[a.index('--data-parallel-size')+1]=='8'
        assert a[a.index('--all2all-backend')+1]=='allgather_reducescatter'
        if arm=='off':assert '--enable-eplb' not in a and '--eplb-config' not in a
        else:assert json.loads(a[a.index('--eplb-config')+1])=={'num_redundant_experts':0 if arm=='b1' else 32}


def test_native_reference_does_not_inherit_plugin():
    e=clean({'CS471_GATE_MODE':'b4','FULL_SERVING_CUSTOM_MODE':'resident_v1','VLLM_PLUGINS':'custom','PYTHONPATH':'custom','FULL_SERVING_RUN_DIR':'retained'})
    assert e=={'FULL_SERVING_RUN_DIR':'retained'}
