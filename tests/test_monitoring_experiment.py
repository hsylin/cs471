import json
from full_serving.monitoring_experiment import ARMS, server_arguments, clean
from full_serving.proposal_experiment import server_arguments as original


def test_original_cli_unchanged_and_light_retains_budget():
    assert len(ARMS)==9
    for arm in ARMS:
        old=arm.removesuffix('_light')
        assert server_arguments(arm,'snapshot','127.0.0.1','primary')==original(old,'snapshot','127.0.0.1','primary')


def test_clean_rejects_monitoring_environment_leakage():
    assert clean({'CS471_GATE_HISTORY':'rolling','VLLM_PLUGINS':'cs471_monitoring','PYTHONPATH':'x'})=={}
