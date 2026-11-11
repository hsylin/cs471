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


def test_rejected_light_gate_keeps_original_commit_protocol():
    # Engineering patch must not override native async completion or change the
    # decision function. GPU shadow tests validate count equality separately.
    from pathlib import Path
    source=Path('full_serving/monitoring_gate_plugin.py').read_text()
    assert 'from full_serving.proposal_gate import decide' in source
    assert 'original_move(ms,ep_rank)' in source
    assert 'event.query()' in source
    assert 'all_gather_object' in source
    assert 'drain' not in source[source.index('def register'):]
