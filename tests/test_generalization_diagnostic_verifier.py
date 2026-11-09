"""Independent diagnostic reconstruction, especially phase/window boundaries."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
v=load('independent_diagnostics',ROOT/'scripts/verify_generalization_diagnostics.py')
fixtures=load('independent_main_fixtures',ROOT/'tests/test_generalization_records_verifier.py')
captured=fixtures.captured
route_fixture=fixtures.route_fixture


def test_frozen_state_and_order_reconstruction():
    pilot=json.loads((ROOT/'configs/generalization-pilot-v2.json').read_text())
    plan=json.loads((ROOT/'configs/generalization-diagnostics-v1.json').read_text())
    assert v.states(pilot)==plan['states']
    assert {k:v.sequence(k) for k in ('main','cycle0','cycle16')}==plan['sequence_definitions']


def test_cycle_phase_changes_window_but_not_frequency():
    a,b=v.sequence('cycle0'),v.sequence('cycle16')
    assert Counter(a)==Counter(b)==Counter({i:8 for i in range(16,48)})
    assert a[-1]==47 and b[-1]==31
    assert a[48:64]==list(range(32,48)) and b[48:64]==list(range(16,32))
    assert len(set(v.sequence('main')))==256


def diagnostic_row(route_fixture,captured,name):
    original,state,base=route_fixture;row=copy.deepcopy(original);state=dict(state)
    state['indices']=v.v.job_indices(256,'g')[0]
    seq=v.sequence(name);row.update(condition=name+'_combined',sequence_name=name,dispatch_mode='online',measured_raw_indices=seq,
        unique_measured_job_combinations=len(set(seq)),raw_job_frequency={str(k):n for k,n in Counter(seq).items()},
        executed_logical_routing_sha256=v.v.array_hash(state['logical'][seq]),
        executed_combining_weights_sha256=v.v.array_hash(captured[1]['combining_weights'][state['indices']][seq]),
        assignment_metrics=[state['diagnostics']['online'][i] for i in seq],terminal_physical_window=state['counts']['online'][seq[-16:]].tolist())
    for d in row['decision_log']:
        raw=seq[d['step']-15:d['step']+1];co=state['counts']['online'][raw]
        d.update(current_counts=(np.bincount(state['logical'][raw].ravel(),minlength=128)*2).tolist(),
            physical_slot_assignments_window=co.tolist(),physical_slot_assignments_sha256=v.v.array_hash(co),
            physical_owner_assignments_per_step=co.reshape(16,2,80).sum(2).tolist())
    row['decision_trajectory_sha256']=v.digest([{k:d[k] for k in ('step','current_counts','current_map','proposed_map','committed_map')} for d in row['decision_log']])
    return row,state,base,captured[1]['combining_weights']


@pytest.mark.parametrize('name',['main','cycle0','cycle16'])
def test_actual_sequence_routes(route_fixture,captured,name):
    v.check_routes(*diagnostic_row(route_fixture,captured,name))


def test_same_frequency_wrong_phase_cannot_relabel_routes(route_fixture,captured):
    args=diagnostic_row(route_fixture,captured,'cycle0');args[0]['condition']='cycle16_combined'
    with pytest.raises(AssertionError):v.check_routes(*args)


def test_cycle_terminal_cannot_use_main_endpoint(route_fixture,captured):
    args=diagnostic_row(route_fixture,captured,'cycle16');args[0]['terminal_physical_window']=args[1]['counts']['online'][-16:].tolist()
    with pytest.raises(AssertionError,match='terminal cycle'):v.check_routes(*args)
