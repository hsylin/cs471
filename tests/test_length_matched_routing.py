import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/report_length_matched_routing.py'
s=importlib.util.spec_from_file_location('length_routing_report',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_exact_equal_prompt_prefix_and_replicated_demand_definition():
    ids=np.tile(np.arange(8),(20,1));ids[8:12]+=64;ids[12:]+=32
    pi=np.array([0]*12+[1]*8);positions=np.r_[np.arange(12),np.arange(8)]
    out=m.summarize(ids,pi,positions,list(range(128))+list(range(32)),documents=2)
    assert out['first8']['tokens']==16 and out['first8']['assignments']==128
    assert out['first8']['replicated_logical_expert_demand_fraction']==.5
    assert out['full_prefix']['replicated_logical_expert_demand_fraction']==.4
    assert out['tv_full_vs_first8']==pytest.approx(.2)


def test_missing_or_duplicate_prefix_position_cannot_be_silently_matched():
    ids=np.tile(np.arange(8),(16,1));pi=np.repeat([0,1],8);pos=np.tile(np.arange(8),2)
    bad=pos.copy();bad[7]=6
    with pytest.raises(ValueError):m.summarize(ids,pi,bad,list(range(128))+list(range(32)),documents=2)
    with pytest.raises(ValueError):m.summarize(ids[:-1],pi[:-1],pos[:-1],list(range(128))+list(range(32)),documents=2)
