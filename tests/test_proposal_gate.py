import numpy as np
import pytest
from full_serving.proposal_gate import decide


def test_proposal_example_and_stability_ablation():
    old=[0,1,2,3];new=[0,2,1,3];recent=[400,300,150,150]
    d=decide([390,305,155,150],recent,old,new,ranks=2,gain_threshold=.1)
    assert d['allow'] and d['tv']==pytest.approx(.01) and d['predicted_gain']==pytest.approx(150/700)
    unstable=[100,150,500,250]
    assert decide(unstable,recent,old,new,ranks=2)['reason']=='unstable'
    assert decide(unstable,recent,old,new,ranks=2,require_stability=False)['allow']


def test_complete_nonzero_and_coverage():
    args=([4,3,2,1],[4,3,2,1],[0,1,2,3],[0,2,1,3])
    assert decide(*args,ranks=2,recent_complete=False)['reason']=='incomplete_recent'
    assert decide(*args,ranks=2,previous_complete=False)['reason']=='incomplete_previous'
    assert decide(*args,ranks=2,previous_complete=False,require_stability=False)['allow']
    assert decide([0]*4,[0]*4,[0,1,2,3],[0,2,1,3],ranks=2)['reason']=='zero_recent'
    with pytest.raises(ValueError):decide([1]*4,[1]*4,[0,0,2,3],[0,1,2,3],ranks=2)


def test_replicas_equal_sharing_and_scale_invariance():
    a=np.array([40,30,15,15]);old=[0,1,0,2,3,3];new=[0,2,3,0,1,3]
    d=decide(a,a,old,new,ranks=2)
    e=decide(100*a,2*a,old,new,ranks=2)
    assert d['tv']==e['tv']==0
    assert d['predicted_gain']==pytest.approx(e['predicted_gain'])
    assert d['current_peak_assignments']==70
