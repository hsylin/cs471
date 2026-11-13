from full_serving.boundary_critical_attribution import resolved_latest

def test_large_uncertainty_of_third_rank_prevents_false_resolution():
    assert resolved_latest({0:100,1:90,2:80},{0:1,1:1,2:25}) is None

def test_all_competitor_upper_bounds_must_be_earlier():
    assert resolved_latest({0:100,1:90,2:80},{0:1,1:1,2:15})==0
    assert resolved_latest({0:100,1:100},{0:0,1:0}) is None
    assert resolved_latest({},{}) is None
