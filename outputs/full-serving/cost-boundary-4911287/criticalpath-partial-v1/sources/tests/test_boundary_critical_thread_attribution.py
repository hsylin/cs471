from full_serving.boundary_critical_attribution import correlate_api_ordinals

def api(tid,start,end,correlation,cat="cuda_runtime"):
    return dict(ph="X",cat=cat,tid=tid,ts=start,dur=end-start,args=dict(correlation=correlation))

def test_background_call_inside_main_scope_is_unassigned():
    e=[api(1,12,13,1),api(2,12,13,2)]
    assert correlate_api_ordinals(e,[(10,20,5,1)])=={1:5}

def test_two_threads_overlapping_execute_scopes_keep_own_ordinals():
    e=[api(1,12,13,1),api(2,12,13,2,"cuda_driver")]
    assert correlate_api_ordinals(e,[(10,20,5,1),(11,21,9,2)])=={1:5,2:9}

def test_cross_boundary_and_gpu_annotations_do_not_correlate():
    e=[api(1,19,23,1),api(1,12,13,2,"gpu_user_annotation"),api(1,22,23,3)]
    assert correlate_api_ordinals(e,[(10,20,5,1)])=={}
