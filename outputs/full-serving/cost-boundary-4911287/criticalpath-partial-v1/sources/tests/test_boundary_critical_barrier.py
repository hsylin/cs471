from full_serving.boundary_critical_barrier import subtract_activity,definite_common,next_execute

def test_overlapping_activity_union_not_double_counted():
 assert subtract_activity(0,10,[(1,4),(3,7),(8,11)])==[(0,1),(7,8)]

def test_activity_touching_boundaries_and_empty_window():
 assert subtract_activity(3,3,[])==[]
 assert subtract_activity(3,7,[(0,3),(7,9)])==[(3,7)]
 assert subtract_activity(3,7,[(0,10)])==[]

def test_conservative_common_interval_shrinks_with_clock_uncertainty():
 assert definite_common({0:(1,3),1:(2,4)},{0:(9,11),1:(7,10)})==(4,7)
 assert definite_common({0:(1,8),1:(2,9)},{0:(8,10),1:(7,10)}) is None

def test_next_execute_requires_same_thread_later_ordinal_and_return():
 s=dict(thread=1,start_ordinal=10,end_wall_ns=100)
 xs=[dict(thread=2,start_ordinal=11,start_wall_ns=101),dict(thread=1,start_ordinal=10,start_wall_ns=102),dict(thread=1,start_ordinal=11,start_wall_ns=99),dict(thread=1,start_ordinal=11,start_wall_ns=110)]
 assert next_execute(s,xs)==xs[3]
 assert next_execute(s,xs[:3]) is None
