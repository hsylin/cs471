from full_serving.boundary_critical_predecessor import predecessor,overlap_other_arrivals,work_at

def test_predecessor_requires_same_stream_and_completed_before_target():
 t=dict(stream=1,start=10,end=11,category='collective');es=[dict(stream=1,start=2,end=4,category='attention_candidate'),dict(stream=2,start=5,end=9,category='other'),dict(stream=1,start=8,end=12,category='other'),t]
 assert predecessor(es,t)==es[0]
 assert predecessor(es,dict(stream=None,start=10)) is None

def test_overlap_must_exceed_all_peer_uncertainty_bounds():
 p=dict(end=10,category='moe_candidate');e={0:dict(start=11),1:dict(start=5),2:dict(start=7)}
 assert overlap_other_arrivals(p,e,0,{0:1,1:1,2:1})
 assert not overlap_other_arrivals(p,e,0,{0:1,1:1,2:4})
 assert not overlap_other_arrivals(dict(end=10,category='collective'),e,0,{0:0,1:0,2:0})

def test_missing_work_is_not_zero_filled_without_dummy_evidence():
 r={i:dict(batches={},iterations={10:dict(zero_actual_work=True)}) for i in range(8)}
 assert work_at(r,10)[0]['real_tokens']==0
 r[3]['iterations']={}
 assert work_at(r,10) is None
