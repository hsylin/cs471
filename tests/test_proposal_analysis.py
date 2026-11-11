import hashlib,json
import numpy as np
import pytest
from full_serving.proposal_analysis import gate_audit
from full_serving.proposal_gate import decide


def test_gate_audit_reconstructs_rejection_and_censors_unobserved_completion(tmp_path):
    current=list(range(128))+list(range(32));counts=[10]*128
    d=decide(counts,counts,current,current)
    proposal={'kind':'decision','step':3750,'seen_steps':3750,'window':1000,'rank_agreement':True,'digest':'agree','window_bounds':{'previous':[1750,2750],'recent':[2750,3750]},'previous_counts':[counts]*48,'recent_counts':[counts]*48,'old_map':[current]*48,'native_candidate':[current]*48,'admitted_map':[current]*48,'layers':[d]*48}
    commit={'kind':'commit_enqueued','cycle_step':3750,'layer':0,'map_sha256':hashlib.sha256(np.asarray(current,dtype=np.int64).tobytes()).hexdigest(),'remote_received_experts':0,'remote_received_tensor_bytes':0,'unchanged_local_slots':20,'wall_ns':2}
    for rank in range(8):
        rows=[{'kind':'bound'},proposal,commit]
        p=tmp_path/f'gate-events-rank{rank}.jsonl';p.write_text(''.join(json.dumps({**r,'rank':rank,'mode':'b4'})+'\n' for r in rows))
    result=gate_audit(tmp_path,'b4');assert result['completion_not_observed_count']==8 and result['all_rank_layer_commits_completed']==0
    assert result['layer_decisions_rank0']=={'unchanged':48}
    p=tmp_path/'gate-events-rank0.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['admitted_map'][0][0]=7;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(AssertionError):gate_audit(tmp_path,'b4')


def test_logged_config_supports_only_known_literal_config_constructors():
    from full_serving.proposal_analysis import logged_config
    d=logged_config("prefix non-default args: {'eplb_config': EPLBConfig(num_redundant_experts=32), 'profiler_config': ProfilerConfig(profiler='torch')}")
    assert d['eplb_config']['num_redundant_experts']==32 and d['profiler_config']['profiler']=='torch'
    with pytest.raises(AssertionError):logged_config("non-default args: {'x': arbitrary_code()}")
