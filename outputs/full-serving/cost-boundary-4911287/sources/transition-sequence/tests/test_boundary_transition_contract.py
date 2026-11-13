import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from full_serving.boundary_transition_contract import (
    Schedule, canonical, freeze_candidate, validate_map, validate_spec, verify_weight_records, native_hash_reference, agreed_marker)

def maps():
    old=[list(range(128))+list(range(32)) for _ in range(48)]
    new=copy.deepcopy(old);new[3][0],new[3][1]=new[3][1],new[3][0]
    return old,new

def weights(mapping):
    return [dict(rank=rank,weights=[
        dict(layer=layer,slot=slot,expert=mapping[layer][slot],
             sha256=[format(mapping[layer][slot]+1,"064x"),format(mapping[layer][slot]+129,"064x")])
        for layer in range(48) for slot in range(rank*20,(rank+1)*20)]) for rank in range(8)]

class Tests(unittest.TestCase):
    def test_queued_warmup_proposal_cannot_be_armed_late(self):
        self.assertFalse(agreed_marker([None]*8))
        self.assertFalse(agreed_marker(["marker"]*7+[None]))
        self.assertTrue(agreed_marker(["marker"]*8))
    def test_conflicting_marker_rejected(self):
        with self.assertRaises(AssertionError):agreed_marker(["a"]*7+["b"])
    def test_graph_oracle_padding_and_replica_hash(self):
        mapping=maps()[0][0]
        self.assertEqual(native_hash_reference([[-1,0],[0,-1]],mapping),[[-1,0],[128,-1]])
        self.assertEqual(native_hash_reference([[-1]*8]*8,mapping),[[-1]*8]*8)
    def test_graph_oracle_rejects_invalid_expert(self):
        with self.assertRaises(AssertionError):native_hash_reference([[128]],maps()[0][0])
    def test_keep_and_preinstalled_are_fixed(self):
        old,new=maps()
        for mode,target in (("keep",old),("preinstalled",new)):
            s=Schedule(mode,old,new)
            self.assertEqual(s.initial(),target)
            for armed in (False,True,True):
                self.assertEqual(s.choose(target,armed),(target,False))
    def test_charged_only_first_measured_proposal(self):
        old,new=maps();s=Schedule("charged",old,new)
        self.assertEqual(s.choose(old,False),(old,False))
        self.assertEqual(s.choose(old,False),(old,False))
        self.assertEqual(s.choose(old,True),(new,True))
        self.assertEqual(s.choose(new,True),(new,False))
    def test_partial_or_stale_map_rejected(self):
        old,new=maps();s=Schedule("charged",old,new);s.choose(old,True)
        with self.assertRaises(AssertionError):s.choose(old,True)
    def test_inputs_cannot_mutate_schedule(self):
        old,new=maps();s=Schedule("charged",old,new);old[0][0]=1
        self.assertEqual(s.initial()[0][0],0)
    def test_missing_expert_or_wrong_capacity_rejected(self):
        old,new=maps();old[0]=[0]*160
        with self.assertRaises(AssertionError):validate_map(old)
        with self.assertRaises(AssertionError):validate_map(new[:47])
    def test_weights_match_before_after_native_migration(self):
        old,new=maps()
        baseline=verify_weight_records(weights(old),old)
        self.assertEqual(verify_weight_records(weights(new),new,baseline),baseline)
    def test_wrong_ownership_or_corrupt_weight_rejected(self):
        old,new=maps();baseline=verify_weight_records(weights(old),old)
        p=weights(new);p[0]["weights"][0]["sha256"][0]="f"*64
        with self.assertRaises(AssertionError):verify_weight_records(p,new,baseline)
        p=weights(new);p[0]["weights"][0]["slot"]=20
        with self.assertRaises(AssertionError):verify_weight_records(p,new)
    def test_full_replica_and_layer_coverage_required(self):
        old,new=maps();p=weights(new);p[-1]["weights"].pop()
        with self.assertRaises(AssertionError):verify_weight_records(p,new)

    def pilot(self,root,disagree=False,no_change=False):
        old,new=maps()
        (root/"results/pilot/b4").mkdir(parents=True)
        (root/"audit.json").write_text('{"status":"passed"}')
        (root/"results/controller-source.json").write_text(json.dumps(dict(declared_git_revision="a"*40)))
        for rank in range(8):
            rows=[]
            for step,admitted in [(750,old),(3750,old if no_change else new),(6750,old if no_change else new)]:
                r=dict(kind="decision",rank=rank,mode="b4",rank_agreement=True,step=step,digest=str(step),
                       old_map=old,admitted_map=admitted,native_candidate=new,previous_counts=[],recent_counts=[])
                if disagree and rank==7 and step==3750:r["digest"]="disagree"
                rows.append(r)
            (root/"results/pilot/b4"/f"gate-events-rank{rank}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    def test_first_changed_candidate_not_best_later(self):
        with tempfile.TemporaryDirectory() as d:
            self.pilot(Path(d));s=freeze_candidate(d)
            self.assertEqual(s["source_step"],3750);self.assertEqual(s["changed_slots"],2)
            validate_spec(s)
            s["candidate_map"][0][0]=99
            with self.assertRaises(AssertionError):validate_spec(s)
    def test_rank_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.pilot(Path(d),disagree=True)
            with self.assertRaises(AssertionError):freeze_candidate(d)
    def test_no_changed_candidate_is_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            self.pilot(Path(d),no_change=True)
            with self.assertRaises(ValueError):freeze_candidate(d)
if __name__=="__main__":unittest.main()
