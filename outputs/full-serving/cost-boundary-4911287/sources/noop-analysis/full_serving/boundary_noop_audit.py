"""Fail-closed all-rank map-elision pilot audit. No serving gain admission."""
import argparse,hashlib,json
from pathlib import Path

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def audit(root):
 root=Path(root);directory=root/"pilot"/"b4"
 assert json.loads((root/"status.json").read_text())["state"]=="comparison_completed"
 allr=[];cross={}
 for rank in range(8):
  path=directory/f"noop-commit-rank{rank}.jsonl";rr=rows(path)
  assert rr and all(r["rank"]==rank for r in rr)
  decisions={r["step"]:r for r in rows(directory/f"gate-events-rank{rank}.jsonl") if r["kind"]=="decision"}
  proofs=[r for r in rr if r["kind"]=="unchanged_proof"]
  assert len(proofs)==len(decisions)>0
  valid={}
  for r in proofs:
   step=r["cycle_step"];d=decisions[step];assert d["rank_agreement"]
   expect=[i for i,(a,b) in enumerate(zip(d["old_map"],d["admitted_map"])) if a==b]
   assert r["shadow"] is True and r["unchanged_layers"]==expect
   assert step not in valid;valid[step]=set(expect)
   fingerprint=(d["digest"],tuple(expect))
   assert cross.setdefault(step,fingerprint)==fingerprint
  skips=[(r["cycle_step"],r["layer"]) for r in rr if r["kind"]=="map_commit_elided"]
  shadows=[(r["cycle_step"],r["layer"]) for r in rr if r["kind"]=="map_shadow_passed"]
  assert skips and len(set(skips))==len(skips) and skips==shadows
  assert all(layer in valid[step] for step,layer in skips)
  weights=[r for r in rr if r["kind"]=="weight_shadow_passed"]
  assert {r["layer"] for r in weights}==set(range(48)),"Incomplete full-weight coverage"
  assert len(weights)==48 and all(r["compared_elements"]>0 for r in weights)
  # The old field is a late pending-slot-empty sample, after weight equality
  # synchronizes. A correctly retired result may already be replaced by the
  # next prefetched layer. Require the pinned worker's forward-progress trace.
  migration_rows=rows(directory/f"migration-events-rank{rank}.jsonl")
  late_prefetch=[]
  for weight in weights:
   if weight["pending_result_retired"]:continue
   prior=[r for r in rr if r["kind"]=="map_commit_elided" and r["layer"]==weight["layer"] and r["wall_ns"]<=weight["wall_ns"]]
   assert prior
   commit=max(prior,key=lambda r:r["wall_ns"])
   following=[r for r in migration_rows if r["kind"]=="transfer_enqueued"
      and r["cycle_step"]==commit["cycle_step"] and r["layer"]==weight["layer"]+1
      and commit["wall_ns"]<r["wall_ns"]<=weight["wall_ns"]]
   assert following,"Nonempty late pending slot lacks next-layer progress evidence"
   late_prefetch.append(dict(layer=weight["layer"],cycle_step=commit["cycle_step"],
      map_elision_wall_ns=commit["wall_ns"],next_transfer_wall_ns=following[0]["wall_ns"],
      late_sample_wall_ns=weight["wall_ns"]))

  observed=[r for r in rows(directory/f"migration-events-rank{rank}.jsonl") if r["kind"]=="commit_completed"]
  # Observer cycle uses the gate's actual proposal clock for this frozen B4 pilot.
  observed_keys={(r["cycle_step"],r["layer"]) for r in observed}
  censored=[list(k) for k in skips if k not in observed_keys]
  allr.append(dict(rank=rank,proof_cycles=len(proofs),elided_map_commits=len(skips),
      exact_metadata_shadows=len(shadows),full_weight_layers=len(weights),
      observed_elided_commits=sum(k in observed_keys for k in skips),
      censored_elided_commits=censored,late_slot_nonempty_with_verified_next_transfer=late_prefetch,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert all(r["observed_elided_commits"]>0 for r in allr)
 return dict(status="passed",ranks=allr,
   pending_semantics="Original pending_result_retired field actually sampled pending slot emptiness after GPU weight comparison. Nonempty samples require subsequent-layer transfer; pinned worker waits consumption and asserts prior pending None before advancing. No drain-as-commit inference.",
   scope="All-rank full-model metadata and full-weight equality for elided commits. GPU completion remains separately observed or censored. This pilot carries shadow overhead; no performance claim.",
   required_separate_checks=["ordinary sustained pilot audit","pinned native metadata graph fixture","final no-shadow matched serving"])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 result=audit(a.root);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"status":result["status"],"ranks":8}))
