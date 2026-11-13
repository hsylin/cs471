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
  assert len(weights)==48 and all(r["compared_elements"]>0 and r["pending_result_retired"] for r in weights)
  observed=[r for r in rows(directory/f"migration-events-rank{rank}.jsonl") if r["kind"]=="commit_completed"]
  # Observer cycle uses the gate's actual proposal clock for this frozen B4 pilot.
  observed_keys={(r["cycle_step"],r["layer"]) for r in observed}
  censored=[list(k) for k in skips if k not in observed_keys]
  allr.append(dict(rank=rank,proof_cycles=len(proofs),elided_map_commits=len(skips),
      exact_metadata_shadows=len(shadows),full_weight_layers=len(weights),
      observed_elided_commits=sum(k in observed_keys for k in skips),
      censored_elided_commits=censored,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert all(r["observed_elided_commits"]>0 for r in allr)
 return dict(status="passed",ranks=allr,
   scope="All-rank full-model metadata and full-weight equality for elided commits. GPU completion remains separately observed or censored. This pilot carries shadow overhead; no performance claim.",
   required_separate_checks=["ordinary sustained pilot audit","pinned native metadata graph fixture","final no-shadow matched serving"])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 result=audit(a.root);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"status":result["status"],"ranks":8}))
