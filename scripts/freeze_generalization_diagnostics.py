#!/usr/bin/env python3
"""Freeze the prespecified 24-trial diagnostic subset without changing main code."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generalization_diagnostics.plan import build

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    plan=build(a.pilot_plan);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(trials=24,states=[s['content_group_id'] for s in plan['states']],sha256=plan['freeze']['sha256'])))
if __name__=='__main__':main()
