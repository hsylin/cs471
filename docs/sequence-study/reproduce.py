"""Offline complete-matrix reconstruction, without rerunning GPU work."""
import argparse,json,sys,math
from pathlib import Path

def compare(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and not isinstance(b,bool) and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-9)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    import os;os.chdir(root)
    from sequence_study.common import load_plan
    from sequence_study.analysis import analyze
    from stock_comparison.analysis import read_records
    from followup.router_tapes import file_sha256
    run=root/'results/sequence-study/final-v2';plan=load_plan(run/'provenance/plan.json');rows,files=read_records(run)
    actual=analyze(plan,rows);saved=json.loads((run/'analysis.json').read_text());saved.pop('record_files',None)
    compare(actual,saved)
    a.output.mkdir(parents=True,exist_ok=False)
    proof=dict(passed=True,records=len(rows),plan_sha256=plan['content_sha256'],source_sha256=plan['source_sha256'],
        analysis_file_sha256=file_sha256(run/'analysis.json'),all_descriptive_outputs_reconstructed=True,
        numerical_comparison=dict(relative_tolerance=2e-12,absolute_tolerance=2e-9),
        raw_files=files,scope='Same-team offline reconstruction from archived raw GPU records; no fresh GPU execution.')
    (a.output/'reproduction.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))

if __name__=='__main__':main()
