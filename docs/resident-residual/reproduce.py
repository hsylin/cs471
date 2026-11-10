#!/usr/bin/env python3
"""Reanalyze archived residual experiments without GPU execution or network."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path


def compare(a,b,path='root'):
    if isinstance(a,dict) and isinstance(b,dict):
        assert set(a)==set(b),path
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        assert math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-9),(path,a,b)
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--banks-root',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from resident_residual.common import source_identity,load_plan
    from resident_residual.analysis import analyze
    from stock_comparison.analysis import read_records
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=False)
    import os;os.chdir(root)
    banks_root=args.banks_root.resolve() if args.banks_root else root/'results/resident-residual'
    source,files=source_identity();results=[]
    expected_source='136259a52849be38f9dc875f6ccef964ba2c92f8cd53ceee2c5e5c51f2e0dce7'
    assert source==expected_source
    base=Path('results/resident-residual')
    for phase in ('calibration-pilot','pilot','calibration-final','final'):
        run=base/(phase+'-v1');plan_path=run/'provenance/plan.json';plan=load_plan(plan_path)
        rows,_=read_records(run);actual=analyze(plan,rows);saved=json.loads((run/'analysis.json').read_text())
        saved.pop('record_files',None);compare(actual,saved);assert actual['valid']
        output=target/(phase+'-independent-statistics.json')
        subprocess.run([sys.executable,str(base/'auditors/audit_residual_statistics.py'),'--repo-root','.',
            '--plan',str(plan_path),'--records',str(run),'--analysis',str(run/'analysis.json'),'--output',str(output)],check=True)
        split='pilot' if 'pilot' in phase else 'final'
        banks=banks_root/('capture-'+split+'-v1')/'banks';bank_audit=False
        if banks.is_dir():
            command=[sys.executable,str(base/'auditors/audit_residual_records.py'),'--repo-root','.',
                '--plan',str(plan_path),'--records',str(run),'--inputs-root',str(run/'inputs'),
                '--capture-root',str(banks),'--binary-root',str(base/'compiled-artifacts'),
                '--source-archive',str(run/'provenance/executed-source.tar.gz'),
                '--output',str(target/(phase+'-independent-execution.json'))]
            if not phase.startswith('calibration'):command+=['--calibration',str(base/('calibration-'+split+'-v1'))]
            subprocess.run(command,check=True);bank_audit=True
        results.append(dict(phase=phase,records=len(rows),production_reconstruction=True,independent_statistics=True,independent_bank_and_execution_audit=bank_audit))
    inventory=json.loads((base/'compiled-artifacts/archive-inventory.json').read_text())
    for row in inventory['files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    result=dict(schema='resident_residual_clean_cpu_reproduction_v1',passed=True,source_sha256=source,source_files=len(files),
        phases=results,compiled_binary_bytes_verified=True,scope='Offline reconstruction from recorded GPU evidence; no new GPU capture or external-laboratory replication.')
    (target/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
