"""Paired ordering intervention; no changes to the three frozen algorithms."""
import argparse,gc,json
from pathlib import Path
import numpy as np
from sequence_study.common import load_plan
from sequence_study.inputs import load_inputs
from stock_comparison.runner import (progress,gather,check_hardware,prepare_references,run_trial)
from stock_comparison.data import digest
from stock_comparison.inputs import WARMUP

def main():
    p=argparse.ArgumentParser()
    for n in ('plan','banks-dir','weights-dir','model-path','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();plan=load_plan(a.plan)
    from stock_comparison.native import StockRuntime,StockLayer
    from resident_residual.native import ResidualLayer
    from skewbench.native import Weights
    rt=StockRuntime(a.model_path)
    try:
        if rt.rank==0:(a.output/'trials').mkdir(parents=True,exist_ok=False)
        rt.barrier();hardware=check_hardware(rt,plan);weights=Weights(a.weights_dir)
        layers={'stockoff':StockLayer(rt,weights,'stockoff'),'stockon32':StockLayer(rt,weights,'stockon32'),
                'fix':ResidualLayer(rt,weights,'fix',**{k:plan['selected_fix'][k] for k in ('alignment_trim','mapper_input_checks','selector_launch')})}
        torch=rt.torch
        if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('FP32 reference precision')
        rt.metadata.update(study_id='sequence_study_v1',input_seed=471902,sequence_intervention=True,
            tf32_allowed=False,float32_matmul_precision='highest',
            precision_per_rank=gather(rt,dict(allow_tf32=False,matmul_precision='highest')))
        from resident_fastpath.fused_validation import verify_adversarial_fused
        rt.metadata['fused_adversarial']=verify_adversarial_fused(rt,layers['fix'].mapper)
        for bi,entry in enumerate(plan['banks']):
            gid=entry['block_id'];trials=[t for t in plan['trials'] if t['block_id']==gid]
            modes=list(dict.fromkeys(t['mode'] for t in trials));prepared={}
            for mode in modes:
                progress(rt,event='sequence_prepare',group=gid,mode=mode)
                hidden,meta=load_inputs(rt,entry,a.banks_dir,a.output,mode=mode)
                if any(l.gate_sha256!=meta['gate_weight_sha256'] for l in layers.values()):raise AssertionError('Wrong gate')
                if bi==0 and len(prepared)==0:
                    on=layers['stockon32'];on.reset_trial()
                    with on.measured():
                        for i in range(750):on.forward(hidden[i%WARMUP]);on.step(i);rt.sync()
                        on.complete_pending()
                    rt.metadata['native_async_preconditioning']=dict(jobs=750,timing_evidence=False,per_rank=gather(rt,on.audit()))
                    on.reset_trial()
                    rt.metadata['output_reduction_probe_per_rank']=gather(rt,layers['stockoff'].attest_output_reduction(hidden[WARMUP]))
                    if len({digest(p['observed_successful_backends']) for p in rt.metadata['output_reduction_probe_per_rank']})!=1:raise AssertionError('Rank reduction disagreement')
                reference,preflight=prepare_references(rt,layers,weights,hidden,a.output,gid+'-'+mode)
                checks=[]
                for raw in range(len(hidden)):checks.append(layers['fix'].verify_online_assignment(hidden[raw]))
                preflight['fix'].update(online_assignment_inputs_checked=len(checks),
                    physical_assignment_digest=digest([r['physical_sha256'] for r in checks]))
                if rt.rank==0:(a.output/'inputs'/(gid+'-'+mode+'-resident-diagnostics.json')).write_text(json.dumps(checks)+'\n')
                prepared[mode]=(hidden,reference,preflight,meta)
            if len({x[3]['measured_occurrence_multiset_sha256'] for x in prepared.values()})!=1:raise AssertionError('Changed multiset')
            if len({x[3]['warmup_indices_sha256'] for x in prepared.values()})!=1:raise AssertionError('Changed conditioning')
            for trial in trials:
                mode,arm=trial['mode'],trial['arm'];hidden,reference,preflight,meta=prepared[mode]
                case=dict(plan['calendars'][gid],arm_order=trial['arm_order'])
                progress(rt,event='sequence_trial_start',group=gid,mode=mode,arm=arm,rep=trial['rep'])
                row=run_trial(rt,layers[arm],hidden,reference,preflight[arm],plan,entry,case,meta,hardware)
                if rt.rank==0:
                    row.pop('record_id');row.update(schema='sequence_study_trial_v1',mode=mode,technical_rep=trial['rep'],evidence_eligible=False,descriptive_supplement=True)
                    row['record_id']=digest(row)
                    with (a.output/'trials'/(gid+'.jsonl')).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    progress(rt,event='sequence_trial_complete',group=gid,mode=mode,arm=arm,rep=trial['rep'],response_ns=row['queue_metrics']['summary_ns']['response']['mean'])
            # Instrumented pass after this group's primary trials. It is not a
            # placement-only causal comparison: last native map came from the
            # recorded last on32 trajectory; no profiling event contaminates it.
            if bi==0:
                from sequence_study.diagnostics import diagnostic
                hidden,reference,_,_=prepared['mixed'];raw=np.linspace(WARMUP,len(hidden)-1,64,dtype=int).tolist()
                probes={}
                for arm in ('stockoff','fix','stockon32'):
                    probes[arm]=diagnostic(rt,layers[arm],hidden,reference,raw,'post_trials_'+arm)
                layers['stockon32'].reset_trial()
                probes['stockon32_initial']=diagnostic(rt,layers['stockon32'],hidden,reference,raw,'initial_stockon32')
                if rt.rank==0:(a.output/'expert-probes.json').write_text(json.dumps(probes,allow_nan=False)+'\n')
            prepared.clear();del hidden,reference;gc.collect();torch.cuda.empty_cache();rt.sync();rt.barrier()
    finally:rt.close()

if __name__=='__main__':main()
