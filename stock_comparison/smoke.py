"""Untimed pilot-only apparatus validation; never primary speedup evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from followup.router_tapes import load_router_bank
from stock_comparison.native import StockRuntime,StockLayer
from skewbench.native import Weights
from stock_comparison.runner import fp32_reference,gather

def main():
    p=argparse.ArgumentParser()
    for n in ('model-path','weights-dir','bank','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    rt=StockRuntime(a.model_path)
    try:
        weights=Weights(a.weights_dir)
        layers={arm:StockLayer(rt,weights,arm) for arm in ('stockoff','stockon32','ours')}
        bank=load_router_bank(a.bank)
        hidden=rt.torch.from_numpy(np.array(bank.hidden_states_bf16[:128])).to(rt.device).view(rt.torch.bfloat16)
        original=hidden.clone()
        ids,combining=layers['stockoff'].routing(hidden)
        reference=layers['stockoff'].forward(hidden,observe=False).clone()
        positions=rt.torch.arange(0,128,8,device=rt.device)
        fp32=fp32_reference(rt,weights,hidden[positions],ids[positions],combining[positions])
        rt.torch.testing.assert_close(reference[positions].float(),fp32,atol=.02,rtol=.03)
        layers['stockoff'].attest_output_reduction(hidden)
        rows=[]
        for arm,layer in layers.items():
            if rt.rank==0:print(json.dumps(dict(event='smoke_start',arm=arm)),flush=True)
            layer.reset_trial()
            with layer.measured():
                for i in range(4096):
                    output=layer.forward(hidden,observe=True);layer.step(i);rt.sync()
                    if i in (0,749,750,1000,3749,3750,4095):
                        rt.torch.testing.assert_close(output.float(),reference.float(),atol=.02,rtol=.03)
                        if rt.rank==0:print(json.dumps(dict(event='smoke_checkpoint',arm=arm,step=i)),flush=True)
                layer.complete_pending()
                end=layer.snapshot()
            audit=layer.audit()
            rt.torch.testing.assert_close(layer.forward(hidden,observe=False).float(),reference.float(),atol=.02,rtol=.03)
            if not rt.torch.equal(original,hidden):raise AssertionError('Input mutation')
            assignment=layer.verify_online_assignment(hidden) if arm=='ours' else None
            rows.append(dict(arm=arm,per_rank=gather(rt,dict(end=end,audit=audit,assignment=assignment))))
        if rt.rank==0:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            if a.output.exists():raise FileExistsError(a.output)
            a.output.write_text(json.dumps(dict(schema='stock_apparatus_smoke_v1',timing_evidence=False,
                bank_sha256=bank.sha256,runtime=rt.metadata,rows=rows),indent=2)+'\n')
    finally:rt.close()
if __name__=='__main__':main()
