"""Fresh hidden-state jobs; native gate and top-k remain inside every forward."""
import json
from pathlib import Path
import numpy as np
from followup.router_tapes import load_router_bank,file_sha256
from stock_comparison.inputs import build_indices,array_digest,WARMUP,BATCH

SEED=471703


def load_inputs(rt,entry,banks_dir,output_dir,*,jobs=4096):
    bank=load_router_bank(Path(banks_dir)/entry['capture_path']);m=bank.metadata
    if (m['selection_content_sha256']!=entry['selection_content_sha256']
            or bank.sha256!=entry['capture_sha256'] or m.get('study_id')!='resident_fix_v1'
            or m['block_id']!=entry['block_id'] or m['dataset']!=entry['dataset']
            or not m['correctness_pass']):
        raise ValueError('Wrong/changed fresh input bank')
    if len(m['prompts'])!=32 or any(not p['native_boundary_exact'] or not p['bf16_roundtrip_exact'] for p in m['prompts']):
        raise ValueError('Expected32 authenticated documents')
    if (np.asarray(bank.hidden_states_bf16,dtype=np.uint16)&0x7f80==0x7f80).any():
        raise ValueError('Nonfinite BF16 input')
    indices,meta=build_indices(len(bank.hidden_states_bf16),entry['block_id'],seed=SEED,jobs=jobs)
    meta={**meta,'schema':'resident_fix_hidden_jobs_v1'}
    hidden=rt.torch.empty((len(indices),BATCH,2048),device=rt.device,dtype=rt.torch.bfloat16)
    for i,idx in enumerate(indices):
        bits=np.asarray(bank.hidden_states_bf16[idx],dtype=np.uint16).copy()
        hidden[i].copy_(rt.torch.from_numpy(bits).view(rt.torch.bfloat16))
    rt.sync()
    target=Path(output_dir)/'inputs'/(entry['block_id']+'.npz')
    if rt.rank==0:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        np.savez_compressed(target,indices=indices,metadata=np.asarray(json.dumps(meta,sort_keys=True)))
    rt.barrier()
    return hidden,dict(**meta,capture_sha256=bank.sha256,source_prompt_ids=m['source_prompt_ids'],
        gate_weight_sha256=m['source_tensor_sha256']['model.layers.0.mlp.gate.weight'],
        index_artifact=str(target),index_artifact_sha256=file_sha256(target),
        executable_input='Exact captured BF16 hidden states only; actual stock gate/topk in measured forward')
