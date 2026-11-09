"""Authentic hidden vectors, fresh deterministic layer-job composition.

HF-captured IDs/weights are provenance only. Actual vLLM computes its own gate
and float32 combining weights inside every timed forward in every arm.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
from followup.router_tapes import load_router_bank, file_sha256
from stock_comparison.data import digest

SEED=471603
BATCH=128
WARMUP=16
JOBS=4096
TOTAL=JOBS+WARMUP

def array_digest(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(digest([a.dtype.str,list(a.shape)]).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def build_indices(record_count,block_id,*,seed=SEED,jobs=JOBS):
    if type(record_count) is not int or record_count<BATCH or type(jobs) is not int or jobs<1:
        raise ValueError('Need at least128 authenticated records and positive job count')
    n=jobs+WARMUP
    if math.comb(record_count,BATCH)<n:raise ValueError('Too few distinct job combinations')
    rows=[];seen=set()
    for raw in range(n):
        for retry in range(1024):
            key=int(digest([seed,block_id,BATCH,raw,retry])[:16],16)
            row=np.random.Generator(np.random.PCG64(key)).permutation(record_count)[:BATCH].astype(np.int64)
            identity=np.sort(row).tobytes()
            if identity not in seen:
                seen.add(identity);rows.append(row);break
        else:raise RuntimeError('Fixed distinct-combination retry bound exhausted')
    indices=np.stack(rows)
    uses=np.bincount(indices[WARMUP:].ravel(),minlength=record_count)
    return indices,dict(schema='stock_hidden_jobs_v1',block_id=block_id,batch_tokens=BATCH,
        warmup_jobs=WARMUP,measured_jobs=jobs,seed=seed,source_records=record_count,
        distinct_job_combinations=len(seen),indices_sha256=array_digest(indices),
        measured_record_occurrences=jobs*BATCH,distinct_measured_records=int((uses>0).sum()),
        minimum_record_uses=int(uses.min()),maximum_record_uses=int(uses.max()),
        not_independent_requests=True,natural_batch_trace=False,
        composition='Independent seeded PCG64 permutations of authenticated record IDs; no repeated record within job; cross-job reuse charged and disclosed')

def load_inputs(rt,entry,banks_dir,output_dir,*,jobs=JOBS):
    bank=load_router_bank(Path(banks_dir)/entry['capture_path'])
    m=bank.metadata
    if m['selection_content_sha256']!=entry['selection_content_sha256']:
        raise ValueError('Capture does not belong to frozen arrival/content selection')
    if bank.sha256!=entry['capture_sha256'] or m.get('study_id')!='stock_comparison_v1':
        raise ValueError('Changed capture or wrong study')
    if m['block_id']!=entry['block_id'] or m['dataset']!=entry['dataset'] or not m['correctness_pass']:
        raise ValueError('Capture provenance differs')
    if len(m['prompts'])!=32 or any(not p['native_boundary_exact'] or not p['bf16_roundtrip_exact'] for p in m['prompts']):
        raise ValueError('Need32 correctly captured fresh documents')
    if (np.asarray(bank.hidden_states_bf16,dtype=np.uint16)&0x7f80==0x7f80).any():raise ValueError('Nonfinite BF16')
    indices,meta=build_indices(len(bank.hidden_states_bf16),entry['block_id'],jobs=jobs)
    # Transfer one job at a time to avoid an additional2GB host temporary/rank.
    hidden=rt.torch.empty((len(indices),BATCH,2048),device=rt.device,dtype=rt.torch.bfloat16)
    for i,idx in enumerate(indices):
        bits=np.asarray(bank.hidden_states_bf16[idx],dtype=np.uint16).copy()
        hidden[i].copy_(rt.torch.from_numpy(bits).view(rt.torch.bfloat16))
    rt.sync()
    target=Path(output_dir)/'inputs'/(entry['block_id']+'.npz')
    if rt.rank==0:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        np.savez_compressed(target,indices=indices,metadata=np.asarray(__import__('json').dumps(meta,sort_keys=True)))
    rt.barrier()
    return hidden,dict(**meta,capture_sha256=bank.sha256,source_prompt_ids=m['source_prompt_ids'],
        gate_weight_sha256=m['source_tensor_sha256']['model.layers.0.mlp.gate.weight'],
        index_artifact=str(target),index_artifact_sha256=file_sha256(target),
        executable_input='Only exact BF16 pre-MoE hidden vectors; gate/topk recomputed inside measured stockblock')
