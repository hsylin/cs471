#!/usr/bin/env python3
"""CPU-only logical-demand comparison after final freeze; no timing inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from followup.router_tapes import load_router_bank
from optimization.common import canonical_hash,manifest_hash

REVISION='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'


def summarize(ids,prompt_index,positions,physical_map,*,documents=64,prefix=8):
    ids=np.asarray(ids);pi=np.asarray(prompt_index);positions=np.asarray(positions)
    if ids.ndim!=2 or ids.shape[1]!=8 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=128):raise ValueError('Expected E128/top8 integer routing')
    if np.any(np.diff(np.sort(ids,axis=1),axis=1)==0):raise ValueError('Logical top-k IDs must be unique per token')
    if pi.shape!=(len(ids),) or positions.shape!=pi.shape or set(np.unique(pi))!=set(range(documents)):raise ValueError('Expected all declared documents')
    mask=np.zeros(len(ids),dtype=bool);lengths=[]
    for doc in range(documents):
        rows=np.flatnonzero(pi==doc);pos=positions[rows]
        if not np.array_equal(np.sort(pos),np.arange(len(rows))) or len(rows)<prefix:raise ValueError('Each prompt must contain every position0..7 and contiguous captured prefix')
        selected=rows[pos<prefix]
        if len(selected)!=prefix or set(positions[selected])!=set(range(prefix)):raise ValueError('Missing or duplicate matched position')
        mask[selected]=True;lengths.append(len(rows))
    if int(mask.sum())!=documents*prefix:raise AssertionError('Unequal matched token counts')
    mapping=np.asarray(physical_map)
    if mapping.shape!=(160,) or set(mapping)!=set(range(128)):raise ValueError('Expected legal fixed160-slot map')
    replicated=np.bincount(mapping,minlength=128)>1
    def distribution(x):
        counts=np.bincount(x.ravel(),minlength=128);p=counts/counts.sum();nonzero=p[p>0]
        return dict(tokens=len(x),assignments=int(counts.sum()),logical_counts=counts.tolist(),logical_probability=p.tolist(),
            entropy_bits=float(-np.sum(nonzero*np.log2(nonzero))),active_logical_experts=int(np.count_nonzero(counts)),
            replicated_logical_expert_demand_fraction=float(p[replicated].sum()))
    full,matched=distribution(ids),distribution(ids[mask])
    tv=float(.5*np.abs(np.asarray(full['logical_probability'])-matched['logical_probability']).sum())
    return dict(documents=documents,matched_prefix_tokens_per_document=prefix,matched_documents_covered=documents,
        full_captured_tokens_per_document=lengths,full_prefix=full,first8=matched,tv_full_vs_first8=tv,
        entropy_difference_bits_first8_minus_full=matched['entropy_bits']-full['entropy_bits'],
        replicated_demand_fraction_difference_first8_minus_full=matched['replicated_logical_expert_demand_fraction']-full['replicated_logical_expert_demand_fraction'])


def analyze(plan,capture_root):
    if plan.get('phase')!='final' or plan.get('split')!='final' or plan.get('freeze',{}).get('sha256')!=manifest_hash(plan):raise ValueError('An already frozen final plan is required')
    groups={g['id']:g['dataset'] for g in plan['groups']};rows=[]
    if len(groups)!=len(plan['groups']) or len(groups) not in (18,24):raise ValueError('Declared selected final groups required')
    if len(plan['banks'])!=len(groups):raise ValueError('Capture inventory differs')
    for entry in plan['banks']:
        gid=entry['block_id'];bank=load_router_bank(Path(capture_root)/entry['capture_path']);m=bank.metadata
        expected=dict(block_id=gid,dataset=groups[gid],split='final',source_model='Qwen/Qwen3-30B-A3B',source_revision=REVISION,
            layer=0,num_experts=128,top_k=8,hidden_size=2048,selection_content_sha256=plan['content_selection_sha256'],correctness_pass=True)
        if bank.sha256!=entry['capture_sha256'] or any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong frozen capture or model/content provenance')
        if len(m['prompts'])!=64 or m['source_prompt_ids']!=entry['source_prompt_ids']:raise ValueError('Wrong document identities or count')
        for p in m['prompts']:
            if any(p.get(k) is not True for k in ('native_boundary_exact','native_router_exact','manual_router_exact','bf16_roundtrip_exact')):raise ValueError('Missing capture correctness attestation')
        summary=summarize(bank.expert_ids,bank.prompt_index,bank.token_position,plan['physical_to_logical_map'])
        rows.append(dict(content_group_id=gid,dataset=groups[gid],capture_sha256=bank.sha256,**summary))
    dataset_summaries=[]
    for dataset in sorted(set(groups.values())):
        members=[r for r in rows if r['dataset']==dataset];out=dict(dataset=dataset,content_groups=len(members),documents=64*len(members),first8_tokens=512*len(members),
            full_prefix_tokens=sum(r['full_prefix']['tokens'] for r in members),
            mean_within_group_tv_full_vs_first8=float(np.mean([r['tv_full_vs_first8'] for r in members])),
            within_group_tv_range=[min(r['tv_full_vs_first8'] for r in members),max(r['tv_full_vs_first8'] for r in members)])
        for version in ('full_prefix','first8'):
            p=np.mean([r[version]['logical_probability'] for r in members],axis=0);nonzero=p[p>0]
            replicated=np.bincount(plan['physical_to_logical_map'],minlength=128)>1
            out[version]=dict(equal_group_mean_logical_probability=p.tolist(),entropy_of_mean_distribution_bits=float(-np.sum(nonzero*np.log2(nonzero))),
                mean_group_entropy_bits=float(np.mean([r[version]['entropy_bits'] for r in members])),
                replicated_logical_expert_demand_fraction=float(p[replicated].sum()))
        out['tv_between_equal_group_mean_distributions']=float(.5*np.abs(np.asarray(out['full_prefix']['equal_group_mean_logical_probability'])-out['first8']['equal_group_mean_logical_probability']).sum())
        dataset_summaries.append(out)
    cross=[]
    for i,a in enumerate(dataset_summaries):
        for b in dataset_summaries[i+1:]:
            cross.append(dict(dataset_a=a['dataset'],dataset_b=b['dataset'],**{version:float(.5*np.abs(np.asarray(a[version]['equal_group_mean_logical_probability'])-b[version]['equal_group_mean_logical_probability']).sum()) for version in ('full_prefix','first8')}))
    return dict(schema='length_matched_routing_descriptive_v1',final_plan_sha256=plan['freeze']['sha256'],groups=len(rows),documents=64*len(rows),
        first8_tokens=512*len(rows),replicated_logical_experts=np.flatnonzero(np.bincount(plan['physical_to_logical_map'],minlength=128)>1).tolist(),
        group_summaries=rows,dataset_summaries=dataset_summaries,cross_dataset_tv=cross,
        timing_data_used=False,confidence_intervals=None,policy_selection=False,
        interpretation='Logical top-k assignment counts only; every selected prompt contributes its first8 captured positions. No matched-latency or causal performance claim.',
        limitations=['Full prefixes weight longer prompts more within a group; first8 gives equal token counts per prompt and changes token-position composition.',
            'Dataset summaries weight content groups equally; these distributions are not natural serving demand.',
            'Entropy uses log2; replica-demand fraction counts logical assignments to experts with multiple resident copies, not actual dispatched-copy use.',
            'Capture attestations and file hashes are checked; this CPU report does not rerun the model or measure GPU time.'])


def markdown(result):
    lines=['# Equal-length routing supplement','',
        f"The selected frozen final inputs contain {result['documents']:,} documents in {result['groups']} groups. This CPU-only comparison uses exactly eight captured token positions per document ({result['first8_tokens']:,} tokens). It makes no latency or causal claim.",'',
        'Each row averages group distributions equally. TV ranges refer to within-group full-prefix versus first-eight differences; entropy is for the dataset mean distribution. Replica demand is the share of logical top-k assignments belonging to replicated experts, not measured physical-copy traffic.','',
        '| Dataset | Groups | Full tokens | Mean within-group TV | TV range | Entropy full → first8 (bits) | Replicated demand full → first8 |','|---|---:|---:|---:|---:|---:|---:|']
    for d in result['dataset_summaries']:
        a,b=d['full_prefix'],d['first8'];lo,hi=d['within_group_tv_range']
        lines.append(f"| {d['dataset']} | {d['content_groups']} | {d['full_prefix_tokens']:,} | {d['mean_within_group_tv_full_vs_first8']:.4f} | {lo:.4f}–{hi:.4f} | {a['entropy_of_mean_distribution_bits']:.4f} → {b['entropy_of_mean_distribution_bits']:.4f} | {a['replicated_logical_expert_demand_fraction']:.4f} → {b['replicated_logical_expert_demand_fraction']:.4f} |")
    lines+=['','| Dataset pair | Full-prefix TV | First8 TV |','|---|---:|---:|']
    for x in result['cross_dataset_tv']:lines.append(f"| {x['dataset_a']} / {x['dataset_b']} | {x['full_prefix']:.4f} | {x['first8']:.4f} |")
    lines+=['','The first-eight construction controls retained token count per prompt, while changing token-position composition and removing the full-prefix weighting of longer prompts. Remaining dataset differences are descriptive routing differences in these captures. They are not proof that length caused any runtime change. No timing outcomes, confidence intervals or policy decisions enter this report.','',f"Frozen final plan: `{result['final_plan_sha256']}`. Full per-group counts, probability vectors and exact source hashes are retained in the adjacent JSON."]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();result=analyze(json.loads(a.plan.read_text()),a.capture_root)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['plan_file_sha256']=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output_dir/'report.md').write_text(markdown(result))
    print(json.dumps({k:result[k] for k in ('groups','documents','first8_tokens','timing_data_used')}))
if __name__=='__main__':main()
