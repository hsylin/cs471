#!/usr/bin/env python3
"""Prepare fresh GSM8K-train/WildChat sources and disjoint Azure row blocks."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resident_residual.data import CONTENT_SEED,ARRIVAL_SEED,DATASETS,ALLOCATION,digest,exclude_and_deduplicate,split_sources,combine_prior_arrivals,scan_fresh_arrivals,assign_arrivals
from scripts.prepare_stock_comparison_data import prior_exclusions,candidates
from followup.router_tapes import MODEL,REVISION,file_sha256
TRAIN_SHA='17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
TRAIN_URL='https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl'


def sealed(path):
    x=json.loads(path.read_text())
    if x['content_sha256']!=digest({k:v for k,v in x.items() if k!='content_sha256'}):raise ValueError('Prior manifest hash differs: '+str(path))
    return x


def references(prior_paths,corpora,additional,downloads,tokenizer):
    refs,provenance,counts=prior_exclusions(prior_paths,corpora,tokenizer)
    for path,expected,schema in zip(additional,(768,704),('stock_comparison_sources_v1','resident_fix_sources_v1')):
        source=sealed(path);rows=[r for b in source['blocks'] for r in b['sources']]
        if source['schema']!=schema or len(rows)!=expected:raise ValueError('Incomplete preceding selection')
        seen={(r['dataset'],r['source_id']) for r in refs}
        for r in rows:
            ids=tokenizer.encode(r['text'],add_special_tokens=False).ids[:512]
            if (r['dataset'],r['source_id']) in seen or digest(r['text'].encode())!=r['text_sha256'] or ids!=r['token_ids']:raise ValueError('Prior content changed/reused')
            seen.add((r['dataset'],r['source_id']))
            refs.append(dict(dataset=r['dataset'],source_id=r['source_id'],text=r['text'],prefix_views=[tokenizer.decode(ids,skip_special_tokens=False)]))
        provenance.append(dict(path=path.parent.name+'/'+path.name,sha256=file_sha256(path),content_sha256=source['content_sha256'],documents=expected))
    if len(refs)!=3455:raise ValueError('Require3399 selected plus56 old capture documents')
    seen={(r['dataset'],r['source_id']) for r in refs};extra=0
    for i,line in enumerate((downloads/'gsm8k.jsonl').read_text().splitlines()):
        row=json.loads(line);key=('gsm8k',f'test-{i}')
        if key in seen:continue
        ids=tokenizer.encode(row['question'],add_special_tokens=False).ids[:512]
        refs.append(dict(dataset=key[0],source_id=key[1],text=row['question'],prefix_views=[tokenizer.decode(ids,skip_special_tokens=False)]));extra+=1
    if i+1!=1319 or extra!=103 or len(refs)!=3558:raise ValueError('Complete cross-split test exclusion required')
    provenance.append(dict(path='gsm8k.jsonl',sha256=file_sha256(downloads/'gsm8k.jsonl'),documents=1319,additional_never_selected_documents=extra,purpose='Exclude entire original test split from the new training-question prompt frame'))
    counts.update(stock_documents=768,resident_fix_documents=704,prior_unique_selected_documents=3399,prior_old_capture_documents=56,additional_unselected_gsm8k_test_documents=103,entire_gsm8k_test_frame=1319,total_reference_documents=3558)
    return refs,provenance,counts


def new_candidates(downloads,train_path,tokenizer):
    raw,short=candidates(downloads,tokenizer)
    raw=[r for r in raw if r['dataset']=='wildchat'];short=[r for r in short if r[0]=='wildchat']
    if file_sha256(train_path)!=TRAIN_SHA or train_path.stat().st_size!=4166206:raise ValueError('Changed pinned training source')
    for i,line in enumerate(train_path.read_text().splitlines()):
        row=json.loads(line);text=row['question'];ids=tokenizer.encode(text,add_special_tokens=False).ids
        if len(ids)<8:short.append(['gsm8k_train',f'train-{i}']);continue
        raw.append(dict(dataset='gsm8k_train',source_id=f'train-{i}',text=text,source_row_kind='question_only',original_split='train',full_token_count=len(ids),captured_token_count=min(512,len(ids)),token_ids=ids[:512],token_ids_sha256=digest(ids[:512]),captured_prefix_text=tokenizer.decode(ids[:512],skip_special_tokens=False)))
    if i+1!=7473:raise ValueError('Expected all7473 original training rows')
    return raw,short


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('downloads','train-source','output','public-output','stock-selection','resident-selection'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-selections',type=Path,nargs=3,required=True)
    p.add_argument('--old-corpora',type=Path,nargs=3,default=[ROOT/name/'corpus/sources.json' for name in ('followup','optimization','resident_dispatch')])
    p.add_argument('--prior-arrivals',type=Path,nargs=3,default=[ROOT/'results/generalization/arrivals-v1/arrival-manifest.json',ROOT/'results/stock-comparison/content-v2/arrival-manifest.json',ROOT/'results/resident-fix/content-v1/arrival-manifest.json'])
    p.add_argument('--content-only',action='store_true');a=p.parse_args()
    if a.output.exists() or a.public_output.exists():raise FileExistsError('Fresh output directories required')
    import tokenizers
    previous=sealed(a.resident_selection)
    for name,pin in previous['raw_files'].items():
        path=a.downloads/name
        if path.stat().st_size!=pin['bytes'] or file_sha256(path)!=pin['sha256']:raise ValueError('Changed pinned raw source '+name)
    tokenizer=tokenizers.Tokenizer.from_file(str(a.downloads/'qwen-tokenizer.json'))
    refs,provenance,counts=references(a.prior_selections,a.old_corpora,[a.stock_selection,a.resident_selection],a.downloads,tokenizer)
    raw,short=new_candidates(a.downloads,a.train_source,tokenizer);eligible,rejected=exclude_and_deduplicate(raw,refs)
    capacity=dict(Counter(r['dataset'] for r in eligible))
    print(json.dumps({'phase':'capacity','fresh_eligible':capacity,'references':counts}),flush=True)
    blocks=split_sources(eligible)
    if any(sum(r['captured_token_count'] for r in b['sources'])<128 for b in blocks):raise ValueError('Selected bank cannot provide a B128 job; no replacement')
    source_files=['resident_residual/data.py','scripts/prepare_residual_data.py','scripts/capture_residual_inputs.py','stock_comparison/data.py','scripts/prepare_stock_comparison_data.py','generalization/external_arrivals.py','followup/router_tapes.py','scripts/capture_optimization_tapes.py']
    rawfiles={**previous['raw_files'],'gsm8k-train.jsonl':dict(bytes=a.train_source.stat().st_size,sha256=file_sha256(a.train_source),url=TRAIN_URL,revision='3101c7d5072418e28b9008a6636bde82a006892c',license='MIT')}
    manifest=dict(schema='resident_residual_sources_v1',study_id='resident_residual_v1',seed=CONTENT_SEED,arrival_seed=ARRIVAL_SEED,source_model=MODEL,source_revision=REVISION,formatting=previous['formatting'],
        source_frame_change='Main GSM8K source is explicitly original train questions only; earlier test-frame results are not pooled. The fixed model is not trained and answers are never captured.',
        selection=dict(seed=CONTENT_SEED,allocation=ALLOCATION,documents_per_block=32,minimum_tokens=8,maximum_captured_tokens_per_document=512,minimum_bank_records=128,near_duplicate_jaccard=.8,shingle_width=5,eligible_counts=capacity,deduplicated_before_split=True,captured_prefix_deduplicated=True,wildchat_frame=previous['selection']['wildchat_frame'],prefix_decoder_tokenizers_version=tokenizers.__version__,main_datasets=list(DATASETS),no_routing_or_performance_selection=True,mbpp_omitted_before_timing=dict(remaining_documents=9,remaining_captured_records=171,reason='Insufficient fresh source material; no MBPP sentinel or population claim')),
        raw_files=rawfiles,prior_exclusions=provenance,exclusion_counts=counts,blocks=blocks,duplicate_rejections=rejected,short_rejections=short,preparation_source_files={name:file_sha256(ROOT/name) for name in source_files},
        capture_limit_note='Authentic separate causal-prefill layer0 hidden states. No answers/continuations/fabricated routing. Runtime consumes hidden states only and recomputes stock routing. Fresh within declared exclusion inventory, not proven absent from checkpoint pretraining.')
    manifest['content_sha256']=digest(manifest);a.output.mkdir(parents=True);a.public_output.mkdir(parents=True)
    (a.output/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    public=json.loads(json.dumps(manifest))
    for b in public['blocks']:
        for r in b['sources']:del r['text'];del r['token_ids']
    public['public_manifest_sha256']=digest(public);(a.public_output/'source-manifest.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps({'phase':'content_ready','content_sha256':manifest['content_sha256'],'groups':len(blocks),'documents':640,'tokens_by_group':{b['block_id']:sum(r['captured_token_count'] for r in b['sources']) for b in blocks}}),flush=True)
    if a.content_only:return
    old=[sealed(path) for path in a.prior_arrivals];prior=combine_prior_arrivals(old);scans=[]
    for trace,name in [('code','AzureLLMInferenceTrace_code_1week.csv'),('conversation','AzureLLMInferenceTrace_conv_1week.csv')]:
        pins={b['source_sha256'] for b in prior if b['trace']==trace}
        def progress(t,day,stat):print(json.dumps({'phase':'arrival_scan','trace':t,'day':day,**stat}),flush=True)
        scan=scan_fresh_arrivals(a.downloads/name,trace=trace,expected_sha256=next(iter(pins)),previous_blocks=prior,jobs=4096,per_day=4,seed=ARRIVAL_SEED,progress=progress)
        scans.append(scan);(a.public_output/(trace+'-scan.json')).write_text(json.dumps(scan,indent=2)+'\n')
    assigned=assign_arrivals(scans,blocks)
    arrival=dict(schema='resident_residual_arrivals_v1',study_id='resident_residual_v1',seed=ARRIVAL_SEED,jobs_per_trial=4096,source_selection_sha256=manifest['content_sha256'],previous_manifests=[dict(path=str(path.relative_to(ROOT)),sha256=file_sha256(path),content_sha256=m['content_sha256'],blocks=len(m['blocks'])) for path,m in zip(a.prior_arrivals,old)],previous_blocks_excluded=184,source_rows_and_boundaries_disjoint=True,new_calendar_days=False,
        assignment_method='Per dataset source-ID hash order; cumulative round-robin dates within split; conversation rotation2; first2 dates pilot, last5 final in each service.',blocks=assigned,assignments=[dict(block_id=b['content_group_id'],trace=b['trace'],arrival_block_id=b['id']) for b in assigned],scans={s['trace']:{k:v for k,v in s.items() if k!='retained_candidates'} for s in scans},scan_files=[dict(path=t+'-scan.json',sha256=file_sha256(a.public_output/(t+'-scan.json'))) for t in ('code','conversation')],mapping='One source timestamp releases one constructed fixed128-token layer job; original request lengths retained as metadata, not executed.',date_scope='Fresh rows within previously used public weeks/services, not new production days.')
    arrival['content_sha256']=digest(arrival);(a.public_output/'arrival-manifest.json').write_text(json.dumps(arrival,indent=2)+'\n')
    print(json.dumps({'phase':'arrivals_ready','content_sha256':arrival['content_sha256'],'blocks':len(assigned),'pilot_blocks':sum(b['split']=='pilot' for b in assigned),'final_blocks':sum(b['split']=='final' for b in assigned),'prior_blocks_excluded':184}),flush=True)

if __name__=='__main__':main()
