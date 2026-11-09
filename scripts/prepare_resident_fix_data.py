#!/usr/bin/env python3
"""Prepare source-disjoint content and4096-job calendars without GPU work."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from resident_fastpath.data import CONTENT_SEED,ARRIVAL_SEED,DATASETS,ALLOCATION,STOCK_SELECTION,digest,exclude_and_deduplicate,split_sources,combine_prior_arrivals,scan_fresh_arrivals,assign_arrivals
from scripts.prepare_stock_comparison_data import prior_exclusions,candidates
from followup.router_tapes import MODEL,REVISION,file_sha256


def sealed(path):
    x=json.loads(path.read_text())
    if x['content_sha256']!=digest({k:v for k,v in x.items() if k!='content_sha256'}):raise ValueError('Prior manifest hash differs: '+str(path))
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('downloads','output','public-output','stock-selection'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-selections',type=Path,nargs=3,required=True)
    p.add_argument('--old-corpora',type=Path,nargs=3,default=[ROOT/name/'corpus/sources.json' for name in ('followup','optimization','resident_dispatch')])
    p.add_argument('--prior-arrivals',type=Path,nargs=2,default=[ROOT/'results/generalization/arrivals-v1/arrival-manifest.json',ROOT/'results/stock-comparison/content-v2/arrival-manifest.json'])
    p.add_argument('--content-only',action='store_true')
    a=p.parse_args()
    if a.output.exists() or a.public_output.exists():raise FileExistsError('Fresh output directories required')
    import tokenizers
    stock=sealed(a.stock_selection)
    if stock['schema']!='stock_comparison_sources_v1' or stock['content_sha256']!=STOCK_SELECTION:raise ValueError('Expected complete pinned stock content selection')
    for name,pin in stock['raw_files'].items():
        path=a.downloads/name
        if path.stat().st_size!=pin['bytes'] or file_sha256(path)!=pin['sha256']:raise ValueError('Changed pinned raw source '+name)
    tokenizer=tokenizers.Tokenizer.from_file(str(a.downloads/'qwen-tokenizer.json'))
    refs,provenance,counts=prior_exclusions(a.prior_selections,a.old_corpora,tokenizer)
    stock_refs=[]
    for b in stock['blocks']:
        for r in b['sources']:
            if digest(r['text'].encode())!=r['text_sha256']:raise ValueError('Stock private text hash differs')
            ids=tokenizer.encode(r['text'],add_special_tokens=False).ids[:512]
            if ids!=r['token_ids']:raise ValueError('Stock tokenizer prefix changed')
            stock_refs.append(dict(dataset=r['dataset'],source_id=r['source_id'],text=r['text'],prefix_views=[tokenizer.decode(ids,skip_special_tokens=False)]))
    if len(stock_refs)!=768 or len({(r['dataset'],r['source_id']) for r in stock_refs})!=768:raise ValueError('All768 stock documents required')
    prior_ids={(r['dataset'],r['source_id']) for r in refs}
    if any((r['dataset'],r['source_id']) in prior_ids for r in stock_refs):raise ValueError('Earlier stock selection was not source-disjoint')
    refs.extend(stock_refs)
    provenance.append(dict(path=a.stock_selection.parent.name+'/'+a.stock_selection.name,sha256=file_sha256(a.stock_selection),content_sha256=stock['content_sha256'],documents=768))
    counts.update(stock_selected_documents=768,total_prior_generalization_and_stock_documents=2695,prior_all_capture_and_selected_documents=2751)
    raw,short=candidates(a.downloads,tokenizer);eligible,rejected=exclude_and_deduplicate(raw,refs)
    capacity=dict(Counter(r['dataset'] for r in eligible))
    print(json.dumps({'phase':'capacity','fresh_eligible':capacity,'exclusions':counts}),flush=True)
    blocks=split_sources(eligible)
    source_files=['resident_fastpath/data.py','scripts/prepare_resident_fix_data.py','scripts/capture_resident_fix_inputs.py','stock_comparison/data.py','scripts/prepare_stock_comparison_data.py','generalization/external_arrivals.py','followup/router_tapes.py']
    manifest=dict(schema='resident_fix_sources_v1',study_id='resident_fix_v1',seed=CONTENT_SEED,arrival_seed=ARRIVAL_SEED,
        source_model=MODEL,source_revision=REVISION,formatting=stock['formatting'],
        selection=dict(seed=CONTENT_SEED,allocation=ALLOCATION,documents_per_block=32,minimum_tokens=8,maximum_captured_tokens_per_document=512,near_duplicate_jaccard=.8,shingle_width=5,eligible_counts=capacity,deduplicated_before_split=True,captured_prefix_deduplicated=True,wildchat_frame=stock['selection']['wildchat_frame'],prefix_decoder_tokenizers_version=tokenizers.__version__,no_routing_or_performance_selection=True,main_datasets=['gsm8k','wildchat'],mbpp_role='separate descriptive sentinel; insufficient remaining source-disjoint tasks for a main stratum'),
        raw_files=stock['raw_files'],prior_exclusions=provenance,exclusion_counts=counts,blocks=blocks,duplicate_rejections=rejected,short_rejections=short,
        preparation_source_files={name:file_sha256(ROOT/name) for name in source_files},
        capture_limit_note='Authentic separate causal-prefill layer0 hidden states. No answers/continuations/fabricated routing. GPU execution consumes hidden states only and recomputes stock routing. Fresh within project, not proven absent from checkpoint pretraining.')
    manifest['content_sha256']=digest(manifest);a.output.mkdir(parents=True);a.public_output.mkdir(parents=True)
    (a.output/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    public=json.loads(json.dumps(manifest))
    for block in public['blocks']:
        for r in block['sources']:del r['text'];del r['token_ids']
    public['public_manifest_sha256']=digest(public)
    (a.public_output/'source-manifest.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps({'phase':'content_ready','content_sha256':manifest['content_sha256'],'groups':len(blocks),'documents':sum(len(b['sources']) for b in blocks),'tokens_by_group':{b['block_id']:sum(r['captured_token_count'] for r in b['sources']) for b in blocks}}),flush=True)
    if a.content_only:return
    old=[sealed(path) for path in a.prior_arrivals];prior=combine_prior_arrivals(old)
    scans=[]
    for trace,name in (('code','AzureLLMInferenceTrace_code_1week.csv'),('conversation','AzureLLMInferenceTrace_conv_1week.csv')):
        source_hashes={b['source_sha256'] for b in prior if b['trace']==trace}
        def progress(t,day,stat):print(json.dumps({'phase':'arrival_scan','trace':t,'day':day,**stat}),flush=True)
        scan=scan_fresh_arrivals(a.downloads/name,trace=trace,expected_sha256=next(iter(source_hashes)),previous_blocks=prior,jobs=4096,per_day=4,seed=ARRIVAL_SEED,progress=progress)
        scans.append(scan);(a.public_output/(trace+'-scan.json')).write_text(json.dumps(scan,indent=2)+'\n')
    assigned=assign_arrivals(scans,blocks)
    arrival=dict(schema='resident_fix_arrivals_v1',study_id='resident_fix_v1',seed=ARRIVAL_SEED,jobs_per_trial=4096,source_selection_sha256=manifest['content_sha256'],
        previous_manifests=[dict(path=str(path.relative_to(ROOT)),sha256=file_sha256(path),content_sha256=m['content_sha256'],blocks=len(m['blocks'])) for path,m in zip(a.prior_arrivals,old)],
        previous_blocks_excluded=140,source_rows_and_boundaries_disjoint=True,new_calendar_days=False,
        assignment_method='Per dataset source-ID hash order; cumulative round-robin dates within split; conversation rotation2; first2 dates pilot, last5 final in each service.',
        blocks=assigned,assignments=[dict(block_id=b['content_group_id'],trace=b['trace'],arrival_block_id=b['id']) for b in assigned],
        scans={s['trace']:{k:v for k,v in s.items() if k!='retained_candidates'} for s in scans},scan_files=[dict(path=t+'-scan.json',sha256=file_sha256(a.public_output/(t+'-scan.json'))) for t in ('code','conversation')],
        mapping='One source timestamp releases one constructed fixed128-token layer job; original context/output lengths retained as metadata, not executed request lengths.',
        date_scope='Fresh rows and boundary rows within the already used public weeks/services; not new production days or independent organizations.')
    arrival['content_sha256']=digest(arrival);(a.public_output/'arrival-manifest.json').write_text(json.dumps(arrival,indent=2)+'\n')
    print(json.dumps({'phase':'arrivals_ready','content_sha256':arrival['content_sha256'],'blocks':len(assigned),'pilot_blocks':sum(b['split']=='pilot' for b in assigned),'final_blocks':sum(b['split']=='final' for b in assigned),'prior_blocks_excluded':140}),flush=True)

if __name__=='__main__':main()
