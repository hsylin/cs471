"""Outcome-independent 512-request cohorts and one contiguous Azure calendar.

Private text remains outside Git. This is a larger, overlapping sample, not a
fresh-content claim. The resident placement and gate thresholds remain frozen.
"""
import argparse
import csv
import json
from pathlib import Path

from full_serving.data import digest, file_digest, verify_pinned_file
from full_serving.prepare_suite import select_cohort, public_request, write_private
from generalization.external_arrivals import parse_timestamp

ROOT = Path(__file__).resolve().parents[1]
SEED = 471111512
CAP = 8192


def prepare(sources, tokenizer_dir, azure, private, output):
    from transformers import AutoTokenizer
    assert not output.exists() and not private.exists()
    assert ROOT not in private.resolve().parents
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        verify_pinned_file(tokenizer_dir / name, name)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    previous = json.loads((ROOT/'results/full-serving/suite-v1/suite-manifest.json').read_text())
    pins = {c['dataset']: c for c in previous['cohorts']}
    azure_sha = '71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f'
    assert file_digest(azure) == azure_sha
    # Fixed row position chosen without examining timings or routing. Includes
    # the following boundary row; no sorting, concatenation or repeated blocks.
    start = 100000
    timestamps = []
    with azure.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if start <= i <= start + 512:
                timestamps.append(row['TIMESTAMP'])
            if i == start + 512:
                break
    cache = {}
    times = [parse_timestamp(t, cache)[0] for t in timestamps]
    assert len(times) == 513 and times == sorted(times) and times[-1] > times[0]
    offsets = [x-times[0] for x in times[:-1]]
    horizon = times[-1]-times[0]
    private.mkdir(mode=0o700, parents=True)
    output.mkdir(parents=True)
    old = json.loads((ROOT/'results/full-serving/two-workloads-v1/frozen-v1/protocol.json').read_text())
    rates = {d['dataset']:d['arrival_rate_requests_per_s'] for d in old['datasets']}
    manifest = {'schema':'cs471_sustained_data_v1', 'seed':SEED, 'max_tokens':CAP,
        'max_model_len':32768, 'datasets':{}, 'arrival_source':{
            'sha256':azure_sha, 'trace':'Azure code one-week 2024',
            'source_rows_half_open':[start,start+512], 'boundary_row':start+512,
            'timestamps':timestamps, 'original_offsets_ns':offsets, 'original_horizon_ns':horizon},
        'limitations':['Overlaps previous cohorts; not fresh-content generalization.',
            'One session per arm/dataset; descriptive ratios only.',
            'Independent content and arrival sources; not original joint production workload.',
            'Natural output cap8192 replaces earlier16/2048; phase dominance must be remeasured.']}
    for dataset in ('longbench','numinamath'):
        path = sources / ('longbench-candidates.jsonl' if dataset=='longbench' else 'candidates/numinamath-candidates.jsonl')
        assert file_digest(path) == pins[dataset]['candidate_file_sha256']
        candidates = [json.loads(line) for line in path.open()]
        rows, summary = select_cohort(candidates,dataset,tokenizer,seed=SEED,pilot=8,final=512,max_output=CAP)
        ds={'selection':summary,'source_manifest':pins[dataset]['source_manifest_path'],
            'source_manifest_sha256':pins[dataset]['source_manifest_sha256'],
            'candidate_sha256':file_digest(path), 'requests':[public_request(r) for r in rows],
            'arrival_rate':rates[dataset], 'items':{}}
        for label, subset in [('warmup', rows[:8]),('measured',rows[8:])]:
            relative=f'{dataset}/{label}.jsonl'
            sha=write_private(private/relative,subset)
            item={'requests_file':relative, 'requests_sha256':sha,
                'request_ids':[r['id'] for r in subset], 'request_count':len(subset)}
            if label=='measured':
                target_horizon=round(512/rates[dataset]*1e9)
                calendar={'kind':'contiguous_external_uniformly_rescaled',
                    'request_ids':item['request_ids'],
                    'arrival_offsets_ns':[round(x*target_horizon/horizon) for x in offsets],
                    'horizon_ns':target_horizon,'rate_requests_per_s':rates[dataset],
                    'source_sha256':azure_sha,'source_rows_half_open':[start,start+512]}
                calendar_path=output/f'{dataset}-calendar.json'
                calendar_path.write_text(json.dumps(calendar,indent=2)+'\n')
                item.update(calendar_file=calendar_path.name,calendar_sha256=file_digest(calendar_path))
            ds['items'][label]=item
        assert not ({r['group_id'] for r in rows[:8]} & {r['group_id'] for r in rows[8:]})
        manifest['datasets'][dataset]=ds
    manifest['seal']=digest(manifest)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({d:{'pilot':8,'final':512,'lengths':v['selection']['lengths']} for d,v in manifest['datasets'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('sources','tokenizer-dir','azure','private','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.sources,a.tokenizer_dir,a.azure,a.private,a.output)
