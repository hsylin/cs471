#!/usr/bin/env python3
"""Independent, descriptive work accounting for the frozen resident study.

No production modules, policy selection, confidence intervals or new GPU runs.
Physical-count NPZs are the work-accounting source. Nsight linked kernel rows
are reaggregated independently; their source report hashes are checked. FLOPs
are geometry-based proxies, never measured Tensor Core instructions or HBM bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HIDDEN, INTERMEDIATE, TOP_K, SLOTS, WORLD = 2048, 768, 8, 160, 2
FLOPS_PER_ROUTE = 6 * HIDDEN * INTERMEDIATE
BYTES_PER_EXPERT = 3 * HIDDEN * INTERMEDIATE * 2
SOURCES = {
    'native_kernel': 'https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/fused_moe.py',
    'profiling_guide': 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html',
    'metric_query_and_replay_cli': 'https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html',
    'counter_permission': 'https://developer.nvidia.com/ERR_NVGPUCTRPERM',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def stats(values):
    values = [float(x) for x in values]
    require(bool(values), 'No observations')
    require(all(np.isfinite(x) for x in values), 'Nonfinite observation')
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values))


def work_metrics(counts, batch_tokens, config):
    """One global physical-assignment vector; do not multiply by replicated ranks."""
    c = np.asarray(counts)
    require(c.shape == (SLOTS,) and np.issubdtype(c.dtype, np.integer)
            and np.all(c >= 0), 'Require160 nonnegative integer slot counts')
    require(int(c.sum()) == batch_tokens * TOP_K, 'Top-k assignment conservation failed')
    bm, bn, bk = (int(config[k]) for k in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K'))
    require(min(bm, bn, bk) > 0, 'Invalid tiles')
    require(HIDDEN % bn == 0 and 2 * INTERMEDIATE % bn == 0
            and HIDDEN % bk == 0 and INTERMEDIATE % bk == 0,
            'This work formula requires exact N/K divisibility')
    require(config.get('SPLIT_K', 1) == 1, 'Split-K accounting not supported')
    per_rank = c.reshape(WORLD, SLOTS // WORLD)
    tiles = ((per_rank + bm - 1) // bm).sum(axis=1)
    tokens = per_rank.sum(axis=1)
    active = (per_rank > 0).sum(axis=1)
    padded = tiles * bm
    useful = tokens * FLOPS_PER_ROUTE
    implied = padded * FLOPS_PER_ROUTE
    return {
        'per_rank_tokens': tokens.tolist(), 'per_rank_active_copies': active.tolist(),
        'per_rank_m_tiles': tiles.tolist(), 'per_rank_padded_rows': padded.tolist(),
        'per_rank_useful_gemm_flops': useful.tolist(),
        'per_rank_tile_implied_gemm_flops_proxy': implied.tolist(),
        'total_active_copies': int(active.sum()), 'peak_rank_active_copies': int(active.max()),
        'total_m_tiles': int(tiles.sum()), 'peak_rank_m_tiles': int(tiles.max()),
        'total_padded_rows': int(padded.sum()), 'peak_rank_tokens': int(tokens.max()),
        'peak_rank_padded_rows': int(padded.max()),
        'useful_row_fraction': float(tokens.sum() / padded.sum()),
        'useful_gemm_flops': int(useful.sum()),
        'tile_implied_gemm_flops_proxy': int(implied.sum()),
        'padded_only_gemm_flops_proxy': int(implied.sum() - useful.sum()),
        'active_expert_parameter_footprint_bytes_proxy': int(active.sum()) * BYTES_PER_EXPERT,
    }


def summarize_work(rows):
    scalar = [k for k, v in rows[0].items() if not isinstance(v, list)]
    result = {k: stats(row[k] for row in rows) for k in scalar}
    result['per_rank'] = [{k: stats(row[k][rank] for row in rows)
                           for k in rows[0] if k.startswith('per_rank_')}
                          for rank in range(WORLD)]
    return result


def resolve_artifact(entry, roots):
    name = Path(entry['path']).name
    paths = list(dict.fromkeys(path for root in roots for path in Path(root).rglob(name)))
    require(bool(paths), f'Missing artifact {name}')
    require(all(digest(path) == entry['sha256'] for path in paths), f'Artifact hash mismatch {name}')
    return paths[0]


def read_records(paths):
    rows = []
    for path in paths:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt') as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    require(len({r['record_id'] for r in rows}) == len(rows), 'Duplicate record IDs')
    return rows


def record_analysis(rows, roots):
    grouped = defaultdict(list)
    for row in rows:
        require(row['status'] == 'complete' and row['correctness_pass'] is True
                and row['arm'] in ('native', 'online'), 'Invalid record')
        require(row['migration_bytes'] == 0 and row['source_map'] == row['final_map'], 'Placement changed')
        key = tuple(row[k] for k in ('config_hash', 'phase', 'block_id', 'regime', 'batch_tokens', 'arm'))
        grouped[key].append(row)
    states, mapper_rows, artifacts = [], {}, {}
    for key, repeated in sorted(grouped.items()):
        row = repeated[0]
        for other in repeated[1:]:
            for field in ('assignment_sha256', 'source_map_sha256', 'source_weights_sha256',
                          'resolved_kernel_config', 'assignment_metrics', 'selector_diagnostic'):
                require(row[field] == other[field], f'Repetitions differ in {field}')
            require(row['assignment_artifact']['sha256'] == other['assignment_artifact']['sha256'],
                    'Repetitions use different input/assignment artifact')
        path = resolve_artifact(row['assignment_artifact'], roots)
        artifacts[str(path)] = row['assignment_artifact']['sha256']
        with np.load(path, allow_pickle=False) as data:
            counts = data[row['arm'] + '_physical_counts'].copy()
            require(np.array_equal(data['physical_to_logical_map'], row['source_map']), 'Artifact map differs')
        if row['phase'] == 'final':
            sequence = list(range(32, 544))
            require(counts.shape == (544, SLOTS), 'Wrong final input count')
        else:
            require(row['phase'] == 'validation' and counts.shape == (48, SLOTS), 'Wrong development inputs')
            sequence = [16 + i % 32 for i in range(256)]
        require(len(row['assignment_metrics']) == len(sequence) == row['steps_completed'], 'Wrong trajectory length')
        metrics = [work_metrics(counts[i], row['batch_tokens'], row['resolved_kernel_config']) for i in sequence]
        for saved, calculated in zip(row['assignment_metrics'], metrics):
            for field in ('per_rank_tokens', 'per_rank_active_copies', 'per_rank_m_tiles', 'per_rank_padded_rows'):
                require(saved[field] == calculated[field], f'Saved metric disagrees with NPZ: {field}')
        states.append({k: row[k] for k in ('config_hash', 'phase', 'block_id', 'regime', 'batch_tokens', 'arm')} |
                      dict(repetitions_checked=len(repeated), steps=len(sequence),
                           distinct_raw_indices=len(set(sequence)), resolved_kernel_config=row['resolved_kernel_config'],
                           metrics=summarize_work(metrics)))
        if row['selector_diagnostic']:
            mapper_key = tuple(row[k] for k in ('block_id', 'regime', 'batch_tokens'))
            previous = mapper_rows.setdefault(mapper_key, row['selector_diagnostic'])
            require(previous == row['selector_diagnostic'], 'Duplicated mapper diagnostics differ')
    summaries = []
    for group in sorted({(r['phase'], r['regime'], r['batch_tokens'], r['arm']) for r in states}):
        chosen = [r for r in states if tuple(r[k] for k in ('phase', 'regime', 'batch_tokens', 'arm')) == group]
        summaries.append(dict(zip(('phase', 'regime', 'batch_tokens', 'arm'), group)) |
                         dict(source_states=len(chosen), metrics={k: stats(r['metrics'][k]['mean'] for r in chosen)
                         for k in chosen[0]['metrics'] if k != 'per_rank'}))
    contrasts = []
    for phase, regime, batch in sorted({(s['phase'], s['regime'], s['batch_tokens']) for s in summaries}):
        paired = {s['arm']: s for s in summaries if (s['phase'], s['regime'], s['batch_tokens']) == (phase, regime, batch)}
        require(set(paired) == {'native', 'online'}, 'Missing architecture comparator')
        contrast = dict(phase=phase, regime=regime, batch_tokens=batch)
        for metric in ('total_active_copies', 'peak_rank_active_copies', 'total_m_tiles',
                       'peak_rank_padded_rows', 'tile_implied_gemm_flops_proxy', 'peak_rank_tokens'):
            contrast[metric + '_online_over_native'] = paired['online']['metrics'][metric]['mean'] / paired['native']['metrics'][metric]['mean']
        contrasts.append(contrast)
    mapper = []
    for batch in sorted({k[2] for k in mapper_rows}):
        for enabled in (False, True):
            values = defaultdict(list)
            for key, diag in mapper_rows.items():
                if key[2] == batch:
                    for item in diag:
                        if item['record'] == enabled:
                            values[item['arm']].extend(1e6 * x for x in item['per_rank_stream_seconds_per_call'])
            mapper.append(dict(batch_tokens=batch, recording=enabled,
                               native_us=stats(values['native']), online_us=stats(values['online']),
                               difference_of_means_us=statistics.mean(values['online'])-statistics.mean(values['native'])))
    return dict(records_checked=len(rows), unique_state_arm_trajectories=len(states), states=states,
                summaries=summaries, contrasts=contrasts, mapper_only_diagnostics=mapper, artifact_sha256=artifacts)


def union_ns(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        require(b >= a, 'Negative kernel interval')
        total += b - a if end is None else max(0, b - max(a, end))
        end = b if end is None else max(end, b)
    return total


def profile_analysis(path, root):
    data = json.loads(Path(path).read_text())
    profiles, source_hashes = [], {}
    for p in data['profiles']:
        for oldpath, expected in p['sources'].items():
            original = Path(oldpath)
            if original.suffix == '.sqlite':
                continue  # not packaged; raw .nsys-rep can regenerate this intermediate
            matches = list(Path(root).rglob(original.name))
            require(len(matches) == 1 and digest(matches[0]) == expected, 'Raw profile provenance hash mismatch')
            source_hashes[str(matches[0])] = expected
        artifact = resolve_artifact(p['assignment_artifact'], [root])
        source_hashes[str(artifact)] = p['assignment_artifact']['sha256']
        method = 'native' if p['arm'] in ('native', 'pre_native') else p['arm']
        with np.load(artifact, allow_pickle=False) as arrays:
            physical = arrays[method + '_physical_ids'].copy()
            logical = arrays['logical_ids'].copy()
            mapping = arrays['physical_to_logical_map'].copy()
        require(np.array_equal(mapping[physical], logical), 'Profile assignment residency/logical IDs failed')
        steps = []
        for index in range(32):
            kernels = sorted((k for k in p['kernels'] if k['profile_step'] == index), key=lambda k: k['gpu_start_epoch_ns'])
            for k in kernels:
                require(abs(k['duration_us'] - (k['gpu_end_epoch_ns']-k['gpu_start_epoch_ns'])/1000) < 1e-8,
                        'Kernel duration mismatch')
            gemms = [k for k in kernels if k['kernel'] == 'fused_moe_kernel']
            expert = [k for k in kernels if k['phase'] == 'fused_experts']
            nccl = [k for k in kernels if k['kernel'].startswith('ncclDevKernel_AllReduce')]
            require(len(gemms) == 2 and len(nccl) == 1 and len(kernels) == 11, 'Unexpected profiled kernel sequence')
            counts = np.bincount(physical[16 + index].ravel(), minlength=SLOTS)
            work = work_metrics(counts, p['batch_tokens'], p['resolved_kernel_config'])
            rank = p['rank']
            duration = sum(k['duration_us'] for k in gemms)
            union = union_ns((k['gpu_start_epoch_ns'], k['gpu_end_epoch_ns']) for k in expert)/1000
            envelope = (max(k['gpu_end_epoch_ns'] for k in expert)-min(k['gpu_start_epoch_ns'] for k in expert))/1000
            steps.append(dict(step=index, rank=rank, gemm_first_us=gemms[0]['duration_us'],
                gemm_second_us=gemms[1]['duration_us'], gemm_sum_us=duration,
                mapper_kernel_us=sum(k['duration_us'] for k in kernels if k['phase']=='mapping'),
                expert_kernel_union_us=union, expert_interkernel_gap_us=envelope-union,
                nccl_kernel_residence_us=nccl[0]['duration_us'],
                expert_end_ns=max(k['gpu_end_epoch_ns'] for k in expert),
                nccl_start_ns=nccl[0]['gpu_start_epoch_ns'], nccl_end_ns=nccl[0]['gpu_end_epoch_ns'],
                rank_active_copies=work['per_rank_active_copies'][rank],
                rank_m_tiles=work['per_rank_m_tiles'][rank], rank_padded_rows=work['per_rank_padded_rows'][rank],
                rank_tokens=work['per_rank_tokens'][rank],
                rank_useful_fraction=work['per_rank_tokens'][rank]/work['per_rank_padded_rows'][rank],
                useful_gemm_tflops_per_kernel_residence=work['per_rank_useful_gemm_flops'][rank]/duration/1e6,
                tile_proxy_tflops_per_kernel_residence=work['per_rank_tile_implied_gemm_flops_proxy'][rank]/duration/1e6,
                gemm_grids=[k['grid'] for k in gemms], work=work))
        fields = [k for k,v in steps[0].items() if isinstance(v,(int,float)) and k not in ('step','rank') and not k.endswith('_ns')]
        profiles.append(dict(profile_stem=p['profile_stem'], batch_tokens=p['batch_tokens'], arm=p['arm'], rank=p['rank'],
                             resolved_kernel_config=p['resolved_kernel_config'], steps=steps,
                             metrics={k:stats(s[k] for s in steps) for k in fields}))
    pairs = []
    for batch, arm in sorted({(p['batch_tokens'],p['arm']) for p in profiles}):
        ranks = sorted((p for p in profiles if (p['batch_tokens'],p['arm'])==(batch,arm)),key=lambda p:p['rank'])
        require([p['rank'] for p in ranks] == [0,1], 'Missing rank profile')
        rows=[]
        for a,b in zip(ranks[0]['steps'],ranks[1]['steps']):
            rows.append(dict(step=a['step'], largest_gemm_residence_rank=int(b['gemm_sum_us']>a['gemm_sum_us']),
                later_expert_completion_rank=int(b['expert_end_ns']>a['expert_end_ns']),
                max_rank_gemm_sum_us=max(a['gemm_sum_us'],b['gemm_sum_us']),
                expert_completion_skew_us=abs(a['expert_end_ns']-b['expert_end_ns'])/1000,
                nccl_arrival_skew_us=abs(a['nccl_start_ns']-b['nccl_start_ns'])/1000,
                nccl_tail_after_both_arrivals_us=(max(a['nccl_end_ns'],b['nccl_end_ns'])-max(a['nccl_start_ns'],b['nccl_start_ns']))/1000))
        pairs.append(dict(batch_tokens=batch,arm=arm,steps=rows,
                          largest_gemm_residence_rank_counts=dict(Counter(x['largest_gemm_residence_rank'] for x in rows)),
                          later_expert_completion_rank_counts=dict(Counter(x['later_expert_completion_rank'] for x in rows)),
                          metrics={k:stats(x[k] for x in rows) for k in rows[0] if k.endswith('_us')}))
    contrasts=[]
    for batch in (128,1024):
        base=next(x for x in pairs if x['batch_tokens']==batch and x['arm']=='pre_native')
        for arm in ('native','one_copy_home','token_balanced'):
            value=next(x for x in pairs if x['batch_tokens']==batch and x['arm']==arm)
            contrasts.append(dict(batch_tokens=batch,arm=arm,reference='pre_native',
                max_rank_gemm_residence_ratio=value['metrics']['max_rank_gemm_sum_us']['mean']/base['metrics']['max_rank_gemm_sum_us']['mean']))
    return dict(profiles=profiles,paired_rank_summaries=pairs,contrasts=contrasts,
                source_sha256=source_hashes,collection_warnings=data['collection_warnings'],
                scope='Existing development profiles: one bank/domain, four offline-dispatch arms, two batches, both ranks. No online-selector Nsight timeline was collected.',
                source_limitation='Kernel linkage reused from hash-pinned independent Nsight SQLite extraction; this script independently recomputes kernel and work aggregates and verifies original report/metadata hashes. It does not reparse proprietary .nsys-rep files.')


def markdown(report):
    lines=['# Resident-dispatch architectural accounting', '',
        'Descriptive follow-up to the completed frozen experiment; no new timing, threshold, policy selection or significance test.', '',
        'Each top-8 assignment performs two GEMMs: H→2I costs 4HI FLOPs and I→H costs 2HI. With H=2048, I=768, the useful total is 9,437,184 FLOPs per assignment (multiply-add counts as two). Activation, reduction and routing arithmetic are excluded. We compute tile-implied work by replacing useful rows with Σ ceil(tokens_in_physical_slot/BM)×BM. N/K dimensions divide the recorded tiles exactly. This is a padded-GEMM work proxy, not a hardware instruction count.', '',
        'Replicated ranks hold the same global assignment list. Each slot is owned by one rank: global useful work is B×8×6HI, not twice that quantity. Active-expert parameter footprint is a separate residency-access proxy; it is not measured HBM traffic.', '',
        '## Saved final assignments', '',
        '| Regime | Batch | Arm | Active copies | M tiles | Useful-row fraction | Useful GFLOPs | Tile-implied GFLOPs |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in report['record_accounting']['summaries']:
        if row['phase']!='final':continue
        m=row['metrics'];mean=lambda k:m[k]['mean']
        lines.append(f"| {row['regime']} | {row['batch_tokens']} | {row['arm']} | {mean('total_active_copies'):.2f} | {mean('total_m_tiles'):.2f} | {mean('useful_row_fraction'):.3%} | {mean('useful_gemm_flops')/1e9:.3f} | {mean('tile_implied_gemm_flops_proxy')/1e9:.3f} |")
    lines += ['', 'Each row averages source-state means; identical technical-repetition assignment artifacts are verified and counted once. Final summaries use all 512 measured inputs per state. Work proxies come from saved per-slot assignment arrays checked against recorded metrics, not from elapsed-time inference.', '',
        '## Existing development kernel profiles', '',
        '| Batch | Arm | Rank | Two GEMMs (µs) | Mapper kernel (µs) | Expert inter-kernel gaps (µs) | NCCL residence (µs) | Useful / tile-proxy TFLOP/s over GEMM residence |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for p in report['profile_accounting']['profiles']:
        m=p['metrics'];v=lambda k:m[k]['mean']
        lines.append(f"| {p['batch_tokens']} | {p['arm']} | {p['rank']} | {v('gemm_sum_us'):.2f} | {v('mapper_kernel_us'):.2f} | {v('expert_interkernel_gap_us'):.2f} | {v('nccl_kernel_residence_us'):.2f} | {v('useful_gemm_tflops_per_kernel_residence'):.1f} / {v('tile_proxy_tflops_per_kernel_residence'):.1f} |")
    lines += ['', 'These are kernel-residence and geometry statistics from one development domain, not wall-time fractions or measured hardware utilization. `one_copy_home` here uses the offline DP assignment; it is not the online greedy kernel. The raw linked rows contain two fused GEMMs per input. Largest GEMM-residence rank and later expert-completion rank are both retained in JSON; profiling-induced rank start/arrival skew prevents treating either alone as an uninstrumented critical path. NCCL residence includes possible peer waiting.', '',
        '## Online selector diagnostic', '',
        '| Batch | Recording | Native stream µs/call | Online stream µs/call | Difference µs |',
        '|---:|---|---:|---:|---:|']
    for d in report['record_accounting']['mapper_only_diagnostics']:
        lines.append(f"| {d['batch_tokens']} | {d['recording']} | {d['native_us']['mean']:.3f} | {d['online_us']['mean']:.3f} | {d['difference_of_means_us']:.3f} |")
    lines += ['', 'CUDA-stream intervals include launch gaps and use the existing fixed diagnostic order. They directly show combined selector/launch cost, but cannot isolate atomic-counter overhead or predict a fused implementation’s gain. Recording-on versus recording-off differences are not a randomized estimate of counter cost.', '',
        '## Hardware-counter gap and bounded diagnostic', '',
        'The installed absolute-path Nsight Compute is 2024.2.1.0; its CPU-only GH100 catalog query confirms all requested metric names. GPU counter permission and compatibility with the actual job remain unverified. A login-node query is not a successful counter capture. No GPU jobs were launched by this analysis.', '',
        'A bounded follow-up would use only old development input raw16, both B128/B1024, both ranks, native/online, with identical map/weights and a warmed path. Collect the two expert GEMMs per case, explicitly excluding NCCL replay. Query the local metric catalog, then request DRAM read/write bytes, DRAM/SM throughput percentages, Tensor-pipe active cycles, achieved warp occupancy, issue activity, L2 hit ratio and kernel duration. Keep the counter run separate from primary latency. Use a one-kernel permission/compatibility probe first; if denied, retain the failure and request normal administrator support rather than changing system configuration.', '',
        'Nsight replay can serialize kernels, change cache state and add passes. The old installed tool must not be assumed to support newer multi-process communicator features. Filter to local expert GEMMs; preserve/declare cache policy, use clock-control none if clocks cannot be set, and log tool/driver/kernel identity. One input is mechanism evidence only, not a new population performance claim.', '',
        'Required metrics (GH100 names checked against the installed tool):', '']
    for metric in report.get('ncu_availability',{}).get('metrics',{}):lines.append(f'- `{metric}`')
    lines += ['', 'Measured DRAM bytes would permit useful-FLOPs/DRAM-byte arithmetic intensity. Tensor-pipe active cycles are not identical to achieved FLOP utilization; SM throughput is a composite peak-normalized metric, not `nvidia-smi` utilization; occupancy is active warps relative to the supported maximum. Report all with kernel identity and duration, not as interchangeable percentages.', '',
        f"NVIDIA explains replay/cache effects in the [profiling guide]({SOURCES['profiling_guide']}) and supported metric/replay options in the [CLI reference]({SOURCES['metric_query_and_replay_cli']}). Counter access can require administrator configuration; see [NVIDIA’s permission documentation]({SOURCES['counter_permission']}). The pinned [vLLM kernel source]({SOURCES['native_kernel']}) defines the expert GEMM path.", '',
        'The completed study’s acceptance/failure remains unchanged. These descriptive metrics explain candidate mechanisms and identify missing counters; they do not turn a 3.4% result into a passed 5% criterion.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,nargs='+',required=True)
    parser.add_argument('--inputs-root',type=Path,nargs='+',required=True)
    parser.add_argument('--profile-analysis',type=Path,required=True)
    parser.add_argument('--profile-root',type=Path,required=True)
    parser.add_argument('--ncu-availability',type=Path)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--output-markdown',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output_json.exists() and not args.output_markdown.exists(), 'Refusing existing outputs')
    report=dict(schema_version=1,analysis_kind='descriptive_architectural_work_accounting',
        hardware_counters_collected=False,geometry=dict(hidden=HIDDEN,intermediate=INTERMEDIATE,top_k=TOP_K,
        physical_slots=SLOTS,world_size=WORLD,flops_per_assignment=FLOPS_PER_ROUTE),
        input_sha256={str(p):digest(p) for p in [*args.records,args.profile_analysis]},
        script_sha256=digest(__file__),sources=SOURCES,
        record_accounting=record_analysis(read_records(args.records),args.inputs_root),
        profile_accounting=profile_analysis(args.profile_analysis,args.profile_root))
    if args.ncu_availability:
        report['ncu_availability']=json.loads(args.ncu_availability.read_text())
        report['input_sha256'][str(args.ncu_availability)]=digest(args.ncu_availability)
    for out in (args.output_json,args.output_markdown):out.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    args.output_markdown.write_text(markdown(report))
    print(json.dumps(dict(records=report['record_accounting']['records_checked'],
                         unique_state_arm_trajectories=report['record_accounting']['unique_state_arm_trajectories'],
                         profiles=len(report['profile_accounting']['profiles']),output=str(args.output_json))))


if __name__=='__main__':main()
