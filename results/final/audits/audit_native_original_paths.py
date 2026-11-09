"""Read-only independent audit; run against the owner's remote final artifacts."""
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path('/work/cs-471/skewbench-hlin/repo')
RESULTS = Path('/work/cs-471/skewbench-hlin/results/final')
EXPECTED = '894bf6474c57655c15d41709ffdb9e84f9e9e16aaece6d7de6a4fc88c94594cb'
issues = []
complete = '--complete' in sys.argv


def check(value, label):
    if not value:
        issues.append(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


source_paths = sorted((ROOT / 'skewbench').glob('*.py')) + [ROOT / 'skewbench/native_sources.json']
source_hashes = {'skewbench/' + p.name: sha(p.read_bytes()) for p in source_paths}
harness = sha(json.dumps(source_hashes, sort_keys=True, separators=(',', ':')).encode())
check(harness == EXPECTED, 'actual harness hash')
native = json.loads((ROOT / 'skewbench/native_sources.json').read_text())
vllm_root = Path(importlib.util.find_spec('vllm').origin).parent
for rel, expected in native['sha256'].items():
    check(sha((vllm_root / Path(rel).relative_to('vllm')).read_bytes()) == expected, 'native source ' + rel)
manifest = json.loads((ROOT / 'configs/frozen/final.json').read_text())
manifest_digest = sha(json.dumps({k: v for k, v in manifest.items() if k != 'freeze'},
                                sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
check(manifest_digest == manifest['freeze']['sha256'], 'manifest digest')
check(manifest['environment']['harness_sha256'] == harness, 'manifest harness binding')
rows = []
partial_lines = 0
lines = (RESULTS / 'records-final.jsonl').read_text().splitlines()
for i, line in enumerate(lines):
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        partial_lines += 1
        check(i == len(lines) - 1, 'nonterminal malformed record')
check(len({r.get('record_id') for r in rows}) == len(rows), 'duplicate record IDs')
W = manifest['trace']['window_steps']
R = manifest['policy']['rebalance_interval']
N = manifest['trace']['num_steps']
G = manifest['environment']['world_size']
E = manifest['trace']['num_experts']
K = manifest['trace']['top_k']
decisions = list(range(R - 1, N - 1, R))
num_decisions = len(decisions)
selected = {}
decisions_checked = 0
all_artifact_rows = []
for r in rows:
    tag = r.get('record_id', '<missing>')[:12]
    check(r.get('status') == 'complete' and r.get('correctness_pass') is True, tag + ' completion')
    if r.get('status') != 'complete':
        continue
    for key, expected in [('config_hash', manifest_digest), ('harness_sha256', harness),
                          ('eplb_mode', 'synchronous'), ('dispatch_backend', 'replicated_input_allreduce'),
                          ('weight_communicator', 'torch_nccl'), ('native_threshold_active', False),
                          ('correctness_forward_batch_tokens', r['batch_tokens']),
                          ('correctness_reference_tokens', 4), ('steps_completed', N),
                          ('raw_trace_steps', N + 2 * W), ('measured_horizon_steps', N),
                          ('terminal_proposals_suppressed', 1), ('replica_budget_actual', 32)]:
        check(r.get(key) == expected, tag + ' ' + key)
    check(r.get('harness_file_sha256') == source_hashes, tag + ' file hashes')
    check(r['schema_version'] == 1 and r['split'] == 'final' and
          r['measurement_kind'] == 'gpu' and r['evidence_eligible'] is True, tag + ' evidence schema')
    check(r['policy_config'] == manifest['policy'], tag + ' frozen policy')
    check(r['weights_source']['source_model'] == manifest['environment']['model_id'] and
          r['weights_source']['source_revision'] == manifest['environment']['model_revision'] and
          r['weights_source']['layer'] == manifest['environment']['layer_id'] and
          r['weights_source']['weights_kind'] == 'pretrained', tag + ' checkpoint provenance')
    warm = r.get('process_transport_warmup', {})
    check(warm.get('completed') is True and warm.get('migration_count') == 2 and
          warm.get('shared_native_eplb_process_group_verified') is True and
          warm.get('original_map_and_weights_restored') is True, tag + ' transport warmup')
    check(all(b > 0 for b in warm.get('forward_sent_bytes_by_rank', []) +
              warm.get('rollback_sent_bytes_by_rank', [])), tag + ' both warmup directions')
    check(warm.get('sent_bytes') == sum(warm.get('sent_bytes_by_rank', [])) > 0,
          tag + ' warmup byte sum')
    check(np.isfinite(r['correctness_max_abs_error']), tag + ' finite numerical error')
    check(r['correctness_atol'] == manifest['correctness_criteria']['absolute_tolerance'] and
          r['correctness_rtol'] == manifest['correctness_criteria']['relative_tolerance'], tag + ' numerical tolerance')
    check(np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0 and
          len(r['per_rank_elapsed_seconds']) == G and len(r['step_seconds_slowest_rank']) == N,
          tag + ' completed timing schema')
    b4 = r['method'] == 'B4'
    check(r['logical_histogram_collection_enabled'] == b4, tag + ' history mode')
    check(r['logical_histogram_measured_capture_steps'] == (N // R * W if b4 else 0), tag + ' measured captures')
    check(r['logical_histogram_warmup_capture_steps'] == (2 * W if b4 else 0), tag + ' warmup captures')
    check(r['previous_window_timestamp_checks'] == (num_decisions if b4 else 0), tag + ' timestamp checks')
    check(r['observation_conservation_checks'] == num_decisions * (2 if b4 else 1), tag + ' exact window checks')
    log = r['decision_log']
    check([d['step'] for d in log] == decisions, tag + ' schedule')
    check(r['accepted'] == sum(d['allowed'] for d in log), tag + ' admitted count')
    check(r['rejected'] == sum(not d['allowed'] for d in log), tag + ' rejected count')
    check(r['migration_bytes'] == sum(d['transfer_sent_bytes'] for d in log), tag + ' total payload')
    current = np.concatenate((np.arange(E), np.arange(32))).tolist()
    for d in log:
        decisions_checked += 1
        dt = tag + ':' + str(d['step'])
        check(d['current_map'] == current, dt + ' map chain')
        check(d['committed_map'] == (d['proposed_map'] if d['allowed'] else d['current_map']), dt + ' commit map')
        check(d['commit_completed'] is True, dt + ' commit completion')
        check(d['committed_map_sha256'] == sha(np.asarray(d['committed_map'], dtype=np.int64).tobytes()), dt + ' map digest')
        check(d['transfer_sent_bytes'] == sum(d['transfer_sent_bytes_by_rank']), dt + ' rank payload sum')
        if not d['allowed']:
            check(d['transfer_sent_bytes'] == 0, dt + ' rejection payload')
        physical = np.asarray(d['physical_slot_assignments_window'], dtype=np.int32)
        check(physical.shape == (W, E + 32), dt + ' physical shape')
        check(d['physical_slot_assignments_sha256'] == sha(physical.tobytes()), dt + ' physical digest')
        check(d['physical_counter_input_multiplicity'] == G, dt + ' physical multiplicity')
        expected_logical = np.zeros(E, dtype=np.int64)
        np.add.at(expected_logical, np.asarray(d['current_map']), physical.sum(axis=0))
        check(np.array_equal(expected_logical * G, d['current_counts']), dt + ' physical-to-logical counts')
        owners = physical.reshape(W, G, -1).sum(axis=2)
        check(np.array_equal(owners, d['physical_owner_assignments_per_step']), dt + ' ownership counts')
        check(np.all(owners.sum(axis=1) == r['batch_tokens'] * K), dt + ' per-step assignments')
        current = d['committed_map']
    check(r['final_map'] == current, tag + ' final map')
    key = (r['regime'], r['trace_variant'], r['batch_tokens'], r['method'])
    selected.setdefault(key, r)
    all_artifact_rows.append(r)

job_fields = ('regime', 'trace_variant', 'phase_steps', 'method', 'seed', 'repetition', 'batch_tokens')
expected_jobs = set()
for case in manifest['matrix']['cases']:
    for method in manifest['matrix']['methods']:
        for seed in manifest['matrix']['seeds']:
            for rep in range(manifest['matrix']['repetitions']):
                for batch in manifest['matrix']['batch_tokens']:
                    job = {**case, 'method': method, 'seed': seed, 'repetition': rep, 'batch_tokens': batch}
                    expected_jobs.add(tuple(job[k] for k in job_fields))
actual_jobs = set()
for r in rows:
    job = {k: r[k] for k in job_fields}
    actual_jobs.add(tuple(job[k] for k in job_fields))
    check(r['record_id'] == sha(json.dumps([manifest_digest, job], sort_keys=True).encode()),
          r['record_id'][:12] + ' record ID')
check(actual_jobs <= expected_jobs, 'unexpected primary matrix cells')
if complete:
    check(actual_jobs == expected_jobs and len(rows) == len(expected_jobs), 'complete primary matrix')
    check(partial_lines == 0, 'no partial final record')

branches = []
branch_path = RESULTS / 'branches.jsonl'
if branch_path.exists():
    branches = [json.loads(line) for line in branch_path.read_text().splitlines() if line.strip()]
if complete:
    check(len(branches) == 8, 'eight diagnostic pairs')
branch_orders = []
branch_keys = set()
for r in branches:
    tag = r['record_id'][:12]
    check(r['schema_version'] == 1 and r['status'] == 'complete' and r['correctness_pass'] is True and
          r['measurement_kind'] == 'gpu_branch' and r['split'] == 'final', tag + ' branch schema')
    check(r['harness_sha256'] == harness and r['harness_file_sha256'] == source_hashes and
          r['config_hash'] == manifest_digest, tag + ' branch provenance')
    check(r['eplb_mode'] == 'synchronous' and r['dispatch_backend'] == 'replicated_input_allreduce', tag + ' branch path')
    warm = r['process_transport_warmup']
    check(warm['completed'] is True and warm['migration_count'] == 2 and
          warm['shared_native_eplb_process_group_verified'] is True and
          warm['original_map_and_weights_restored'] is True and
          all(x > 0 for x in warm['forward_sent_bytes_by_rank'] + warm['rollback_sent_bytes_by_rank']), tag + ' branch warmup')
    check(r['decision_step'] == 63 and r['decision_trace_step'] == 95 and
          r['future_steps'] == 128 and r['future_rebalancing_enabled'] is False, tag + ' branch horizon')
    check(r['diagnostic_prefix_histogram_capture_steps'] == 16 and
          r['previous_window_timestamp_checks'] == 1, tag + ' branch prefix collection')
    check(r['source_map_sha256'] == sha(np.asarray(r['current_map'], dtype=np.int64).tobytes()), tag + ' branch source map')
    check(r['candidate_map_sha256'] == sha(np.asarray(r['proposed_map'], dtype=np.int64).tobytes()), tag + ' branch candidate map')
    check(len(r['source_weight_rank_sha256']) == G and all(re.fullmatch('[0-9a-f]{64}', h)
          for h in r['source_weight_rank_sha256']), tag + ' branch rank weight digests')
    check(r['source_weights_sha256'] == sha(json.dumps(r['source_weight_rank_sha256']).encode()), tag + ' branch weight aggregate')
    check(sorted(r['branch_order']) == ['allow', 'skip'] and
          [b['branch'] for b in r['branches']] == r['branch_order'], tag + ' branch execution order')
    branch_orders.append(r['branch_order'][0])
    key = (r['regime'], r['trace_variant'], r['batch_tokens'], r['seed'], r['repetition'], r['decision_step'])
    branch_keys.add(key)
    branch_job = {k: r[k] for k in job_fields}
    branch_job.update(decision_step=r['decision_step'], branch_order=r['branch_order'])
    check(r['record_id'] == sha(json.dumps([manifest_digest, branch_job], sort_keys=True).encode()), tag + ' branch record ID')
    for b in r['branches']:
        check(b['correctness_pass'] is True and b['future_steps'] == 128 and
              b['correctness_forward_batch_tokens'] == r['batch_tokens'] and
              b['correctness_reference_tokens'] == 4 and np.isfinite(b['correctness_max_abs_error']), tag + ' branch numerical check')
        check(b['correctness_atol'] == manifest['correctness_criteria']['absolute_tolerance'] and
              b['correctness_rtol'] == manifest['correctness_criteria']['relative_tolerance'], tag + ' branch tolerance')
        check(np.isfinite(b['elapsed_seconds']) and b['elapsed_seconds'] > 0, tag + ' branch completed timing schema')
        for field in ('source_map_sha256', 'source_weights_sha256', 'candidate_map_sha256', 'trace_sha256'):
            check(re.fullmatch('[0-9a-f]{64}', b[field]) and b[field] == r[field], tag + ' common branch state ' + field)
        if b['branch'] == 'skip':
            check(b['migration_bytes'] == 0, tag + ' branch skip bytes')
        else:
            check(b['migration_bytes'] >= 0, tag + ' branch allow bytes')
    all_artifact_rows.append(r)
if complete:
    expected_branch_keys = {(c['regime'], c['trace_variant'], batch, 1001, 0, 63)
                           for c in manifest['diagnostics']['cases']
                           for batch in manifest['diagnostics']['batch_tokens']}
    check(branch_keys == expected_branch_keys and len(branches) == len(branch_keys), 'complete branch states')
    check(collections.Counter(branch_orders) == {'allow': 4, 'skip': 4}, 'balanced branch order')

preflight = []
if complete:
    preflight_path = RESULTS / 'preflight.jsonl'
    if preflight_path.exists():
        preflight = [json.loads(line) for line in preflight_path.read_text().splitlines() if line.strip()]
        for r in preflight:
            check(r['status'] == 'complete' and r['correctness_pass'] is True, 'preflight success')
            all_artifact_rows.append(r)

groups = collections.defaultdict(list)
for r in (all_artifact_rows if complete else selected.values()):
    groups[r['trace_artifact']['path']].append(r)
tape_count = 0
exact_vector_checks = 0
artifact_inventory = []
for filename, group in groups.items():
    path = Path(filename)
    artifact_digest = sha(path.read_bytes())
    artifact_inventory.append({
        'basename': path.name, 'server_path': filename, 'sha256': artifact_digest,
        'trace_sha256': group[0]['trace_sha256'], 'bytes': path.stat().st_size,
        'references': len(group), 'methods': sorted({r['method'] for r in group}),
        'referenced_by': [{k: r[k] for k in ('record_id', 'measurement_kind', 'method',
                                          'regime', 'trace_variant', 'batch_tokens',
                                          'seed', 'repetition')} for r in group],
    })
    with np.load(path, allow_pickle=False) as raw:
        ids = raw['expert_ids']
        combining = raw['combining_weights']
        digest = hashlib.sha256(str(raw['metadata']).encode())
        digest.update(memoryview(ids).cast('B'))
        digest.update(memoryview(combining).cast('B'))
        trace_digest = digest.hexdigest()
        check(trace_digest == str(raw['trace_sha256']), filename + ' embedded trace hash')
        check(raw['hidden_states'].shape == (group[0]['batch_tokens'], 2048), filename + ' hidden shape')
        check(ids.shape == (group[0]['trace_config']['num_steps'], group[0]['batch_tokens'], K), filename + ' raw tape shape')
        for r in group:
            tag = r['record_id'][:12]
            check(r['trace_artifact']['sha256'] == artifact_digest, tag + ' artifact digest')
            check(r['trace_sha256'] == trace_digest == r['trace_artifact']['trace_sha256'], tag + ' route fingerprint')
            log = r.get('decision_log', [{**r, 'step': r.get('decision_step')}])
            window = r['trace_config']['window_steps']
            for d in log:
                end = 2 * window + d['step'] + 1
                for field, stop in [('current_counts', end), ('previous_counts', end - window)]:
                    if d[field] is None:
                        continue
                    expected = np.bincount(ids[stop-window:stop].reshape(-1), minlength=E) * G
                    check(np.array_equal(d[field], expected), tag + ':' + str(d['step']) + ' exact ' + field)
                    exact_vector_checks += 1
    tape_count += 1
inventory_files = {str(p) for p in (RESULTS / 'traces').glob('*.npz')}
if complete:
    check(inventory_files == set(groups), 'trace archive inventory equals all referenced tapes including preflight')
    check(not list((RESULTS / 'traces').glob('*.tmp')), 'no unfinished trace artifacts')

tracked_weight_files = None
if (ROOT / '.git').exists():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    tracked_weight_files = [p for p in tracked if Path(p).suffix in ('.safetensors', '.pt', '.pth', '.bin', '.ckpt')]
    check(not tracked_weight_files, 'no raw checkpoint weights tracked by Git')
job_accounting = None
if complete:
    job_accounting = subprocess.check_output(
        ['sacct', '-j', '4863402', '-n', '-P', '-X', '--format=JobID,State,ExitCode,NodeList'], text=True).strip()
    check(job_accounting.split('|')[1:3] == ['COMPLETED', '0:0'], 'Slurm job completed normally')

model_config_path = Path('/work/cs-471/skewbench-hlin/models/qwen3-layer0/config.json')
model_config = json.loads(model_config_path.read_text())
for r in rows + branches:
    check(r['weights_source'] == model_config, r['record_id'][:12] + ' shared exact model config')
model_provenance = {
    'server_config_path': str(model_config_path),
    'config_sha256': sha(model_config_path.read_bytes()),
    'config': model_config,
    'native_source_pin': native,
    'framework_metadata': {k: rows[0][k] for k in ('native_vllm_commit', 'vllm_version', 'torch_version',
                                                 'cuda_version', 'dtype', 'gpu_name', 'world_size')},
    'weight_hash_validation_scope': 'The attested Runtime Weights loader verifies the exported safetensors SHA-256 before GPU execution. This audit independently checks the config provenance and agreement of every run; it does not download the checkpoint or rehash the large weights file.',
}
reject_then_accept = 0
for r in rows:
    if r.get('method') != 'B4' or r.get('status') != 'complete':
        continue
    saw_rejection = False
    for d in r['decision_log']:
        if d['allowed'] and saw_rejection:
            reject_then_accept += 1
            break
        saw_rejection |= not d['allowed']

summary = {
    'schema_version': 1,
    'passed': not issues,
    'audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'harness_sha256': harness,
    'manifest_sha256': manifest_digest,
    'native_source_files_verified': len(native['sha256']),
    'slurm_job_accounting': job_accounting,
    'tracked_weight_files': tracked_weight_files,
    'git_weight_check_scope': 'Remote Git metadata checked if available; the server source tree is normally rsynced without .git. A local Git check is added when packaging this audit.',
    'rows_snapshot': len(rows),
    'complete_matrix_requested': complete,
    'expected_primary_records': len(expected_jobs),
    'missing_primary_records': len(expected_jobs - actual_jobs),
    'branch_pairs_checked': len(branches),
    'branch_order_counts': dict(collections.Counter(branch_orders)),
    'branch_b3_eligible_count': sum(r['b3_would_allow'] for r in branches),
    'preflight_rows_checked': len(preflight),
    'partial_last_lines': partial_lines,
    'method_counts': dict(collections.Counter(r.get('method') for r in rows)),
    'batch_tokens_verified': sorted({r['batch_tokens'] for r in rows}),
    'decision_structures_checked': decisions_checked,
    'b4_runs_with_scheduled_reject_then_accept': reject_then_accept,
    'raw_replay_records_checked': sum(len(group) for group in groups.values()),
    'primary_method_case_batch_strata': len(selected),
    'raw_tapes_checked': tape_count,
    'artifact_bytes_checked': sum(a['bytes'] for a in artifact_inventory),
    'artifact_inventory': artifact_inventory,
    'model_provenance': model_provenance,
    'exact_logical_vectors_recomputed': exact_vector_checks,
    'representative_strata': sorted({(r['regime'], r['trace_variant'], r['batch_tokens']) for r in selected.values()}),
    'issues': issues,
}
print(json.dumps(summary, indent=2))
