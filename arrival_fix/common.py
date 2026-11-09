"""Provenance and fail-closed contracts for the response-time follow-up."""
from pathlib import Path
import hashlib
import json
import uuid

from optimization.common import canonical_hash, manifest_hash
from resident_dispatch.common import load_manifest, execution_fingerprint

ROOT = Path(__file__).resolve().parents[1]
HARDWARE_HOST = 'kh014'
GPU_UUIDS = ['GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3',
             'GPU-7d5c9009-305b-9341-957f-1deb96611ed5']


def canonical_gpu_uuid(value):
    """Normalize Torch's bare UUID and NVML's GPU-prefixed UUID identically."""
    return 'GPU-' + str(uuid.UUID(str(value).removeprefix('GPU-')))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    names = ['arrival_fix/' + n for n in ('__init__.py', 'common.py', 'calendars.py', 'runner.py', 'noop.py', 'proof.py')]
    names += ['arrival_study/' + n for n in ('__init__.py', 'common.py', 'schedules.py', 'runner.py')]
    names += ['scripts/freeze_arrival_fix.py', 'scripts/run-arrival-fix-kuma.sh']
    files = {name: file_hash(ROOT/name) for name in names}
    return canonical_hash(files), files


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('arrival_fix_schema_version') != 1 or plan.get('freeze', {}).get('sha256') != manifest_hash(plan):
        raise ValueError('Unfrozen or changed response-fix plan')
    base = load_manifest(ROOT/plan['base_manifest']['path'])
    if base['freeze']['sha256'] != plan['base_manifest']['sha256']:
        raise ValueError('Underlying resident manifest changed')
    if execution_fingerprint()[0] != base['environment']['resident_dispatch_sha256']:
        raise ValueError('Frozen native/online implementation changed')
    if source_identity()[0] != plan['arrival_fix_source_sha256']:
        raise ValueError('Response-fix implementation changed')
    if file_hash(ROOT/'docs/arrival-response-fix-protocol.md') != plan['protocol_file_sha256']:
        raise ValueError('Protocol changed after freeze')
    if file_hash(ROOT/'arrival_fix/analysis.py') != plan['analysis_file_sha256']:
        raise ValueError('Analysis changed after freeze')
    if sorted(plan['hardware']['gpu_uuids']) != sorted(GPU_UUIDS) or plan['hardware']['hostname'] != HARDWARE_HOST:
        raise ValueError('This experiment requires the exact allocated two GPUs on the declared host')
    return plan, base
