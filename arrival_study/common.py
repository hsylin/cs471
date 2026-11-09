"""Independent provenance for the supplementary arrival experiment."""
from pathlib import Path
import hashlib
import json

from optimization.common import canonical_hash, manifest_hash
from resident_dispatch.common import load_manifest as load_base, execution_fingerprint as resident_fingerprint

ROOT=Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths=['arrival_study/'+name for name in ('__init__.py','common.py','schedules.py','runner.py')]
    paths+=['scripts/run-instructor-arrivals-kuma.sh','scripts/run-instructor-arrival-study-kuma.sh',
            'scripts/freeze_instructor_arrivals.py']
    files={name:file_hash(ROOT/name) for name in paths}
    return canonical_hash(files),files


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('arrival_schema_version')!=1 or plan.get('freeze',{}).get('sha256')!=manifest_hash(plan):
        raise ValueError('Unfrozen or altered arrival plan')
    base=load_base(ROOT/plan['base_manifest']['path'])
    if base['freeze']['sha256']!=plan['base_manifest']['sha256']:
        raise ValueError('Underlying resident study changed')
    if resident_fingerprint()[0]!=base['environment']['resident_dispatch_sha256']:
        raise ValueError('Frozen native/online execution changed')
    if source_identity()[0]!=plan['arrival_source_sha256']:
        raise ValueError('Arrival execution code changed')
    if file_hash(ROOT/'docs/instructor-arrival-protocol.md')!=plan['protocol_file_sha256']:
        raise ValueError('Arrival protocol changed after freeze')
    return plan,base
