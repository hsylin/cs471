"""Fail-closed source, arrival and fixed-map identity for the new study."""
from pathlib import Path
import json
from arrival_combo.common import source_identity as previous_identity
from arrival_fix.common import file_hash, GPU_UUIDS, HARDWARE_HOST, canonical_gpu_uuid
from optimization.common import canonical_hash, manifest_hash
from resident_dispatch.common import load_manifest, execution_fingerprint

ROOT=Path(__file__).resolve().parents[1]
ARMS=['native','shortcut','combined']


def source_identity():
    _,files=previous_identity()
    names=['generalization/'+n for n in ('__init__.py','data.py','common.py','inputs.py','runner.py','external_arrivals.py')]
    names += ['scripts/freeze_generalization.py','scripts/run-generalization-kuma.sh']
    files.update({n:file_hash(ROOT/n) for n in names})
    return canonical_hash(files),files


def validate_calendar(calendar):
    offsets=calendar['arrival_offsets_ns'];h=calendar['horizon_ns']
    if len(offsets)!=256 or not isinstance(h,int) or h<=0:
        raise ValueError('Exactly256 arrivals and positive integer horizon required')
    if any(not isinstance(x,int) or isinstance(x,bool) or x<0 or x>=h for x in offsets):
        raise ValueError('Arrival outside offered interval')
    if offsets!=sorted(offsets):raise ValueError('Nonmonotone arrival calendar')
    if calendar['sha256']!=canonical_hash({k:v for k,v in calendar.items() if k!='sha256'}):
        raise ValueError('Changed arrival calendar')


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('generalization_schema_version')!=1 or plan.get('freeze',{}).get('sha256')!=manifest_hash(plan):
        raise ValueError('Changed/unfrozen generalization plan')
    base=load_manifest(ROOT/plan['base_manifest']['path'])
    if base['freeze']['sha256']!=plan['base_manifest']['sha256'] or execution_fingerprint()[0]!=base['environment']['resident_dispatch_sha256']:
        raise ValueError('Changed frozen resident dispatcher')
    if source_identity()[0]!=plan['execution_source_sha256']:raise ValueError('Changed execution source')
    for field,name in [('analysis_sha256','generalization/analysis.py'),('inference_sha256','generalization/inference.py')]:
        if file_hash(ROOT/name)!=plan[field]:raise ValueError('Changed frozen analysis/inference')
    for name,expected in plan['protocol_files'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Changed protocol')
    if plan['hardware']['hostname']!=HARDWARE_HOST or sorted(plan['hardware']['gpu_uuids'])!=sorted(GPU_UUIDS):
        raise ValueError('Wrong physical configuration')
    if plan['arms']!=ARMS or plan['batch_tokens']!=128:raise ValueError('Three arms and B128 required')
    expected=list(range(128))+list(range(32))
    if plan['physical_to_logical_map']!=expected:raise ValueError('Test-dependent placement forbidden')
    for case in plan.get('cases',[]):validate_calendar(case['calendar'])
    return plan,base
