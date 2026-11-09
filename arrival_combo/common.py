"""Separate provenance for the three-arm composition; old experiments stay frozen."""
from pathlib import Path
import json

from arrival_fix.common import file_hash, GPU_UUIDS, HARDWARE_HOST, canonical_gpu_uuid
from optimization.common import canonical_hash, manifest_hash
from resident_dispatch.common import load_manifest, execution_fingerprint

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['native', 'shortcut', 'combined']


def source_identity():
    previous = json.loads((ROOT/'configs/arrival-response-calibration-v4.json').read_text())
    files = {name: file_hash(ROOT/name) for name in previous['arrival_fix_source_files']}
    if files != previous['arrival_fix_source_files']:
        raise ValueError('Previously frozen unchanged-map/arrival source changed')
    files['arrival_fix/analysis.py'] = file_hash(ROOT/'arrival_fix/analysis.py')
    if files['arrival_fix/analysis.py'] != previous['analysis_file_sha256']:
        raise ValueError('Previously frozen analysis helper changed')
    names = ['arrival_combo/'+n for n in ('__init__.py', 'common.py', 'calendars.py', 'runner.py', 'proof.py')]
    names += ['scripts/freeze_arrival_combo.py', 'scripts/run-arrival-combo-kuma.sh']
    files.update({name:file_hash(ROOT/name) for name in names})
    return canonical_hash(files), files


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('arrival_combo_schema_version') != 1 or plan.get('freeze', {}).get('sha256') != manifest_hash(plan):
        raise ValueError('Unfrozen or changed composition plan')
    base = load_manifest(ROOT/plan['base_manifest']['path'])
    if base['freeze']['sha256'] != plan['base_manifest']['sha256'] or execution_fingerprint()[0] != base['environment']['resident_dispatch_sha256']:
        raise ValueError('Frozen native/online implementation changed')
    if source_identity()[0] != plan['arrival_combo_source_sha256']:
        raise ValueError('Composition implementation changed')
    for field, name in [('protocol_file_sha256', 'docs/arrival-combined-protocol.md'),
                        ('analysis_file_sha256', 'arrival_combo/analysis.py')]:
        if file_hash(ROOT/name) != plan[field]:
            raise ValueError('Frozen protocol or analysis changed')
    if sorted(plan['hardware']['gpu_uuids']) != sorted(GPU_UUIDS) or plan['hardware']['hostname'] != HARDWARE_HOST:
        raise ValueError('The exact reserved pair is required')
    if plan['matrix']['arms'] != ARMS or plan['matrix']['reps'] != 2:
        raise ValueError('Three fixed arms and two reversed repeats required')
    return plan, base
