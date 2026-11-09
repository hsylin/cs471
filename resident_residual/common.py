"""Frozen source identities for the bounded residual-overhead investigation."""
import json
from pathlib import Path
from followup.router_tapes import file_sha256
from stock_comparison.common import HOST,GPU_UUIDS,MODEL_REVISION,VLLM_COMMIT,seal,scale_calendar,validate_calendar
from stock_comparison.data import digest
from resident_fastpath.common import source_identity as previous_identity

PREVIOUS_CLOSURE='08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff'
ARMS=('stockoff','previous','fix')

def source_identity(*,diagnostic=False):
    old,files=previous_identity()
    if old!=PREVIOUS_CLOSURE:raise ValueError('Previous83-file execution closure changed')
    root=Path(__file__).resolve().parents[1]
    if diagnostic:
        paths=[root/p for p in ('resident_residual/__init__.py','resident_residual/common.py',
            'resident_residual/diagnose.py','resident_residual/alignment.py','scripts/run-residual-diagnostic-kuma.sh')]
    else:
        paths=list((root/'resident_residual').glob('*.py'))+list((root/'scripts').glob('*residual*'))
        paths += [root/'scripts/capture_optimization_tapes.py',root/'generalization/external_arrivals.py']
    files={**files,**{str(p.relative_to(root)):file_sha256(p) for p in paths if p.is_file()}}
    return digest(files),dict(sorted(files.items()))

def load_plan(path):
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='resident_residual_plan_v1' or seal(p)!=p:raise ValueError('Invalid/changed plan')
    if p['source_sha256']!=source_identity()[0] or p['previous_closure_sha256']!=PREVIOUS_CLOSURE:raise ValueError('Source changed after freeze')
    if p['arms']!=list(ARMS) or p['jobs']!=4096 or p['batch_tokens']!=128:raise ValueError('Wrong scope')
    if p['hardware']!={'hostname':HOST,'gpu_uuids':GPU_UUIDS,'allocation_id':'4872248'}:raise ValueError('Wrong pair')
    root=Path(__file__).resolve().parents[1]
    for key,rel in [('protocol_sha256','docs/resident-residual/protocol-v1.md'),('selected_fix_sha256','configs/resident-residual-selected-v1.json')]:
        if p[key]!=file_sha256(root/rel):raise ValueError('Freeze changed: '+rel)
    chosen=json.loads((root/'configs/resident-residual-selected-v1.json').read_text())
    if p['selected_fix']!=chosen or p['selected_fix_content_sha256']!=digest(chosen):raise ValueError('Selected fix dictionary changed')
    return p
