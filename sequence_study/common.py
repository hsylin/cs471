"""Separate bounded input-construction diagnostic; prior study remains frozen."""
import json
from pathlib import Path
from resident_residual.common import source_identity as parent_identity
from stock_comparison.data import digest
from stock_comparison.common import seal
from followup.router_tapes import file_sha256

ARMS=('stockoff','stockon32','fix')
PARENT='136259a52849be38f9dc875f6ccef964ba2c92f8cd53ceee2c5e5c51f2e0dce7'

def source_identity():
    identity,files=parent_identity()
    if identity!=PARENT:raise ValueError('Frozen parent changed')
    root=Path(__file__).resolve().parents[1]
    files.update({str(p.relative_to(root)):file_sha256(p) for p in (root/'sequence_study').iterdir() if p.is_file() and p.suffix in ('.py','.sh')})
    return digest(files),dict(sorted(files.items()))

def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if seal(plan)!=plan or plan['schema']!='sequence_study_plan_v1':raise ValueError('Changed plan')
    if plan['source_sha256']!=source_identity()[0]:raise ValueError('Changed execution source')
    if plan['jobs']!=4096 or plan['arms']!=list(ARMS) or len(plan['banks'])!=4 or len(plan['trials'])!=54:raise ValueError('Wrong scope')
    root=Path(__file__).resolve().parents[1]
    if file_sha256(root/'docs/sequence-study/protocol-v1.md')!=plan['protocol_sha256']:raise ValueError('Changed protocol')
    from resident_residual.common import load_plan as load_parent
    parent=load_parent(root/'configs/resident-residual-final-v1.json')
    if (plan['parent_plan_sha256']!=parent['content_sha256'] or plan['hardware']!=parent['hardware']
        or plan['selected_fix']!=parent['selected_fix'] or plan['input_seed']!=471902):raise ValueError('Changed parent link')
    for b in plan['banks']:
        if b not in parent['banks'] or plan['calendars'][b['block_id']] not in parent['cases']:raise ValueError('Changed content/calendar link')
    return plan
