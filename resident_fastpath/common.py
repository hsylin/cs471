"""Source freeze and exact one-layer hardware contract for the resident fix."""
import json
from pathlib import Path

from followup.router_tapes import file_sha256
from stock_comparison.common import (HOST,GPU_UUIDS,MODEL_REVISION,VLLM_COMMIT,
    seal,scale_calendar,validate_calendar,source_identity as old_source_identity)
from stock_comparison.data import digest

ARMS=('stockoff','current','fix')
OLD_CLOSURE='9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4'


def source_identity():
    old,files=old_source_identity()
    if old!=OLD_CLOSURE:raise ValueError('Previous frozen execution source was changed')
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'resident_fastpath').glob('*.py'))
    paths+=list((root/'scripts').glob('*resident_fix*'))
    paths+=list((root/'scripts').glob('run-resident-fix-*'))
    files={**files,**{str(p.relative_to(root)):file_sha256(p) for p in paths if p.is_file()}}
    return digest(files),dict(sorted(files.items()))


def load_plan(path):
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='resident_fix_plan_v1' or seal(p)!=p:
        raise ValueError('Changed/invalid resident fix plan')
    if p['source_sha256']!=source_identity()[0] or p['old_closure_sha256']!=OLD_CLOSURE:
        raise ValueError('Source changed after resident fix freeze')
    if p['arms']!=list(ARMS) or p['jobs']!=4096 or p['batch_tokens']!=128:
        raise ValueError('Wrong one-layer scope')
    if p['hardware']!={'hostname':HOST,'gpu_uuids':GPU_UUIDS,'allocation_id':'4872248'}:
        raise ValueError('Wrong reserved pair')
    root=Path(__file__).resolve().parents[1]
    if p['protocol_sha256']!=file_sha256(root/'docs/resident-fix/protocol-v1.md'):
        raise ValueError('Protocol changed after freeze')
    selected=root/'configs/resident-fix-selected-v1.json'
    if p['selected_fix_sha256']!=file_sha256(selected) or p['selected_fix']!=json.loads(selected.read_text()):
        raise ValueError('Selected implementation changed after freeze')
    return p
