"""Frozen plan, source identities and externally shaped absolute calendars."""
from pathlib import Path
from fractions import Fraction
import json
from stock_comparison.data import digest
from followup.router_tapes import file_sha256

ARMS=('stockoff','stockon32','ours')
HOST='kh014'
GPU_UUIDS=['GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3','GPU-7d5c9009-305b-9341-957f-1deb96611ed5']
MODEL_REVISION='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
VLLM_COMMIT='db9527a46873454610df6dbedf79a36d6bf1a7f6'

def source_identity():
    root=Path(__file__).resolve().parents[1]
    dirs=('stock_comparison','skewbench','arrival_fix','resident_dispatch','arrival_study','optimization','followup')
    paths=sorted(p for d in dirs for p in (root/d).glob('*.py'))
    paths += [root/'skewbench/native_sources.json']
    paths += sorted((root/'scripts').glob('*stock*'))
    paths += [root/'scripts/cluster-env.sh']
    paths = sorted(set(p for p in paths if p.is_file()))
    files={str(p.relative_to(root)):file_sha256(p) for p in paths}
    return digest(files),files

def seal(plan):
    p={k:v for k,v in plan.items() if k!='content_sha256'}
    return {**p,'content_sha256':digest(p)}

def load_plan(path):
    p=json.loads(Path(path).read_text())
    if p.get('schema')!='stock_comparison_plan_v1' or seal(p)!=p:raise ValueError('Changed/invalid frozen plan')
    if p['source_sha256']!=source_identity()[0]:raise ValueError('Execution source changed after plan freeze')
    if p['arms']!=list(ARMS) or p['jobs']!=4096 or p['batch_tokens']!=128:raise ValueError('Wrong frozen scope')
    if p['hardware']!={'hostname':HOST,'gpu_uuids':GPU_UUIDS,'allocation_id':'4872248'}:raise ValueError('Wrong hardware')
    return p

def scale_calendar(source,elapsed_ns,*,rho,jobs=4096):
    if len(source['original_offsets_ns'])!=jobs or source['jobs']!=jobs:raise ValueError('Wrong arrival block length')
    if type(elapsed_ns) is not int or elapsed_ns<=0 or rho not in (.5,.85):raise ValueError('Invalid frozen calibration/load')
    scale=Fraction(elapsed_ns,1)/Fraction(str(rho))/source['original_horizon_ns']
    offsets=[round(x*scale) for x in source['original_offsets_ns']]
    horizon=round(source['original_horizon_ns']*scale)
    if offsets[0]!=0 or offsets[-1]>=horizon or any(a>b for a,b in zip(offsets,offsets[1:])):raise ValueError('Invalid calendar')
    return seal(dict(schema='stock_absolute_calendar_v1',source_block_id=source['id'],
        source_rows_sha256=source['source_rows_sha256'],arrival_type=source['trace'],
        arrival_date=source['source_date_utc'],rho=rho,n_jobs=jobs,arrival_offsets_ns=offsets,horizon_ns=horizon,
        source_horizon_ns=source['original_horizon_ns'],stockoff_calibration_elapsed_ns=elapsed_ns,
        actual_absolute_rate_jobs_per_second=jobs*1e9/horizon,rounding='nearest_ns_ties_even',
        unit='one fixed128token preassembled layer job per original request timestamp; original request lengths not executed'))

def validate_calendar(calendar,source):
    expected=scale_calendar(source,calendar['stockoff_calibration_elapsed_ns'],rho=calendar['rho'],jobs=calendar['n_jobs'])
    if calendar!=expected:raise ValueError('Changed absolute arrival calendar')
