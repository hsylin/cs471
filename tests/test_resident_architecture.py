"""Independent arithmetic checks for descriptive architectural accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('resident_architecture',Path(__file__).parents[1]/'scripts/analyze_resident_architecture.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(bm=64):
    return {'BLOCK_SIZE_M':bm,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'SPLIT_K':1}


def test_two_gemm_geometry_and_replicated_rank_conservation():
    counts=np.zeros(160,dtype=np.int64);counts[0]=512;counts[80]=512
    result=MODULE.work_metrics(counts,128,config())
    assert result['useful_gemm_flops']==128*8*(4*2048*768+2*768*2048)
    assert result['useful_gemm_flops']==result['tile_implied_gemm_flops_proxy']
    assert result['useful_row_fraction']==1
    assert result['per_rank_m_tiles']==[8,8]


def test_boundary_padding_uses_individual_slots_not_rank_total():
    counts=np.zeros(160,dtype=np.int64);counts[0]=65;counts[1]=959
    result=MODULE.work_metrics(counts,128,config())
    assert result['per_rank_m_tiles']==[17,0]
    assert result['total_padded_rows']==1088
    assert result['useful_row_fraction']==1024/1088
    assert result['active_expert_parameter_footprint_bytes_proxy']==2*3*2048*768*2


def test_small_groups_cannot_be_mistaken_for_full_tile_use():
    counts=np.zeros(160,dtype=np.int64);counts[:128]=8
    result=MODULE.work_metrics(counts,128,config())
    assert result['total_active_copies']==128
    assert result['total_m_tiles']==128
    assert result['useful_row_fraction']==1/8
    assert result['tile_implied_gemm_flops_proxy']==8*result['useful_gemm_flops']


def test_batch1024_uses_recorded_m128():
    counts=np.zeros(160,dtype=np.int64);counts[0]=129;counts[80]=8063
    result=MODULE.work_metrics(counts,1024,config(128))
    assert result['per_rank_m_tiles']==[2,63]
    assert result['total_padded_rows']==8320


def test_bad_conservation_and_unsupported_geometry_fail_closed():
    counts=np.zeros(160,dtype=np.int64);counts[0]=1024
    with pytest.raises(ValueError,match='conservation'):
        MODULE.work_metrics(counts,1024,config())
    bad=config();bad['BLOCK_SIZE_N']=100
    with pytest.raises(ValueError,match='divisibility'):
        MODULE.work_metrics(counts,128,bad)


def test_union_counts_overlapping_kernels_once():
    assert MODULE.union_ns([(10,20),(15,30),(31,35)])==24
