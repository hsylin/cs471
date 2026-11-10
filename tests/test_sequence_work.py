import numpy as np
import pytest

from sequence_study.work import native_physical_ids, rank_work, reconstruct_work, routing_summary


def test_native_hash_uses_token_row_and_sorted_copies():
    ids = np.tile(np.arange(8), (128,1))
    mapping = np.r_[np.arange(128),np.arange(32)]
    physical = native_physical_ids(ids, mapping)
    np.testing.assert_array_equal(physical[::2],ids[::2])
    np.testing.assert_array_equal(physical[1::2],ids[1::2]+128)
    # Three copies exercise uint32 wrap before modulo, not unbounded multiplication.
    mapping = np.r_[np.arange(128), np.zeros(2,int), np.arange(1,31)]
    result = native_physical_ids(ids,mapping)
    for token in (1,2,127):
        expected = [0,128,129][((token*2654435769)&0xffffffff)%3]
        assert result[token,0] == expected


def test_tile_work_and_commit_uses_next_job_and_ignores_tail():
    ids = np.tile(np.arange(8),(4,128,1))
    initial = np.r_[np.arange(128),np.arange(32)]
    changed = initial[::-1]
    events = [dict(commit_step=1,committed_map=changed.tolist(),commit_in_completion_tail=False),
              dict(commit_step=3,committed_map=initial.tolist(),commit_in_completion_tail=True)]
    result = reconstruct_work(ids,'stockon32',initial,events)
    rows = result['records']
    assert rows[0]['map_sha256'] == rows[1]['map_sha256']
    assert rows[2]['map_sha256'] == rows[3]['map_sha256'] != rows[0]['map_sha256']
    assert rows[0]['per_rank_tokens'] == [512,512]
    assert rows[0]['per_rank_active_copies'] == [8,8]
    assert rows[0]['per_rank_padded_rows'] == [512,512]
    off = rank_work(ids[0],physical_slots=128)
    assert off['per_rank_tokens'] == [1024,0]
    assert off['per_rank_padded_rows'] == [1024,0]


def test_constant_demand_tv_and_reject_uncommitted_state():
    ids = np.tile(np.arange(8),(32,128,1))
    summary = routing_summary(ids)
    assert summary['adjacent_job_tv_mean'] == 0
    assert summary['windows']['16']['consecutive_window_tv'] == [0.0]
    with pytest.raises(ValueError,match='Uncommitted'):
        reconstruct_work(ids,'stockon32',np.r_[np.arange(128),np.arange(32)],
                         [dict(commit_step=None,committed_map=None)])
