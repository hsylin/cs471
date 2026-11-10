import hashlib
import json

import pytest

from full_serving.prepare_arrivals import (
    choose_block, digest, extract_blocks, load_exclusions, merged_intervals, overlaps,
)


def test_inclusive_boundary_exclusion():
    intervals = merged_intervals([(10, 20), (21, 24), (40, 50)])
    assert intervals == [(10, 24), (40, 50)]
    assert overlaps(0, 10, intervals)
    assert overlaps(24, 30, intervals)
    assert not overlaps(25, 39, intervals)
    assert not overlaps(51, 80, intervals)


def test_hash_winner_after_exclusions_and_partial_tail():
    # Bins [0,4], [5,9], [10,14], [15,19], with [20,22] discarded.
    block = choose_block("code", "2024-05-10", "final", 4, 0, 23, [(4, 5)], seed=3)
    expected = min([10, 15], key=lambda row: digest([3, "code", "2024-05-10", row]))
    assert block["row_start"] == expected
    assert block["candidate_bins"] == 4
    assert block["excluded_candidate_bins"] == 2
    assert block["discarded_partial_bin_rows"] == 3


def test_no_candidate_is_an_error():
    with pytest.raises(ValueError, match="No unused"):
        choose_block("code", "2024-05-10", "final", 4, 0, 10, [(0, 10)])


def test_load_excludes_all_old_manifests_including_boundaries(tmp_path):
    results = tmp_path / "results"
    old = results / "old/arrival-manifest.json"
    new = results / "new/arrival-manifest.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    document = {"blocks": [{"trace": "conversation", "row_start": 4,
                             "row_stop_exclusive": 8, "boundary_row": 8}]}
    old.write_text(json.dumps(document))
    new.write_text("not parsed because output is excluded")
    intervals, inventory = load_exclusions(results, new)
    assert intervals["conversation"] == [(4, 8)]
    assert len(inventory) == 1
    assert inventory[0]["path"] == "results/old/arrival-manifest.json"


def tiny_source(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                    "2024-05-10 00:00:00.000000001Z,100,20\n"
                    "2024-05-10 00:00:00.000000011Z,200,30\n"
                    "2024-05-10 00:00:00.000000031Z,300,40\n"
                    "2024-05-11 00:00:00.000000001Z,400,50\n")
    bounds = {"2024-05-10": [0, 3], "2024-05-11": [3, 4]}
    block = choose_block("code", "2024-05-10", "final", 2, 0, 3, [])
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), bounds, block


def test_full_stream_extracts_exact_nanoseconds_and_boundary(tmp_path):
    path, sha, bounds, block = tiny_source(tmp_path)
    selected, scan = extract_blocks(path, sha, [block], bounds)
    assert scan["rows"] == 4
    assert selected[0]["row_indices"] == [0, 1]
    assert selected[0]["original_offsets_ns"] == [0, 10]
    assert selected[0]["original_horizon_ns"] == 30
    assert selected[0]["terminal_gap_ns"] == 20
    assert selected[0]["boundary_context_tokens"] == 300


@pytest.mark.parametrize("changed", ["sha", "bounds"])
def test_full_stream_refuses_changed_source_or_selection_inventory(tmp_path, changed):
    path, sha, bounds, block = tiny_source(tmp_path)
    if changed == "sha":
        sha = "0" * 64
    else:
        bounds["2024-05-10"][1] = 2
    with pytest.raises(ValueError, match="mismatch|differ"):
        extract_blocks(path, sha, [block], bounds)
