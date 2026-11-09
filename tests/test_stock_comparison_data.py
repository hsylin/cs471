"""Synthetic source-selection tests; no model or performance observations."""
import csv
import hashlib
from pathlib import Path

import pytest

from stock_comparison.data import (DATASETS, digest, normalized, exclude_and_deduplicate,
    split_sources, fresh_arrival_candidates, scan_fresh_arrivals, assign_fresh_arrivals)


def row(source, text, dataset="mbpp", prefix=None):
    return dict(dataset=dataset, source_id=source, text=text,
                captured_prefix_text=text if prefix is None else prefix)


def ref(source, text, prefix=None):
    return dict(dataset="mbpp", source_id=source, text=text,
                prefix_views=[text if prefix is None else prefix])


def test_source_id_and_normalized_views_excluded():
    prior = [ref("old", "original full body", "VISIBLE prefix"),
             ref("near", " ".join(f"token{i}" for i in range(100)))]
    candidates = [row("old", "entirely changed text"), row("copy", "ORIGINAL   FULL BODY"),
                  row("prefix", "different unseen full body", prefix="visible PREFIX"),
                  row("nearly", " ".join(f"token{i}" for i in range(100))+" new tail"),
                  row("fresh", "a different source without shared five gram sequences")]
    kept, dropped = exclude_and_deduplicate(candidates, prior)
    assert [r["source_id"] for r in kept] == ["fresh"]
    assert {r["source_id"] for r in dropped} == {"old", "copy", "prefix", "nearly"}
    assert next(r for r in dropped if r["source_id"] == "prefix")["view"] == "captured_prefix"


def test_all_prior_views_remain_indexed_not_greedily_removed():
    # B is near A; C is near B but not A. Exclusion references must retain B.
    base = [f"word{i}" for i in range(100)]
    a = " ".join(base)
    b = " ".join(base[10:]+[f"tail{i}" for i in range(10)])
    c = " ".join(base[20:]+[f"tail{i}" for i in range(20)])
    kept, dropped = exclude_and_deduplicate([row("C", c)], [ref("A", a), ref("B", b)])
    assert not kept and dropped[0]["duplicate_of"] == ["mbpp", "B"]


def test_old_capture_long_and_common_prefix_and_whole_source_hash():
    previous = [dict(dataset="old-capture", source_id="book", text="long captured old view",
                     prefix_views=["long captured old view", "common leading prefix"],
                     whole_source_normalized_sha256=digest(normalized("entire old book").encode()))]
    candidates = [row("one", "different full document", prefix="COMMON LEADING PREFIX"),
                  row("two", "Entire Old Book"), row("three", "unrelated unique fresh document")]
    kept, rejected = exclude_and_deduplicate(candidates, previous)
    assert [x["source_id"] for x in kept] == ["three"]
    assert rejected[1]["reason"] == "prior_whole_source_exact"


def test_candidate_duplicates_removed_before_split_and_no_silent_shortfall():
    candidates = [row("a", "a novel distinct task description"), row("b", "a novel distinct task description")]
    kept, rejected = exclude_and_deduplicate(candidates, [])
    assert len(kept) == len(rejected) == 1
    with pytest.raises(ValueError, match="no reuse"):
        split_sources(kept)


def test_hash_split_is_order_invariant_and_disjoint():
    records = [dict(dataset=d, source_id=str(i), text_sha256=digest([d,i]),
                    normalized_text_sha256=digest(["text",d,i])) for d in DATASETS for i in range(256)]
    a, b = split_sources(records), split_sources(records[::-1])
    assert a == b and len(a) == 24
    assert all(len(x["sources"]) == 32 for x in a)
    ids = [(r["dataset"],r["source_id"]) for b in a for r in b["sources"]]
    assert len(ids) == len(set(ids)) == 768
    assert sum(x["split"] == "pilot" for x in a) == 6


def test_arrival_overlap_includes_old_and_new_boundary_rows():
    candidates = [dict(trace="code", source_date_utc="2024-05-10", row_start=i,
                       boundary_row=i+4, original_horizon_ns=4, terminal_gap_ns=1)
                  for i in (0,5,10)]
    old = [dict(trace="code", row_start=4, boundary_row=5)]
    selected, report = fresh_arrival_candidates(candidates, old, trace="code", per_day=1, jobs=4)
    assert selected[0]["row_start"] == 10
    assert report["excluded_prior_overlap"] == 2
    assert report["fresh_calendar_dates"] is False


def test_invalid_hash_winner_is_not_replaced():
    candidates = [dict(trace="code", source_date_utc="2024-05-10", row_start=0,
                       boundary_row=4, original_horizon_ns=4, terminal_gap_ns=0)]
    with pytest.raises(ValueError, match="amendment"):
        fresh_arrival_candidates(candidates, [], trace="code", per_day=1, jobs=4)


def write_trace(path, trace, start_day):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TIMESTAMP","ContextTokens","GeneratedTokens"])
        for day in range(start_day,start_day+7):
            for second in range(45):
                writer.writerow([f"2024-05-{day:02d}T00:00:{second:02d}.000001Z",10,2])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_streamed_source_scan_assignment_and_hash_rejection(tmp_path):
    scans=[]
    for trace,day in (("code",10),("conversation",12)):
        path=tmp_path/(trace+".csv");pin=write_trace(path,trace,day)
        prior=[dict(trace=trace,row_start=0,boundary_row=4)]
        scan=scan_fresh_arrivals(path,trace=trace,expected_sha256=pin,previous_blocks=prior,jobs=4)
        assert scan["rows"]==315 and scan["days"][f"2024-05-{day:02d}"]["excluded_prior_overlap"]==1
        assert all(b["row_start"]>=5 for bs in scan["retained_candidates"].values() for b in bs)
        scans.append(scan)
        with pytest.raises(ValueError, match="source bytes"):
            scan_fresh_arrivals(path,trace=trace,expected_sha256="0"*64,previous_blocks=[],jobs=4)
    blocks=[dict(block_id=f"stock-{d}-{split}-{i}",dataset=d,split=split)
            for d in DATASETS for split,n in (("pilot",2),("final",6)) for i in range(n)]
    assigned=assign_fresh_arrivals(scans,blocks)
    assert len(assigned)==48 and sum(b["split"]=="pilot" for b in assigned)==12
    assert len({(b["trace"],b["row_start"]) for b in assigned})==48
    for group in blocks:
        assert {b["trace"] for b in assigned if b["content_group_id"]==group["block_id"]}=={"code","conversation"}


def test_streaming_rejects_time_inversion_and_missing_day(tmp_path):
    path=tmp_path/"bad.csv"
    pin=write_trace(path,"code",10)
    lines=path.read_text().splitlines()
    lines[2],lines[3]=lines[3],lines[2]
    path.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError,match="inversion"):
        scan_fresh_arrivals(path,trace="code",expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),previous_blocks=[],jobs=4)
    write_trace(path,"code",10)
    path.write_text("\n".join(path.read_text().splitlines()[:1+6*45])+"\n")
    with pytest.raises(ValueError,match="seven days"):
        scan_fresh_arrivals(path,trace="code",expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),previous_blocks=[],jobs=4)


def test_independent_pair_audit_rejects_mutated_prefix():
    from scripts.audit_stock_comparison_data import no_near_pairs
    previous=[("old"," ".join(f"token{i}" for i in range(100)))]
    fresh=[("new","unrelated short source")]
    no_near_pairs(fresh,previous,"prefix")
    with pytest.raises(AssertionError,match="exact"):
        no_near_pairs([("new",previous[0][1].upper())],previous,"prefix")
    with pytest.raises(AssertionError,match="near"):
        no_near_pairs([("new",previous[0][1]+" changed ending")],previous,"prefix")
