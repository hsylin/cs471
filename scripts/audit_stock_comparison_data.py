#!/usr/bin/env python3
"""CPU-only source/row audit, independently implemented from the selector.

This is a same-team artifact check, not an external reproduction or GPU test.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_sealed(path, key="content_sha256"):
    x = json.loads(Path(path).read_text())
    assert x[key] == sha({k: v for k, v in x.items() if k != key}), (path, "canonical hash")
    return x


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def grams(text):
    words = re.findall(r"\w+|[^\w\s]", norm(text))
    return set(tuple(words[i:i+5]) for i in range(max(1, len(words)-4)))


def no_near_pairs(fresh, previous, label):
    """Brute-force length-filtered comparisons; do not reuse selector index."""
    prior = [(key, norm(text), grams(text)) for key, text in previous]
    comparisons = 0
    for key, text in fresh:
        normalized, a = norm(text), grams(text)
        for other, normalized_other, b in prior:
            assert normalized != normalized_other, (label, "exact", key, other)
            if 5 * min(len(a), len(b)) < 4 * max(len(a), len(b)):
                continue
            comparisons += 1
            # Integer form of Jaccard >= 0.8, avoiding float boundary ambiguity.
            assert 5 * len(a & b) < 4 * len(a | b), (label, "near", key, other)
        prior.append((key, normalized, a))
    return comparisons


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def verify_arrival_structure(arrivals, old, scans):
    assert len(old["blocks"]) == 92
    blocks = arrivals["blocks"]
    assert len(blocks) == 48 and Counter(b["split"] for b in blocks) == {"pilot": 12, "final": 36}
    assert arrivals["jobs_per_trial"] == 4096 and arrivals["new_calendar_days"] is False
    assert arrivals["previous_manifest_sha256"] == old["_file_sha256"]
    actual = sorted((x["block_id"], x["trace"], x["arrival_block_id"]) for x in arrivals["assignments"])
    expected = sorted((x["content_group_id"], x["trace"], x["id"]) for x in blocks)
    assert actual == expected and len(set(actual)) == 48
    for trace in ("code", "conversation"):
        prior = [(x["row_start"], x["boundary_row"]+1) for x in old["blocks"] if x["trace"] == trace]
        selected = [b for b in blocks if b["trace"] == trace]
        new = [(b["row_start"], b["boundary_row"]+1) for b in selected]
        assert all(not overlap(a, b) for a in new for b in prior), "old/new boundary overlap"
        assert all(not overlap(a, b) for i, a in enumerate(new) for b in new[i+1:]), "new/new boundary overlap"
        scan = scans[trace]
        day_start = 0
        winners = set()
        for day, stats in sorted(scan["days"].items()):
            bins = [(start, start+4097) for start in range(day_start, day_start+stats["rows"]-4096, 4097)]
            eligible = [b for b in bins if all(not overlap(b, p) for p in prior)]
            assert len(bins) == stats["candidate_bins"] and len(eligible) == stats["eligible_bins"]
            ranked = sorted(eligible, key=lambda b: (sha([471602, trace, day, b[0]]), b[0]))[:4]
            assert [b[0] for b in ranked] == [b["row_start"] for b in scan["retained_candidates"][day]]
            winners.update(b[0] for b in ranked)
            day_start += stats["rows"]
        assert day_start == scan["rows"]
        assert all(b["row_start"] in winners for b in selected)
        days = sorted(scan["days"])
        for b in selected:
            assert b["jobs"] == 4096 and b["boundary_row"] == b["row_start"]+4096
            assert b["source_date_utc"] in (days[:2] if b["split"] == "pilot" else days[2:])
            assert b["original_offsets_ns"][0] == 0
            assert len(b["original_offsets_ns"]) == 4096
            assert b["terminal_gap_ns"] > 0 and b["original_horizon_ns"] > 0


def exact_ns(timestamp):
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|\+00:00)", timestamp)
    assert m, "UTC timestamp"
    day, hh, mm, ss, frac, _ = m.groups()
    seconds = (date.fromisoformat(day)-date(1970,1,1)).days*86400+int(hh)*3600+int(mm)*60+int(ss)
    return seconds*10**9+int((frac or "").ljust(9,"0"))


def verify_selected_texts(downloads, fresh):
    """Check selected private text against the pinned raw document records."""
    import pyarrow.parquet as pq
    wanted = {(r["dataset"], r["source_id"]):r["text"] for r in fresh}
    found = {}
    for i,line in enumerate((downloads/"gsm8k.jsonl").read_text().splitlines()):
        r=json.loads(line);key=("gsm8k",f"test-{i}")
        if key in wanted:found[key]=r["question"]
    for line in (downloads/"mbpp-full.jsonl").read_text().splitlines():
        r=json.loads(line);key=("mbpp",str(r["task_id"]))
        if key in wanted:found[key]=r["text"]
    index=0
    for batch in pq.ParquetFile(downloads/"wildchat-first-shard.parquet").iter_batches(batch_size=1024):
        for r in batch.to_pylist():
            if index>=8192:break
            index+=1;key=("wildchat",r["conversation_id"])
            if key not in wanted:continue
            users=[t for t in r["conversation"] if t["role"]=="user"]
            assert r["language"]=="English" and not r["toxic"] and users and not users[0].get("toxic")
            assert users[0]["content"].strip()
            if key in found:assert found[key]==users[0]["content"]
            found[key]=users[0]["content"]
        if index>=8192:break
    assert found==wanted, "selected text differs from pinned eligible raw source"
    return len(found)


def verify_raw_arrivals(downloads, arrivals, scans):
    result = {}
    for trace, name in (("code", "AzureLLMInferenceTrace_code_1week.csv"), ("conversation", "AzureLLMInferenceTrace_conv_1week.csv")):
        path = downloads/name
        assert file_sha(path) == scans[trace]["source_sha256"], "full weekly bytes"
        selected = sorted((b for b in arrivals["blocks"] if b["trace"] == trace), key=lambda b:b["row_start"])
        current, rows, completed = 0, [], 0
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            assert next(reader) == ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
            for index, r in enumerate(reader):
                if current >= len(selected):
                    continue
                b = selected[current]
                if not b["row_start"] <= index <= b["boundary_row"]:
                    continue
                assert len(r) == 3
                rows.append([r[0], int(r[1]), int(r[2])])
                if index == b["boundary_row"]:
                    assert len(rows) == 4097 and sha(rows) == b["source_rows_sha256"]
                    assert [x[0] for x in rows[:-1]] == b["raw_timestamps"]
                    assert [x[1] for x in rows[:-1]] == b["context_tokens"]
                    assert [x[2] for x in rows[:-1]] == b["generated_tokens"]
                    ns = [exact_ns(x[0]) for x in rows]
                    assert [n-ns[0] for n in ns[:-1]] == b["original_offsets_ns"]
                    assert ns[-1]-ns[0] == b["original_horizon_ns"]
                    assert ns[-1]-ns[-2] == b["terminal_gap_ns"]
                    assert rows[-1] == [b["boundary_timestamp"], b["boundary_context_tokens"], b["boundary_generated_tokens"]]
                    assert all(a <= z for a,z in zip(ns,ns[1:]))
                    current += 1; completed += len(rows); rows = []
        assert current == 24 and index+1 == scans[trace]["rows"]
        result[trace] = dict(source_sha256=file_sha(path), full_rows=index+1, selected_rows_including_boundaries=completed, blocks=current)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("sources", "public", "arrivals", "prior-arrivals", "downloads", "output"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--prior-sources", type=Path, nargs=3, required=True)
    p.add_argument("--old-corpora", type=Path, nargs=3, default=[ROOT/n/"corpus/sources.json" for n in ("followup","optimization","resident_dispatch")])
    a=p.parse_args()
    assert not a.output.exists(), "fresh audit output required"
    source=load_sealed(a.sources); public=load_sealed(a.public,"public_manifest_sha256")
    assert source["schema"] == "stock_comparison_sources_v1"
    expected=json.loads(json.dumps(source))
    for b in expected["blocks"]:
        for r in b["sources"]:
            del r["text"]; del r["token_ids"]
    assert expected == {k:v for k,v in public.items() if k!="public_manifest_sha256"}
    for name,pin in source["raw_files"].items():
        assert file_sha(a.downloads/name)==pin["sha256"] and (a.downloads/name).stat().st_size==pin["bytes"]
    import tokenizers
    tokenizer=tokenizers.Tokenizer.from_file(str(a.downloads/"qwen-tokenizer.json"))
    fresh=[r for b in source["blocks"] for r in b["sources"]]
    assert len(source["blocks"])==24 and len(fresh)==768
    assert Counter((b["dataset"],b["split"]) for b in source["blocks"])=={(d,s):n for d in ("gsm8k","mbpp","wildchat") for s,n in (("pilot",2),("final",6))}
    assert all(len(b["sources"])==32 for b in source["blocks"])
    prior={}
    for path in a.prior_sources:
        old=load_sealed(path)
        for b in old["blocks"]:
            for r in b["sources"]:
                prior[r["dataset"],r["source_id"]]=r
    assert len(prior)==1927
    keys=[(r["dataset"],r["source_id"]) for r in fresh]
    assert len(set(keys))==768 and not set(keys)&prior.keys()
    raw_text_matches=verify_selected_texts(a.downloads,fresh)
    full_prior=[(k,r["text"]) for k,r in prior.items()]
    prefix_prior=[(k,tokenizer.decode(r["token_ids"],skip_special_tokens=False)) for k,r in prior.items()]
    old_sources=0;whole_hashes=set()
    for path in a.old_corpora:
        old=json.loads(path.read_text())
        for b in old["blocks"]:
            for r in b["sources"]:
                raw=path.parent/r["file"];assert file_sha(raw)==r["sha256"]
                text=raw.read_text(encoding="utf-8-sig")
                if r["cleaning"]!="none":
                    start=re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",text)
                    end=re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*",text)
                    assert start and end;text=text[start.end():end.start()].strip()
                whole_hashes.add(sha(norm(text).encode()))
                chars=16384
                while True:
                    ids=tokenizer.encode(text[:chars],add_special_tokens=False).ids
                    if len(ids)>2048 or chars>=len(text):break
                    chars*=2
                assert len(ids)>=2048
                key=("old",r["source_id"])
                long=tokenizer.decode(ids[:2048],skip_special_tokens=False)
                full_prior.append((key,long));prefix_prior.extend([(key,long),(key,tokenizer.decode(ids[:512],skip_special_tokens=False))])
                old_sources+=1
    assert old_sources==56
    fresh_full=[];fresh_prefix=[]
    for key,r in zip(keys,fresh):
        ids=tokenizer.encode(r["text"],add_special_tokens=False).ids
        assert len(ids)==r["full_token_count"] and ids[:512]==r["token_ids"] and sha(r["text"].encode())==r["text_sha256"]
        assert sha(norm(r["text"]).encode()) not in whole_hashes
        fresh_full.append((key,r["text"]));fresh_prefix.append((key,tokenizer.decode(ids[:512],skip_special_tokens=False)))
    comparisons={"full":no_near_pairs(fresh_full,full_prior,"full"),"prefix":no_near_pairs(fresh_prefix,prefix_prior,"prefix")}
    arrivals=load_sealed(a.arrivals)
    assert arrivals["source_selection_sha256"]==source["content_sha256"]
    prior_arrivals=json.loads(a.prior_arrivals.read_text());prior_arrivals["_file_sha256"]=file_sha(a.prior_arrivals)
    scans={}
    for ref in arrivals["preparation_provenance"]["scan_files"]:
        path=a.arrivals.parent/ref["path"];assert file_sha(path)==ref["sha256"]
        scan=json.loads(path.read_text());scans[scan["trace"]]=scan
    verify_arrival_structure(arrivals,prior_arrivals,scans)
    raw=verify_raw_arrivals(a.downloads,arrivals,scans)
    result=dict(passed=True,scope="Independent implementation, same team; source/row evidence only, no GPU or outcome audit",source_sha256=source["content_sha256"],arrival_sha256=arrivals["content_sha256"],documents=768,selected_text_exact_raw_source_matches=raw_text_matches,prior_unique_documents=1927,old_captured_sources=56,full_and_prefix_near_threshold=.8,length_filtered_pair_comparisons=comparisons,arrival_sources=raw,prior_arrival_blocks=92,selected_arrival_blocks=48,source_rows_disjoint=True,new_production_days=False,audit_script_sha256=file_sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
