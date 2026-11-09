#!/usr/bin/env python3
"""Prepare fresh stock-vLLM comparison content and 4096-job source calendars."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stock_comparison.data import (CONTENT_SEED, ARRIVAL_SEED, DATASETS, digest, normalized,
    exclude_and_deduplicate, split_sources, scan_fresh_arrivals, assign_fresh_arrivals)
from followup.router_tapes import MODEL, REVISION, file_sha256


def prefix_ids(tokenizer, text, limit):
    # Tokenize only a sufficiently long leading slice for large historical
    # books. More than limit tokens prevents slice-end tokenization ambiguity.
    chars = 16384
    while True:
        part = text[:chars]
        ids = tokenizer.encode(part, add_special_tokens=False).ids
        if len(ids) > limit or chars >= len(text):
            return ids[:limit]
        chars *= 2


def prior_exclusions(paths, corpus_paths, tokenizer):
    refs, seen, manifests = [], {}, []
    if len(paths) != 3 or len(corpus_paths) != 3:
        raise ValueError("All three prior selections and all three old corpus manifests are required")
    for path in paths:
        v = json.loads(path.read_text())
        if v["content_sha256"] != digest({k: x for k, x in v.items() if k != "content_sha256"}):
            raise ValueError("Prior private selection hash differs")
        n = sum(len(b["sources"]) for b in v["blocks"])
        if n != 1920:
            raise ValueError("Expected complete 1920-document historical selection")
        manifests.append(dict(path=path.parent.name+"/"+path.name, sha256=file_sha256(path), content_sha256=v["content_sha256"], documents=n))
        for b in v["blocks"]:
            for r in b["sources"]:
                key = (r["dataset"], r["source_id"])
                if key in seen:
                    if seen[key] != r["text_sha256"]:
                        raise ValueError("Prior source identity changed text")
                    continue
                if digest(r["text"].encode()) != r["text_sha256"]:
                    raise ValueError("Prior text hash differs")
                ids = tokenizer.encode(r["text"], add_special_tokens=False).ids[:512]
                if ids != r["token_ids"]:
                    raise ValueError("Prior captured prefix tokenization changed")
                refs.append(dict(dataset=r["dataset"], source_id=r["source_id"], text=r["text"],
                                 prefix_views=[tokenizer.decode(ids, skip_special_tokens=False)]))
                seen[key] = r["text_sha256"]
    old_count = 0
    for path in corpus_paths:
        v = json.loads(path.read_text())
        manifests.append(dict(path=str(path.relative_to(ROOT)), sha256=file_sha256(path),
                              documents=sum(len(b["sources"]) for b in v["blocks"])))
        for b in v["blocks"]:
            for r in b["sources"]:
                source = path.parent / r["file"]
                if file_sha256(source) != r["sha256"]:
                    raise ValueError("Old corpus raw bytes changed")
                text = source.read_text(encoding="utf-8-sig")
                if r["cleaning"] == "content_between_gutenberg_start_end_markers":
                    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
                    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
                    if not start or not end or end.start() <= start.end():
                        raise ValueError("Old Gutenberg cleaning boundary changed")
                    text = text[start.end():end.start()].strip()
                elif r["cleaning"] != "none":
                    raise ValueError("Unknown historical source cleaning")
                ids = prefix_ids(tokenizer, text, 2048)
                if len(ids) != 2048:
                    raise ValueError("Historical source lacks its 2048 captured tokens")
                long_view = tokenizer.decode(ids, skip_special_tokens=False)
                refs.append(dict(dataset="old-capture", source_id=r["source_id"], text=long_view,
                    prefix_views=[long_view, tokenizer.decode(ids[:512], skip_special_tokens=False)],
                    whole_source_normalized_sha256=digest(normalized(text).encode())))
                old_count += 1
    if old_count != 56:
        raise ValueError("All 56 old captured sources required")
    return refs, manifests, dict(prior_selection_unique_documents=len(seen),
                                prior_selection_unique_by_dataset=dict(Counter(k[0] for k in seen)),
                                old_capture_sources=old_count,
                                old_prefix_views="actual first2048 tokens and common first512-token view, plus normalized full-source exact hash")


def candidates(downloads, tokenizer):
    import pyarrow.parquet as pq
    raw = []
    wild = []
    for batch in pq.ParquetFile(downloads/"wildchat-first-shard.parquet").iter_batches(batch_size=1024):
        wild.extend(batch.to_pylist())
        if len(wild) >= 8192:
            break
    for r in wild[:8192]:
        if r["language"] != "English" or r["toxic"]:
            continue
        turns = [t for t in r["conversation"] if t["role"] == "user"]
        if not turns or turns[0].get("toxic") or not turns[0]["content"].strip():
            continue
        raw.append(dict(dataset="wildchat", source_id=r["conversation_id"], text=turns[0]["content"],
                        source_row_kind="first_user_turn", original_split="train"))
    for r in map(json.loads, (downloads/"mbpp-full.jsonl").read_text().splitlines()):
        raw.append(dict(dataset="mbpp", source_id=str(r["task_id"]), text=r["text"],
                        source_row_kind="task_description_only", original_split="test" if 11 <= r["task_id"] <= 510 else "other_original_split"))
    for i, r in enumerate(map(json.loads, (downloads/"gsm8k.jsonl").read_text().splitlines())):
        raw.append(dict(dataset="gsm8k", source_id=f"test-{i}", text=r["question"], source_row_kind="question_only", original_split="test"))
    kept, short = [], []
    for r in raw:
        ids = tokenizer.encode(r["text"], add_special_tokens=False).ids
        if len(ids) < 8:
            short.append([r["dataset"], r["source_id"]])
            continue
        kept.append({**r, "full_token_count":len(ids), "captured_token_count":min(512, len(ids)),
                     "token_ids":ids[:512], "token_ids_sha256":digest(ids[:512]),
                     "captured_prefix_text":tokenizer.decode(ids[:512], skip_special_tokens=False)})
    return kept, short


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("downloads", "output", "public-output"):
        p.add_argument("--"+k, type=Path, required=True)
    p.add_argument("--prior-selections", type=Path, nargs=3, required=True)
    p.add_argument("--old-corpus-manifests", type=Path, nargs=3, default=[ROOT/f"{n}/corpus/sources.json" for n in ("followup", "optimization", "resident_dispatch")])
    p.add_argument("--prior-arrivals", type=Path, default=ROOT/"results/generalization/arrivals-v1/arrival-manifest.json")
    p.add_argument("--content-only", action="store_true", help="Prepare content first; leave source-calendar scanning for a separate approved run")
    args = p.parse_args()
    if args.output.exists() or args.public_output.exists():
        raise FileExistsError("Fresh private/public output directories required")
    import tokenizers
    # The accepted previous selection pins the unchanged public files, metadata,
    # license records and exact checkpoint tokenizer. No new download is used.
    prior = json.loads(args.prior_selections[-1].read_text())
    for name, entry in prior["raw_files"].items():
        path = args.downloads/name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("Pinned source file changed: "+name)
    tokenizer = tokenizers.Tokenizer.from_file(str(args.downloads/"qwen-tokenizer.json"))
    refs, exclusions, exclusion_counts = prior_exclusions(args.prior_selections, args.old_corpus_manifests, tokenizer)
    raw, short = candidates(args.downloads, tokenizer)
    eligible, rejected = exclude_and_deduplicate(raw, refs)
    counts = dict(Counter(r["dataset"] for r in eligible))
    print(json.dumps({"phase":"capacity", "fresh_eligible":counts, "exclusions":exclusion_counts}), flush=True)
    blocks = split_sources(eligible)
    selection = dict(seed=CONTENT_SEED, pilot_blocks_per_dataset=2, final_blocks_per_dataset=6,
        documents_per_block=32, minimum_tokens=8, maximum_captured_tokens_per_document=512,
        near_duplicate_jaccard=.8, shingle_width=5, eligible_counts=counts,
        deduplicated_before_split=True, captured_prefix_deduplicated=True,
        wildchat_frame=prior["selection"]["wildchat_frame"], prefix_decoder_tokenizers_version=tokenizers.__version__,
        no_routing_or_performance_selection=True)
    manifest = dict(schema="stock_comparison_sources_v1", study_id="stock_comparison_v1",
        capture_format_note="Same authenticated per-document capture format; distinct study ID, source selection hash and block IDs.",
        seed=CONTENT_SEED, arrival_seed=ARRIVAL_SEED, source_model=MODEL, source_revision=REVISION,
        formatting=prior["formatting"], selection=selection, raw_files=prior["raw_files"],
        prior_exclusions=exclusions, exclusion_counts=exclusion_counts, blocks=blocks,
        duplicate_rejections=rejected, short_rejections=short,
        preparation_source_sha256=file_sha256(__file__), data_contract_sha256=file_sha256(ROOT/"stock_comparison/data.py"),
        capture_limit_note="Prompt-only causal prefill layer0; no answers, continuations or fabricated expert routes. Source-disjoint from all earlier selections/captures, not proven absent from checkpoint pretraining.")
    manifest["content_sha256"] = digest(manifest)
    args.output.mkdir(parents=True)
    args.public_output.mkdir(parents=True)
    (args.output/"sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    public = json.loads(json.dumps(manifest))
    for block in public["blocks"]:
        for r in block["sources"]:
            del r["text"]
            del r["token_ids"]
    public["public_manifest_sha256"] = digest(public)
    (args.public_output/"source-manifest.json").write_text(json.dumps(public, indent=2)+"\n")
    print(json.dumps({"phase":"content_ready", "blocks":len(blocks), "documents":sum(len(b['sources']) for b in blocks),
                      "content_sha256":manifest["content_sha256"],
                      "tokens_by_block":{b['block_id']:sum(r['captured_token_count'] for r in b['sources']) for b in blocks}}), flush=True)
    if args.content_only:
        return
    previous = json.loads(args.prior_arrivals.read_text())
    if len(previous["blocks"]) != 92:
        raise ValueError("Complete previous 92 arrival intervals required")
    scans = []
    def progress(trace, day, stats):
        print(json.dumps({"phase":"arrival_scan", "trace":trace, "day":day, **stats}), flush=True)
    for trace, filename in (("code", "AzureLLMInferenceTrace_code_1week.csv"), ("conversation", "AzureLLMInferenceTrace_conv_1week.csv")):
        hashes = {b["source_sha256"] for b in previous["blocks"] if b["trace"] == trace}
        if len(hashes) != 1:
            raise ValueError("Inconsistent prior weekly source pins")
        scan = scan_fresh_arrivals(args.downloads/filename, trace=trace, expected_sha256=next(iter(hashes)), previous_blocks=previous["blocks"], progress=progress)
        scans.append(scan)
        (args.public_output/f"{trace}-scan.json").write_text(json.dumps(scan, indent=2)+"\n")
    arrivals = dict(schema="stock_comparison_arrivals_v1", seed=ARRIVAL_SEED, jobs_per_trial=4096,
        previous_manifest_sha256=file_sha256(args.prior_arrivals), previous_blocks_excluded=92,
        source_rows_and_boundaries_disjoint=True, new_calendar_days=False,
        mapping="Each source timestamp is one fixed-token layer job; original request token lengths are retained metadata, not executed.",
        date_scope="Fresh source rows from previously used service weeks; conditional calendar evidence, not unseen production dates.",
        blocks=assign_fresh_arrivals(scans, blocks), scans={s['trace']:{k:v for k,v in s.items() if k!='retained_candidates'} for s in scans})
    arrivals["content_sha256"] = digest(arrivals)
    (args.public_output/"arrival-manifest.json").write_text(json.dumps(arrivals, indent=2)+"\n")
    print(json.dumps({"phase":"arrivals_ready", "blocks":len(arrivals['blocks']), "jobs":4096, "fresh_source_rows":True, "fresh_calendar_days":False}), flush=True)


if __name__ == "__main__":
    main()
