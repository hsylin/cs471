"""Prepare separate full-text cohorts; never pool datasets into one workload.

Adapters supply authentic messages and a source/group identity. This selector
uses only source identities, text duplicates and the declared context limit.
No routing or performance observations enter selection. Final candidates remain
reserved until pilot precision and execution duration support a frozen design.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import os
from pathlib import Path
import statistics

from full_serving.data import digest, file_digest, normalized, verify_pinned_file, MODEL_REVISION

DATASETS = ("sharegpt", "instructcoder", "numinamath", "longbench")
SEED = 471111
PUBLIC_FIELDS = {"id", "dataset", "source_id", "source_split", "source_row_index", "source_revision",
                 "group_id", "task", "input_tokens", "messages_sha256", "normalized_messages_sha256",
                 "first_user_normalized_sha256", "prompt_token_ids_sha256", "max_tokens", "split",
                 "content_block", "padding", "truncation", "assistant_initial_segment"}


def public_request(row):
    public = {k: v for k, v in row.items() if k in PUBLIC_FIELDS}
    if any(not isinstance(v, (str, int, bool, type(None))) for v in public.values()):
        raise ValueError("Public request metadata must be scalar")
    return public


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def candidate_frame(path, dataset, *, seed=SEED, keep=1024):
    """Stream large releases, retaining the first fixed hash ranks in bounded RAM."""
    heap, ids = [], set()
    with Path(path).open() as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            identity = row["id"]
            if row["dataset"] != dataset or identity in ids:
                raise ValueError("Mixed source or duplicate candidate identity")
            ids.add(identity)
            rank = int(digest([seed, dataset, identity]), 16)
            entry = (-rank, -index, row)
            if len(heap) < keep:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    return [item[2] for item in heap], {"input_candidate_rows": len(ids), "retained_hash_ranks": len(heap),
                                      "frame_rule": "lowest 1024 SHA256(seed,dataset,ID) ranks; no performance/length filtering"}


def write_private(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return file_digest(path)


def message_key(messages):
    return digest([{**m, "content": normalized(m["content"])} for m in messages])


def first_user_key(messages):
    return digest(normalized(next(m["content"] for m in messages if m["role"] == "user")).encode())


def smoke_ids(rows):
    ordered = sorted(rows, key=lambda r: (r["input_tokens"], r["id"]))
    return [ordered[(i * (len(ordered) - 1)) // 7]["id"] for i in range(8)]


def select_cohort(candidates, dataset, tokenizer, *, seen_content=None, seen_first=None,
                  seed=SEED, pilot=64, final=64, max_context=32768, max_output=2048):
    if dataset not in DATASETS or pilot < 8 or final < 1:
        raise ValueError("Unsupported dataset or cohort allocation")
    seen_content = set(seen_content or ())
    seen_first = set(seen_first or ())
    ordered = sorted(candidates, key=lambda r: (digest([seed, dataset, r["id"]]), r["id"]))
    ids, groups, selected, rejected = set(), set(), [], []
    for row in ordered:
        identity = row["id"]
        if identity in ids:
            raise ValueError("Duplicate adapter identity")
        ids.add(identity)
        if row["dataset"] != dataset:
            raise ValueError("Mixed dataset candidate file")
        messages = row["messages"]
        if (not messages or messages[-1].get("role") != "user"
                or any(set(m) != {"role", "content"} or m["role"] not in {"user", "assistant", "system"}
                       or not isinstance(m["content"], str) for m in messages)):
            raise ValueError("Invalid source message contract")
        group = str(row.get("group_id", row.get("source_id", identity)))
        content, first = message_key(messages), first_user_key(messages)
        reason = ("duplicate_source_group" if group in groups else
                  "duplicate_messages" if content in seen_content else
                  "duplicate_first_user" if first in seen_first else None)
        if reason:
            rejected.append({"id": identity, "reason": reason})
            continue
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                               truncation=False, add_generation_prompt=True,
                                               enable_thinking=False)
        if not tokens or len(tokens) + max_output > max_context:
            rejected.append({"id": identity, "reason": "empty_or_over_context", "input_tokens": len(tokens)})
            continue
        groups.add(group)
        seen_content.add(content)
        seen_first.add(first)
        selected.append({**row, "group_id": group, "input_tokens": len(tokens),
                         "messages_sha256": digest(messages), "normalized_messages_sha256": content,
                         "first_user_normalized_sha256": first, "prompt_token_ids_sha256": digest(tokens),
                         "max_tokens": max_output, "split": "pilot" if len(selected) < pilot else "final",
                         "content_block": dataset + ("-pilot-00" if len(selected) < pilot else "-final-00")})
        if len(selected) == pilot + final:
            break
    if len(selected) != pilot + final:
        raise ValueError(f"{dataset}: only {len(selected)} valid distinct groups; need {pilot+final}")
    lengths = [r["input_tokens"] for r in selected]
    manifest = {"dataset": dataset, "seed": seed, "selection": "ascending SHA256(seed,dataset,source ID)",
                "candidate_count": len(candidates), "examined_count": len(ids),
                "rejected": rejected, "rejection_counts": dict(Counter(r["reason"] for r in rejected)),
                "selected_count": len(selected), "pilot_count": pilot, "final_candidate_count": final,
                "lengths": {"minimum": min(lengths), "maximum": max(lengths), "median": statistics.median(lengths),
                            "mean": statistics.mean(lengths), "over_512": sum(x > 512 for x in lengths),
                            "over_2048": sum(x > 2048 for x in lengths)},
                "task_counts": dict(Counter(r.get("task", dataset) for r in selected)),
                "smoke_request_ids": smoke_ids(selected[:pilot]),
                "requests": [public_request(r) for r in selected]}
    return selected, manifest


def mapping(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError("Duplicate dataset argument")
        result[name] = Path(path)
    if set(result) != set(DATASETS):
        raise ValueError("Provide all four new datasets separately")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", action="append", required=True, help="DATASET=private_candidate_jsonl")
    p.add_argument("--source-manifest", action="append", required=True, help="DATASET=public_source_json")
    p.add_argument("--wildchat-private", type=Path, required=True)
    p.add_argument("--wildchat-manifest", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--public-output", type=Path, required=True)
    args = p.parse_args()
    candidates, sources = mapping(args.candidates), mapping(args.source_manifest)
    root = Path(__file__).resolve().parents[1]
    if root in args.private_output.resolve().parents or args.private_output.resolve() == root:
        raise ValueError("Private texts must remain outside Git")
    if args.private_output.exists() or args.public_output.exists():
        raise FileExistsError("Use fresh output directories")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        verify_pinned_file(args.tokenizer_dir / name, name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    from full_serving.verify_data import verify_bundle
    wildchat_verification = verify_bundle(args.wildchat_private, args.wildchat_manifest)
    all_wildchat = load_rows(args.wildchat_private / "requests.jsonl")
    seen_content = {message_key(r["messages"]) for r in all_wildchat}
    seen_first = {first_user_key(r["messages"]) for r in all_wildchat}
    args.private_output.mkdir(parents=True, mode=0o700)
    args.public_output.mkdir(parents=True)
    cohorts = []
    # Retain original WildChat membership; reserve the other five final bundles.
    wildchat = load_rows(args.wildchat_private / "pilot.jsonl") + load_rows(args.wildchat_private / "final-00.jsonl")
    wild_public = json.loads(args.wildchat_manifest.read_text())
    inputs = [("wildchat", wildchat, {"dataset": "wildchat", "parent_manifest_sha256": file_digest(args.wildchat_manifest),
               "smoke_request_ids": wild_public["smoke_request_ids"], "other_final_bundles": "reserved, not discarded or measured"})]
    for dataset in DATASETS:
        frame, frame_stats = candidate_frame(candidates[dataset], dataset)
        rows, cohort = select_cohort(frame, dataset, tokenizer,
                                     seen_content=seen_content, seen_first=seen_first)
        cohort["streaming_frame"] = frame_stats
        seen_content.update(message_key(r["messages"]) for r in rows)
        seen_first.update(first_user_key(r["messages"]) for r in rows)
        cohort["candidate_file_sha256"] = file_digest(candidates[dataset])
        cohort["source_manifest_sha256"] = file_digest(sources[dataset])
        cohort["source_manifest_path"] = str(sources[dataset].resolve().relative_to(root))
        inputs.append((dataset, rows, cohort))
    for dataset, rows, cohort in inputs:
        private_dir = args.private_output / dataset
        pilot = [r for r in rows if r["split"] == "pilot"]
        final = [r for r in rows if r["split"] == "final"]
        by_id = {r["id"]: r for r in rows}
        cohorts.append({**cohort, "private_files": {
            "requests.jsonl": write_private(private_dir / "requests.jsonl", rows),
            "pilot.jsonl": write_private(private_dir / "pilot.jsonl", pilot),
            "final.jsonl": write_private(private_dir / "final.jsonl", final),
            "smoke.jsonl": write_private(private_dir / "smoke.jsonl", [by_id[i] for i in cohort["smoke_request_ids"]])}})
    manifest = {"schema": "full_serving_separate_suite_v1", "selection_status": "pilot_ready_final_candidates_not_frozen",
                "source_model_revision": MODEL_REVISION, "seed": SEED,
                "datasets_kept_separate": True, "pooled_speedup_claim": False,
                "formatting": {"enable_thinking": False, "add_generation_prompt": True, "truncation": False,
                               "max_model_len": 32768, "phase": "full model prefill and generated decode"},
                "generation": {"max_tokens": 2048, "temperature": 0, "ignore_eos": False},
                "selection_order": ["wildchat", *DATASETS], "cohorts": cohorts,
                "wildchat_admission": wildchat_verification,
                "burstgpt": "Trace-only diagnostic; no BurstGPT prompt dataset or authentic joint content-arrival claim",
                "preparation_sources": {str(p.relative_to(root)): file_digest(p) for p in [Path(__file__), root / "full_serving/data.py"]}}
    manifest["manifest_sha256"] = digest(manifest)
    path = args.public_output / "suite-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"public_manifest": str(path), "datasets": [c["dataset"] for c in cohorts],
                      "pilot_requests": 320, "reserved_final_candidates": 320}))


if __name__ == "__main__":
    main()
