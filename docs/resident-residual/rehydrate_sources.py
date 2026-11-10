#!/usr/bin/env python3
"""Reconstruct the frozen private input manifest; no selection or GPU work.

This standalone support program imports no experiment modules. Only the removed
text and token_ids fields are restored from pinned public source bytes. It is
specific to the resident-residual v1 selection, not a general dataset selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

CONTENT_SHA = "777b3e2752745494c686e71249ebf4bd3b18b172ab4bc3c26f0907ff88fab92c"
PUBLIC_SHA = "74328bff41783e4adf18f1595738062457fb30a3aec7d6991a850e3e8fa308c1"
PRIVATE_FILE_SHA = "050e3e13801b366352e24568f57a9dc18f56b8558afd5f76f3de54bae4fe1d70"
REQUIRED_FILES = (
    "gsm8k-train.jsonl", "wildchat-first-shard.parquet", "qwen-tokenizer.json",
    "gsm8k-license.txt", "wildchat-license.md",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_public(public):
    require(public.get("schema") == "resident_residual_sources_v1"
            and public.get("study_id") == "resident_residual_v1", "Wrong study/schema")
    require(public.get("content_sha256") == CONTENT_SHA, "Wrong frozen selection")
    require(public.get("public_manifest_sha256") == PUBLIC_SHA
            and sha({k: v for k, v in public.items() if k != "public_manifest_sha256"}) == PUBLIC_SHA,
            "Public manifest changed")
    rows = [r for b in public["blocks"] for r in b["sources"]]
    require(len(public["blocks"]) == 20 and len(rows) == 640, "Incomplete frozen matrix")
    require(Counter(r["dataset"] for r in rows) == {"gsm8k_train": 320, "wildchat": 320},
            "Wrong source frames")
    keys = {(r["dataset"], r["source_id"]) for r in rows}
    require(len(keys) == 640, "Duplicate selected identity")
    require(all("text" not in r and "token_ids" not in r for r in rows),
            "Expected public manifest with exactly those fields removed")
    return keys


def restore(manifest_path, raw_dir, train_source=None):
    public = json.loads(Path(manifest_path).read_text())
    wanted = validate_public(public)
    paths = {name: Path(raw_dir) / name for name in REQUIRED_FILES}
    if train_source is not None:
        paths["gsm8k-train.jsonl"] = Path(train_source)
    verified = {}
    for name, path in paths.items():
        pin = public["raw_files"][name]
        require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"],
                "Pinned raw bytes differ: " + name)
        verified[name] = dict(bytes=pin["bytes"], sha256=pin["sha256"])

    # Recover selected identities only. No ranking, deduplication or replacement
    # is performed; the public manifest already fixes all membership and order.
    found = {}
    lines = paths["gsm8k-train.jsonl"].read_text().splitlines()
    require(len(lines) == 7473, "Incomplete training-question source")
    for index, line in enumerate(lines):
        key = ("gsm8k_train", f"train-{index}")
        if key in wanted:
            found[key] = json.loads(line)["question"]

    import pyarrow.parquet as pq
    index = 0
    for batch in pq.ParquetFile(paths["wildchat-first-shard.parquet"]).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if index == 8192:
                break
            index += 1
            key = ("wildchat", row["conversation_id"])
            if key not in wanted:
                continue
            users = [turn for turn in row["conversation"] if turn["role"] == "user"]
            require(row["language"] == "English" and not row["toxic"] and users
                    and not users[0].get("toxic") and users[0]["content"].strip(),
                    "Selected WildChat identity violates the frozen source frame")
            text = users[0]["content"]
            require(key not in found or found[key] == text, "Ambiguous selected source identity")
            found[key] = text
        if index == 8192:
            break
    require(index == 8192 and set(found) == wanted, "Selected public source identity not found")

    import tokenizers
    tokenizer = tokenizers.Tokenizer.from_file(str(paths["qwen-tokenizer.json"]))
    restored = copy.deepcopy(public)
    del restored["public_manifest_sha256"]
    for block in restored["blocks"]:
        for position, row in enumerate(block["sources"]):
            text = found[row["dataset"], row["source_id"]]
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            captured = ids[:512]
            require(sha(text.encode()) == row["text_sha256"], "Selected text digest differs")
            require(len(ids) == row["full_token_count"] and len(captured) == row["captured_token_count"],
                    "Token counts differ")
            require(sha(captured) == row["token_ids_sha256"], "Selected token IDs differ")
            require(sha(normalized(text).encode()) == row["normalized_text_sha256"],
                    "Normalized full text differs")
            prefix = tokenizer.decode(captured, skip_special_tokens=False)
            require(sha(normalized(prefix).encode()) == row["captured_prefix_normalized_sha256"],
                    "Model-visible prefix differs")
            # Restore original field positions as well as values so the
            # deterministic pretty-printed private file is byte-identical.
            filled = {}
            for key, value in row.items():
                filled[key] = value
                if key == "source_id":
                    filled["text"] = text
                elif key == "captured_token_count":
                    filled["token_ids"] = captured
            require({k: v for k, v in filled.items() if k not in ("text", "token_ids")} == row,
                    "A public field changed")
            block["sources"][position] = filled

    require(sha({k: v for k, v in restored.items() if k != "content_sha256"}) == CONTENT_SHA,
            "Rehydrated canonical content differs; do not capture")
    payload = (json.dumps(restored, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    require(sha(payload) == PRIVATE_FILE_SHA, "Original private-file byte identity differs")
    report = dict(schema="resident_residual_rehydration_v1", passed=True,
                  documents=640, groups=20, content_sha256=CONTENT_SHA,
                  public_manifest_sha256=PUBLIC_SHA, private_file_sha256=sha(payload),
                  required_raw_files=verified, tokenizers_version=tokenizers.__version__,
                  no_new_selection=True, no_prior_private_manifests_required=True,
                  restored_fields=["text", "token_ids"], gpu_work=False)
    return payload, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, help="Optional path override for the same pinned training file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.report and args.report.exists()):
        raise FileExistsError("Fresh output/report paths required")
    payload, report = restore(args.manifest, args.raw_dir, args.train_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
