"""Build separate, untokenized InstructCoder and NuminaMath-1.5 input pools.

Raw public files and private prompt JSONL stay outside Git. Selection uses
stable source row IDs only; it does not use lengths, expert routing, outputs,
or timing. Reference output/solution/answer fields never enter the prompt.
The resulting pool is a deterministic sampling frame, not a frozen pilot/final
assignment and not a claim to replicate METRO's undisclosed exact workload.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path


SEED = 471110
SOURCE_PINS = {
    "instructcoder": {
        "repository": "likaixin/InstructCoder",
        "revision": "6a778a720284d6520b56bd03d5c3070930d41071",
        "files": {"train.json": "2a2157045c254968382b28561623d3fc26d36818fac81ea726e0d67452582ef2"},
    },
    "numinamath": {
        "repository": "AI-MO/NuminaMath-1.5",
        "revision": "1b05109f9e5c1ad06c0663519502416c30b300f8",
        "files": {
            "train-00000-of-00003.parquet": "1e37cedcc5104c0d5a9afbf8b26328a59d7005cd37c2bd08352419f5f0e03214",
            "train-00001-of-00003.parquet": "bc48fdddc1ab4e559727ad4f1f03b0263785c1c16b7964fb885fba07e18d6a39",
            "train-00002-of-00003.parquet": "951aa899fe2a1b64f5b40c6636372d35b9db4f0b9c922e874d4bf357db64da66",
        },
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def input_messages(dataset, row):
    """Only source instruction/input or problem fields can affect the prompt."""
    if dataset == "instructcoder":
        instruction, code = row.get("instruction"), row.get("input")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("missing_or_empty_instruction")
        if not isinstance(code, str):
            raise ValueError("nontext_input_code")
        content = instruction + ("\n\nInput code:\n" + code if code else "")
    elif dataset == "numinamath":
        content = row.get("problem")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("missing_or_empty_problem")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return [{"role": "user", "content": content}]


def iter_rows(dataset, source_dir):
    root = Path(source_dir) / dataset
    if dataset == "instructcoder":
        rows = json.loads((root / "train.json").read_text())
        if not isinstance(rows, list):
            raise ValueError("Expected InstructCoder JSON list")
        for index, row in enumerate(rows):
            yield f"train.json:{index}", row
    elif dataset == "numinamath":
        import pyarrow.parquet as pq
        for filename in SOURCE_PINS[dataset]["files"]:
            parquet = pq.ParquetFile(root / filename)
            # Deliberately do not deserialize answers, solutions or chat turns.
            if "problem" not in parquet.schema_arrow.names:
                raise ValueError("NuminaMath shard lacks a problem column")
            columns = [name for name in ("problem", "source") if name in parquet.schema_arrow.names]
            index = 0
            for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    yield f"{filename}:{index}", row
                    index += 1
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def select_candidates(dataset, rows, count=1024, seed=SEED):
    if count <= 0:
        raise ValueError("Positive candidate count required")
    heap, exclusions = [], {}
    seen_ids = set()
    total = 0
    for source_id, row in rows:
        if source_id in seen_ids:
            raise ValueError("Duplicate source identity")
        seen_ids.add(source_id)
        total += 1
        try:
            messages = input_messages(dataset, row)
        except ValueError as error:
            reason = str(error)
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        score = digest(["full-serving-code-math-frame-v1", seed, dataset, source_id])
        record = {"id": f"{dataset}:{source_id}", "dataset": dataset,
                  "messages": messages, "source_id": source_id, "source_split": "train",
                  "source_repository": SOURCE_PINS[dataset]["repository"],
                  "source_revision": SOURCE_PINS[dataset]["revision"],
                  "frame_selection_score": score, "messages_sha256": digest(messages)}
        if dataset == "numinamath" and isinstance(row.get("source"), str):
            record["upstream_subset"] = row["source"]
        entry = (-int(score, 16), source_id, record)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    if len(heap) < count:
        raise ValueError(f"Only {len(heap)} valid candidates, requested {count}")
    selected = sorted((entry[2] for entry in heap), key=lambda row: (row["frame_selection_score"], row["source_id"]))
    return selected, {"source_rows_scanned": total, "schema_exclusions": exclusions,
                      "eligible_rows": total - sum(exclusions.values()), "candidate_count": len(selected)}


def source_evidence(dataset, downloads):
    pin = SOURCE_PINS[dataset]
    result = {
        "metro_source": "https://arxiv.org/html/2512.09277v1#S6.SS1",
        "metro_publication_label": "arXiv v1 preprint (10 December 2025); this record does not establish later publication status",
        "metro_match": "Same named dataset family; exact METRO rows, split, revision, prompt template and real-system generation settings are not disclosed in the cited setup.",
        "repository": pin["repository"], "revision": pin["revision"], "source_split": "train",
        "card_url": f"https://huggingface.co/datasets/{pin['repository']}/blob/{pin['revision']}/README.md",
    }
    if dataset == "instructcoder":
        result.update({
            "dataset_name": "InstructCoder", "variant": "Official linked HF train.json, not EditEval, github_seed or additional_seed",
            "publication": {"title": "InstructCoder: Instruction Tuning Large Language Models for Code Editing",
                            "venue": "ACL 2024 Student Research Workshop", "url": "https://aclanthology.org/2024.acl-srw.52/"},
            "data_origin": "Instruction/input/output triplets synthesized using ChatGPT from GitHub-commit seed tasks; not a production user traffic trace.",
            "prompt_fields": ["instruction", "input"], "omitted_reference_fields": ["output"],
            "prompt_format": "instruction verbatim + '\\n\\nInput code:\\n' + input verbatim when input is nonempty; one user turn",
            "license": {"status": "official repository MIT; separate HF dataset license scope unverified",
                        "repository_license": "MIT", "hf_card_license_field": None,
                        "note": "The linked official GitHub repository has MIT LICENSE and links this HF training dataset. The pinned HF dataset has no LICENSE file or dataset license metadata; its card discusses model licenses, not an explicit separate dataset grant. Preserve this distinction.",
                        "repository_revision": downloads["github_revision"],
                        "url": f"https://github.com/qishenghu/InstructCoder/blob/{downloads['github_revision']}/LICENSE"},
        })
    else:
        result.update({
            "dataset_name": "NuminaMath-1.5", "variant": "NuminaMath-1.5 full train split across all three official Parquet shards; not CoT or TIR variants",
            "data_origin": "Curated math problems from competition/exam/forum sources plus synthetic subsets as documented in the card; not a production arrival trace.",
            "metro_variant_evidence": "METRO introduction explicitly names NuminaMath-1.5, and reference [27] gives its HF URL; section VI-A shortens the name to NuminaMath.",
            "prompt_fields": ["problem"], "omitted_reference_fields": ["solution", "answer", "messages", "ref_solution"],
            "prompt_format": "problem field verbatim; one user turn; no reference CoT or answer appended",
            "license": {"status": "explicit dataset card license", "spdx": "Apache-2.0",
                        "url": "https://www.apache.org/licenses/LICENSE-2.0", "card_section": "Licensing Information"},
        })
    return result


def prepare(source_dir, private_dir, public_dir, count=1024, seed=SEED):
    source_dir, private_dir, public_dir = map(Path, (source_dir, private_dir, public_dir))
    repo = Path(__file__).resolve().parents[1]
    if private_dir.resolve().is_relative_to(repo):
        raise ValueError("Private candidate payloads must be outside the repository")
    downloads_path = source_dir / "code-math-downloads.json"
    downloads = json.loads(downloads_path.read_text())
    download_records = {record["path"]: record for record in downloads["files"]}
    # This file contains only URLs, paths, sizes and digests, never prompt text.
    # Publish it so raw files can be independently re-fetched outside Git.
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "code-math-downloads.json").write_bytes(downloads_path.read_bytes())
    outputs = []
    for dataset, pin in SOURCE_PINS.items():
        metadata_path = source_dir / f"{dataset}-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata["sha"] != pin["revision"] or metadata.get("gated") not in (False, None):
            raise ValueError("Unexpected source revision or gated-access status")
        provenance = []
        for filename, expected in pin["files"].items():
            relative = f"{dataset}/{filename}"
            source = source_dir / relative
            record = download_records[relative]
            actual = file_sha256(source)
            if actual != expected or record["sha256"] != expected:
                raise ValueError(f"Pinned source SHA mismatch: {relative}")
            provenance.append({**record, "verified_sha256": actual})
        supporting = []
        for relative, record in download_records.items():
            if relative.startswith(dataset + "/") and relative not in {item["path"] for item in provenance}:
                if file_sha256(source_dir / relative) != record["sha256"]:
                    raise ValueError(f"Supporting evidence changed: {relative}")
                supporting.append(record)
        selected, counts = select_candidates(dataset, iter_rows(dataset, source_dir), count, seed)
        private_dir.mkdir(parents=True, exist_ok=True)
        payload = private_dir / f"{dataset}-candidates.jsonl"
        with payload.open("w") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.chmod(payload, 0o600)
        manifest = {"schema": "full_serving_code_math_source_v1", "dataset": dataset,
                    **source_evidence(dataset, downloads), **counts,
                    "selection": {"seed": seed, "method": "Lowest SHA256 canonical JSON ['full-serving-code-math-frame-v1',seed,dataset,source_id] across every valid train row; no length/performance filters",
                                  "pilot_final_assignment": "Not assigned here; freeze separately by document and normalized-prompt identity",
                                  "uses_reference_answers": False, "uses_timing_outcomes": False,
                                  "tokenized": False, "truncated": False},
                    "source_files": provenance, "supporting_evidence_files": supporting,
                    "repository_metadata": {"private_relative_path": metadata_path.name,
                                            "sha256": file_sha256(metadata_path),
                                            "repository_id": metadata["id"], "gated": metadata.get("gated"),
                                            "retrieved_url": f"https://huggingface.co/api/datasets/{pin['repository']}?blobs=true",
                                            "pinned_retrieval_url": f"https://huggingface.co/api/datasets/{pin['repository']}/revision/{pin['revision']}?blobs=true"},
                    "methods_evidence_files": [record for relative, record in download_records.items()
                                               if relative.startswith("evidence/")],
                    "private_candidates": {"basename": payload.name, "sha256": file_sha256(payload), "bytes": payload.stat().st_size},
                    "candidate_inventory": [{key: value for key, value in row.items() if key != "messages"} for row in selected],
                    "script_sha256": file_sha256(__file__), "downloads_manifest_sha256": file_sha256(downloads_path)}
        manifest["content_sha256"] = digest(manifest)
        public_dir.mkdir(parents=True, exist_ok=True)
        output = public_dir / f"{dataset}-source.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        outputs.append({"dataset": dataset, "public_manifest": str(output), "private_candidates": str(payload), **counts})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="../full-serving-suite-sources")
    parser.add_argument("--private-dir", default="../full-serving-suite-sources/candidates")
    parser.add_argument("--public-dir", default="results/full-serving/suite-sources")
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_dir, args.private_dir, args.public_dir,
                             args.candidate_count, args.seed), indent=2))


if __name__ == "__main__":
    main()
