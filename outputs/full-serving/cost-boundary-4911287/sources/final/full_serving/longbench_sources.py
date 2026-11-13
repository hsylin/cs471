"""Pinned original LongBench inputs, retaining full official task prompts.

This prepares candidates, not a performance-selected or frozen final sample.
Group-aware pilot/final selection is performed separately by prepare_suite.py.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import ssl
from string import Formatter
import urllib.request
import zipfile

from full_serving.data import (MODEL, MODEL_REVISION, digest, file_digest,
                               length_distribution, text_digest, verify_pinned_file)
from full_serving.prepare_data import private_write, serialize_requests

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
TASKS = ("qasper", "gov_report", "hotpotqa")
CODE_BASE = f"https://raw.githubusercontent.com/THUDM/LongBench/{CODE_REVISION}"
RESOURCES = {
    "data.zip": (f"https://huggingface.co/datasets/zai-org/LongBench/resolve/{DATASET_REVISION}/data.zip", 113932529, "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"),
    "dataset2prompt.json": (f"{CODE_BASE}/LongBench/config/dataset2prompt.json", 5437, "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"),
    "dataset2maxlen.json": (f"{CODE_BASE}/LongBench/config/dataset2maxlen.json", 459, "72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022"),
    "README.md": (f"{CODE_BASE}/LongBench/README.md", 16674, "1e16429a4c8ef3da202855b9e1242943d3987b09b64fc7b8a1c7725a445ab8d5"),
    "pred.py": (f"{CODE_BASE}/LongBench/pred.py", 8874, "fdbb0ab2bb68c822fc35126ac050ec60110396bd14fb6d27bed1969f6361835c"),
    "CODE-LICENSE": (f"{CODE_BASE}/LICENSE", 1075, "bf7f05f274d65931bdf14f060b6855f1c359b8424eb74fd3091d25de89d700c3"),
    "LongBench.py": (f"https://huggingface.co/datasets/zai-org/LongBench/resolve/{DATASET_REVISION}/LongBench.py", 3982, "be9fad2dba34c4ddf77c83640974160bb95c0bb8759e6274ca01a6b2fe9f3372"),
    "qasper-upstream-README.md": ("https://huggingface.co/datasets/allenai/qasper/resolve/fdc9d8214fbab5dd782958601db4d678e6934a54/README.md", 9639, "da348d3f8277f8a93ca397941f6a525433e282858b1c0562aed3f667d8e55b05"),
    "hotpotqa-upstream-README.md": ("https://raw.githubusercontent.com/hotpotqa/hotpot/3635853403a8735609ee997664e1528f4480762a/README.md", 6839, "2a0a1758bb9a2e52b52e6a6528bbf72d46a5a01e0318cdea0473d0660d555998"),
    "govreport-upstream-index.md": ("https://raw.githubusercontent.com/gov-report-data/gov-report-data.github.io/8dfd043210c02f5b425cf3e04667a7488a3f7237/index.md", 2722, "7b8a09ef9d071d3b0fa6f8e861192beba23766538fa917e2d419cd709204d5c9"),
}
TASK_LICENSES = {"qasper": "CC-BY-4.0", "gov_report": "CC-BY-4.0", "hotpotqa": "CC-BY-SA-4.0"}
TASK_REASONS = {
    "qasper": "Scientific single-document QA; shorter long-context prompts and concise answers.",
    "gov_report": "Government-report summarization; long documents and naturally longer generated answers.",
    "hotpotqa": "Wikipedia multi-document QA; long passage sets and concise answers.",
}


def verify_sources(source_dir, *, download=False):
    source_dir = Path(source_dir).resolve()
    if source_dir == ROOT or ROOT in source_dir.parents:
        raise ValueError("Raw dataset sources must stay outside the repository")
    source_dir.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name, (url, size, sha) in RESOURCES.items():
        path = source_dir / name
        if not path.exists() and download:
            context = ssl.create_default_context()
            try:
                import certifi
                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                pass
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, context=context, timeout=120) as response, temporary.open("xb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(path)
        if path.stat().st_size != size or file_digest(path) != sha:
            raise ValueError(f"Pinned source mismatch: {name}")
        verified[name] = {"url": url, "bytes": size, "sha256": sha}
    return verified


def build_candidates(source_dir, tokenizer, *, max_context=32768, max_new_tokens=2048):
    """Retain 200 source records per chosen task, excluding only context overflow."""
    source_dir = Path(source_dir)
    templates = json.loads((source_dir / "dataset2prompt.json").read_text())
    caps = json.loads((source_dir / "dataset2maxlen.json").read_text())
    requests, rejected, members, observed_lengths = [], [], {}, {}
    with zipfile.ZipFile(source_dir / "data.zip") as archive:
        for task in TASKS:
            template = templates[task]
            fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
            if not fields <= {"context", "input"}:
                raise ValueError("Official prompt unexpectedly references target answers")
            raw = archive.read(f"data/{task}.jsonl")
            records = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
            if len(records) != 200:
                raise ValueError(f"Unexpected pinned test-frame size: {task}")
            members[task] = {"zip_member": f"data/{task}.jsonl", "records": len(records),
                             "bytes": len(raw), "sha256": digest(raw)}
            observed_lengths[task] = []
            for position, record in enumerate(records):
                if record["dataset"] != task or record["language"] != "en" or not record["context"].strip():
                    raise ValueError(f"Unexpected original task record: {task}:{position}")
                prompt = template.format(context=record["context"], input=record["input"])
                messages = [{"role": "user", "content": prompt}]
                token_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False, truncation=False, padding=False)
                length = len(token_ids)
                observed_lengths[task].append(length)
                identity = str(record["_id"])
                if length + max_new_tokens > max_context:
                    rejected.append({"task": task, "source_id": identity, "input_tokens": length,
                                     "reason": "context_overflow", "max_tokens": max_new_tokens,
                                     "max_model_len": max_context, "truncation": False})
                    continue
                requests.append({"id": f"longbench-{task}-{identity}", "dataset": "longbench",
                    "task": task, "source_id": identity, "source_split": "test", "source_row": position,
                    "source_revision": DATASET_REVISION, "messages": messages, "input_tokens": length,
                    "max_tokens": max_new_tokens, "group_id": text_digest(record["context"]),
                    "source_context_sha256": digest(record["context"].encode()),
                    "source_record_sha256": digest(record), "messages_sha256": digest(messages),
                    "normalized_text_sha256": text_digest(prompt), "prompt_token_ids_sha256": digest(token_ids),
                    "official_prompt_template_sha256": digest(template.encode()),
                    "original_length_field": record["length"], "original_length_unit": "English words, not Qwen tokens",
                    "official_task_max_new_tokens": caps[task], "dataset_license": TASK_LICENSES[task],
                    "truncation": False, "padding": False, "reference_answers_in_prompt": False})
    requests.sort(key=lambda row: (row["task"], row["source_id"]))
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate original task identity")
    summary = {"source_members": members, "source_rows": 600, "eligible_candidates": len(requests),
               "rejection_counts": dict(Counter(r["reason"] for r in rejected)), "rejections": rejected,
               "by_task": {task: {"eligible": sum(r["task"] == task for r in requests),
                   "unique_context_groups": len({r["group_id"] for r in requests if r["task"] == task}),
                   "source_prompt_lengths": length_distribution(observed_lengths[task]),
                   "eligible_prompt_lengths": length_distribution([r["input_tokens"] for r in requests if r["task"] == task])}
                   for task in TASKS}}
    return requests, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "tokenizer-dir", "output", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Fresh candidate and manifest output paths required")
    if args.output.resolve() == ROOT or ROOT in args.output.resolve().parents:
        raise ValueError("Candidate payload text must remain outside Git")
    files = verify_sources(args.source_dir, download=args.download)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        verify_pinned_file(args.tokenizer_dir / name, name)
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    requests, summary = build_candidates(args.source_dir, tokenizer)
    payload = serialize_requests(requests)
    manifest = {"schema": "full_serving_longbench_candidates_v1", "prepared_on": "2026-11-10",
        "status": "candidate_pool_only_not_frozen_final", "dataset": "longbench",
        "official_dataset_name": "THUDM/LongBench (redirects to zai-org/LongBench)",
        "version": "Original LongBench v1, builder1.0.0; not LongBench-E and not LongBench-v2",
        "dataset_revision": DATASET_REVISION, "prompt_code_revision": CODE_REVISION,
        "tasks": list(TASKS), "task_choice_before_performance": TASK_REASONS,
        "source_split": "test", "benchmark_paper": {"title": "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding",
            "publication": "ACL2024 long paper", "doi": "10.18653/v1/2024.acl-long.172",
            "url": "https://aclanthology.org/2024.acl-long.172/"},
        "code_license": "MIT; does not replace underlying dataset licenses",
        "task_licenses": TASK_LICENSES,
        "attribution": "LongBench: Bai et al., ACL2024; Qasper: Dasigi et al., NAACL2021; GovReport: Huang et al., NAACL2021; HotpotQA: Yang et al., EMNLP2018. Preserve task-specific attribution and license terms; raw payloads are not published.",
        "raw_files": files, **summary,
        "model": MODEL, "model_revision": MODEL_REVISION,
        "formatting": {"official_task_template_preserved": True, "chat_wrapper": "Pinned Qwen chat template",
            "enable_thinking": False, "add_generation_prompt": True, "truncation": False, "padding": False,
            "max_model_len": 32768, "max_new_tokens": 2048, "reference_answers_in_input": False,
            "departure_from_original_evaluator": "Reject over-limit prompts instead of middle-truncation; use common2048 output allowance rather than official per-task generation caps; actual generation ends at EOS or cap."},
        "split_unit": "group_id is normalized full context SHA256; retain at most one question per context before pilot/final splitting",
        "environment": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                        "chat_template_sha256": digest(tokenizer.chat_template.encode())},
        "candidate_payload_sha256": digest(payload.encode()),
        "candidates": [{k: v for k, v in row.items() if k != "messages"} for row in requests],
        "preparation_source_sha256": file_digest(__file__),
        "limitations": "Only three of21 original tasks; long-context workload coverage, not an original-score benchmark or FreeBalance replication. No performance measurements used; final group-disjoint sample and arrival assignment remain separate."}
    manifest["public_manifest_sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private_write(args.output, payload)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidates": len(requests), "by_task": summary["by_task"],
                      "rejections": summary["rejection_counts"], "output": str(args.output),
                      "manifest": str(args.manifest), "seal": manifest["public_manifest_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
