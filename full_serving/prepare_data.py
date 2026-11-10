"""Prepare real full-conversation inputs; private payloads must remain outside Git."""
import argparse
import json
import os
from pathlib import Path

from full_serving.data import (MODEL_REVISION, DATASET_REVISION, SEED, digest, file_digest,
                               load_prior_manifests, prepare_dataset, verify_pinned_file)

ROOT = Path(__file__).resolve().parents[1]


def private_write(path, value):
    """Create once with owner-only permissions; refuse accidental overwrite."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value)


def serialize_requests(rows):
    # Escape Unicode line/paragraph separators as well as embedded newlines.
    # Decoding preserves every source character, while each request stays one
    # physical line even for downstream consumers using str.splitlines().
    return "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows)


def prepare(args):
    private = args.private_output.resolve()
    public = args.public_output.resolve()
    if private == ROOT or ROOT in private.parents:
        raise ValueError("Private conversations must be written outside the repository")
    if private.exists() or public.exists():
        raise FileExistsError("Fresh private and public output directories required")
    files = {
        "wildchat-first-shard.parquet": args.raw_parquet,
        "wildchat-license.md": args.license_file,
        "tokenizer.json": args.tokenizer_dir / "tokenizer.json",
        "tokenizer_config.json": args.tokenizer_dir / "tokenizer_config.json",
    }
    pins = {name: verify_pinned_file(path, name) for name, path in files.items()}
    config = json.loads(files["tokenizer_config.json"].read_text())
    if "enable_thinking" not in config.get("chat_template", ""):
        raise ValueError("Pinned non-thinking template is required")
    import pyarrow.parquet as pq
    import pyarrow
    import jinja2
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True,
                                               trust_remote_code=False)
    prior_paths = sorted(ROOT.glob("results/*/content-v*/source-manifest.json"))
    if len(prior_paths) < 6:
        raise ValueError("Complete earlier public selection inventory is required")
    prior = load_prior_manifests(prior_paths, repo_root=ROOT, old_corpus_paths=[
        ROOT / name / "corpus/sources.json" for name in ("followup", "optimization", "resident_dispatch")])
    parquet = pq.ParquetFile(args.raw_parquet)
    rows = (row for batch in parquet.iter_batches(batch_size=1024,
             columns=["conversation_id", "conversation", "language", "toxic"])
            for row in batch.to_pylist())
    requests, manifest = prepare_dataset(rows, tokenizer, prior, seed=args.seed)
    if manifest["long_prompt_check"]["selected_over_512"] == 0:
        raise ValueError("Natural sample contains no prompts above 512 tokens; add a separately labelled length diagnostic before proceeding")
    manifest["raw_files"] = pins
    manifest["source_urls"] = {
        "parquet": f"https://huggingface.co/datasets/allenai/WildChat/resolve/{DATASET_REVISION}/data/train-00000-of-00006.parquet",
        "license": f"https://huggingface.co/datasets/allenai/WildChat/resolve/{DATASET_REVISION}/LICENSE.md",
        "tokenizer_config": f"https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/{MODEL_REVISION}/tokenizer_config.json",
    }
    manifest["license"] = {"identifier": "ODC-By-1.0", "attribution": "WildChat, Wenting Zhao et al., Allen Institute for AI; retain the pinned license and dataset attribution with derived manifests", "raw_payloads_published": False}
    manifest["environment"] = {"transformers_version": transformers.__version__,
                               "tokenizers_version": tokenizers.__version__,
                               "pyarrow_version": pyarrow.__version__, "jinja2_version": jinja2.__version__,
                               "chat_template_sha256": digest(tokenizer.chat_template.encode())}
    manifest["preparation_source_files"] = {str(p.relative_to(ROOT)): file_digest(p) for p in (Path(__file__), ROOT / "full_serving/data.py")}
    payload = serialize_requests(requests)
    by_id = {row["id"]: row for row in requests}
    smoke = serialize_requests(by_id[identity] for identity in manifest["smoke_request_ids"])
    # Files are split so pilot tooling never needs to open held-out final text.
    private.mkdir(parents=True, mode=0o700)
    os.chmod(private, 0o700)
    private_write(private / "requests.jsonl", payload)
    private_write(private / "smoke.jsonl", smoke)
    private_write(private / "pilot.jsonl", serialize_requests(row for row in requests if row["split"] == "pilot"))
    for block in sorted({row["content_block"] for row in requests}):
        text = serialize_requests(row for row in requests if row["content_block"] == block)
        private_write(private / f"{block}.jsonl", text)
    manifest["private_payload_sha256"] = digest(payload.encode())
    manifest["public_manifest_sha256"] = digest(manifest)
    public.mkdir(parents=True)
    (public / "request-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": manifest["selection_status"], "private_output": str(private),
                      "public_manifest": str(public / "request-manifest.json"), "counts": manifest["counts"],
                      "lengths": manifest["length_distributions"]["selected"],
                      "public_manifest_sha256": manifest["public_manifest_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-parquet", "license-file", "tokenizer-dir", "private-output", "public-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
