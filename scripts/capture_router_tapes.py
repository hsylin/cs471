#!/usr/bin/env python3
"""Capture real layer-0 Qwen routing; construct explicitly repeated prefill replay.

Only embedding, attention, norms and router checkpoint tensors are needed. The
installed, hash-pinned Transformers decoder and router execute unchanged; the
expert computation is replaced by a recorder after the routing result exists.
No benchmark performance is measured by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup.router_tapes import (MODEL, REVISION, canonical_json, compose_router_tape,
                                   file_sha256, save_router_bank)

TRANSFORMERS_VERSION = "5.17.0"
MODELING_SHA256 = "b2666af64b7b31a02a152e9187ed2c5671262543b61510d0359c26fa491105ca"
MASKING_SHA256 = "50a737f63d8c778a5597fa34ac139049af921f44958205e3e1c29fe2bae77254"


def source_text(source: dict, directory: Path) -> str:
    path = directory / source["file"]
    if file_sha256(path) != source["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if source["cleaning"] == "content_between_gutenberg_start_end_markers":
        start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
        end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", text)
        if start is None or end is None or end.start() <= start.end():
            raise ValueError("Unrecognized Gutenberg boundaries; no silent extraction")
        text = text[start.end():end.start()].strip()
    elif source["cleaning"] != "none":
        raise ValueError("Unknown source cleaning rule")
    return text


def validate_source_manifest(manifest: dict) -> None:
    seen = set()
    counts = {"pilot": 0, "final": 0}
    blocks = set()
    for block in manifest["blocks"]:
        if block["block_id"] in blocks:
            raise ValueError("Repeated source block")
        blocks.add(block["block_id"])
        counts[block["split"]] += 1
        if [s["domain"] for s in block["sources"]] != ["prose", "code"]:
            raise ValueError("Each block must contain prose then code")
        for source in block["sources"]:
            identity = source["source_id"]
            if identity in seen:
                raise ValueError("Source documents overlap across blocks or splits")
            seen.add(identity)
    if counts != {"pilot": 2, "final": 8}:
        raise ValueError("Expected the fixed two-pilot/eight-final document design")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[1]/"followup/corpus/sources.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tokens-per-domain", type=int, default=2048)
    p.add_argument("--split", choices=("pilot", "final", "all"), default="all")
    p.add_argument("--compose", action="store_true", help="Also make explicitly constructed default replay schedules")
    p.add_argument("--num-steps", type=int, default=672)
    p.add_argument("--warmup-steps", type=int, default=32)
    p.add_argument("--phase-steps", type=int, default=120)
    p.add_argument("--periodic-phase-steps", type=int, default=112)
    p.add_argument("--duration-steps", type=int, default=16)
    p.add_argument("--repeat-steps", type=int, default=128)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    if args.tokens_per_domain < 1024:
        p.error("At least 1024 captured records per domain are required")
    import numpy as np
    import torch
    import transformers
    import transformers.masking_utils as masks
    from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Expected Transformers {TRANSFORMERS_VERSION}, got {transformers.__version__}")
    for module, digest in ((qwen, MODELING_SHA256), (masks, MASKING_SHA256)):
        if file_sha256(inspect.getfile(module)) != digest:
            raise RuntimeError(f"Pinned implementation changed: {module.__name__}")
    manifest = json.loads(args.sources.read_text())
    validate_source_manifest(manifest)
    root = args.sources.parent
    if file_sha256(root/"raw/cpython_LICENSE") != manifest["cpython_license_sha256"]:
        raise ValueError("Archived CPython license checksum mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_args = dict(revision=REVISION, cache_dir=str(args.cache_dir), local_files_only=args.local_files_only)
    def get(name: str) -> Path:
        return Path(hf_hub_download(MODEL, name, **download_args))
    config_path, index_path = get("config.json"), get("model.safetensors.index.json")
    config = AutoConfig.from_pretrained(MODEL, **download_args, trust_remote_code=False)
    config._attn_implementation = "eager"
    if (config.hidden_size, config.num_experts, config.num_experts_per_tok) != (2048, 128, 8):
        raise ValueError("Unexpected checkpoint geometry")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **download_args, trust_remote_code=False)
    index = json.loads(index_path.read_text())
    prefix = "model.layers.0."
    keys = ["model.embed_tokens.weight", prefix+"input_layernorm.weight", prefix+"post_attention_layernorm.weight",
            prefix+"mlp.gate.weight"]
    keys += [prefix+"self_attn."+name+".weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
    tensors, sources = {}, []
    for name in sorted({index["weight_map"][key] for key in keys}):
        path = get(name)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                if index["weight_map"][key] == name:
                    tensors[key] = f.get_tensor(key)
        sources.append({"file": name, "sha256": file_sha256(path)})
    tensor_hashes = {key: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                     for key, value in tensors.items()}
    if any(t.dtype != torch.bfloat16 for t in tensors.values()):
        raise ValueError("Expected actual BF16 checkpoint tensors")

    class ExpertRecorder(torch.nn.Module):
        def forward(self, hidden_states, top_k_index, top_k_weights):
            self.record = (hidden_states.detach().clone(), top_k_index.detach().clone(), top_k_weights.detach().clone())
            return torch.zeros_like(hidden_states)

    # The native decoder and sparse MoE router stay intact. No expert tensors
    # are allocated or loaded, because the observer replaces only the experts.
    with torch.device("meta"):
        embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        layer = qwen.Qwen3MoeDecoderLayer(config, 0)
    recorder = ExpertRecorder()
    layer.mlp.experts = recorder
    embedding = embedding.to_empty(device=args.device).to(dtype=torch.bfloat16).eval()
    layer = layer.to_empty(device=args.device).to(dtype=torch.bfloat16).eval()
    embedding.load_state_dict({"weight": tensors["model.embed_tokens.weight"]}, strict=True)
    layer.load_state_dict({key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}, strict=True)
    rotary = qwen.Qwen3MoeRotaryEmbedding(config=config, device=args.device).eval()
    for key, actual in layer.state_dict().items():
        if not torch.equal(actual.cpu(), tensors[prefix+key]):
            raise AssertionError(f"Loaded checkpoint tensor differs: {key}")
    if not torch.equal(embedding.weight.cpu(), tensors["model.embed_tokens.weight"]):
        raise AssertionError("Loaded embedding differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    tokenizer_records = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = get(name)
        tokenizer_records.append({"file": name, "sha256": file_sha256(path)})
    index_file = args.output_dir/"capture-index.json"
    run_records = json.loads(index_file.read_text()) if index_file.exists() else []
    if len({r["block_id"] for r in run_records}) != len(run_records):
        raise ValueError("Existing capture index has duplicate source blocks")
    for block in manifest["blocks"]:
        if args.split != "all" and block["split"] != args.split:
            continue
        target = args.output_dir/"banks"/(block["block_id"]+".npz")
        if target.exists():
            raise FileExistsError(f"Capture already exists: {target}; use a new output directory")
        pieces = {k: [] for k in ("hidden_states_bf16", "expert_ids", "combining_weights", "token_ids",
                                 "prompt_index", "token_position", "domain_ids")}
        prompts = []
        for domain, source in enumerate(block["sources"]):
            text = source_text(source, root)
            all_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(all_tokens) < args.tokens_per_domain:
                raise ValueError(f"Source has too few tokens: {source['source_id']} ({len(all_tokens)})")
            token_list = all_tokens[:args.tokens_per_domain]
            tokens = torch.tensor([token_list], device=args.device, dtype=torch.int64)
            positions = torch.arange(tokens.shape[1], device=args.device).unsqueeze(0)
            embedded = embedding(tokens)
            mask_fn = masks.create_causal_mask if config.sliding_window is None else masks.create_sliding_window_causal_mask
            mask = mask_fn(config=config, inputs_embeds=embedded, attention_mask=None,
                           past_key_values=None, position_ids=positions)
            pos_embed = rotary(embedded, position_ids=positions)
            layer(embedded, attention_mask=mask, position_ids=positions,
                  past_key_values=None, use_cache=False, position_embeddings=pos_embed)
            hidden, ids, weights = recorder.record
            # Independent call sequence checks the captured decoder boundary;
            # actual router is recomputed from the exact BF16 captured input.
            attn, _ = layer.self_attn(layer.input_layernorm(embedded), pos_embed, mask, past_key_values=None)
            expected_hidden = layer.post_attention_layernorm(embedded + attn).reshape(-1, config.hidden_size)
            if not torch.equal(expected_hidden, hidden):
                raise AssertionError("Native decoder boundary differs from direct attention/residual/norm")
            logits, expected_weights, expected_ids = layer.mlp.gate(hidden)
            if not torch.equal(ids, expected_ids) or not torch.equal(weights, expected_weights):
                raise AssertionError("Native router recomputation differs")
            # Formula check is intentionally additional to executing the native
            # router; it catches changes in normalization or rounding semantics.
            probs = torch.softmax(torch.nn.functional.linear(hidden, layer.mlp.gate.weight), dim=-1, dtype=torch.float32)
            values, check_ids = torch.topk(probs, config.num_experts_per_tok, dim=-1)
            if config.norm_topk_prob:
                values = values / values.sum(dim=-1, keepdim=True)
            values = values.to(logits.dtype)
            if not torch.equal(ids, check_ids) or not torch.equal(weights, values):
                raise AssertionError("Router formula/rounding check differs")
            bits = hidden.contiguous().view(torch.uint16).cpu().numpy().copy()
            restored = torch.from_numpy(bits.copy()).to(args.device).view(torch.bfloat16)
            if not torch.equal(hidden, restored):
                raise AssertionError("BF16 bank serialization is not exact")
            n = len(token_list)
            for name, value in {
                "hidden_states_bf16": bits, "expert_ids": ids.cpu().numpy().astype(np.int64),
                "combining_weights": weights.float().cpu().numpy(), "token_ids": np.asarray(token_list, np.int64),
                "prompt_index": np.full(n, domain, np.int32), "token_position": np.arange(n, dtype=np.int32),
                "domain_ids": np.full(n, domain, np.int8),
            }.items():
                pieces[name].append(value)
            prompts.append({**source, "prompt_id": source["source_id"], "total_source_tokens": len(all_tokens),
                            "captured_tokens": n, "token_offset": 0, "position_start": 0,
                            "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "token_ids_sha256": hashlib.sha256(np.asarray(token_list, np.int64).tobytes()).hexdigest(),
                            "native_boundary_exact": True, "native_router_exact": True,
                            "manual_router_exact": True, "bf16_roundtrip_exact": True})
        arrays = {name: np.concatenate(value, axis=0) for name, value in pieces.items()}
        metadata = {
            "schema": "qwen_router_bank_v1", "block_id": block["block_id"], "split": block["split"],
            "source_model": MODEL, "source_revision": REVISION, "layer": 0,
            "num_experts": config.num_experts, "top_k": config.num_experts_per_tok, "hidden_size": config.hidden_size,
            "origin": "teacher_forced_causal_prefill_layer0", "native_serving_trace": False,
            "source_prompt_ids": [s["source_id"] for s in block["sources"]], "prompts": prompts,
            "attention_backend": "eager", "hidden_dtype": "bfloat16_exact_uint16_bits",
            "combining_weights_dtype": "native_bfloat16_values_stored_losslessly_as_float32",
            "special_tokens_added": False, "chat_template": None, "context": "one full causal prefix per source document",
            "source_manifest_sha256": file_sha256(args.sources), "source_shards": sources,
            "source_tensor_sha256": tensor_hashes, "config_sha256": file_sha256(config_path),
            "index_sha256": file_sha256(index_path), "tokenizer_files": tokenizer_records,
            "transformers_version": transformers.__version__, "modeling_sha256": MODELING_SHA256,
            "masking_sha256": MASKING_SHA256, "torch_version": torch.__version__,
            "capture_script_sha256": file_sha256(__file__), "correctness_pass": True,
        }
        bank = save_router_bank(target, metadata, arrays)
        record = {"block_id": block["block_id"], "split": block["split"], "bank_file": str(target),
                  "capture_sha256": bank.sha256, "tokens": len(bank.expert_ids), "correctness_pass": True}
        if args.compose:
            record["replays"] = []
            for batch in (128, 1024):
                for regime in ("stable", "transient", "persistent", "switching"):
                    out = args.output_dir/"tapes"/f"{block['block_id']}-{regime}-b{batch}.npz"
                    tape = compose_router_tape(target, out, regime=regime, batch_tokens=batch,
                        num_steps=args.num_steps, warmup_steps=args.warmup_steps,
                        phase_steps=args.periodic_phase_steps if regime == "switching" else args.phase_steps,
                        duration_steps=args.duration_steps, repeat_steps=args.repeat_steps)
                    record["replays"].append({"file": str(out), "sha256": tape.sha256})
        run_records.append(record)
        temporary_index = index_file.with_suffix(".json.tmp")
        temporary_index.write_text(json.dumps(run_records, indent=2)+"\n")
        temporary_index.replace(index_file)
        print(canonical_json(record), flush=True)


if __name__ == "__main__":
    main()
