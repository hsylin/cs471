#!/usr/bin/env python3
"""Capture disjoint public prompts through pinned checkpoint layer0 attention/router.

No timings are used for content selection. No answers are generated or consumed.
Only post-attention inputs/routes are saved; measured expert execution is separate.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from followup.router_tapes import MODEL, REVISION, file_sha256, save_router_bank
from generalization.data import digest, validate_splits
from scripts.capture_optimization_tapes import TRANSFORMERS_VERSION, MODELING_SHA256, MASKING_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--split',choices=['pilot','final'],required=True)
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError('Fresh capture directory required')
    manifest=json.loads(args.sources.read_text())
    if manifest['schema']!='generalization_sources_v1' or manifest['content_sha256']!=digest({k:v for k,v in manifest.items() if k!='content_sha256'}):
        raise ValueError('Changed/unfrozen source selection')
    validate_splits(manifest['blocks'])
    if (manifest['source_model'],manifest['source_revision'])!=(MODEL,REVISION):raise ValueError('Wrong checkpoint')
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
    run_records=[]
    for block in manifest['blocks']:
        if block['split']!=args.split:continue
        target=args.output_dir/'banks'/(block['block_id']+'.npz')
        pieces={k:[] for k in ('hidden_states_bf16','expert_ids','combining_weights','token_ids','prompt_index','token_position','domain_ids')}
        prompts=[]
        for domain,source in enumerate(block['sources']):
            text=source['text']
            if digest(text.encode())!=source['text_sha256']:raise ValueError('Source text changed')
            all_tokens=tokenizer(text,add_special_tokens=False)['input_ids']
            token_list=all_tokens[:512]
            if token_list!=source['token_ids'] or len(all_tokens)!=source['full_token_count']:
                raise ValueError('Pinned tokenizer differs from frozen content selection')
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
                "domain_ids": np.full(n, domain % 2, np.int8),
            }.items():
                pieces[name].append(value)
            prompts.append({**{k:v for k,v in source.items() if k not in ("text", "token_ids")}, "prompt_id": source["source_id"], "total_source_tokens": len(all_tokens),
                            "captured_tokens": n, "token_offset": 0, "position_start": 0,
                            "cleaned_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "token_ids_sha256": hashlib.sha256(np.asarray(token_list, np.int64).tobytes()).hexdigest(),
                            "native_boundary_exact": True, "native_router_exact": True,
                            "manual_router_exact": True, "bf16_roundtrip_exact": True})
        arrays={name:np.concatenate(value,axis=0) for name,value in pieces.items()}
        metadata={
            'schema':'qwen_router_bank_v1','block_id':block['block_id'],'split':block['split'],'dataset':block['dataset'],
            'source_model':MODEL,'source_revision':REVISION,'layer':0,'num_experts':128,'top_k':8,'hidden_size':2048,
            'origin':'prompt_causal_prefill_layer0','native_serving_trace':False,
            'source_prompt_ids':[r['dataset']+':'+r['source_id'] for r in block['sources']], 'prompts':prompts,
            'domain_ids_meaning':'legacy bank serialization: prompt index modulo2, not semantic domains; never an inference stratum',
            'attention_backend':'eager','hidden_dtype':'bfloat16_exact_uint16_bits',
            'combining_weights_dtype':'native_bfloat16_values_stored_losslessly_as_float32',
            'special_tokens_added':False,'chat_template':None,'context':'each prompt evaluated independently; prefix capped512; no generated/teacher-forced answer',
            'source_manifest_sha256':file_sha256(args.sources),'selection_content_sha256':manifest['content_sha256'],
            'source_shards':sources,'source_tensor_sha256':tensor_hashes,
            'config_sha256':file_sha256(config_path),'index_sha256':file_sha256(index_path),'tokenizer_files':tokenizer_records,
            'transformers_version':transformers.__version__,'modeling_sha256':MODELING_SHA256,'masking_sha256':MASKING_SHA256,
            'torch_version':torch.__version__,'capture_script_sha256':file_sha256(__file__),'correctness_pass':True,
            'study_id':'generalization_v1','timing_evidence':False}
        bank=save_router_bank(target,metadata,arrays)
        record={'block_id':block['block_id'],'split':block['split'],'dataset':block['dataset'],'bank_file':str(target),
                'capture_sha256':bank.sha256,'tokens':len(bank.expert_ids),'documents':len(prompts),'correctness_pass':True}
        run_records.append(record)
        (args.output_dir/'capture-index.json').write_text(json.dumps(run_records,indent=2)+'\n')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
