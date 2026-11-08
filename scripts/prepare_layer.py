#!/usr/bin/env python3
"""Export actual Qwen expert weights from only the checkpoint shards needed.

Downloads are pinned; the exported artifact contains no random weights. This
does not turn synthetic activation/routing replay into a natural serving trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    args.output.mkdir(parents=True, exist_ok=True)
    def get(name: str) -> Path:
        return Path(hf_hub_download(MODEL, name, revision=REVISION,
                                   cache_dir=str(args.cache_dir)))
    config = json.loads(get("config.json").read_text())
    index = json.loads(get("model.safetensors.index.json").read_text())
    if not 0 <= args.layer < config["num_hidden_layers"]:
        raise ValueError("Layer outside source model")
    prefix = f"model.layers.{args.layer}.mlp.experts."
    n = config["num_experts"]
    keys = [f"{prefix}{e}.{projection}.weight" for e in range(n)
            for projection in ("gate_proj", "up_proj", "down_proj")]
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)
    tensors = {}
    shard_records = []
    for name, selected in sorted(by_shard.items()):
        path = get(name)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in selected:
                tensors[key] = f.get_tensor(key)
        shard_records.append({"file": name, "sha256": sha256(path)})
    w13 = torch.stack([
        torch.cat([tensors[f"{prefix}{e}.gate_proj.weight"],
                   tensors[f"{prefix}{e}.up_proj.weight"]], dim=0)
        for e in range(n)
    ]).contiguous()
    w2 = torch.stack([tensors[f"{prefix}{e}.down_proj.weight"]
                      for e in range(n)]).contiguous()
    h, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    assert w13.shape == (n, 2 * intermediate, h)
    assert w2.shape == (n, h, intermediate)
    assert w13.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    target = args.output / "layer.safetensors"
    save_file({"w13_weight": w13, "w2_weight": w2}, str(target),
              metadata={"source_model": MODEL, "source_revision": REVISION,
                        "layer": str(args.layer)})
    manifest = {
        "schema_version": 1, "source_model": MODEL,
        "source_revision": REVISION, "layer": args.layer,
        "num_experts": n, "hidden_size": h,
        "intermediate_size": intermediate,
        "moe_intermediate_size": intermediate,
        "top_k": config["num_experts_per_tok"],
        "dtype": "bfloat16", "activation": config["hidden_act"],
        "weights_file": target.name, "weights_sha256": sha256(target),
        "source_shards": shard_records,
        "weights_kind": "pretrained",
        "replay_scope": "Controlled synthetic activations and logical routes; actual pretrained expert weights",
    }
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
