# Reconstruct and recapture the frozen input selection

The private `sources.json` can be reconstructed from the public source manifest and five pinned public files. Earlier private selection manifests are **not required**. The support script restores only `text` and `token_ids`; it does not select replacement documents, change splits, or use routing/performance observations. It verifies all public fields, the original canonical content hash, and the original private-file byte hash before writing a new file with mode `0600`.

The frozen selection contains 640 documents: 320 GSM8K training questions and 320 WildChat first-user turns, in four pilot and 16 final groups. The model is not trained, and answer fields are not captured. Reconstructing this fixed selection is distinct from independently repeating its original all-prior-source exclusion audit.

## CPU reconstruction

Run from the repository root in a CPU environment with `tokenizers==0.23.3` and `pyarrow==26.0.1` (the versions used for the successful local check). Download these files under the exact names shown into a fresh `raw/` directory. Source hashes and lengths are verified against the public manifest; filenames alone are not trusted.

| Local filename | Pinned public source |
|---|---|
| `gsm8k-train.jsonl` | [GSM8K training questions](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl) |
| `gsm8k-license.txt` | [GSM8K MIT license](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE) |
| `wildchat-first-shard.parquet` | [WildChat shard, revision f66566c](https://huggingface.co/datasets/allenai/WildChat/resolve/f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4/data/train-00000-of-00006.parquet) |
| `wildchat-license.md` | [WildChat ODC-By terms at the same revision](https://huggingface.co/datasets/allenai/WildChat/resolve/f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4/LICENSE.md) |
| `qwen-tokenizer.json` | [Qwen tokenizer, revision ad44e77](https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/tokenizer.json) |

The WildChat database license does not establish separate rights over every conversation's contents. The reconstructed manifest contains original prompt text and token IDs; keep it outside public result bundles. Requiring the pinned license bytes documents the terms used for this release, not a new rights assessment. Network access and continuing availability of these public revisions are external dependencies.

```bash
python docs/resident-residual/rehydrate_sources.py \
  --manifest results/resident-residual/content-v1/source-manifest.json \
  --raw-dir raw \
  --output private-recapture/sources.json \
  --report private-recapture/rehydration.json
```

The command refuses existing output paths and fails on missing/changed raw bytes, an unknown selected identity, changed text/tokenization, or a changed public manifest. The expected canonical content SHA256 is `777b3e2752745494c686e71249ebf4bd3b18b172ab4bc3c26f0907ff88fab92c`. The expected private-file SHA256 is `050e3e13801b366352e24568f57a9dc18f56b8558afd5f76f3de54bae4fe1d70`. The script was tested against the existing pinned raw files: all 640 documents reproduced the original private manifest byte for byte. [The check](../../results/resident-residual/rehydration-check-v1.json) and [mutation checks](../../results/resident-residual/rehydration-validation-v1.json) record that same-team CPU validation. No new download or GPU recapture was performed for this test.

## GPU input recapture

Provision the pinned Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` and a compatible GPU environment. The [recorded cluster environment](../../results/resident-fix/final-v1/provenance/environment-lock.txt) is a reference for the shared environment; the capture entry point itself additionally requires Transformers 5.17.0 and exact hashes of the Qwen modeling and masking implementations. It loads actual BF16 checkpoint tensors and verifies boundary/router/serialization identities. This is not a CPU-only command or a claim that arbitrary hardware/software produces identical hidden bits.

With checkpoint assets already cached, use fresh directories:

```bash
python scripts/capture_residual_inputs.py \
  --sources private-recapture/sources.json \
  --output-dir recapture/pilot --cache-dir /path/to/huggingface-cache \
  --device cuda:0 --split pilot --local-files-only
python scripts/capture_residual_inputs.py \
  --sources private-recapture/sources.json \
  --output-dir recapture/final --cache-dir /path/to/huggingface-cache \
  --device cuda:0 --split final --local-files-only
```

Archive the new hardware/software identity, source hashes, logs, capture indexes, and all bank bytes. Compare exact source tokens/positions, tensor encodings and checkpoint pins, then assess hidden-state and numerical differences explicitly. A new recapture does not inherit the original run's passing attestations by name. The measured runtime consumes hidden states only and recomputes its own native routing; captured HuggingFace expert IDs and weights remain provenance.

These direct capture commands avoid impersonating the original Slurm allocation. The original cluster wrapper and capture audit intentionally enforce kh014/allocation 4872248; they must not be made to pass by forging that identity on a different system. A performance rerun on another pair needs a newly declared hardware identity, source freeze and baseline calibration. This recipe only removes the private-source-file obstacle to independently reconstructing and recapturing inputs; it is not evidence of external GPU reproduction or unchanged performance.
