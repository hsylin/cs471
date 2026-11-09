# Reproducing the stock-layer comparison

CPU reconstruction uses the archived clocks, plans and saved input/route artifacts. It does not require CUDA, model weights or the private full-text selection file. Captured banks are also included for GPU replay, a separate operation requiring the pinned model/environment and an honestly identified hardware allocation. Neither operation is an external-laboratory replication merely because a second implementation checks the arithmetic.

This guide specifies commands and artifact requirements without asserting an experimental outcome. Use the complete archive containing `results/stock-comparison/final-v1`, not a partially copied running directory. All commands below run from the repository root and use fresh output paths.

The completed clean-source CPU reconstruction is archived under `results/stock-comparison/clean-cpu-reproduction-v1/`. It verified all 62 frozen source files, 18 calibrations and 216 final trials. Regenerated analyses matched 961 calibration and 53,507 final scalar fields with zero float difference after the declared path-metadata normalization; both execution audits and the separate statistical reconstruction passed. This records arithmetic/artifact reproducibility, not another physical experiment or external-laboratory replication.

## Frozen identities and archive layout

The executed-source checkout at final submission was `dfd419f7b4e62fdeec408581c572bf4d3a7a3065`. Later documentation/data commits may have different Git identities while retaining the same measured-file inventory. The authoritative pins are:

| Identity | SHA256 or revision |
|---|---|
| Final plan canonical hash | `96b7cce1bb268e038991f13740d426405708bf3962e76fcacfef7aad4bcaa46e` |
| Frozen execution-source inventory | `9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4` |
| Protocol file | `54a796b09b22fc02bcb792baa28278a5a0c2db57a65f154fc8c4ab1ee0cae89e` |
| Fresh content selection | `6020e89fc0b814460c6e2b4c2e9423a6956f390ca850cfe96d45cfe5687ef20f` |
| vLLM | `db9527a46873454610df6dbedf79a36d6bf1a7f6`, package `0.31.0+cu129` |
| Qwen3-30B-A3B | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, layer 0 |

`results/stock-comparison/plans/final-v1.json` and `plans/final-calibration-v1.json` are the canonical final treatment/calibration plans. Each run also retains its exact plan in `provenance/plan.json`. The run directories `calibration-final-v1` and `final-v1` contain per-group `trials/*.jsonl`, authenticated `inputs/`, `analysis.json`, logs and provenance. Pilot directories follow the same pattern. Preserve each calibration file's original basename and decompressed bytes: the treatment plan records their individual hashes. Combining all calibrations into a new file is insufficient for that byte-level provenance check.

Captured banks are in `capture-pilot-v1/banks/` and `capture-final-v1/banks/`, with their respective `capture-index.json`. The public source and arrival manifests are in `content-v2/`. Historical `content-v1/` scans remain authenticated provenance for the schema-only preparation correction. Raw expert weights are not in Git.

## CPU analysis and independent audits

Python 3.12 and NumPy are sufficient for the following analysis/audit commands; CUDA/Torch are not imported by these CPU entry points. NumPy 2.3.5 matches the GPU analysis environment. The standalone statistical verifier was additionally tested with NumPy 2.5.4; its exact draw-table digest is checked rather than assuming different environments produce identical draws.

```bash
python3.12 -m venv .venv-stock-cpu
.venv-stock-cpu/bin/python -m pip install 'numpy==2.3.5'
test ! -e work/stock-cpu-reproduction-new
mkdir -p work/stock-cpu-reproduction-new
```

Check the included capture files against their indexes and the frozen selection before a bank replay. This verifies bank identity; it does not rerun attention or reconstruct omitted full text:

```bash
.venv-stock-cpu/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
import numpy as np

base = Path('results/stock-comparison')
selection = json.loads((base/'content-v2/source-manifest.json').read_text())['content_sha256']
for split, expected in [('pilot', 6), ('final', 18)]:
    root = base/f'capture-{split}-v1'
    rows = json.loads((root/'capture-index.json').read_text())
    assert len(rows) == expected
    for row in rows:
        bank = root/'banks'/Path(row['bank_file']).name
        with bank.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == row['capture_sha256']
        with np.load(bank, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            assert meta['selection_content_sha256'] == selection
            assert meta['block_id'] == row['block_id'] and len(meta['prompts']) == 32
    print(split, 'capture file identities match')
PY
```

Recompute the 18 native-only final calibrations and the complete 216-trial final analysis:

```bash
.venv-stock-cpu/bin/python scripts/analyze_stock_comparison.py \
  --plan results/stock-comparison/calibration-final-v1/provenance/plan.json \
  --records results/stock-comparison/calibration-final-v1 \
  --output work/stock-cpu-reproduction-new/calibration-analysis.json

.venv-stock-cpu/bin/python scripts/analyze_stock_comparison.py \
  --plan results/stock-comparison/final-v1/provenance/plan.json \
  --records results/stock-comparison/final-v1 \
  --output work/stock-cpu-reproduction-new/final-analysis.json
```

Run the separate execution auditor against the saved **stock-vLLM route artifacts**, not the HuggingFace route fields in the capture banks:

```bash
.venv-stock-cpu/bin/python scripts/audit_stock_comparison_records.py \
  --plan results/stock-comparison/calibration-final-v1/provenance/plan.json \
  --records results/stock-comparison/calibration-final-v1 \
  --inputs-dir results/stock-comparison/calibration-final-v1/inputs \
  --repo-root . \
  --output work/stock-cpu-reproduction-new/calibration-execution-audit.json

.venv-stock-cpu/bin/python scripts/audit_stock_comparison_records.py \
  --plan results/stock-comparison/final-v1/provenance/plan.json \
  --records results/stock-comparison/final-v1 \
  --inputs-dir results/stock-comparison/final-v1/inputs \
  --calibration results/stock-comparison/calibration-final-v1 \
  --repo-root . \
  --output work/stock-cpu-reproduction-new/final-execution-audit.json
```

This auditor checks current files against the plan's execution-source inventory, calendars, raw clock envelopes, calibration provenance, saved routes, observation windows and native state. It does not rerun the CUDA kernels. A source mismatch must be investigated against the frozen inventory; changing the archived expected hashes is not reproduction.

Run the statistical reimplementation, which imports no production analysis/inference code and uses the regenerated analysis only as a comparison target:

```bash
.venv-stock-cpu/bin/python results/stock-comparison/reproduction/verify_stock_final_statistics.py \
  --plan results/stock-comparison/final-v1/provenance/plan.json \
  --records results/stock-comparison/final-v1 \
  --summary work/stock-cpu-reproduction-new/final-analysis.json \
  --protocol docs/stock-comparison/protocol-v1.md \
  --output work/stock-cpu-reproduction-new/final-statistical-audit.json
```

It reconstructs every trial's mean and p95 response from immutable arrivals and completion clocks, 72 matched triplets, all three contrasts, the 100,000 content-bundle draws, 14 guards, linked-date deletions and the hierarchical verdict. The final draw digest must be `0b20e84dbe760e5c0cca1895574655c48e13bafe9d5625e026e9156fc4442ab8`. Trials, controller steps and repeated token records are not bootstrap units. The intervals remain conditional on this content/calendar construction and one allocation.

Compare regenerated analyses with their archived counterparts. Only record-file path/compression metadata is normalized; original decompressed file digests, all other identities and numerical values remain checked:

```bash
.venv-stock-cpu/bin/python - <<'PY'
import json, math
from pathlib import Path

def compare(a, b, path='root'):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: compare(a[k], b[k], path+'.'+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path+f'[{i}]')
    elif isinstance(a, float):
        assert isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b), path
        assert math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-10), path
    else:
        assert type(a) is type(b) and a == b, path

for stage, fresh in [('calibration-final-v1', 'calibration-analysis.json'),
                     ('final-v1', 'final-analysis.json')]:
    old = json.loads((Path('results/stock-comparison')/stage/'analysis.json').read_text())
    new = json.loads((Path('work/stock-cpu-reproduction-new')/fresh).read_text())
    old_files, new_files = old.pop('record_files'), new.pop('record_files')
    assert sorted(x['uncompressed_bytes_sha256'] for x in old_files) == \
           sorted(x['uncompressed_bytes_sha256'] for x in new_files)
    compare(old, new)
    print(stage, 'analysis matches archived identities and numerical fields')
PY
```

Pilot reconstruction uses the same commands with `pilot-v1` and `calibration-pilot-v1`. It expects six calibrations and 72 treatment trials. The independent statistical verifier supports pilot mode and produces no confirmatory intervals. Do not substitute pilot results for missing final rows.

The separate native lifecycle audit and descriptive mechanism report can also be regenerated:

```bash
.venv-stock-cpu/bin/python results/stock-comparison/reproduction/audit_stock_lifecycle.py \
  --plan results/stock-comparison/final-v1/provenance/plan.json \
  --records results/stock-comparison/final-v1 \
  --inputs-dir results/stock-comparison/final-v1/inputs \
  --output work/stock-cpu-reproduction-new/lifecycle-audit.json \
  --markdown work/stock-cpu-reproduction-new/lifecycle-audit.md

.venv-stock-cpu/bin/python results/stock-comparison/reproduction/stock_mechanisms.py \
  --records results/stock-comparison/final-v1 \
  --inputs-dir results/stock-comparison/final-v1/inputs \
  --validated-summary work/stock-cpu-reproduction-new/final-analysis.json \
  --output-dir work/stock-cpu-reproduction-new/mechanism
```

Use the complete-matrix lifecycle mode shown here, without `--allow-partial`. Mechanism outputs are descriptive: active/padded counts are not measured GPU utilization, boundary excess is not isolated control cost, and asynchronous tail time is not an exclusive GPU-controller duration. The implementation-level precision evidence is archived separately under `results/stock-comparison/precision-audit/`.

## Data access, capture and privacy limits

The archive includes all 24 captured banks, 49,312 authentic token records, saved job-index artifacts and measured stock route artifacts. It also includes selected Azure timestamps, boundaries and source-row hashes. These are sufficient for replaying the prepared hidden inputs and reconstructing the reported clocks without obtaining private Azure prompt text; no such text was available to this study.

The private `stock-comparison-private-v2/sources.json`, containing full selected text and tokenizer lists, is not included. Public source manifests retain dataset revisions, source IDs, text/prefix hashes, counts and selection metadata. Raw-source recapture requires retrieving the pinned public datasets and reconstructing the selected text by those IDs, then reproducing the private manifest/tokenization hashes. The existing preparation script additionally expects the three earlier private selection manifests for exclusions. A clean-download, one-command reconstruction of all private manifests has not been established; the included banks are the verified GPU replay input.

The capture entry point is `scripts/capture_stock_comparison_inputs.py`; its cluster wrapper is `scripts/run-stock-comparison-capture-kuma.sh`. After obtaining a correctly reconstructed private manifest, the Python interface is:

```bash
python scripts/capture_stock_comparison_inputs.py \
  --sources /path/to/reconstructed-stock-sources.json \
  --output-dir /path/to/fresh-capture-output \
  --cache-dir "$SKB_ROOT/cache/huggingface" \
  --device cuda:0 --split final --local-files-only
```

This command requires the pinned GPU capture environment and checkpoint. It is a recapture procedure, not a promise of byte-identical fresh GPU output without verification. The accepted source-selection canonical hash must remain `6020e89f…`; every new bank and capture attestation needs verification before reuse in a new execution plan.

The complete source/version/license discussion is in [data-provenance.md](data-provenance.md). GSM8K is MIT, MBPP dataset terms are CC BY 4.0, the selected WildChat release records ODC-By database terms, and Azure records CC-BY attribution terms. Database terms do not establish separate rights over every user-authored conversation. Model weights remain subject to their checkpoint terms. Omitting the full-text manifest does not anonymize the archive: token IDs can reconstruct captured prefixes, and hidden states are not established anonymization. Original source attribution and applicable terms still matter when redistributing derived artifacts.

## GPU replay and hardware identity

The measured hardware contract is hostname `kh014`, Slurm allocation `4872248`, and exactly these two H100 94 GB GPUs with the archived NV6 topology:

```text
GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3
GPU-7d5c9009-305b-9341-957f-1deb96611ed5
```

Both the plan loader and runtime enforce the physical identity. The original allocation is not a reusable credential or a generic cluster setting. A different allocation, even on the same node, requires an explicit new hardware manifest/source amendment, a new source freeze and fresh native-only calibration. A different pair additionally requires documenting the actual GPUs/topology. Preserve the original artifacts and identify the new run as a replication under its real hardware conditions. Do not spoof `SLURM_JOB_ID`, hostname or GPU UUIDs, or weaken checks to make an old manifest pass.

The full executed environment lock is `results/stock-comparison/calibration-final-v1/provenance/environment-lock.txt`, with a lock repeated in each treatment run's provenance. Important versions include Python 3.12.14, vLLM 0.31.0+cu129, PyTorch 2.13.0+cu129, NumPy 2.3.5, Triton 3.7.1, NCCL 2.29.7, Transformers 5.17.0 and FlashInfer 0.7.0.post1. The package lock alone does not reproduce the driver, devices or topology; retain the archived `nvidia-smi.txt`, `topology.txt` and Slurm metadata. Native vLLM source hashes are checked before timing. Do not load a different CUDA module over the pinned runtime environment.

For a fresh authorized cluster installation, use a new project root. Bootstrap specifies the official vLLM cu129 wheel; then synchronize the full archived lock rather than accepting current dependency versions:

```bash
export SKB_ROOT=/path/to/new-stock-replay-project
bash scripts/bootstrap-kuma.sh
uv pip sync --python "$SKB_ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  results/stock-comparison/calibration-final-v1/provenance/environment-lock.txt
source scripts/cluster-env.sh
python scripts/prepare_layer.py \
  --output "$SKB_ROOT/models/qwen3-layer0" \
  --cache-dir "$SKB_ROOT/cache/huggingface" --layer 0
```

The exported `layer.safetensors` SHA256 is `80046eef8f68a7afd8d165e582190900854faa44076bd22c6a869730e26d4096`. The local model snapshot used by the wrapper is:

```text
$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
```

It must include the pinned config and original gate shard, `model-00001-of-00016.safetensors`. The independently checked BF16 gate byte hash is `5a165372095204aaa0343a67d2d2afde1086b730a825998b59264fb9c49a216d`. `prepare_layer.py` downloads pinned config/index and required expert shards; verify the required local snapshot files exist before launch. Fresh capture additionally requires tokenizer and attention/embedding tensors from that snapshot. No random-weight fallback is provided.

Record the actual checkout identity in `SOURCE_COMMIT` (`git rev-parse HEAD > SOURCE_COMMIT` for a Git checkout). For an extracted archive, retain the archive/source identity and explicitly label its lack of Git metadata. The wrapper expects `SLURM_SUBMIT_DIR` to identify the checkout, writes environment/hardware/source provenance and refuses an existing run output directory. It launches one `torchrun` process group with two ranks.

The following illustrates final-set replay **only while the original allocation and exact frozen identity remain valid**. For any later allocation, first perform the explicit hardware/source amendment described above and use its actual allocation ID in the launch commands. Output paths and derived plans must be new:

```bash
test ! -e work/stock-gpu-replay-new
mkdir -p work/stock-gpu-replay-new
git rev-parse HEAD > SOURCE_COMMIT
export SLURM_SUBMIT_DIR="$PWD"

python scripts/freeze_stock_comparison.py \
  --phase calibration --split final \
  --capture-index results/stock-comparison/capture-final-v1/capture-index.json \
  --arrivals results/stock-comparison/content-v2/arrival-manifest.json \
  --output work/stock-gpu-replay-new/calibration-plan.json

srun --jobid=4872248 --nodes=1 --ntasks=1 --gpus=2 --cpus-per-task=8 \
  bash scripts/run-stock-comparison-kuma.sh \
  work/stock-gpu-replay-new/calibration-plan.json \
  results/stock-comparison/capture-final-v1/banks \
  work/stock-gpu-replay-new/calibration

python scripts/freeze_stock_comparison.py \
  --phase final --split final \
  --capture-index results/stock-comparison/capture-final-v1/capture-index.json \
  --arrivals results/stock-comparison/content-v2/arrival-manifest.json \
  --calibration work/stock-gpu-replay-new/calibration \
  --output work/stock-gpu-replay-new/final-plan.json

srun --jobid=4872248 --nodes=1 --ntasks=1 --gpus=2 --cpus-per-task=8 \
  bash scripts/run-stock-comparison-kuma.sh \
  work/stock-gpu-replay-new/final-plan.json \
  results/stock-comparison/capture-final-v1/banks \
  work/stock-gpu-replay-new/final
```

The freezer validates the complete native-only calibrations before deriving treatment rates. New calibration values legitimately create new calendar/plan hashes; they do not replace the historical final plan. Preserve preparation, normal asynchronous completion and every trial, then run the CPU audits against the new plans and input artifacts. Reusing these banks tests repeatability of this fixed captured workload, not another fresh-content experiment. A new hardware replication also cannot be relabeled as the original single-allocation measurement.
