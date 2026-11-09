# Reproduce the payback diagnostic

The archived inputs and records support two different checks: recomputing the
reported statistics on a CPU, and repeating the native GPU measurements. The
first should recover the saved numbers; the second produces new timing
observations and need not recover identical elapsed times. No independent
external laboratory has reproduced this experiment.

## Offline analysis

From the repository root, use Python 3.10 or later with NumPy and Matplotlib:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install 'numpy>=1.26' 'matplotlib>=3.8'
PYTHON=.venv-analysis/bin/python bash scripts/reproduce-payback-reports.sh /tmp/payback-reproduction
```

Use a fresh output path. This command checks the archived file hashes, analyses
both pilots and the final matrix, draws the primary figure, and independently
reconstructs ratios and intervals from raw arm records. It also independently
reconstructs physical dispatch from archived logical IDs and checks the complete
raw-state/commit/payload record contract. These CPU checks can take several
minutes. The independent statistics checker
does not import the production analysis module. It is a second implementation
within this project, not external human replication. Analysis requires no model
weights, CUDA, vLLM, Hugging Face access, or server credentials.

The authoritative final input is
`configs/frozen/payback-final-v1.json`, whose canonical configuration digest is
`ee0236aa6d7f466e8bbfb8452b9983df3790d7447fba742d93b82151ab2ac706`.
This digest excludes the `freeze` field, as specified by the frozen analysis
module; it is not the SHA of the formatted JSON file. Raw-file SHA values are in
`results/payback/SHA256SUMS`. Frozen pilot manifests are retained separately.

Inspect `summary.json`, `pairs.csv`, `cells.csv`, and `aggregates.csv`. For primary
criteria use `family_adjusted_ci`, not the ordinary `ci95` field. Eight source
blocks determine uncertainty; 1,536 arm records are not 1,536 independent
workloads. H128 is descriptive; H512 is primary. The bootstrap describes the
fixed corpus under block resampling, not a production population.

## Native measurement

The executed stack is vLLM commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6` (0.31.0+cu129), Torch
2.13.0+cu129, CUDA 12.9, Python 3.12.14 and two H100 94 GB GPUs on one node with
NV6 connectivity. Full installed-package, GPU, topology and Slurm records are
under each run's `provenance/`. This is synchronous native `torch_nccl` weight
movement with replicated token inputs and output all-reduce. It does not
exercise request-sharded all-to-all serving. The execution log reports a missing
`E=80,N=768,device_name=NVIDIA_H100.json` tuning file, so all arms use the same
native default MoE kernel configuration. The study does not claim an optimized
H100 latency bound.

Prepare actual Qwen3-30B-A3B layer-0 weights using the existing pinned exporter:

```bash
python scripts/prepare_layer.py --output /path/to/qwen3-layer0 --cache-dir /path/to/hf-cache
```

The source checkpoint revision is
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; the exported expert file SHA is
`80046eef8f68a7afd8d165e582190900854faa44076bd22c6a869730e26d4096`.
Weights are fetched from the original model source rather than duplicated in
this repository. Inspect their original license before redistribution.

With the recorded GPU environment, run from the repository root:

```bash
torchrun --standalone --nproc-per-node=2 -m followup.payback \
  --manifest configs/frozen/payback-final-v1.json \
  --banks-dir results/payback/inputs \
  --weights-dir /path/to/qwen3-layer0 \
  --output /path/to/new-final/records.jsonl
```

The runner checks native-source, harness, checkpoint, capture and replay pins.
It must fail on a mismatched execution environment rather than silently run a
different experiment. The source commit frozen before final timing is
`2ca454aa0a1a772287240d15c25878786e42b864`; the measured follow-up fingerprint is
`ba8c8ab0dc4519a78564d25f3acb301fab5b6e4e5d30631bed99cd9ba9e6fc61`.
Later report, packaging and independent-audit files do not alter that measured
fingerprint. The legacy study's source fingerprint remains unchanged.

On Kuma, `scripts/run-payback-kuma.sh` additionally records the allocation and
environment and refuses to overwrite its output directory. Read
`scripts/cluster-env.sh` to configure the project root for another installation.
Each KEEP/PREINSTALLED/CHARGED triple must run in one allocation and process;
partial triples cannot be resumed or assembled from separate runs. Preserve a
failed attempt and repeat its complete triple in a fresh output location.

## Inputs and timing boundary

`results/payback/inputs/banks/` contains ten authenticated captures, one per
document pair, with lossless BF16 hidden-state bits, logical top-8 expert IDs,
native combining weights, token IDs, original positions, and source metadata.
There are 2,048 tokens from each of two domains in every bank: 40,960 captured
token records altogether. `tapes-v2/` contains all 80 executed replay files.
The original source texts and licenses are under `followup/corpus/`.

Use these exact archived files for the frozen measurement. The capture script
and `scripts/compose_router_tapes.py` provide a recapture recipe, but a new
capture may differ in floating-point outputs or metadata. A changed file must
receive a new manifest and be reported as a new experiment. Do not edit the
frozen hashes to make a different capture appear identical. Original capture
and later composition script hashes are retained separately in the inventories.

The native prefix runs B3 through its first opportunity at step 63; it may
accept or reject. The state after step 127 is restored identically for all
three arms. The same candidate is installed before timing in PREINSTALLED and
inside timing in CHARGED. KEEP retains the source map. The timer ends only after
the fixed future and device completion, with no later balancing. It covers
Python replay, mapping, expert computation, output reduction and, for CHARGED,
the normal native transfer, map commit and synchronization.

Common prefix, proposal formation, state restoration, preflight, conditioning,
post-run correctness checks and diagnostic replays are outside these conditional
branch timers. Their recorded costs and roles must not be confused with a
complete online-policy elapsed time. CHARGED minus PREINSTALLED includes
transition/cache effects and is not an isolated network bandwidth measurement.

Stable, transient, persistent and window-aligned switching timelines are
constructed from captured teacher-forced causal-prefill tokens. Sampling is
without replacement inside a batch, but tokens repeat across steps. These are
authentic numerical token records with constructed temporal demand, not
autoregressive sessions or full-model serving traces.
