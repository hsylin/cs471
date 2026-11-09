# Reproduce the bounded kernel-configuration study

This guide separates recomputing the archived analysis on a CPU from making new
GPU timing observations. CPU reproduction should recover the recorded numerical
analysis; a GPU rerun will produce new elapsed times. Neither check is an
external laboratory replication. The design and decision rules are in
[optimization-protocol.md](optimization-protocol.md).

All v1 stages are complete: 192 tuning trials, 64 policy validation runs,
512 held-out policy runs and 16 recorded rank-level profile traces. Clean
CPU reproduction from data commit `4e1b1de4d31ecc0c0476534a4d42e92f37856ac8`
recovered the committed final summary byte for byte; the independent numerical,
input and routing audits also passed. See the
[reproduction record](../results/optimization/audits/clean-reproduction-verification.json).
This reproduces archived calculations, not a new GPU timing experiment.

## Archive and identity

Run commands from the repository root. The archive is:

```text
configs/optimization-tuning-frozen-v1.json
configs/optimization-validation-frozen-v1.json
configs/optimization-final-frozen-v1.json
results/optimization/
  capture/                 # capture-index.json, banks/*.npz, tapes/*.npz
  tuning/                  # records.jsonl.gz, selection.json, tuning-inputs/
  validation/              # records.jsonl.gz, profile/, provenance/
  final/                   # records.jsonl.gz, provenance/
  analysis/                # final summary, paired/cell/aggregate tables, figures
  validation-analysis/     # separate pilot validation report
  audits/                  # input, statistics and mechanism verification
```

The frozen manifest's `freeze.sha256` hashes canonical JSON excluding the
`freeze` field. It differs from the hash of the formatted manifest file. Source,
checkpoint, capture, replay and selection pins are retained separately. Read the
actual archived manifest and provenance rather than infer a source version from
the latest Git branch.

The tuning matrix contains 192 trials: two pilot banks × two domains × two batch
sizes × eight declared choices × three technical repetitions. Validation has 64
complete policy runs. The final matrix has 512 runs: eight held-out blocks × four
regimes × two batches × two repetitions × four DEFAULT/TUNED and B3/SKIP arms.
Every declared run completed and passed its required checks.

## CPU reproduction

Python 3.10 or later, NumPy and Matplotlib are sufficient for these checks. CUDA,
vLLM, model weights, Hugging Face access and server credentials are unnecessary.

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install 'numpy>=1.26' 'matplotlib>=3.8'
mkdir -p work/optimization-reproduction-new
```

Use a new destination: analysis reports and independent verifiers refuse to
overwrite existing outputs. Do not run the input auditor with Python's `-O`
option, because its assertions are part of the verification contract.

First validate the actual captured arrays, file/content hashes, source-document
exclusions, recorded native-capture attestations and independently regenerated
replay indices:

```bash
.venv-analysis/bin/python scripts/audit_optimization_inputs.py \
  --capture-dir results/optimization/capture \
  --output-prefix work/optimization-reproduction-new/inputs
```

An optional `--previous-bank results/payback/inputs/banks/pilot-01.npz` also checks
checkpoint/tokenizer provenance against a predecessor capture. The source
exclusion check already uses the archived original corpus manifest. This is an
artifact audit; it does not rerun pretrained attention or routing on a CPU.

Recompute the pilot selection from all 192 raw trials:

```bash
.venv-analysis/bin/python -m optimization.analysis select \
  --manifest configs/optimization-tuning-frozen-v1.json \
  --records results/optimization/tuning/records.jsonl.gz \
  --output work/optimization-reproduction-new/selection.json
```

The selected configuration IDs, dictionaries and numerical candidate results
should match the archived `results/optimization/tuning/selection.json`. A
regenerated selection file may differ in file-path metadata and the hashes of
compressed versus original uncompressed record files. Keep the **archived
selection file** for verifying the frozen final manifest's file-level linkage;
do not replace it with a reformatted or recompressed reconstruction.

Independently verify the pilot selection, actual saved input indices and original
decompressed timing payload:

```bash
.venv-analysis/bin/python scripts/audit_optimization_tuning.py \
  --manifest configs/optimization-tuning-frozen-v1.json \
  --records results/optimization/tuning/records.jsonl.gz \
  --selection results/optimization/tuning/selection.json \
  --tuning-dir results/optimization/tuning \
  --output work/optimization-reproduction-new/independent-tuning.json
```

Recompute validation and final reports separately:

```bash
.venv-analysis/bin/python -m optimization.analysis report \
  --manifest configs/optimization-validation-frozen-v1.json \
  --records results/optimization/validation/records.jsonl.gz \
  --output-dir work/optimization-reproduction-new/validation-analysis

.venv-analysis/bin/python -m optimization.analysis report \
  --manifest configs/optimization-final-frozen-v1.json \
  --records results/optimization/final/records.jsonl.gz \
  --output-dir work/optimization-reproduction-new/analysis
```

The report writes `summary.json`, `pairs.csv`, `cells.csv`, `aggregates.csv` and
ratio figures. Use `--no-plots` when only numerical reproduction is needed.
Incomplete or invalid groups must remain visible; do not remove them to obtain
an acceptable result.

The independent statistics verifier is a separate standard-library
implementation that does not import the production analysis module:

```bash
.venv-analysis/bin/python scripts/verify_optimization_results.py \
  --manifest configs/optimization-final-frozen-v1.json \
  --records results/optimization/final/records.jsonl.gz \
  --report work/optimization-reproduction-new/analysis/summary.json \
  --selection results/optimization/tuning/selection.json \
  --pilots configs/optimization-tuning-frozen-v1.json \
           configs/optimization-validation-frozen-v1.json \
  --output work/optimization-reproduction-new/independent-statistics.json
```

This checks raw-record consistency, matched groups, pilot/final provenance,
paired ratios, finite-corpus block-bootstrap intervals and declared criteria. It
does not independently execute GPU kernels or recompare expert-weight tensors.
The primary comparison is TUNED+B3 / DEFAULT+B3; TUNED+SKIP / DEFAULT+SKIP is the
declared secondary control. Eight source-document blocks, not 512 runs, determine
the final resampling units. Technical repeats and related regime/batch cells
are correlated. Cell-level claims use the report's adjusted intervals; the
finite hand-selected corpus is not a random production-traffic sample.

Reconstruct the exact observed routing windows, physical replica assignments,
complete mapping chains and sender payloads from the archived arrays:

```bash
.venv-analysis/bin/python scripts/audit_optimization_routing.py \
  --manifest configs/optimization-final-frozen-v1.json \
  --records results/optimization/final/records.jsonl.gz \
  --capture-dir results/optimization/capture \
  --output work/optimization-reproduction-new/independent-routing.json
```

Recompute the mechanism diagnostics from the sixteen raw selected-profile
traces and their exact saved input-index files:

```bash
.venv-analysis/bin/python scripts/audit_optimization_mechanism.py \
  --old-profile-dir results/payback/profile \
  --new-profile-dir results/optimization/validation/profile \
  --manifest configs/optimization-validation-frozen-v1.json \
  --output work/optimization-reproduction-new/mechanism.json
```

The verifier checks new trace and input SHA values, same-state pairing, physical
assignment conservation, active-expert counts, padding arithmetic, resolved
configurations and observed launch thread counts. It counts only actual Chrome
events with `cat=kernel` and `ph=X`, excluding nested operator annotations. Old
profiles provide the prior diagnosis; different sources/maps make them unsuitable
as a paired old-versus-new speedup baseline. For old traces, this command computes
their hashes; add `--old-evidence-json PATH` if a separately saved prior trace-hash
inventory is available. New trace hashes are checked against their saved profile
summary. Kernel residence is not an exclusive wall-time share, a bandwidth
measurement or proof of a hardware bottleneck classification.

## Native GPU rerun

The fixed setup is Qwen/Qwen3-30B-A3B checkpoint revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, layer 0, BF16, hidden size 2048,
expert intermediate size 768, 128 logical experts and top-8 routing. The replica
budget is 32 extra copies: 160 physical slots, 80 per GPU. The recorded stack uses
two H100 94 GB GPUs on one node with NV6 connectivity, vLLM 0.31.0+cu129 at commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, PyTorch 2.13.0+cu129 and CUDA 12.9.
The original driver is 560.35.03; its `nvidia-smi` CUDA 12.6 label describes driver
capability rather than the PyTorch build. Preserve the complete actual package,
GPU, topology, allocation and source records under each run's `provenance/`.

Runtime source attestation includes the native fused-MoE selector and
implementation. `optimization/kernel_config.py` restricts the candidate set,
checks the actually resolved configuration and restores the process-global
override even on exceptions. DEFAULT means the unchanged native lookup; for
the executed E=80/N=768 H100 shape, the recorded tuning table was absent and the
native fallback was used. A different source revision or resolved kernel setting
must fail the declared configuration checks. TUNED uses the archived pilot
selection and is not retuned from validation or final outcomes.

Prepare the checkpoint's expert tensors with the pinned exporter if they are not
already present:

```bash
python scripts/prepare_layer.py \
  --output /path/to/qwen3-layer0 \
  --cache-dir /path/to/huggingface-cache
```

The expected exported expert-file SHA256 is
`80046eef8f68a7afd8d165e582190900854faa44076bd22c6a869730e26d4096`.
The repository contains provenance and an exporter, not redistributed model
weights. Obtain weights from the original model source and retain its license.

With the recorded environment and exact archived input files, a direct final
rerun is:

```bash
torchrun --standalone --nproc-per-node=2 -m optimization.runner \
  --manifest configs/optimization-final-frozen-v1.json \
  --banks-dir results/optimization/capture \
  --weights-dir /path/to/qwen3-layer0 \
  --output /path/to/new-optimization-final/records.jsonl
```

On Kuma, use the wrapper from the repository directory. It records the
environment/allocation and refuses an existing output directory:

```bash
mkdir -p logs
sbatch scripts/run-optimization-kuma.sh final \
  /work/cs-471/skewbench-hlin/results/optimization-final-rerun-new \
  configs/optimization-final-frozen-v1.json \
  results/optimization/capture
```

`scripts/cluster-env.sh` activates `$SKB_ROOT/venv` and the wrapper expects expert
weights at `$SKB_ROOT/models/qwen3-layer0`, plus a `SOURCE_COMMIT` file in the
submitted checkout. Its default project root is
`/work/cs-471/skewbench-hlin`; set `SKB_ROOT` for another installation. The wrapper
requests one node, two H100s, eight CPUs, account cs471 and its configured QoS.
Adapt scheduler/account paths only for an authorized equivalent installation,
recording the differences; do not silently relax runtime or data pins.

For an exact rerun use the existing selected configuration and frozen final
manifest. To repeat the entire pilot procedure independently, run the tuning
manifest into fresh outputs, regenerate selection, then pin new validation/final
manifests. Those new selection-file hashes constitute a separately versioned
experiment even if the chosen tile dictionaries agree. The `validation-profile`
wrapper mode runs the 64 policy validations and then launches profiling in a
separate `torchrun` process, preventing profiler instrumentation from entering
policy timing. Profiling is restricted to the pilot validation manifest.

There is no splice/resume workflow for partial matched groups. Preserve failed
attempts and rerun the complete declared matrix in a new output location. Do not
combine DEFAULT and TUNED arms from different allocations into a paired group.

## Captures and measurement boundary

Ten authenticated banks contain 4096 complete token records each: 2048 from prose
and 2048 from Python source. The two pilot and eight final document pairs are
new to this optimization study. `optimization/corpus/` archives source texts,
licenses and their identity/content hashes. Banks retain exact BF16 hidden bits,
native logical top-8 IDs, rounded combining weights, token positions and source
metadata. The eighty policy tapes select whole captured records without
replacement inside a batch; records recur across steps. Temporal schedules are
constructed from teacher-forced causal-prefill observations, not natural online
or autoregressive sessions.

Each tuning cell has one saved 48-input index file reused across every candidate
and repetition: 16 conditioning forwards then 32 timed forwards on an initial,
fixed map. Retain all 192 declared trials in the archive, including the explicit
configuration that duplicates DEFAULT at batch 128. JIT, input materialization,
reference construction and conditioning are outside the tuning timer and their
costs are recorded. Selection uses completed forward time including physical
mapping and output allreduce, not isolated GEMM durations.

Each policy arm uses 32 untimed observation inputs and 512 measured inputs. The
timer includes logical-observation handling, physical mapping, fused experts,
output allreduce, Python replay, native planning/admission and accepted
synchronous weight transfer/map commit, with device completion before stopping.
The native window is 16 and cadence 64; a proposal after the last measured input
is suppressed consistently. Same-policy DEFAULT/TUNED arms must reproduce the
same observations, candidate/commit trajectory and actual sender payloads.

Full input materialization, initialization/JIT, preflight and post-run checks are
excluded and logged. Numerical checks compare complete endpoint batches with
DEFAULT outputs plus 16 evenly spaced independent FP32 reference tokens per
endpoint; they do not independently reference every token at every step. Exact
resident-weight/map and cross-rank checks complement those sampled references.
Per-step CUDA events are shared instrumentation; phase timers overlap device
work and are not an additive component decomposition.

The executed communication path is replicated input with NCCL output allreduce
and synchronous native `torch_nccl` weight migration. It omits online logical
router computation, attention, token all-to-all, serving scheduling, cold start
and input-feeding latency. The study can establish a configuration effect on
this layer/path; it does not establish a new routing algorithm or full-serving
speedup.

## Optional input regeneration

The exact archived NPZ files are authoritative for a frozen rerun. To recapture
the fresh corpus in the pinned GPU/Transformers environment, use a new directory:

```bash
python scripts/capture_optimization_tapes.py \
  --output-dir /path/to/new-optimization-capture \
  --cache-dir /path/to/huggingface-cache \
  --local-files-only --compose
```

This requires the original checkpoint/tokenizer files in the local cache;
omitting `--local-files-only` permits fetching the pinned revision. Capture pins
Transformers 5.17.0 and the exact decoder/masking source files. It checks loaded
checkpoint tensors, the captured post-attention normalization boundary, native
router recomputation, independent routing arithmetic and BF16 round-trip
identity. It does not measure expert runtime.

To recompute policy schedules from already authenticated banks without
recapturing the model boundary:

```bash
python scripts/compose_optimization_tapes.py \
  --banks-dir results/optimization/capture/banks \
  --output-dir /path/to/new-optimization-tapes
```

Composition uses raw 544 / warm 32 / measured 512, seed 10471, phase 120 for
transient/persistent, phase 112 for switching, duration 16 and repeat 1024.
Recreated files may have different file bytes or capture metadata even when
array contents agree. Changed artifacts require newly versioned manifests;
never edit frozen hashes to present them as the original inputs.
