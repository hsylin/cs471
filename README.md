# SkewBench

**Instructor-feedback supplement.** The bounded arrival study completed eight native calibrations and 128 evaluation trials on two H100s. Online dispatch reduced mean service in 27/32 descriptive cells, but mean response in only 21/32; it did not establish a general latency improvement. Independent raw-data audits and all 425 CPU tests pass. Architectural accounting explains reduced padded work without claiming measured hardware utilization; the counter probe was denied by node permissions. The historical 5% target failure remains unchanged. Read the [results and feedback response](docs/instructor-feedback-results.md), [notation and parameter history](docs/instructor-feedback-response.md), and [reproduction guide](docs/instructor-supplement-reproduction.md).

**Completed resident-dispatch follow-up.** One fixed GPU dispatcher and 256
fresh runs reduced completed changing-workload layer time by 3.40% (95% block
interval: 2.99–3.85%), missing the frozen 5% target. Batch 1024 improved
6.70–7.29% across regimes; stable batch 128 was 2.93% slower, with the 2%
regression guard inconclusive. Both final allocations used the same two H100s.
Read the [result and limits](docs/resident-dispatch-results.md),
[frozen protocol amendment](docs/resident-online-validation-protocol.md), and
[reproduction guide](docs/resident-dispatch-reproduction.md). Native SKIP remains
the unchanged baseline; this is a fixed-map layer experiment, not a serving claim.

**Completed kernel-optimization follow-up.** A bounded eight-option native
kernel search and 512 fresh held-out policy runs produce a 0.42% lower B3 runtime
estimate (95% block interval: 0.30–0.55% lower), below the frozen 5% target. The
stable guard passes. Identical-config controls also show sub-percent differences,
limiting causal attribution of that small estimate. The selected batch-128
N-tile change saves only 4–6 microseconds inside the expert GEMMs. At batch 1024,
selection retains DEFAULT. Fixed placement with default kernels is descriptively
2.90% lower elapsed time than B3 on these changing traces; it remains an existing control,
not a new algorithm or a full-serving claim.

Read the [kernel evaluation](docs/optimization-evaluation.md),
[frozen protocol](docs/optimization-protocol.md),
[raw artifacts](results/optimization/README.md), and
[reproduction instructions](docs/optimization-reproduction.md).
All 206 CPU tests and independent statistics, routing, input and profile audits
pass; clean-checkout CPU reproduction recovers the final summary byte for byte.

**Completed follow-up: candidate payback.** The literature-guided follow-up uses
authentic Qwen layer inputs and 1,536 held-out three-arm measurements on two
H100s. Native and single-copy placements have small resident gains (0.77% and
0.20%), but charging their transitions makes changing-workload elapsed time
2.99% and 2.45% worse at H512. Both fail the frozen 5% useful-gain and 2% stable
regression criteria. Rare profitable native cases are retained; this is a
candidate-specific boundary for the pinned default kernel and constructed
prefill replay, not a universal result about migration.

Read the [follow-up evaluation](docs/payback-final-evaluation.md),
[exact inputs and raw records](results/payback/README.md),
[reproduction instructions](docs/payback-reproduction.md), and
[literature synthesis](docs/payback-literature-review.md). The full CPU-only
reproduction command and independent arithmetic/dispatch audits pass.

**Original TV-gate study.**

SkewBench tests whether two-window demand stability improves expert-migration
admission beyond an improvement threshold and a fixed cooldown. The required
study uses one distributed MoE layer, one hardware configuration, and one fixed
replica budget. Performance claims require completed GPU runs; CPU unit tests
and smoke runs are not performance evidence.

The completed held-out study contains **1,400 valid policy runs and eight
same-state decision pairs** on two H100 GPUs. The TV gate did not meet the frozen
5% useful-gain criterion: burst B4/B3 was 1.0040 (95% CI 0.9990–1.0098), and B4/B5
was 1.0116 (1.0094–1.0136). Stable B4/B3 passed the 2% regression margin despite
about 1.15% extra elapsed time. All eight forced migrations lost time over their
128-step replay horizons. This is a bounded negative result, with a complete
experiment and passing correctness checks.

Read the [final evaluation](docs/final-evaluation.md),
[raw-data index](results/README.md), and
[frozen analysis](results/final/analysis/summary.json).

## Original-study scope

- vLLM v0.31.0, source commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Explicitly synchronous native EPLB layer execution, with its execution and
  dispatch path recorded in every run.
- Actual expert weights from layer 0 of `Qwen/Qwen3-30B-A3B`, revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- 128 logical experts, top-8 routing, hidden size 2048, expert intermediate size
  768, BF16 weights. Controlled synthetic activations and logical routes are
  replayed identically across methods. This is not a natural-text serving trace.
- Course allocation: Kuma, account `cs471`; the initial hardware check used two
  H100 94 GB GPUs on one node, connected by NVLink. Actual run topology is logged.

The core comparison uses B2 (native balancing with replicas), B3 (improvement
gate), B4 (stability plus improvement), B5 (one frozen cooldown plus improvement),
and SKIP (fixed placement with the same replica budget). B0 and B1 are
no-additional-replica contextual controls.

## Original-study reproducibility contract

Pilot and final trace seeds are disjoint. Thresholds, cadence, cooldown, batch
sizes, trace parameters, measurement horizon, and replicate count must be frozen
and hashed before final measurements. Incomplete runs and failed correctness
checks cannot enter the performance analysis. The primary useful-gain endpoint
is the prespecified short-hotspot regime; persistent changes and window-phase
tests characterize limitations. Stable skew tests regression.

The frozen acceptance margins are a 5% useful elapsed-time improvement over
both B3 and B5 on the target regime and at most 2% stable-workload regression
against B3. These are design choices, not expected outcomes. Analysis uses paired
log runtime ratios and resamples independent trace-seed blocks. Technical repeats
within a seed do not create independent workload samples.

Correctness requires preserved logical routes, reference numerical agreement,
consistent expert identities/maps/replica counts, unchanged state after rejected
moves, and complete installation of accepted weights. A discarded async result
must never be counted as a committed migration. The initial study is synchronous.

The primary timer covers 512 completed layer steps after 32 untimed observation
steps and process-local native transport warmup. Candidate opportunities occur
every 64 measured steps; a fresh proposal after the final input is suppressed.
B4 pays for collecting the required previous window and calculating stability;
it reuses native EPLB's current window. Rejections return the current map through
the native synchronous no-op path, including that path's overhead. SKIP uses this
same admission point and therefore still pays for native monitoring and planning.

The inputs are replicated on both ranks, local expert computation uses native
fused MoE kernels, and outputs use an NCCL all-reduce. Expert weights move using
the native `torch_nccl` EPLB communicator. This isolates layer admission; it does
not measure DeepEP token all-to-all, distributed serving, or full-model latency.
Native observation totals contain one copy per input rank; normalization cancels
this factor, and recorded totals explicitly retain its multiplicity.

Exact input tapes, numerical input tensors, source hashes, per-decision mappings,
transfer payloads, commit checks, and step timings are retained outside Git.
Branch diagnostics restore identical recorded weights and mappings from a B3
trajectory, force allow or skip, and replay 128 future steps with later balancing
disabled. Four pairs run allow first and four run skip first, with their order
shuffled using the frozen diagnostic order seed. Positive `skip_time - allow_time` means the move repaid its cost over
that particular future; these few cases do not establish population accuracy.

## Cluster setup

The course handout requires `--account=cs471`; do not substitute a paid account.
Use no more than four GPUs per job. Keep files within the group's project
directory; do not inspect other groups' files. Credentials are not repository
configuration and must never be committed.

```bash
bash scripts/bootstrap-kuma.sh
source scripts/cluster-env.sh
python scripts/prepare_layer.py \
  --output "$SKB_ROOT/models/qwen3-layer0" \
  --cache-dir "$SKB_ROOT/cache/huggingface"
```

`prepare_layer.py` downloads only checkpoint shards needed for the selected
layer, exports exact pretrained expert weights, and records source and output
hashes. Model weights and large raw results remain outside Git.

## CPU validation

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Tests cover controller arithmetic and boundaries, deterministic top-k traces,
sampling-domain separation, data validation, and paired statistics. Synthetic
test numbers are never presented as GPU results.

## Related primary sources

The benchmark isolates admission rather than implementing these complete systems.
These are cited as preprints/manuscripts, not independently reproduced findings:
[METRO](https://doi.org/10.48550/arXiv.2512.09277),
[Gimbal](https://doi.org/10.48550/arXiv.2606.15177),
[TEMPO](https://doi.org/10.48550/arXiv.2608.13057), and
[FreeBalance](https://doi.org/10.48550/arXiv.2608.14205).

## Reproduce the pilot

The frozen manifests contain source hashes; editing the harness requires a new
manifest. Earlier diagnostic pilots remain identifiable by their separate
manifests; see `docs/execution-notes.md` for corrections. The complete cooldown
pilot uses the `pilot-v3` manifests and source checkout `de304a6`. Use that
checkout to repeat the commands below. The final source additionally avoids
collecting unused B4 histograms; the B3/B5 selection paths are unchanged.

```bash
mkdir -p logs
sbatch scripts/run-suite-kuma.sh "$SKB_ROOT/results/pilot-v3" \
  configs/frozen/pilot-v3-k128.json \
  configs/frozen/pilot-v3-k64.json \
  configs/frozen/pilot-v3-k256.json
```

The suite runs a correctness preflight, then all three parameter candidates
sequentially on one two-GPU allocation. It refuses to overwrite existing results.
Use a fresh output directory for independent replication. Preserve the complete
environment lock and hardware logs alongside the records. The CUDA histogram
preflight checks exact counts and uses PyTorch's synchronization debug guard;
that prototype guard is not an exhaustive proof about every CUDA operation.

```bash
python -m skewbench.analysis cooldown-select \
  --manifests configs/frozen/pilot-v3-k128.json \
              configs/frozen/pilot-v3-k64.json \
              configs/frozen/pilot-v3-k256.json \
  --records "$SKB_ROOT/results/pilot-v3/records-pilot-v3-k128.jsonl" \
            "$SKB_ROOT/results/pilot-v3/records-pilot-v3-k64.jsonl" \
            "$SKB_ROOT/results/pilot-v3/records-pilot-v3-k256.jsonl" \
  --output-dir "$SKB_ROOT/results/pilot-v3/selection"
```

The code must not consume held-out final records to select the cooldown or
thresholds.

## Reproduce the final reports

With Python 3.10 or newer, NumPy, and Matplotlib installed, the archived JSONL
records can regenerate the tables, confidence intervals, branch summaries, and
figures locally without GPUs, vLLM, model weights, or network access. Install the
small analysis package beforehand if needed (`python -m pip install -e .`).
Scikit-learn is not required.

```bash
bash scripts/reproduce-reports.sh ../skewbench-final-reproduction
```

The wrapper reads `results/final/records-final.jsonl.gz`,
`results/final/branches.jsonl.gz`, and `configs/frozen/final.json`. It decompresses
the records into the new output directory, runs the current primary and branch
analyzers, then runs the descriptive diagnostic report. Any existing output path
is refused, including an empty directory. If a command fails, partial outputs
remain available for inspection; use another new directory for the next attempt.
Set `PYTHON=/path/to/python` when using a particular analysis environment.

The main outputs are `analysis/summary.json`, `analysis/cells.csv`,
`analysis/methods.csv`, `analysis/runtime-ratios.png`, the compact
`analysis/primary-runtime-ratios.png` and PDF, `analysis/branch-summary.json`,
`analysis/branches.csv`, and `diagnostics/diagnostic-summary.json` with PNG/PDF figures. These commands
recompute the archived observations; they do not collect independent GPU data.

An independent arithmetic implementation checks the declared paired estimates
and seed-block bootstrap against the generated summary without importing the
main analyzer:

```bash
python scripts/verify_final_ratios.py \
  --manifest configs/frozen/final.json \
  --records ../skewbench-final-reproduction/records-final.jsonl \
  --report ../skewbench-final-reproduction/analysis/summary.json \
  --output ../skewbench-final-reproduction/ratio-audit.json
```

The separate branch/full-policy correspondence audit requires the original raw
NPZ input tapes, which remain outside Git. Supply their containing directory;
the script checks their recorded hashes, exact input arrays and logical windows,
the matching B3 proposal states, and the prespecified branch order:

```bash
python scripts/verify_final_branch_correspondence.py \
  --manifest configs/frozen/final.json \
  --records ../skewbench-final-reproduction/records-final.jsonl \
  --branches ../skewbench-final-reproduction/branches.jsonl \
  --trace-dir /path/to/original/raw-traces \
  > ../skewbench-final-reproduction/branch-correspondence-audit.json
```

Run these audit scripts without Python's `-O` flag; they use assertions. Raw-tape
correspondence does not independently rerun numerical or GPU weight checks;
those remain recorded runtime attestations.

## Rerun the final GPU experiment

The original frozen execution checkout is
`b4a837dea77409c693731c71b9dd8c43ae244c21` (`b4a837d`). The measured package and
native-source pin retain harness SHA-256
`894bf6474c57655c15d41709ffdb9e84f9e9e16aaece6d7de6a4fc88c94594cb`;
later report scripts and documentation do not change that measured-package hash.
Use `configs/frozen/final.json` as committed, rather than regenerating a manifest.
The full executed Python environment lock is
[`results/provenance/environment-lock.txt`](results/provenance/environment-lock.txt),
with wheel provenance in
[`results/provenance/package-provenance.json`](results/provenance/package-provenance.json).
The lock, pinned CUDA wheel, driver, and recorded GPU topology are part of GPU
reproduction; the lightweight CPU analysis dependencies are not a replacement.
In a fresh replication environment, synchronize the installed packages to that
lock after bootstrap (set `SKB_ROOT` to your own project directory):

```bash
uv pip sync --python "$SKB_ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match results/provenance/environment-lock.txt
```

After the cluster and pretrained-layer setup above, run from the repository root
on Kuma using a fresh result directory:

```bash
source scripts/cluster-env.sh
mkdir -p logs
final_rerun_directory="$SKB_ROOT/results/final-independent-rerun"
if [ ! -e "$final_rerun_directory" ] && [ ! -L "$final_rerun_directory" ]; then
  sbatch scripts/run-final-kuma.sh "$final_rerun_directory" \
    configs/frozen/final.json
else
  echo "Choose a new output directory: $final_rerun_directory" >&2
fi
```

The job uses one two-GPU allocation, runs the correctness preflight, the frozen
matrix and eight paired branch states, then generates reports. Choose a different
directory for each repetition and preserve its environment and hardware logs.
The runner checks the measured-package and installed native-source hashes against
the frozen configuration. A new hardware allocation collects independent timings;
it does not guarantee identical timings or conclusions.
