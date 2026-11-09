# Reproducing the instructor-feedback supplement

The supplement adds externally scheduled layer jobs and architectural accounting to the completed resident-dispatch experiment. Its observations are descriptive. It does not change the original migration thresholds, resident dispatcher, final workloads, confidence intervals, or acceptance decision.

The execution source was frozen at `9e020d0`. The calibration manifest is `configs/instructor-arrival-calibration-v1.json`, whose canonical hash is `875419578f9a9d6ccd4cee216c85753dc977f10673fb21cc505ef4355cb7e1a2`. The manifest pins the arrival implementation, the complete protocol, and the descriptive analysis. The underlying resident execution fingerprint and model/software pins are inherited from `configs/resident-online-validation-frozen-v1.json`.

The [protocol](instructor-arrival-protocol.md) defines the arrival and completion clocks, queue, native-only calibration, fixed load fractions, and finite-run accounting. A layer job contains a preassembled captured token batch. It is not a full-model user request. All 256 jobs must finish; the nominal release horizon cannot censor late completions.

## GPU execution

The new study requires the same two-H100 configuration, pinned environment and model weights as the [resident reproduction](resident-dispatch-reproduction.md), plus the existing `optimization-capture-v1` development banks. Calibration and evaluation must share one allocation. The master wrapper first runs eight native-only calibrations, then deterministically freezes four calendars per state, then runs all 128 evaluation trials forming 64 native/online pairs. The online result cannot influence those calendars.

```bash
sbatch --exclude=kh024 scripts/run-instructor-arrival-study-kuma.sh \
  configs/instructor-arrival-calibration-v1.json \
  /work/cs-471/skewbench-hlin/results/optimization-capture-v1 \
  /work/cs-471/skewbench-hlin/results/instructor-arrival-study-reproduction
```

The output directory must be new. The wrapper records the source commit, environment lock, allocation, topology and GPU identities before timing. It creates `calibration/records.jsonl`, `evaluation-plan.json`, `evaluation/records.jsonl` and both descriptive summaries. Each phase stores its input/physical-assignment artifacts. No profiler or concurrent diagnostic process should run during this allocation.

The reproducible intervention is unchanged native SKIP versus the previously frozen online resident dispatcher. Both retain native observation and nonterminal planning, fixed resident weights and map, zero migration, the same job sequence, and identical arrival offsets within a pair. The wrapper charges CPU commands, GPU completion synchronization and rank acknowledgements. It therefore has a different timing boundary from the historical asynchronous host replay; its reference capacity is measured anew.

## CPU checks

The repository test suite exercises calendar construction, altered-calendar rejection, incomplete-job handling, timing envelopes, occupancy identities, analysis pairing and independent auditing:

```bash
python -m pytest -q
```

Production analysis can be rerun without a GPU:

```bash
python -m arrival_study.analysis \
  --plan results/instructor/arrivals-v1/provenance/calibration-plan.json \
  --records results/instructor/arrivals-v1/calibration/records.jsonl.gz \
  --output /tmp/instructor-calibration-summary.json
python -m arrival_study.analysis \
  --plan results/instructor/arrivals-v1/evaluation-plan.json \
  --records results/instructor/arrivals-v1/evaluation/records.jsonl.gz \
  --output /tmp/instructor-evaluation-summary.json
```

The independent verifier imports no production arrival module. It reconstructs integer calendars, all timestamp-derived summaries, queue occupancy, horizon/drain accounting, rank completion envelopes, source/order identities and raw logical/physical routing windows. It also checks each evaluation row against its referenced native calibration allocation. The earlier independent CPU route oracle supplies physical routing checks.

```bash
python scripts/verify_instructor_arrivals.py \
  --plan results/instructor/arrivals-v1/provenance/calibration-plan.json \
  --records results/instructor/arrivals-v1/calibration/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --summary results/instructor/arrivals-v1/calibration/summary.json \
  --output /tmp/instructor-calibration-audit.json
python scripts/verify_instructor_arrivals.py \
  --plan results/instructor/arrivals-v1/evaluation-plan.json \
  --records results/instructor/arrivals-v1/evaluation/records.jsonl.gz \
  --calibration-plan results/instructor/arrivals-v1/provenance/calibration-plan.json \
  --calibration-records results/instructor/arrivals-v1/calibration/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --summary results/instructor/arrivals-v1/evaluation/summary.json \
  --output /tmp/instructor-evaluation-audit.json
```

The evaluation plan hashes the original uncompressed calibration bytes. The independent verifier accepts the byte-preserving gzip archive and checks the decompressed content against that hash. Do not reserialize records before archiving. The raw assignments and capture inputs must remain available; a summary alone is insufficient for this audit.

## Architectural evidence

The architecture analyzer reconstructs per-slot work from the existing saved assignments and checks the recorded metrics before aggregating active physical experts, M tiles, useful rows, useful GEMM FLOPs and tile-implied work. It deduplicates technical repetitions with identical assignment artifacts. The separate GPU-profile aggregation retains per-rank kernel residence and peer-waiting limitations. The exact input hashes are recorded in `results/instructor/architecture-v1/analysis.json`; its original pre-probe summary and the updated post-probe note are retained separately.

```bash
python scripts/analyze_resident_architecture.py \
  --records results/resident/online-final-v1/records.jsonl.gz \
    results/resident/online-validation-v1/records.jsonl.gz \
  --inputs-root results/resident/online-final-v1/rep0/inputs \
    results/resident/online-final-v1/rep1/inputs \
    results/resident/online-validation-v1/inputs \
  --profile-analysis results/resident/profile-v1/analysis.json \
  --profile-root results/resident/profile-v1 \
  --ncu-availability results/instructor/architecture-v1/ncu-availability-before-probe.json \
  --output-json /tmp/instructor-architecture.json \
  --output-markdown /tmp/instructor-architecture.md
```

This rerun reproduces the geometry, all assignment/profile aggregates, tool-catalog data and analyzer hash exactly. The catalog input's path spelling differs from the original working path; its file hash is unchanged. The saved `cpu-reproduction.json` records this comparison.

Nsight Compute counter capture was attempted separately on allocation `4870934`, using an inexpensive permission probe before model loading. The cluster returned `ERR_NVGPUCTRPERM`; `results/instructor/counters-unavailable-v1` retains the raw error, tool version, driver setting and environment. This is a failed counter measurement, not a measured HBM or Tensor Core utilization result. No administrator settings were changed.

Reconstructing these artifacts is computational reproducibility of the saved evidence. An independently executed study on another allocation or by another group remains a separate empirical check.
