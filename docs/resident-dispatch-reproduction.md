# Reproduce the resident-dispatch experiment

This guide separates CPU reconstruction of archived evidence from a new GPU
measurement. The fixed question is whether current-batch selection among resident
expert copies improves completed layer time after charging selection and native
accounting. The offline headroom assignments and the online greedy policy are
different implementations; neither an active-copy optimum nor a CPU token-load
comparison is a runtime bound.

Use the complete archive accompanying the report, including its later audit
scripts. The frozen online execution snapshot is Git commit `4e92b1c`; validation
ran from `0d54786`, with the same measured source fingerprint. No execution or
policy change separates validation and final. The final allocation identities
are recorded with the final results, rather than inferred from this document.

## Identity and retained evidence

Run all commands from the repository root. The principal paths are:

```text
configs/resident-headroom-frozen-v1.json
configs/resident-headroom-frozen-v2.json
configs/resident-online-validation-frozen-v1.json
configs/resident-online-final-frozen-v1.json
configs/resident-headroom-analysis-v1.json
configs/resident-online-validation-analysis-v1.json
configs/resident-online-final-analysis-v1.json
results/resident/
  headroom-invalidated-v1/          # entire contaminated attempt retained
  headroom-calibration-failed-v1/   # all 80 completed rows; failed A/A retained
  headroom-passed-v2/               # revised 80-row headroom experiment
  profile-v1/                      # original Nsight reports and descriptive analysis
  approximation-v1/                # old-input DP-versus-greedy count comparison
  capture-v1/                      # eight fresh banks and 64 replay tapes
  online-validation-v1/            # 32 rows, eight input artifacts, provenance
  online-final-v1/
    rep0/                          # first allocation, 128 rows, input artifacts
    rep1/                          # second allocation, 128 rows, input artifacts
    records.jsonl.gz               # complete 256-row union
    summary.json
    allocation-map.json            # repetitions 0/1: jobs 4869986/4870000
```

The final commands below require both complete allocation outputs. They must not
be used to turn an incomplete or invalid matrix into an accepted result.

| Identity | SHA-256 |
|---|---|
| Revised headroom manifest | `559dbcdd6ebd01d9b5c29036fee831fc4e53cb9cca61f23fe337fb8cc33941c8` |
| Online validation manifest | `3b1a64b3966358bbb18a1c4581d5775212f91f32491f35b658e819f7c2c8154a` |
| Online final manifest | `5c39e99cb063a4998753d653c16aa736ba74f569b13714618dc7ac00e6ac1860` |
| Shared online execution fingerprint | `e5677540ad9d0cf7cb824f68ebcf2b5e5c5256ae62c03e425df23010bce0462f` |
| Legacy native harness fingerprint | `894bf6474c57655c15d41709ffdb9e84f9e9e16aaece6d7de6a4fc88c94594cb` |

Manifest hashes use canonical JSON excluding `freeze`, not the formatted file's
byte hash. Each manifest also pins source files, checkpoint/capture/tape files,
the resolved native kernel configurations and the analysis source. Preserve the
original manifest and archive files. Reformatting or recompressing an artifact
can change a file hash even when its decoded data is unchanged.

The initial attempt on allocation 4868958 was invalidated in its entirety because
an overlapping monitoring process could have disturbed measurement. The next,
complete attempt on 4869036 failed the frozen A/A calibration. The single allowed
apparatus revision on 4869257 is separately archived; the original attempts were
not pooled or filtered to create it. Read their `invalidation.md` and
`interpretation.json`. Commit `0b9fbe1` contains the revised historical headroom
execution; `016748f` archives its predecessors. The earlier execution is also
preserved in the `source-commit.txt` of each attempt. Historical GPU reruns need
those checkouts because adding the online modules changes the measured source
fingerprint. Use separate worktrees, never replace the frozen hashes to make a
new checkout masquerade as an old one.

The standalone bundle also includes the historical execution source at
`results/resident/headroom-passed-v2/source-snapshot-0b9fbe1.tar.gz`.
Extract it separately for that pilot; supply the two archived development banks
from the main bundle as `--banks-dir`. The snapshot has no online modules.

## Clean CPU reconstruction

The source-and-results tarball contains no `.git` directory. Extract it into a
fresh directory and enter its `cs471` directory before running the commands
below. For a Git clone instead, create a clean worktree of the complete archive
commit accompanying the report; this optional step requires Git metadata:

```bash
git worktree add --detach ../cs471-resident-reproduction HEAD
cd ../cs471-resident-reproduction
```

The recorded CPU analysis environment used Python 3.13.7, NumPy 2.5.4,
Matplotlib 3.11.2 and pytest 9.2.0. From either fresh source directory:

```bash
python3.13 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install \
  'numpy==2.5.4' 'matplotlib==3.11.2' 'pytest==9.2.0'
mkdir -p work/resident-reproduction-new
.venv-analysis/bin/python -m pytest -q
```

The CPU checks need no CUDA, vLLM, model weights or cluster credentials. Run from
the root: the existing package installer only discovers `skewbench`, while these
commands also use the repository's `resident_dispatch`, `optimization` and
`followup` packages. Do not use Python `-O`; several independent audits rely on
assertions. Choose fresh audit destinations and preserve existing reports.

Verify the fresh source exclusions, capture provenance, actual arrays and all
independently reconstructed replay indices:

```bash
.venv-analysis/bin/python scripts/audit_resident_inputs.py \
  --capture-dir results/resident/capture-v1 \
  --previous-bank results/optimization/capture/banks/opt-pilot-01.npz \
  --output-prefix work/resident-reproduction-new/fresh-inputs
```

This checks eight banks and 64 tapes, including native-capture attestations and
whole-document exclusions against both prior corpora. It does not independently
rerun pretrained attention or the learned router. Document disjointness is not a
claim that prose, comments or programming constructs never overlap.

Recompute the revised headroom statistics, then independently check its
selection and raw physical records:

```bash
.venv-analysis/bin/python -m resident_dispatch.analysis \
  --plan configs/resident-headroom-analysis-v1.json \
  --records results/resident/headroom-passed-v2/records.jsonl.gz \
  --output work/resident-reproduction-new/headroom-summary.json

.venv-analysis/bin/python scripts/verify_resident_headroom.py \
  --manifest configs/resident-headroom-frozen-v2.json \
  --records results/resident/headroom-passed-v2/records.jsonl.gz \
  --summary work/resident-reproduction-new/headroom-summary.json \
  --allocation-id 4869257 \
  --output work/resident-reproduction-new/headroom-statistics.json

.venv-analysis/bin/python scripts/verify_resident_pilot.py \
  --manifest configs/resident-headroom-frozen-v2.json \
  --records results/resident/headroom-passed-v2/records.jsonl.gz \
  --artifacts-dir results/resident/headroom-passed-v2 \
  --banks-dir results/optimization/capture \
  --output work/resident-reproduction-new/headroom-routing.json
```

The statistics verifier independently reconstructs all 80 keys, A/A calibration,
reserves and the recorded selection. The raw verifier reads NPZ arrays directly,
checks exact native hashing, assignment eligibility/count objectives, all logged
physical windows, fixed maps, zero migration, local order and revised setup
metadata. Neither interprets free offline selection as an online speedup.

Recompute the online development summary separately:

```bash
.venv-analysis/bin/python -m resident_dispatch.analysis \
  --plan configs/resident-online-validation-analysis-v1.json \
  --records results/resident/online-validation-v1/records.jsonl.gz \
  --output work/resident-reproduction-new/validation-summary.json
```

Independently check those descriptive ratios and their recorded contract:

```bash
.venv-analysis/bin/python scripts/verify_resident_validation.py \
  --manifest configs/resident-online-validation-frozen-v1.json \
  --records results/resident/online-validation-v1/records.jsonl.gz \
  --summary work/resident-reproduction-new/validation-summary.json \
  --allocation-id 4869806 \
  --output work/resident-reproduction-new/validation-statistics.json
```

Its 32 rows are descriptive development evidence. There is no development
confidence interval or confirmatory performance acceptance. The documented
validation amendment records the user's authorization to proceed to the bounded
fresh evaluation after correctness validation without retuning the policy.

For the complete final archive, reproduce the frozen block-bootstrap analysis
and independently recompute its statistics:

```bash
.venv-analysis/bin/python -m resident_dispatch.analysis \
  --plan configs/resident-online-final-analysis-v1.json \
  --records results/resident/online-final-v1/records.jsonl.gz \
  --output work/resident-reproduction-new/final-summary.json

.venv-analysis/bin/python scripts/verify_resident_final.py \
  --manifest configs/resident-online-final-frozen-v1.json \
  --records results/resident/online-final-v1/records.jsonl.gz \
  --summary work/resident-reproduction-new/final-summary.json \
  --allocation-map results/resident/online-final-v1/allocation-map.json \
  --output work/resident-reproduction-new/final-statistics.json
```

The archived allocation map binds repetition `"0"` to job `4869986` and `"1"` to
job `4870000`. For a new GPU reproduction, use its newly declared allocation map.
The verifier checks 256 rows, 128 pairs, all paired source/state identities,
local arm order, the bootstrap intervals, practical-gain criterion and stable
regression guard. The bootstrap resamples eight source blocks; ranks, steps and
technical repetitions are not additional independent workloads.

Independently audit the online records and reconstruct their physical routes
from the captured arrays:

```bash
.venv-analysis/bin/python scripts/verify_resident_online_records.py \
  --manifest configs/resident-online-validation-frozen-v1.json \
  --records results/resident/online-validation-v1/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --output work/resident-reproduction-new/validation-routing.json

.venv-analysis/bin/python scripts/verify_resident_online_records.py \
  --manifest configs/resident-online-final-frozen-v1.json \
  --records results/resident/online-final-v1/records.jsonl.gz \
  --capture-root results/resident/capture-v1 \
  --inputs-root results/resident/online-final-v1/rep0/inputs \
                results/resident/online-final-v1/rep1/inputs \
  --output work/resident-reproduction-new/final-routing.json
```

These checks use saved NPZs and an independent implementation of the physical
assignment rule, without importing the production selector or analysis. A
single complete allocation can be checked separately with `--repetition 0` or
`--repetition 1` and its records/input directory. That partial-stage audit does
not satisfy the complete final statistical matrix. Recorded numerical
attestations are checked, but expert arithmetic is not independently executed.

An optional descriptive approximation audit is reproducible without GPUs:

```bash
.venv-analysis/bin/python scripts/analyze_resident_approximation.py \
  --manifest configs/resident-online-validation-frozen-v1.json \
  --banks-dir results/optimization/capture \
  --pilot-inputs results/resident/headroom-passed-v2/inputs \
  --output work/resident-reproduction-new/approximation.json \
  --markdown work/resident-reproduction-new/approximation.md
```

This reuses only the old development inputs, checks the exact original 48 input
indices and saved DP assignments, and compares counts against the frozen CPU
reference for the online greedy policy. It neither changes the policy nor
estimates selector runtime.

## New GPU measurements

Use two H100 94 GB GPUs on one node with NV6 connectivity, one rank per GPU. The
recorded path is synchronous EPLB, replicated input tensors and output allreduce;
it is not an all-to-all serving engine. Keep DEFAULT fused BF16 kernels, 128
logical experts, top-8, 160 physical slots and 80 slots per rank. The 32 redundant
copies are an extra budget, not the total expert count. The physical map stays
`[0..127,0..31]`; in the online arm, both ranks choose the same global physical
home per logical expert. No checkpoints, precision settings, kernels or
thresholds are changed.

The same-batch contract matters. With different router inputs at different
source ranks, independent local selections would not guarantee the global
active-copy optimum measured here. A serving integration would need an agreed
assignment or demand exchange, and its coordination cost must be measured. This
prototype supports only the frozen map, top-8 and the two declared batch sizes.

The native source pin is vLLM
`db9527a46873454610df6dbedf79a36d6bf1a7f6` (`0.31.0+cu129`). The full executed lock is
`results/resident/online-validation-v1/provenance/environment-lock.txt`; retain the
final allocation locks alongside it. Key versions are Python 3.12.14,
PyTorch 2.13.0+cu129, Triton 3.7.1, NCCL 2.29.7, Transformers 5.17.0 and
TorchCodec 0.16.0+cu129. Driver 560.35.03 reports CUDA 12.6 in `nvidia-smi`; the
PyTorch/runtime build is CUDA 12.9. Preserve both facts and the actual topology.
The CPU analysis environment is not a GPU environment specification.

For a new authorized project root, set `SKB_ROOT` before using
`scripts/bootstrap-kuma.sh` and `scripts/cluster-env.sh`. Bootstrap installs the
explicit official vLLM cu129 wheel; synchronize to the archived full lock rather
than resolving the current dependency releases:

```bash
export SKB_ROOT=/path/to/new-resident-project
bash scripts/bootstrap-kuma.sh
uv pip sync --python "$SKB_ROOT/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  results/resident/online-validation-v1/provenance/environment-lock.txt
```

Obtain the original model subject to its license. Expert weights are not stored
in Git. Export layer 0 from `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`:

```bash
source scripts/cluster-env.sh
python scripts/prepare_layer.py \
  --output "$SKB_ROOT/models/qwen3-layer0" \
  --cache-dir "$SKB_ROOT/cache/huggingface" \
  --layer 0
```

The exported `layer.safetensors` SHA-256 must be
`80046eef8f68a7afd8d165e582190900854faa44076bd22c6a869730e26d4096`.
Its gate/up tensor is `[128,1536,2048]`, down tensor `[128,2048,768]`, BF16. Source,
model, tensor geometry and actual resolved native tile settings are checked by
the runner before timing. A mismatch is not repaired by editing a frozen hash.

Use the exact archived capture files for a timing rerun. A Git clone at the
complete archive commit is the simplest GPU reproduction path. Create
`SOURCE_COMMIT` with its actual checkout identity; the wrapper copies it into each
provenance directory. For a tarball, first verify the unchanged measured files
against the frozen source hashes, then use the archived source identity from the
bundle metadata and explicitly record that this is an extracted archive without
Git metadata. Do not run `git rev-parse` in that case or claim a fresh Git checkout.
The wrapper also records the installed packages, hardware, topology and Slurm
allocation, and refuses an existing output directory.

```bash
git rev-parse HEAD > SOURCE_COMMIT
mkdir -p logs
sbatch scripts/run-resident-kuma.sh online-validation \
  "$SKB_ROOT/results/resident-online-validation-rerun-new" \
  configs/resident-online-validation-frozen-v1.json \
  results/optimization/capture

sbatch scripts/run-resident-kuma.sh final \
  "$SKB_ROOT/results/resident-online-final-rerun-new-rep0" \
  configs/resident-online-final-frozen-v1.json \
  results/resident/capture-v1 --repetition 0
```

After the first allocation completes, submit the second independently, with a
new output directory and the other repetition:

```bash
sbatch scripts/run-resident-kuma.sh final \
  "$SKB_ROOT/results/resident-online-final-rerun-new-rep1" \
  configs/resident-online-final-frozen-v1.json \
  results/resident/capture-v1 --repetition 1
```

The wrapper requests account `cs471`, partition `h100`, one node, two GPUs and
eight CPUs; adapt scheduler/account paths only for another authorized equivalent
installation and record the differences. Each final allocation contains 128
rows. Repetition 0 and 1 reverse each cell's local native/online order. A final run
without `--repetition` is rejected. Do not splice arms from separate attempts or
pool historical calibration attempts into the new result. Keep failed runs and
logs. Combine only the two complete declared repetition files, then rerun all
input, record and statistics audits on the new outputs. GPU timings are new
observations and need not reproduce archived elapsed numbers exactly.

## Inputs, timing and correctness coverage

Final inputs come from eight new document pairs, disjoint by document/content
identity from the prior 40 documents. The banks retain actual BF16 hidden-state
bits, learned-router top-8 IDs and BF16-rounded combining weights from
teacher-forced causal prefill. Each tape constructs 544 input steps: 32 initial
conditioning and 512 measured, with fixed stable, transient, persistent and
switching schedules. These are constructed repeated-prefill workloads, not
observed serving traffic. Cached GPU materialization is common and excluded;
every online assignment is computed from the current step's IDs inside timing.

The measured region includes current-batch histogram, home selection, physical-ID
production, native physical recording, expert execution, output collective,
native SKIP observation/planning and final synchronous completion. It uses the
maximum elapsed time across the ranks. All timing events are initialized before
measurement. Common prefix conditioning and 64 full-SKIP conditioning steps are
excluded and logged, then state is reset. GC stays enabled. The last input's
statistics are recorded but a terminal post-input proposal is suppressed. Final
ordinary decision steps are 63, 127, 191, 255, 319, 383, 447; weights never migrate.

Actual GPU physical assignments for every input in the saved sequence are
checked against the CPU rule, hashed and reduced into saved per-slot counts.
Runtime assertions check
logical-route preservation, rank agreement, residency, exact reconstructed
windows, native planner trajectory, zero migration and unchanged maps/weights.
All nine adversarial cases run with recording enabled and disabled, including
partial/zero real-token masks; their hashes and assertions are recorded.

Numerical coverage is deliberately narrower than checking every output element
of every final forward. Validation compares full-batch outputs for all 48 unique
inputs before timing, with independent FP32 references on 16 evenly spaced tokens
at raw inputs 16 and 47, then repeats endpoint checks after timing. Final full-batch
and FP32 checks use raw inputs 32, 160, 288, 543 before and after timing. Tolerances
remain `atol=0.02`, `rtol=0.03`. Exact resident-weight checks are runtime
attestations; the CPU auditors do not reload checkpoint weights or recompute
GPU expert outputs. Saved source hashes and successful records are evidence of
those checks, not an independent rerun of them.

To regenerate the capture computation itself, use the pinned capture script,
original model revision and source documents in a separate GPU job/output:

```bash
python scripts/capture_resident_tapes.py \
  --sources resident_dispatch/corpus/sources.json \
  --output-dir /fresh/path/resident-capture \
  --cache-dir "$SKB_ROOT/cache/huggingface" \
  --split final --compose --local-files-only
```

The required model/tokenizer checkpoint files must already be in that cache when
using `--local-files-only`. Capture regeneration is stronger evidence than merely
re-reading recorded native-router attestations. Audit the regenerated arrays and
compare them with the archive. If file or content hashes differ, document and
explain the difference; do not silently substitute new artifacts under an old
manifest. Reconstructing scripts and records within this project is same-team
reproduction. External reproduction requires a separate person/environment to
rerun the pinned computation and report its own evidence.
