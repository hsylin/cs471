# SkewBench

SkewBench tests whether two-window demand stability improves expert-migration
admission beyond an improvement threshold and a fixed cooldown. The required
study uses one distributed MoE layer, one hardware configuration, and one fixed
replica budget. Performance claims require completed GPU runs; CPU unit tests
and smoke runs are not performance evidence.

## Scope

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

## Reproducibility contract

Pilot and final trace seeds are disjoint. Thresholds, cadence, cooldown, batch
sizes, trace parameters, measurement horizon, and replicate count must be frozen
and hashed before final measurements. Incomplete runs and failed correctness
checks cannot enter the performance analysis. The primary useful-gain endpoint
is the prespecified short-hotspot regime; persistent changes and window-phase
tests characterize limitations. Stable skew tests regression.

The suggested acceptance margins are a 5% useful elapsed-time improvement over
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
thresholds. Final results and their exact commands are added after execution.
