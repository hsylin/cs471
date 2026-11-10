# Reproducing the residual-overhead study

The source used for fresh measurement was committed before those measurements at `fbae52ae021d0571bfd2b8f53b66f41303007b54`. Its 107-file execution inventory is `136259a52849be38f9dc875f6ccef964ba2c92f8cd53ceee2c5e5c51f2e0dce7`. Later report/auditor additions do not change that inventory. The previous 83-file inventory remains `08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff`.

## CPU reconstruction from a clean checkout

Use Python with NumPy. The recorded GPU environment used NumPy 2.3.5; the local independent reconstruction uses NumPy 2.5.4. Run from the repository root, using a new output path:

```bash
python docs/resident-residual/reproduce.py --output /absolute/new/reanalysis
```

This checks the frozen source inventory, reconstructs all four fresh-stage production summaries, reruns a separately implemented raw-clock/statistical audit, and hashes the archived selector PTX/cubin bytes. It does not execute a model or silently claim a new GPU experiment.

The large authenticated input banks are retained in the original workspace/server and excluded from Git. When the exact banks are available, pass a directory containing `capture-pilot-v1/banks` and `capture-final-v1/banks`:

```bash
python docs/resident-residual/reproduce.py \
  --banks-root /absolute/resident-residual-inputs \
  --output /absolute/new/full-cpu-audit
```

That additionally runs the independent input/assignment/geometry/state/clock auditor on each phase. The output explicitly records whether that additional bank-based audit ran. It still authenticates recorded numerical checks rather than rerunning the GPU kernels. The [public-source rehydration and GPU recapture recipe](recapture-inputs.md) explains how to reconstruct the exact 640-document private manifest without any preceding private selections. A new GPU recapture is a distinct experiment and must retain its own source, environment, hardware and numerical provenance.

## GPU performance replay

The four plan files and each run's `provenance/plan.json` contain exact bank hashes, input seeds, treatment orders, source rows, baseline-capacity calibration records and immutable absolute calendars. Source weights, native vLLM source pins, environment lock, Slurm allocation, GPU UUIDs, topology and compiled specializations are archived with the runs. [The executed phase driver](executed-phase-driver.sh) records the actual sequence: pilot baseline calibration, pilot treatments, final baseline calibration and final treatments. Fresh performance runs must use new output directories.

The original wrappers intentionally enforce allocation `4872248`, node `kh014`, the two recorded H100 94 GB UUIDs, model revision, source inventory and fixed B128/top-8 layer geometry. Do not fake those identities on another system. For an independent hardware replay, make a separate checkout, declare the actual equivalent hardware/allocation contract in the new residual common/freezer configuration, and freeze a new source inventory and plans before timing. Keep the archived source and plans unchanged. This is prospective replication on a new allocation, not byte-for-byte reproduction of its timing values. Changing to another MoE backend, model, batch size, graph mode or hardware interconnect is an extension rather than this replay.

The selected implementation uses a pinned Triton 3.7.1 compiled launcher. It rejects unsupported signatures, pointer alignment, instrumentation/JIT hooks and mutable JIT globals. It supports sequential calls on its constructing stream, with the original GPU algorithm and safety checks. This is an explicit one-layer research fast path, not an integrated full-model vLLM serving configuration.

## Evidence boundaries

The 56-row diagnostic actually executed the archived `diagnostic-freeze-v2` source. V1 was a draft; V3 queued but was cancelled before acquiring GPUs. The nonauthoritative text log was truncated by that queued launch; all 56 hashed raw records and exact executed source remain intact and independently audited. See `results/resident-residual/diagnostic-execution-note.json`.

Precomputed physical-ID replay is diagnostic only. The 24-row development experiment and isolated host-submission timings use earlier pilot inputs and have no final inferential claim. The 192 final treatments use new-to-this-experiment content and disjoint external arrival rows. They measure constructed layer-job latency with replicated input/native output all-reduce, not complete request generation or a token all-to-all serving path. Preserve all negative results and report the registered guard failures as well as aggregate effects.

## Regenerate the figures and report

After all four stage audits and the clean CPU reconstruction have passed, archive that reconstruction's `reproduction.json` under `results/resident-residual/reproduction-v1/`. With Matplotlib installed, regenerate the result artifacts into a fresh local directory:

```bash
python docs/resident-residual/plot_results.py --output /absolute/new/report
python docs/resident-residual/build_report.py --output /absolute/new/report
```

The report builder refuses to describe an incomplete final matrix or claim passing independent audits that are absent. The plot shows source-disjoint content-bundle effects descriptively and only the registered primary response endpoint's confidence interval. It does not attach unregistered intervals to service or to a gate-closed secondary comparison.
