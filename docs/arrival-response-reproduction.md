# Reproducing the response-time fix

This experiment compares native SKIP with an unchanged-map movement/commit fast path. It retains native physical routing, planning, observations and all 160 resident expert slots. The [protocol](arrival-response-fix-protocol.md) defines the finite FCFS queue, fresh synthetic arrival calendars, paired order, primary endpoint and regression guards. The [diagnosis](arrival-response-diagnosis.md) separately reconstructs the earlier native/online response-time reversal.

The measured source is commit `3f3feb8`. The accepted pilot and calibration manifests are `configs/arrival-response-pilot-v4.json` and `configs/arrival-response-calibration-v4.json`. Their canonical SHA-256 hashes are `a4ad6ec65a1daae362df921c98673f0515532bfd6d971828432d6e9f85ed21a7` and `1561aacf4a0b30eb96bd63185d2ef0427ed23b7125d445037a2dbc760070aca1`. The latter archives every gamma draw before GPU evaluation. The final manifest deterministically resolves nanosecond calendars using eight native-only calibrations; treatment measurements cannot change those calendars.

The GPU environment and checkpoint are inherited from the [resident study](resident-dispatch-reproduction.md). The vLLM source remains pinned to `db9527a46873454610df6dbedf79a36d6bf1a7f6`. A job is one preassembled batch of captured tokens through a single layer, not a complete language-model request. Payloads are reused development captures; only arrival calendars are new. Two reversed repetitions remain nested within each of eight calendar blocks, conditional on four fixed payload states and one allocation.

The user authorized an equivalent H100 pair and a 48-hour reservation. Allocation `4872248` holds `kh014` from 2026-11-09 16:45:31 through 2026-11-11 16:45:31 Zurich time. The two UUIDs, memory sizes and NV6 topology are archived under `results/arrival-response/reservation`. Both arms and all phases must use the same devices. Reproduction on a different pair requires an explicit new hardware manifest; changing the device pin does not reproduce this exact physical allocation.

The launcher requires a fresh output directory and `SLURM_SUBMIT_DIR` pointing to the checkout. It must run as an ordinary step inside the held allocation, without a concurrent GPU probe or profiler:

```bash
export SLURM_SUBMIT_DIR=/work/cs-471/skewbench-hlin/repo-arrival-fix
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-arrival-fix-kuma.sh \
  configs/arrival-response-pilot-v4.json \
  /work/cs-471/skewbench-hlin/results/optimization-capture-v1 \
  /work/cs-471/skewbench-hlin/results/arrival-response-pilot-reproduction
```

After pilot correctness and its independent audit pass, run the same command with `configs/arrival-response-calibration-v4.json` and a new output directory. The launcher runs eight native calibrations, writes `final-plan.json`, executes all 256 final trials, and runs the frozen statistical analysis. Pilot timing is descriptive and does not select the intervention or gate final execution. The 48-hour parent reservation is deliberately retained after experiments, as requested.

The pilot exercises a real cross-rank weight swap, native commit, restoration and unchanged skips before timing. Mixed-rank fallback is covered by CPU contract tests. Separate diagnostic timings include CPU function residence and natural blocking; they are excluded from primary performance evidence. The primary native arm installs no wrapper. The treatment's equality check and unanimous rank agreement remain charged.

CPU reproduction uses Python, NumPy and the archived raw assignments and captures. The full suite is `python -m pytest -q`. No Torch import or GPU is needed for the analysis or independent verifiers. JSONL archives must preserve decompressed bytes rather than reserialize rows.

```bash
python -m arrival_fix.analysis \
  --plan results/arrival-response/final-v4/final-plan.json \
  --records results/arrival-response/final-v4/final/records.jsonl.gz \
  --output /tmp/arrival-response-summary.json
python scripts/verify_arrival_fix.py \
  --plan results/arrival-response/pilot-v4/provenance/initial-plan.json \
  --records results/arrival-response/pilot-v4/pilot/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --diagnostics results/arrival-response/pilot-v4/pilot/diagnostics.jsonl.gz \
  --proof results/arrival-response/pilot-v4/pilot/gpu-correctness-proof.json \
  --output /tmp/arrival-response-pilot-audit.json
python scripts/verify_arrival_fix.py \
  --plan results/arrival-response/final-v4/final-plan.json \
  --records results/arrival-response/final-v4/final/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --calibration-plan results/arrival-response/final-v4/provenance/initial-plan.json \
  --calibration-records results/arrival-response/final-v4/calibration/records.jsonl.gz \
  --summary results/arrival-response/final-v4/final/summary.json \
  --output /tmp/arrival-response-final-audit.json
python scripts/report_arrival_fix.py \
  --summary results/arrival-response/final-v4/final/summary.json \
  --records results/arrival-response/final-v4/final/records.jsonl.gz \
  --output-dir /tmp/arrival-response-figures
python scripts/verify_arrival_fix_statistics.py \
  --plan results/arrival-response/final-v4/final-plan.json \
  --records results/arrival-response/final-v4/final/records.jsonl.gz \
  --summary results/arrival-response/final-v4/final/summary.json \
  --output /tmp/arrival-response-statistics-audit.json
```

The independent execution audit imports no production arrival modules. It reconstructs raw routes, windows, calendar arithmetic, rank agreement, all timestamp summaries, paired order and calibration links. A separate statistics verifier reconstructs the confirmatory estimates from timestamps. Agreement of these programs is computational reproducibility of recorded evidence; independent GPU execution remains a stronger external check.

All attempts remain in the archive. The initial exact-device job was cancelled while pending. The first equivalent-pair step failed on launcher command separation before initialization. The next failed before any measurement because Torch reports bare UUIDs while NVML includes a `GPU-` prefix. Version 4 normalizes representation before the same exact-device comparison. Neither failure supplied timing observations or changed endpoints, arrival weights or treatment parameters.
