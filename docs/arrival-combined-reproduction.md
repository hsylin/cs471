# Reproducing the three-arm composition

The [frozen protocol](arrival-combined-protocol.md) compares native SKIP, the unchanged-map shortcut with native dispatch, and that shortcut combined with the existing resident dispatcher. The measured implementation is commit `7620d2c`. Both components and the earlier studies remain unchanged. The new experiment runs in `arrival_combo/`; it uses the same absolute-clock FCFS wrapper and checks logical equivalence while allowing the combined arm's physical assignments to differ.

The pilot manifest is `configs/arrival-combined-pilot-v1.json`, canonical hash `f9698a244d680dd921709f2396b0b67d36a214b317549a5f75bdb10cea066ca2`. The native calibration manifest is `configs/arrival-combined-calibration-v1.json`, hash `97a9bfb2e26175570be25facc649a46f580907db3292d3d44b97d3c59a22f289`. Both pin execution sources, the protocol and statistical analyzer. Final gamma draws and three-arm orders are frozen before pilot outcomes. Native-only calibration resolves the final nanosecond calendars before any final treatment timing.

GPU reproduction requires the earlier pinned Qwen layer weights, vLLM environment and capture banks described in the [resident reproduction guide](resident-dispatch-reproduction.md). The held allocation is 4872248 on kh014, with the two exact H100 UUIDs recorded in the protocol. Different physical devices require a declared new hardware manifest; they cannot be silently substituted. All comparisons and new calibrations share this reservation, with no concurrent profiler or GPU probe.

Run from the isolated server checkout and use a fresh output directory:

```bash
export SLURM_SUBMIT_DIR=/work/cs-471/skewbench-hlin/repo-arrival-combo
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-arrival-combo-kuma.sh \
  configs/arrival-combined-pilot-v1.json \
  /work/cs-471/skewbench-hlin/results/optimization-capture-v1 \
  /work/cs-471/skewbench-hlin/results/arrival-combined-pilot-reproduction
```

The pilot runs both batch-size composition proofs and all 24 old-calendar trials. Once correctness passes, repeat the same command with `configs/arrival-combined-calibration-v1.json` and another fresh output directory. The launcher runs eight native-only calibration trials, freezes `final-plan.json`, completes all 384 final trials and writes the frozen analysis. No pilot timing threshold selects or tunes the combined treatment. The parent 48-hour reservation remains held afterward as requested.

The final matrix has 128 matched three-arm cases. Each case has identical payloads and arrivals in native, shortcut and combined. Native and shortcut must have identical physical routes; combined must match the frozen online oracle. All arms must have identical logical demand, planner trajectories, source/final placement and weights. The three-arm design cannot identify synergy or an interaction without a contemporaneous online-only fourth arm.

From the repository root, CPU checks and reanalysis require NumPy and the archived captures/assignment artifacts. Run the full test suite with `python -m pytest -q` in the full repository checkout. The focused download is validated by the current-experiment reanalysis commands below. The independent verifiers import no production composition analysis. Byte-preserving gzip compression retains the original JSONL checksum; never reserialize raw rows to normalize them.

```bash
python -m arrival_combo.analysis \
  --plan results/arrival-combined/final-v1/final-plan.json \
  --records results/arrival-combined/final-v1/final/records.jsonl.gz \
  --output /tmp/arrival-combined-summary.json
python scripts/verify_arrival_combo.py \
  --plan configs/arrival-combined-pilot-v1.json \
  --records results/arrival-combined/pilot-v1/pilot/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --proof results/arrival-combined/pilot-v1/pilot/gpu-composition-proof-b128.json \
    results/arrival-combined/pilot-v1/pilot/gpu-composition-proof-b1024.json \
  --output /tmp/arrival-combined-pilot-audit.json
python scripts/verify_arrival_combo.py \
  --plan results/arrival-combined/final-v1/final-plan.json \
  --records results/arrival-combined/final-v1/final/records.jsonl.gz \
  --capture-root results/optimization/capture \
  --calibration-plan results/arrival-combined/final-v1/provenance/initial-plan.json \
  --calibration-records results/arrival-combined/final-v1/calibration/records.jsonl.gz \
  --summary results/arrival-combined/final-v1/final/summary.json \
  --output /tmp/arrival-combined-execution-audit.json
python scripts/verify_arrival_combo_statistics.py \
  --plan results/arrival-combined/final-v1/final-plan.json \
  --records results/arrival-combined/final-v1/final/records.jsonl.gz \
  --summary results/arrival-combined/final-v1/final/summary.json \
  --output /tmp/arrival-combined-statistics-audit.json
python scripts/report_arrival_combo.py \
  --summary results/arrival-combined/final-v1/final/summary.json \
  --records results/arrival-combined/final-v1/final/records.jsonl.gz \
  --output-dir /tmp/arrival-combined-figures
```

The primary claim concerns combined versus shortcut, including three shortcut-relative regression guards. The secondary native comparison is confirmatory only if that incremental gate passes. The complete 384-trial archive, all three contrasts and all eight descriptive workload cells remain available regardless of the outcome. Fresh synthetic calendars over reused layer payloads do not establish full-model serving performance. CPU reconstruction checks computational reproducibility; an independent GPU replay remains a separate empirical test.

The downloadable composition bundle is focused on this CPU reproduction: it includes the repository's source and configurations, this experiment's complete results, the required captured inputs, and the previous calendar manifest used only for the pilot. Older studies' result directories and model weights are not included. Their complete history remains in the Git repository. The bundle checksum and a clean-extraction audit identify the exact exported files and reproduce this experiment without relying on the working checkout.
