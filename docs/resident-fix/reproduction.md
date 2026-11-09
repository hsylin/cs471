# Reproducing the resident-path repair

The CPU procedure reconstructs archived observations. It does not rerun GPU arithmetic, capture new inputs, or constitute external-laboratory replication. Run it from a complete checkout containing all four resident-fix run directories, their capture banks and frozen plans. An incomplete live directory is not a valid substitute.

The completed clean replay is archived in `results/resident-fix/clean-cpu-reproduction-v1/`. It extracted a fresh Git archive of `1ddf5af3d16d2f9dc0eabc891816d771e056b1b4`, checked all 83 measured source files, and passed all four phases: 5 pilot calibrations, 60 pilot treatments, 17 final calibrations and 204 final treatments. Every archived analysis matched exactly after path/compression normalization. Both independent audits passed; the final statistical reimplementation compared 47,936 scalar fields with maximum numerical difference `5.55e-17`. The archive's commit, path selection and SHA256 are recorded in `archive-provenance.json`. This demonstrates artifact reconstruction, not another physical experiment.

## Exact implementation and scope

The measurement source was committed as `331cc69f2820d58b4bde2b509f6b3d3c193da894` before fresh timing. Later artifact/documentation commits retain the same 83-file execution inventory, `08d4b09f110b0ca225eb95073723b4c93a3ab799a0cf52b9c04ed0322ceba1ff`. The preceding 62-file closure remains `9c6d6a7e1e94f4be769d435fbf7652e0cbb86501637c046ae078a0ccef1015d4`. The protocol and selected implementation are separately hashed inside every plan.

Workload preparation/capture also has separate source manifests, including `generalization/external_arrivals.py` and `scripts/capture_optimization_tapes.py`. These preparation-only dependencies are authenticated by the selection/capture provenance; they are not invoked by measured forward execution or CPU timing reconstruction.

The GPU environment is archived per run: vLLM `0.31.0+cu129`, commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`; Torch `2.13.0+cu129`; Triton `3.7.1`; NumPy `2.3.5`. The model is Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, layer 0, BF16/B128/top-8. Hardware checks require allocation 4872248 on kh014 and the two recorded H100 UUIDs. This is an eager replicated-input/native-output-all-reduce layer harness, not a full serving engine or token-all-to-all path.

## CPU replay

The recorded local analysis environment uses Python 3.13.7 and NumPy 2.5.4. CUDA, model weights and the private full-text selection file are unnecessary for this procedure. The archived banks and stock-route arrays are necessary.

```bash
python3.13 -m venv .venv-resident-cpu
.venv-resident-cpu/bin/python -m pip install 'numpy==2.5.4'
.venv-resident-cpu/bin/python results/resident-fix/reproduction/reproduce_cpu.py \
  --output-dir work/resident-cpu-reproduction-new
```

The output directory must not exist. The helper checks its tool manifest, all executed source files, the protocol, selected implementation and executed plans. It then recomputes production analyses, independently checks raw clocks/assignments/state, and independently reconstructs statistics. It compares all archived analysis values exactly, normalizing only relocatable file paths and gzip storage metadata while preserving decompressed record-byte hashes. Its final `reproduction-report.json` reports each phase separately; do not interpret a partial report as success.

The four phases contain five pilot calibrations, 60 pilot treatments, 17 final calibrations and 204 final treatments. Use `--phases calibration-pilot`, `--phases pilot`, `--phases calibration-final`, or `--phases final` for a bounded replay. Calibration/pilot phases must have no confirmatory result. The final bootstrap includes 16 GSM8K/WildChat content bundles; the one MBPP bundle remains descriptive.

Individual commands, audit scope and mutation tests are documented in the [independent-tool README](../../results/resident-fix/reproduction/README.md). The service/work-proxy report is descriptive and separate from acceptance inference:

```bash
.venv-resident-cpu/bin/python results/resident-fix/reproduction/summarize_service.py \
  --plan results/resident-fix/plans/final-v1.json \
  --records results/resident-fix/final-v1/trials \
  --inputs-root results/resident-fix/final-v1/inputs \
  --output work/resident-final-mechanism-new.json \
  --markdown work/resident-final-mechanism-new.md
```

Numerical checks recorded by the GPU run cover 4,112 whole prepared batches per group against stockoff, 64 sampled FP32 token positions per group, 16 whole postflight batches per trial and all resident weights/maps. CPU reconstruction authenticates these records and independently regenerates routing assignments; it does not expand that numerical coverage or establish downstream answer quality.

## GPU replay and fresh replication

Use the pinned environment, checkpoint and exported layer weights described in the existing cluster setup. `scripts/run-resident-fix-kuma.sh` invokes the runner with a frozen plan, capture-bank directory and new output directory. It refuses to overwrite an existing run. The exact original-pair replay command is:

```bash
git rev-parse HEAD > SOURCE_COMMIT
export SLURM_SUBMIT_DIR="$PWD"
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus-per-node=2 \
  bash scripts/run-resident-fix-kuma.sh \
  results/resident-fix/plans/final-v1.json \
  results/resident-fix/capture-final-v1/banks \
  work/resident-final-gpu-replay-new
```

This command is valid only while that allocation is active and the recorded GPUs are assigned. A different allocation, GPU pair, software build, workload or source change requires an explicitly labeled new protocol/plan and new calibration, rather than changing archived expected hashes to pass a check. Replay of the same banks/calendars is a reproducibility check, not fresh generalization evidence. The source-selection scripts and public manifests describe recapture, including dataset access/license requirements and the explicit prior-data exclusion inventory.

The custom fused path requires the exact frozen 160-slot map and sequential use of its output buffer on one CUDA inference stream. It is not a general multistream dispatcher or a supported upstream EPLB-off configuration. Genuine stockoff retains 128 slots; a common maximum replica allowance does not imply equal resident memory.

## Evidence inventory

- `diagnostic-v1` and `fused-development-v1`: 60 and 36 reused-input development runs, executed source snapshots and separate audits.
- `content-v1`, `capture-pilot-v1`, `capture-final-v1`: source/arrival selection, authenticated hidden banks and capture provenance. The capture ledger preserves the archive-only failure after successful pilot capture; no bank was silently recaptured.
- `plans`: four sealed calibration/treatment plans, including frozen absolute arrivals and treatment orders.
- `calibration-pilot-v1`, `pilot-v1`, `calibration-final-v1`, `final-v1`: original per-group JSONL records, job compositions, actual stock routes, physical-assignment diagnostics where applicable, logs, environment and hardware provenance.
- `reproduction`: separately implemented raw/statistical auditors, CPU replay helper, source audit and source-derived launch-geometry evidence.

Keep calibration file basenames and original uncompressed bytes intact: treatment plans bind capacities to those exact files and capture identities. Preserve earlier negative results and every prespecified final case. Lower imbalance, fewer useful tiles, shorter service, finite-episode throughput and full-model speedup are different claims.
