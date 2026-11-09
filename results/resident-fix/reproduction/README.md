# Independent CPU verification

These tools sit outside the frozen execution-source inventory. They use only Python's standard library and NumPy. They neither launch GPU work nor import the production analysis or inference modules. `stock_raw_primitives.py` is a vendored copy of the earlier independently written stock-record auditor; the new static-resident checks and greedy assignment oracle are implemented separately.

Run from the repository root with a Python environment containing NumPy. Replace the uppercase paths with actual archived paths. `RECORDS` and `CALIBRATION` can be a run directory containing `trials/*.jsonl[.gz]`, or a single JSONL/JSONL.gz file. `INPUTS` is the corresponding run's saved `inputs/` directory. `BANKS` contains the authenticated capture NPZ files named by the plan. The optional `--repo-root` points to the repository holding the exact frozen sources and protocol.

```sh
python results/resident-fix/reproduction/audit_resident_fix_records.py \
  --plan PLAN.json --records RECORDS --inputs-dir INPUTS \
  --banks-dir BANKS --calibration CALIBRATION --output record-audit.json

python results/resident-fix/reproduction/verify_resident_fix_statistics.py \
  --plan PLAN.json --records RECORDS --analysis VALIDATED-SUMMARY.json \
  --output statistics-audit.json
```

For a baseline calibration, omit `--calibration`; its records contain only stockoff. Pilot and calibration statistics remain descriptive, with no confirmatory inference. For treatment records, the raw audit requires the actual baseline calibration records and compares their uncompressed file hashes to the frozen plan. Gzip storage changes are reported separately from original record-byte identity.

The raw audit verifies exact selected source, model, hardware, calibration, arrival and input hashes; it regenerates every one of the 4,112 prepared job compositions per content group. Saved **actual stock gate/top-k outputs**, rather than historical captured router outputs, drive a separate scalar greedy oracle for all current/fix assignments. It compares physical-ID and float32-weight byte hashes, full physical counts and rank loads; reconstructs current-arm observation windows; and checks the static arm has no control, recording or migration work. It also validates the fused stream/storage contract, precision settings and the recorded numerical coverage.

The statistical verifier recomputes FCFS summaries, all descriptive paired ratios and the registered final endpoints directly from clocks. It uses two main eight-content-group strata, 100,000 PCG64 draws with seed 471712, ten one-sided guards, date-deletion point sensitivity, the gated secondary contrast and six descriptive MBPP sentinel checks. Production output is read only after reconstruction, as the comparison target. The main bootstrap excludes the single MBPP group.

This is a separately implemented audit within the same team, not independent external replication. Stored arrays and recorded GPU checks are authenticated and checked, but GPU arithmetic is not rerun. Numerical coverage remains 4,112 whole prepared batches per group against stockoff, 64 sampled FP32 positions per group, 16 whole postflight batches per trial, and resident-weight/map checks. The confidence intervals remain conditional on the selected inputs, arrival material and one hardware allocation.

## CPU QA

The tests use explicitly synthetic evidence, never experimental observations. Production helpers are imported **only by the test module** as comparison targets. To reproduce the tests in the full checkout:

```sh
python -m pytest results/resident-fix/reproduction/test_independent_verifiers.py -q
```

Tests cover the full 100,000-draw table, gates and sentinel separation, unestimable date deletion, complete phase matrices and intentional record/calendar/assignment/clock corruption. Test fixtures do not estimate GPU performance.
