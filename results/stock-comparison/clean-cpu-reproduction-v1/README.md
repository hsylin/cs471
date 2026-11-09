# Completed clean-source CPU reconstruction

The reconstruction passed using source extracted from commit `dfd419f7b4e62fdeec408581c572bf4d3a7a3065`. All 62 measured-source files and the frozen protocol matched their declared hashes before and after execution. The complete 18 native-only calibrations and 216 final trials were reconstructed; no trials were excluded.

The regenerated calibration analysis matched 961 scalar fields, and the final analysis matched 53,507 scalar fields, with **zero floating-point difference** from the archived analyses. Only top-level record-file path/compression metadata was normalized, after exact comparison of decompressed record-byte hashes. All other identities and integer values were compared exactly.

Both independent execution audits passed. The separate statistical reimplementation compared 5,710 scalar fields, with maximum numerical difference `1.7763568394002505e-15`. Its exact bootstrap draw digest was `0b20e84dbe760e5c0cca1895574655c48e13bafe9d5625e026e9156fc4442ab8`. These checks reproduce the archived arithmetic and decision logic; they do not add a new GPU measurement.

`verification.json` records commands, source/plan/protocol identities, comparison counts and limitations. `staged-artifact-manifest.json` identifies all 140 input/provenance files, which were hardlinked into the temporary work directory and hashed before and after the run. The raw files and NPZ inputs are not duplicated in this compact archive. `reports/` and `logs/` retain actual regenerated outputs. `reproduce_stock_clean_cpu.py` is the exact harness, and its five synthetic harness tests are retained separately. `archive-files.json` inventories the archived evidence files.

To repeat with another fresh output directory, first extract the source-only paths from the pinned commit. Keep the complete original artifacts in the main checkout:

```bash
set -euo pipefail
mkdir work/stock-clean-source-new
git archive dfd419f7b4e62fdeec408581c572bf4d3a7a3065 \
  followup resident_dispatch generalization optimization tests arrival_fix \
  docs skewbench scripts stock_comparison arrival_study \
  | tar -x -C work/stock-clean-source-new

python results/stock-comparison/clean-cpu-reproduction-v1/reproduce_stock_clean_cpu.py \
  --source-root work/stock-clean-source-new \
  --repo-root . \
  --artifact-root results/stock-comparison \
  --python /absolute/path/to/cpu-venv/bin/python \
  --output work/stock-clean-output-new \
  --final-complete-confirmed
```

The selected Python environment needs NumPy. The exact Git commit must be available locally because the harness extracts the independent statistical auditor from it. The explicit completion flag is appropriate only for the complete frozen matrix; it does not waive the harness's count/hash checks.

This is same-team CPU verification with an independently extracted source tree and independent audit implementations. The data were shared with the original archive rather than independently acquired. It is not an external-organization replication, and it does not extend inference beyond the frozen layer workload, selected arrivals and one hardware allocation.
