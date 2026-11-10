# Reproducing the input-construction supplement

The executed plan is `configs/sequence-study-v2.json`, frozen at commit `dd1754e334d40618dc20aebece1b54460fcb47c7`. The v1 plan is an unexecuted draft; v2 adds zero-safe handling of descriptive drain ratios before any GPU timing. The execution closure is `69cced71025c74d2bbf28f1df1ae2d14d6a14ad4fda260d5eb619b50ed35319d` (116 files). Its parent 107-file execution closure remains unchanged.

The first upload included macOS metadata sidecars. Source verification rejected that upload before any GPU launch. Removing only those extra sidecars restored the exact frozen source inventory; no production code changed. The actual run uses the clean Python-created source archive and archives its own verified source inventory.

For CPU-only reanalysis from a clean checkout, run `python docs/sequence-study/reproduce.py --output /absolute/new/directory`. This reconstructs descriptive statistics from raw clocks and validates the complete paired matrix. Independently implemented audit commands are archived alongside the results. This is recorded-evidence reconstruction, not an external laboratory replication or a fresh GPU measurement.

Authentic input banks are the previously captured final banks in `results/resident-residual/capture-final-v1/banks`, verified by the immutable plan. They are intentionally excluded from Git because of size. The parent recapture recipe explains how to reconstruct the pinned public source documents and capture the same checkpoint/layer boundary. A fresh capture may require rerunning provenance/numerical checks; never silently replace a bank under the old hash.

GPU reproduction requires the pinned environment, checkpoint export and declared pair/host/allocation. The current plan deliberately rejects different hardware. Run under the existing allocation with `SLURM_SUBMIT_DIR` set to the repository directory:

```sh
srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 \
  bash sequence_study/run-kuma.sh configs/sequence-study-v2.json \
  results/resident-residual/capture-final-v1/banks /absolute/fresh/run
python -m sequence_study.analysis --plan configs/sequence-study-v2.json \
  --records /absolute/fresh/run --output /absolute/fresh/run/analysis.json
```

Do not rerun over an existing result directory or concurrently on the same GPUs. For an independent machine/allocation, create and document a new plan rather than weakening the identity check. The 54 paired timing trials are separate from initialization, common warmup, correctness passes and the instrumented expert diagnostics. These setup/probe passes are excluded from primary timing. Every admitted timing job and native normal-commit tail completes.

The supplement reuses content and arrival blocks: only its constructed orderings are new. It does not add a fresh-content confirmation, dynamic batcher, decode path, token all-to-all backend or full-model serving measurement. Native-on versus off changes both control activity and physical-slot geometry. Cumulative episode service break-even is not isolated migration payback.
