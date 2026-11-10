#!/usr/bin/env bash
set -euo pipefail
cd /work/cs-471/skewbench-hlin/repo-resident-residual
export SLURM_SUBMIT_DIR="$PWD"
source scripts/cluster-env.sh
for split in pilot final; do
  capture="results/resident-residual/capture-${split}-v1"
  calibration_plan="configs/resident-residual-calibration-${split}-v1.json"
  calibration="results/resident-residual/calibration-${split}-v1"
  plan="configs/resident-residual-${split}-v1.json"
  result="results/resident-residual/${split}-v1"
  python scripts/freeze_residual.py --phase calibration --split "$split" --capture-index "$capture/capture-index.json" --arrivals results/resident-residual/content-v1/arrival-manifest.json --output "$calibration_plan"
  srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 bash scripts/run-residual-kuma.sh "$calibration_plan" "$capture/banks" "$calibration"
  python scripts/analyze_residual.py --plan "$calibration_plan" --records "$calibration" --output "$calibration/analysis.json"
  python scripts/freeze_residual.py --phase "$split" --split "$split" --capture-index "$capture/capture-index.json" --arrivals results/resident-residual/content-v1/arrival-manifest.json --calibration "$calibration" --output "$plan"
  srun --jobid=4872248 --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=2 bash scripts/run-residual-kuma.sh "$plan" "$capture/banks" "$result"
  python scripts/analyze_residual.py --plan "$plan" --records "$result" --output "$result/analysis.json"
done
