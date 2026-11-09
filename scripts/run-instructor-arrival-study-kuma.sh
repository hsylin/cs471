#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:25:00
#SBATCH --job-name=skb-instructor-study
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
calibration_plan="$1"; banks="$2"; output="$3"
if [ -e "$output" ]; then echo 'Refusing existing output directory' >&2; exit 2; fi
mkdir -p "$output/provenance"
nvidia-smi -q > "$output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/provenance/slurm-job.txt"
cp SOURCE_COMMIT "$output/provenance/source-commit.txt"
cp "$calibration_plan" "$output/provenance/calibration-plan.json"
python - <<'PY' > "$output/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
torchrun --standalone --nproc-per-node=2 -m arrival_study.runner \
  --plan "$calibration_plan" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output/calibration/records.jsonl"
python scripts/freeze_instructor_arrivals.py --phase evaluation \
  --calibration-plan "$calibration_plan" --calibration-records "$output/calibration/records.jsonl" \
  --output "$output/evaluation-plan.json"
torchrun --standalone --nproc-per-node=2 -m arrival_study.runner \
  --plan "$output/evaluation-plan.json" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output/evaluation/records.jsonl"
python -m arrival_study.analysis --plan "$calibration_plan" \
  --records "$output/calibration/records.jsonl" --output "$output/calibration/summary.json"
python -m arrival_study.analysis --plan "$output/evaluation-plan.json" \
  --records "$output/evaluation/records.jsonl" --output "$output/evaluation/summary.json"
