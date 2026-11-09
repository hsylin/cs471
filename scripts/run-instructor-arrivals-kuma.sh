#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --job-name=skb-instructor-arrivals
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
plan="$1"; banks="$2"; output="$3"
if [ -e "$output" ]; then echo 'Refusing existing output directory' >&2; exit 2; fi
mkdir -p "$output/provenance"
nvidia-smi -q > "$output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/provenance/slurm-job.txt"
cp SOURCE_COMMIT "$output/provenance/source-commit.txt"
cp "$plan" "$output/provenance/executed-plan.json"
python - <<'PY' > "$output/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
exec torchrun --standalone --nproc-per-node=2 -m arrival_study.runner \
  --plan "$plan" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output/records.jsonl"
