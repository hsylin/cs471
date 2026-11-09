#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --job-name=skb-router-capture
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the follow-up repository directory}"
source scripts/cluster-env.sh
if [ "$#" -ne 1 ]; then
  echo 'Usage: run-capture-kuma.sh FRESH_CAPTURE_DIRECTORY' >&2
  exit 2
fi
if [ -e "$1" ] || [ -L "$1" ]; then
  echo "Refusing to overwrite capture directory: $1" >&2
  exit 2
fi
python scripts/capture_router_tapes.py \
  --cache-dir "$SKB_ROOT/cache/huggingface" --output-dir "$1" \
  --device cuda:0 --compose
