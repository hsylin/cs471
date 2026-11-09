#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --job-name=skb-payback-profile
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the follow-up repository directory}"
source scripts/cluster-env.sh
if [ "$#" -ne 2 ]; then
  echo 'Usage: profile-payback-kuma.sh PILOT_BANK FRESH_OUTPUT_DIRECTORY' >&2
  exit 2
fi
if [ -e "$2" ] || [ -L "$2" ]; then
  echo "Refusing to overwrite profile directory: $2" >&2
  exit 2
fi
mkdir -p "$2"
nvidia-smi -q > "$2/nvidia-smi.txt"
nvidia-smi topo -m > "$2/topology.txt"
python -m pip freeze > "$2/environment-lock.txt"
exec torchrun --standalone --nproc-per-node=2 -m followup.profile_layer \
  --bank "$1" --weights-dir "$SKB_ROOT/models/qwen3-layer0" --output-dir "$2"
