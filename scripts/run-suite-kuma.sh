#!/usr/bin/env bash
# One allocation keeps the complete suite on the same two GPUs.
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=03:00:00
#SBATCH --job-name=skb-suite
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster-env.sh
output_directory="$1"
shift
if [ "$#" -eq 0 ]; then
  echo 'Usage: run-suite-kuma.sh OUTPUT_DIRECTORY FROZEN_MANIFEST [...]' >&2
  exit 2
fi
mkdir -p "$output_directory"
if [ -e "$output_directory/preflight.jsonl" ]; then
  echo "Refusing to overwrite prior preflight in $output_directory" >&2
  exit 2
fi
bash scripts/run-kuma.sh --mode smoke --method B4 --batch-tokens 1024 \
  --num-steps 128 --window-steps 16 --replicas 32 \
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output_directory/preflight.jsonl"
for manifest in "$@"; do
  manifest_name="$(basename "$manifest" .json)"
  result_file="$output_directory/records-$manifest_name.jsonl"
  if [ -e "$result_file" ]; then
    echo "Refusing to append an unreviewed rerun: $result_file" >&2
    exit 2
  fi
  bash scripts/run-kuma.sh --mode matrix \
    --config "$manifest" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
    --output "$result_file"
done
