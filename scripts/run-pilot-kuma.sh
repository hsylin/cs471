#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=03:00:00
#SBATCH --job-name=skb-pilot-v3
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster-env.sh
if [ "$#" -ne 1 ]; then
  echo 'Usage: run-pilot-kuma.sh FRESH_OUTPUT_DIRECTORY' >&2
  exit 2
fi
output_directory="$1"
manifests=(configs/frozen/pilot-v3-k128.json configs/frozen/pilot-v3-k64.json configs/frozen/pilot-v3-k256.json)
bash scripts/run-suite-kuma.sh "$output_directory" "${manifests[@]}"
bash scripts/run-kuma.sh --mode branch --config "${manifests[0]}" \
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output_directory/branches.jsonl"
python -m skewbench.analysis cooldown-select --manifests "${manifests[@]}" \
  --records "$output_directory/records-pilot-v3-k128.jsonl" \
            "$output_directory/records-pilot-v3-k64.jsonl" \
            "$output_directory/records-pilot-v3-k256.jsonl" \
  --output-dir "$output_directory/selection"
python -m skewbench.analysis branch-report --manifest "${manifests[0]}" \
  --records "$output_directory/branches.jsonl" \
  --output-dir "$output_directory/branch-audit"
