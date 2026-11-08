#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=03:00:00
#SBATCH --job-name=skb-final
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster-env.sh
if [ "$#" -ne 2 ]; then
  echo 'Usage: run-final-kuma.sh OUTPUT_DIRECTORY FROZEN_FINAL_MANIFEST' >&2
  exit 2
fi
output_directory="$1"
manifest="$2"
manifest_name="$(basename "$manifest" .json)"
bash scripts/run-suite-kuma.sh "$output_directory" "$manifest"
bash scripts/run-kuma.sh --mode branch --config "$manifest" \
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output_directory/branches.jsonl"
python -m skewbench.analysis report --manifest "$manifest" \
  --records "$output_directory/records-$manifest_name.jsonl" \
  --output-dir "$output_directory/analysis"
python -m skewbench.analysis branch-report --manifest "$manifest" \
  --records "$output_directory/branches.jsonl" \
  --output-dir "$output_directory/analysis"
python scripts/diagnostic_report.py \
  --records "$output_directory/records-$manifest_name.jsonl" \
  --branches "$output_directory/branches.jsonl" \
  --output-dir "$output_directory/diagnostics"
