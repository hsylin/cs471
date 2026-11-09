#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
output="$1"
if [ -e "$output" ]; then echo 'Fresh smoke output directory required' >&2; exit 2; fi
mkdir -p "$output"
model="$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
torchrun --standalone --nproc_per_node=2 -m stock_comparison.smoke --model-path "$model" --weights-dir "$SKB_ROOT/models/qwen3-layer0" --bank results/stock-capture-pilot-v1/banks/stock-gsm8k-pilot-01.npz --output "$output/smoke.json" 2>&1 | tee "$output/run.log"
