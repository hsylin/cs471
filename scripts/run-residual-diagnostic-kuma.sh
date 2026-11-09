#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
model="$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
torchrun --standalone --nproc_per_node=2 -m resident_residual.diagnose \
 --model-path "$model" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
 --banks-dir "$SKB_ROOT/repo-resident-fix/results/resident-fix/capture-pilot-v1/banks" \
 --capture-index "$SKB_ROOT/repo-resident-fix/results/resident-fix/capture-pilot-v1/capture-index.json" "$@"
