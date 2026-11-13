#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
export CUDA_HOME="$SKB_ROOT/tools/cuda-12.9.1"
export PATH="$CUDA_HOME/bin:$PATH"
node=${SLURMD_NODENAME:-$(hostname -s)}
export FLASHINFER_WORKSPACE_BASE="$SKB_ROOT/cache/flashinfer-jit/${SLURM_JOB_ID:?}/$node"
case ${SLURM_PROCID:?} in 0) role=primary;; 1) role=worker;; *) exit 2;; esac
printf '%s.%s\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$FULL_SERVING_RUN_DIR/server-step-$role.txt"
python full_serving/check_toolchain.py --output-dir "$FULL_SERVING_RUN_DIR/flashinfer-prebuild"
exec python -m full_serving.proposal_experiment node "$1" "$role" "$2" "$3" --protocol "$4"
