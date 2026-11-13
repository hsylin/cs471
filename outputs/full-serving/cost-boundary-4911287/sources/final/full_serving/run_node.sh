#!/usr/bin/env bash
# One task per node; launched only by the allocation-specific controller.
set -euo pipefail
arm=$1
snapshot=$2
primary_ip=$3
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
export CUDA_HOME="${FULL_SERVING_CUDA_HOME:-$SKB_ROOT/tools/cuda-12.9.1}"
export PATH="$CUDA_HOME/bin:$PATH"
# FlashInfer honors this variable at import time. Each node has a distinct
# Ninja workspace; HOME and the original shared cache are left untouched.
node=${SLURMD_NODENAME:-$(hostname -s)}
[[ $node =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'Invalid node name.\n' >&2; exit 2; }
export FLASHINFER_WORKSPACE_BASE="$SKB_ROOT/cache/flashinfer-jit/${SLURM_JOB_ID:?}/$node"
case ${SLURM_PROCID:?} in
  0) role=primary ;;
  1) role=worker ;;
  *) printf 'Expected exactly two server tasks.\n' >&2; exit 2 ;;
esac
printf '%s.%s\n' "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" > "${FULL_SERVING_RUN_DIR:?}/server-step-$role.txt"
# One process per node builds and exercises the native sampler before four
# vLLM workers can import it concurrently. This cost belongs to startup.
python full_serving/check_toolchain.py --output-dir "$FULL_SERVING_RUN_DIR/flashinfer-prebuild"
exec bash full_serving/launch_native_draft.sh "$arm" "$role" "$snapshot" "$primary_ip" --execute
