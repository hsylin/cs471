#!/usr/bin/env bash
# Pilot-only wrapper; it never admits final dataset blocks.
set -euo pipefail
arm=$1
snapshot=$2
primary_ip=$3
config=$4
sources=$5
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
unset FULL_SERVING_CUSTOM_MODE VLLM_PLUGINS PYTHONPATH
export CUDA_HOME="${FULL_SERVING_CUDA_HOME:-$SKB_ROOT/tools/cuda-12.9.1}"
export PATH="$CUDA_HOME/bin:$PATH"
node=${SLURMD_NODENAME:-$(hostname -s)}
[[ $node =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
export FLASHINFER_WORKSPACE_BASE="$SKB_ROOT/cache/flashinfer-jit/${SLURM_JOB_ID:?}/$node"
case ${SLURM_PROCID:?} in
  0) role=primary ;;
  1) role=worker ;;
  *) exit 2 ;;
esac
printf '%s.%s\n' "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" > "${FULL_SERVING_RUN_DIR:?}/server-step-$role.txt"
python full_serving/check_toolchain.py --output-dir "$FULL_SERVING_RUN_DIR/flashinfer-prebuild"
exec python -m full_serving.custom_pilot launch-node "$arm" "$role" "$snapshot" "$primary_ip" --config "$config" --sources "$sources"
