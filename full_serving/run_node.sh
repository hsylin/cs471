#!/usr/bin/env bash
# One task per node; launched only by the allocation-specific controller.
set -euo pipefail
arm=$1
snapshot=$2
primary_ip=$3
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
case ${SLURM_PROCID:?} in
  0) role=primary ;;
  1) role=worker ;;
  *) printf 'Expected exactly two server tasks.\n' >&2; exit 2 ;;
esac
printf '%s.%s\n' "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" > "${FULL_SERVING_RUN_DIR:?}/server-step-$role.txt"
exec bash full_serving/launch_native_draft.sh "$arm" "$role" "$snapshot" "$primary_ip" --execute
