#!/usr/bin/env bash
# Full-model launcher. Default: print only; never allocates GPUs.
# Run once per node inside the existing allocation, with all four GPUs visible.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch_native_draft.sh off|on primary|worker MODEL_SNAPSHOT PRIMARY_IP [--execute]

Default is a dry run. MODEL_SNAPSHOT must be the complete pinned local snapshot.
Start primary and worker concurrently in one two-task Slurm step on their
respective nodes. This script does not submit jobs, choose nodes, open tunnels or download data.
Execution requires FULL_SERVING_CHECKPOINT_MANIFEST from preflight checkpoint
and FULL_SERVING_RUN_DIR for per-node runtime records. The initial backend is
allgather_reducescatter, not optimized DeepEP all-to-all. Wait for /health, check
/v1/models and complete a generation before timing. Drain all admitted requests
before SIGTERM to the serving step, then wait for both nodes to exit; do not
cancel the parent reservation or call asynchronous drain a committed EPLB map.
EOF
}

if (( $# < 4 || $# > 5 )); then usage >&2; exit 2; fi
arm=$1
role=$2
model_snapshot=$3
primary_ip=$4
execute=${5:-}
[[ $arm == off || $arm == on ]] || { usage >&2; exit 2; }
[[ $role == primary || $role == worker ]] || { usage >&2; exit 2; }
[[ -z $execute || $execute == --execute ]] || { usage >&2; exit 2; }

command=(vllm serve "$model_snapshot"
  --served-model-name qwen3-full
  --dtype bfloat16
  --max-model-len 32768
  --generation-config vllm
  --tensor-parallel-size 1
  --data-parallel-size 8
  --data-parallel-size-local 4
  --data-parallel-backend mp
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --data-parallel-address "$primary_ip"
  --data-parallel-rpc-port 13345)

if [[ $arm == on ]]; then
  command+=(--enable-eplb --eplb-config '{"num_redundant_experts":32}')
fi
if [[ $role == primary ]]; then
  # One API process still routes to all eight DP ranks in internal LB mode.
  command+=(--host 0.0.0.0 --port 8000 --api-server-count 1)
else
  command+=(--data-parallel-start-rank 4 --headless)
fi

printf 'Full-model command (%s, %s):\n' "$arm" "$role"
printf '%q ' "${command[@]}"
printf '\n'
[[ $execute == --execute ]] || exit 0

[[ ${SLURM_JOB_ID:-} == 4882284 ]] || {
  printf 'Refusing launch outside reservation 4882284.\n' >&2; exit 1;
}
[[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}} == 2 ]] || {
  printf 'Expected a two-node allocation.\n' >&2; exit 1;
}
[[ ${model_snapshot##*/} == ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 ]] || {
  printf 'Expected the pinned checkpoint snapshot directory.\n' >&2; exit 1;
}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
: "${FULL_SERVING_CHECKPOINT_MANIFEST:?Run full checkpoint preflight first}"
: "${FULL_SERVING_RUN_DIR:?Set a unique run directory}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python "$script_dir/preflight.py" runtime --snapshot "$model_snapshot" \
  --checkpoint-manifest "$FULL_SERVING_CHECKPOINT_MANIFEST" \
  --role "$role" --primary-ip "$primary_ip" \
  --output "$FULL_SERVING_RUN_DIR/preflight-$role.json"
# exec preserves the Slurm task PID so normal signals reach vLLM's handlers.
exec "${command[@]}"
