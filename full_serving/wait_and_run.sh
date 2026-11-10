#!/usr/bin/env bash
# Lightweight allocation-specific waiter. GPU work runs only in Slurm steps.
# Does not submit a new allocation or use any other account.
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
run_id=${1:?provide a unique run ID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
job=4882284
state="$SKB_ROOT/full-serving/$run_id"
mkdir "$state"
printf '%s\n' "$$" > "$state/waiter.pid"
trap 'printf "interrupted\n" > "$state/waiter-state.txt"; exit 130' INT
trap 'printf "interrupted\n" > "$state/waiter-state.txt"; exit 143' TERM
deadline=$(( $(date +%s) + 86400 ))
while true; do
  now=$(date +%s)
  (( now < deadline )) || { printf 'queue wait exceeded 24h\n' > "$state/waiter-state.txt"; exit 1; }
  scheduler_state=$(squeue -h -j "$job" -o '%T')
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$scheduler_state" > "$state/waiter-state.txt"
  case "$scheduler_state" in
    RUNNING) break ;;
    PENDING|CONFIGURING) sleep 60 ;;
    *) printf 'allocation unavailable: %s\n' "$scheduler_state" >> "$state/waiter-state.txt"; exit 1 ;;
  esac
done
primary=$(scontrol show hostnames "$(squeue -h -j "$job" -o '%N')" | head -1)
# The complete snapshot must exist by startup; preflight fails closed otherwise.
printf 'launching controller on %s at %s\n' "$primary" "$(date --iso-8601=seconds)" > "$state/waiter-state.txt"
srun --jobid="$job" --nodes=1 --ntasks=1 --nodelist="$primary" --exact \
  --cpus-per-task=4 --gpus=0 --mem=16G --input=none \
  --output="$state/controller.log" --error="$state/controller.err" \
  python -m full_serving.orchestrate \
  --snapshot "$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" \
  --pilot "$SKB_ROOT/private/full-serving-data-v1/pilot.jsonl" \
  --smoke "$SKB_ROOT/private/full-serving-data-v1/smoke.jsonl" \
  --data-manifest results/full-serving/data-v1/request-manifest.json \
  --arrival-manifest results/full-serving/arrivals-v1/arrival-manifest.json \
  --output "$state/results" --private-output "$SKB_ROOT/private/$run_id-outputs"
printf 'controller completed at %s\n' "$(date --iso-8601=seconds)" > "$state/waiter-state.txt"
