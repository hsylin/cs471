#!/usr/bin/env bash
# Submit only steps inside the existing eight-GPU allocation; no new job.
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
source scripts/cluster-env.sh
run_id=${1:?provide a unique run ID}
[[ $run_id =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
: "${FULL_SERVING_SOURCE_REVISION:?Set the deployed full Git SHA}"
[[ $FULL_SERVING_SOURCE_REVISION =~ ^[0-9a-f]{40}$ ]] || exit 2
job=4882284
[[ $(squeue -h -j "$job" -o '%T') == RUNNING ]] || { printf 'Allocation is not running.\n' >&2; exit 1; }
primary=$(scontrol show hostnames "$(squeue -h -j "$job" -o '%N')" | head -1)
state="$SKB_ROOT/full-serving/$run_id"
mkdir "$state"
date --iso-8601=seconds > "$state/launched.txt"
printf '%s\n' "$FULL_SERVING_SOURCE_REVISION" > "$state/source-revision.txt"
set +e
srun --jobid="$job" --mpi=none --exact --nodes=1 --ntasks=1 --nodelist="$primary" \
  --cpus-per-task=4 --gpus=0 --mem=16G --input=none \
  --output="$state/controller.log" --error="$state/controller.err" \
  python -m full_serving.run_suite \
  --snapshot "$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" \
  --private-data "$SKB_ROOT/private/full-serving-suite-v1" \
  --suite-manifest results/full-serving/suite-v1/suite-manifest.json \
  --admission results/full-serving/suite-v1/admission.json \
  --arrivals results/full-serving/arrivals-v1/arrival-manifest.json \
  --burstgpt-block results/full-serving/suite-sources/burstgpt-diagnostic-block.json \
  --output "$state/results" --private-output "$SKB_ROOT/private/$run_id-outputs" \
  --source-revision "$FULL_SERVING_SOURCE_REVISION"
suite_exit=$?
srun --jobid="$job" --mpi=none --exact --nodes=1 --ntasks=1 --nodelist="$primary" \
  --cpus-per-task=4 --gpus=0 --mem=16G --input=none \
  --output="$state/report.log" --error="$state/report.err" \
  python -m full_serving.report_pilots \
  "$state/results/wildchat" "$state/results/sharegpt" "$state/results/instructcoder" \
  "$state/results/numinamath" "$state/results/longbench" "$state/results/sharegpt-burstgpt-diagnostic" \
  --output "$state/results/report.json" --allow-partial
report_exit=$?
printf 'suite_exit=%s report_exit=%s\n' "$suite_exit" "$report_exit" > "$state/completed.txt"
exit "$suite_exit"
