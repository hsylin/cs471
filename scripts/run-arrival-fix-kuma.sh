#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=kh083
#SBATCH --time=00:20:00
#SBATCH --job-name=skb-arrival-fix
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
plan="$1"; banks="$2"; output="$3"
if [ -e "$output" ]; then echo 'Refusing existing output directory' >&2; exit 2; fi
mkdir -p "$output/provenance"+nvidia-smi -q > "$output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/provenance/slurm-job.txt"
cp SOURCE_COMMIT "$output/provenance/source-commit.txt"
cp "$plan" "$output/provenance/initial-plan.json"
python - <<'PY' > "$output/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
python - "$plan" "$output" <<'PY'
import json,pathlib,sys
from arrival_fix.common import load_plan
from arrival_fix.runner import hardware_check
plan,_=load_plan(sys.argv[1]);out=pathlib.Path(sys.argv[2])
try: uuids=hardware_check(plan)
except Exception as exc:
 (out/'unmeasured-rejection.json').write_text(json.dumps({'measurement_started':False,'reason':str(exc)},indent=2)+'\n')
 raise
(out/'provenance/validated-gpu-pair.json').write_text(json.dumps({'gpu_uuids':uuids,'pass':True},indent=2)+'\n')
PY
phase=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["phase"])' "$plan")
torchrun --standalone --nproc-per-node=2 -m arrival_fix.runner \
  --plan "$plan" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output/$phase/records.jsonl"
if [ "$phase" = calibration ]; then
  python scripts/freeze_arrival_fix.py --phase final --calibration-plan "$plan" \
    --calibration-records "$output/calibration/records.jsonl" --output "$output/final-plan.json"
  torchrun --standalone --nproc-per-node=2 -m arrival_fix.runner \
    --plan "$output/final-plan.json" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
    --output "$output/final/records.jsonl"
  python -m arrival_fix.analysis --plan "$output/final-plan.json" \
    --records "$output/final/records.jsonl" --output "$output/final/summary.json"
fi
