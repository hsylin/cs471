#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
sources="$1"; output="$2"; split="$3"
if [ -e "$output" ]; then echo 'Fresh output directory required' >&2; exit 2; fi
python - <<'PY'
import os,socket,subprocess
from arrival_fix.common import GPU_UUIDS,HARDWARE_HOST
actual=sorted(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split())
assert socket.gethostname().split('.')[0]==HARDWARE_HOST
assert actual==sorted(GPU_UUIDS)
assert os.environ['SLURM_JOB_ID']=='4872248'
print({'capture_only':True,'host':HARDWARE_HOST,'gpu_uuids':actual},flush=True)
PY
python scripts/capture_generalization_inputs.py --sources "$sources" \
  --output-dir "$output" --cache-dir "$SKB_ROOT/cache/huggingface" \
  --device cuda:0 --split "$split" --local-files-only
cp SOURCE_COMMIT "$output/source-commit.txt"
nvidia-smi -q > "$output/nvidia-smi.txt"
