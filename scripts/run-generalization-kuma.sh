#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
plan="$1"; banks="$2"; output="$3"
if [ -e "$output" ]; then echo 'Fresh output directory required' >&2; exit 2; fi
mkdir -p "$output/provenance"
cp "$plan" "$output/provenance/plan.json"
cp SOURCE_COMMIT "$output/provenance/source-commit.txt"
nvidia-smi -q > "$output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/provenance/slurm-job.txt"
python - <<'PY' > "$output/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
torchrun --standalone --nproc-per-node=2 -m generalization.runner \
  --plan "$plan" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output/records.jsonl"
