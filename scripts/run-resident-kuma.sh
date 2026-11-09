#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --job-name=skb-resident
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
mode="$1"; output_directory="$2"; shift 2
if [ -e "$output_directory" ]; then
  echo "Refusing existing output: $output_directory" >&2; exit 2
fi
mkdir -p "$output_directory/provenance"
nvidia-smi -q > "$output_directory/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output_directory/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output_directory/provenance/slurm-job.txt"
cp SOURCE_COMMIT "$output_directory/provenance/source-commit.txt"
python - <<'PY' > "$output_directory/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
if [ "$mode" = capture ]; then
  exec python scripts/capture_resident_tapes.py --output-dir "$output_directory" \
    --cache-dir "$SKB_ROOT/cache/huggingface" --local-files-only --compose "$@"
fi
manifest="$1"; banks_directory="$2"; shift 2
cp "$manifest" "$output_directory/provenance/executed-manifest.json"
module_name=resident_dispatch.pilot
if [ "$mode" = final ] || [ "$mode" = online-validation ]; then module_name=resident_dispatch.runner; fi
exec torchrun --standalone --nproc-per-node=2 -m "$module_name" \
  --manifest "$manifest" --banks-dir "$banks_directory" \
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" --output "$output_directory/records.jsonl" "$@"
