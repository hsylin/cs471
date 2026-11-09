#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=03:00:00
#SBATCH --job-name=skb-payback
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the follow-up repository directory}"
source scripts/cluster-env.sh
if [ "$#" -lt 3 ]; then
  echo 'Usage: run-payback-kuma.sh FRESH_OUTPUT_DIRECTORY MANIFEST BANKS_DIRECTORY [RUNNER_OPTIONS...]' >&2
  exit 2
fi
output_directory="$1"
manifest="$2"
banks_directory="$3"
shift 3
if [ -e "$output_directory" ] || [ -L "$output_directory" ]; then
  echo "Refusing to overwrite output: $output_directory" >&2
  exit 2
fi
mkdir -p "$output_directory/provenance"
cp "$manifest" "$output_directory/provenance/executed-manifest.json"
nvidia-smi -q > "$output_directory/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output_directory/provenance/topology.txt"
python - <<'PY' > "$output_directory/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
scontrol show job "$SLURM_JOB_ID" > "$output_directory/provenance/slurm-job.txt"
if [ -f SOURCE_COMMIT ]; then
  cp SOURCE_COMMIT "$output_directory/provenance/source-commit.txt"
fi
python - "$output_directory/provenance/source-files.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
paths = sorted([*Path('followup').rglob('*.py'), *Path('skewbench').glob('*.py'),
                Path('skewbench/native_sources.json'), Path('scripts/run-payback-kuma.sh')])
data = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
Path(sys.argv[1]).write_text(json.dumps(data, indent=2)+'\n')
PY
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==2; print({"torch":torch.__version__, "cuda":torch.version.cuda, "gpus":torch.cuda.device_count()})'
exec torchrun --standalone --nproc-per-node=2 -m followup.payback \
  --manifest "$manifest" --banks-dir "$banks_directory" \
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
  --output "$output_directory/records.jsonl" "$@"
