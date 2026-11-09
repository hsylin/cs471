#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --job-name=skb-resident-profile
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the resident-dispatch repository}"
source scripts/cluster-env.sh
if [ "$#" -lt 3 ]; then
  echo 'Usage: profile-resident-kuma.sh PILOT_MANIFEST BANKS_DIR FRESH_OUTPUT_DIR [--domain 0|1] [--block-id PILOT_BLOCK]' >&2
  exit 2
fi
profile_manifest="$1"; profile_banks="$2"; profile_output="$3"; shift 3
if [ -e "$profile_output" ] || [ -L "$profile_output" ]; then
  echo "Refusing existing profile directory: $profile_output" >&2
  exit 2
fi
mkdir -p "$profile_output/provenance" "$profile_output/nsys-traces"
nsight_bin='/ssoft/spack/pinot-noir/kuma-h100/v2/spack/opt/spack/linux-rhel9-zen4/nvhpc-24.7/cuda-12.5.1-ulebdd3d2frpxeygqmhuv4kovkoywbq3/bin'
nsys_bin="$nsight_bin/nsys"
ncu_bin="$nsight_bin/ncu"
nvidia-smi -q > "$profile_output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$profile_output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$profile_output/provenance/slurm-job.txt"
cp "$profile_manifest" "$profile_output/provenance/executed-manifest.json"
if [ -f SOURCE_COMMIT ]; then cp SOURCE_COMMIT "$profile_output/provenance/source-commit.txt"; fi
python - <<'PY' > "$profile_output/provenance/environment.json"
import importlib.metadata as m
import json, os, sys
print(json.dumps({'python': sys.executable, 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
    'packages': sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name']),
    'ncu_scope': 'availability/version only; counter permission and replay safety unverified; no NCCL replay'}, indent=2))
PY
# Use absolute tools; do not load modules or replace the existing runtime libraries.
"$ncu_bin" --version > "$profile_output/provenance/ncu-version.txt" 2>&1 || true
profile_backend="${RESIDENT_PROFILE_BACKEND:-auto}"
case "$profile_backend" in auto|nsys|torch) ;; *) echo 'Invalid RESIDENT_PROFILE_BACKEND' >&2; exit 2;; esac
nsys_available=0
if "$nsys_bin" --version > "$profile_output/provenance/nsys-version.txt" 2>&1; then nsys_available=1; fi
profile_args=(--manifest "$profile_manifest" --banks-dir "$profile_banks"
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" "$@")
if [ "$profile_backend" != torch ] && [ "$nsys_available" = 1 ]; then
  # Each torchrun child owns a rank-specific Nsight collector. Repeat honors all
  # eight cudaProfilerStart/Stop regions; RANK substitution avoids collisions.
  nsys_command=(torchrun --standalone --nproc-per-node=2 --no-python "$nsys_bin" profile
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
    --capture-range=cudaProfilerApi --capture-range-end=repeat:8
    --output="$profile_output/nsys-traces/rank%q{RANK}" --force-overwrite=false
    python -m resident_dispatch.profile "${profile_args[@]}"
    --output-dir "$profile_output/nsys-data" --backend nsys)
  printf '%q ' "${nsys_command[@]}" > "$profile_output/provenance/nsys-command.txt"
  printf '\n' >> "$profile_output/provenance/nsys-command.txt"
  set +e
  "${nsys_command[@]}" > "$profile_output/nsys-run.log" 2>&1
  nsys_status=$?
  set -e
  if [ "$nsys_status" = 0 ]; then
    # A successful launcher alone is not evidence that both-rank traces exist.
    set +e
    python - "$profile_output" <<'PY' > "$profile_output/provenance/nsys-artifacts.json"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reports = sorted((root/'nsys-traces').glob('*.nsys-rep'))
counts = {str(rank): sum(p.name.startswith(f'rank{rank}') and p.stat().st_size > 0
                         for p in reports) for rank in range(2)}
indices = []
for rank in range(2):
    path = root/'nsys-data'/f'profile-index-rank{rank}.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    rows = data.get('profiles', [])
    indices.append(len(rows) == 8 and all(r.get('correctness_pass') is True for r in rows))
okay = all(n >= 8 for n in counts.values()) and all(indices)
print(json.dumps({'complete': okay, 'nonempty_reports_by_rank': counts,
                  'rank_profile_indices_complete': indices,
                  'reports': [{'path': str(p), 'bytes': p.stat().st_size} for p in reports]}, indent=2))
sys.exit(0 if okay else 3)
PY
    nsys_status=$?
    set -e
  fi
  printf '%s\n' "$nsys_status" > "$profile_output/provenance/nsys-exit-status.txt"
  if [ "$nsys_status" = 0 ]; then
    echo 'Nsight Systems profile completed; diagnostic records and both-rank traces retained.'
    exit 0
  fi
  if [ "$profile_backend" = nsys ]; then exit "$nsys_status"; fi
  echo "Nsight collection failed ($nsys_status); preserving logs and trying a separate Torch profiler process." >&2
elif [ "$profile_backend" = nsys ]; then
  echo 'Requested Nsight Systems executable is unavailable; see provenance/nsys-version.txt' >&2
  exit 2
fi
torch_command=(torchrun --standalone --nproc-per-node=2 -m resident_dispatch.profile
  "${profile_args[@]}" --output-dir "$profile_output/torch-data" --backend torch)
printf '%q ' "${torch_command[@]}" > "$profile_output/provenance/torch-command.txt"
printf '\n' >> "$profile_output/provenance/torch-command.txt"
exec "${torch_command[@]}"
