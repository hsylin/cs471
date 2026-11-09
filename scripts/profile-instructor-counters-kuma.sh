#!/usr/bin/env bash
#SBATCH --account=cs471
#SBATCH --partition=h100
#SBATCH --qos=cs471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --job-name=skb-instructor-ncu
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/cluster-env.sh
if [ "$#" != 3 ]; then
  echo 'Usage: profile-instructor-counters-kuma.sh VALIDATION_MANIFEST OLD_BANKS_DIR FRESH_OUTPUT_DIR' >&2
  exit 2
fi
counter_manifest="$1"; counter_banks="$2"; counter_output="$3"
if [ -e "$counter_output" ] || [ -L "$counter_output" ]; then
  echo "Refusing existing output: $counter_output" >&2; exit 2
fi
mkdir -p "$counter_output/provenance" "$counter_output/data"
ncu_bin='/ssoft/spack/pinot-noir/kuma-h100/v2/spack/opt/spack/linux-rhel9-zen4/nvhpc-24.7/cuda-12.5.1-ulebdd3d2frpxeygqmhuv4kovkoywbq3/bin/ncu'
"$ncu_bin" --version > "$counter_output/provenance/ncu-version.txt"
nvidia-smi -q > "$counter_output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$counter_output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$counter_output/provenance/slurm-job.txt"
cp "$counter_manifest" "$counter_output/provenance/executed-manifest.json"
if [ -f SOURCE_COMMIT ]; then cp SOURCE_COMMIT "$counter_output/provenance/source-commit.txt"; fi
if [ -r /proc/driver/nvidia/params ]; then
  grep 'RmProfilingAdminOnly' /proc/driver/nvidia/params > "$counter_output/provenance/counter-permission-setting.txt" || true
fi
python - <<'PY' > "$counter_output/provenance/environment-lock.txt"
import importlib.metadata as m
print('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name'])))
PY
# This probe imports Torch only and profiles one local fill. No model/NCCL work.
set +e
"$ncu_bin" --replay-mode kernel --cache-control none --clock-control none \
  --profile-from-start off --metrics gpu__time_duration.sum,dram__bytes_write.sum \
  --export "$counter_output/permission-probe.ncu-rep" \
  python -m arrival_study.counter_profile --probe --probe-output "$counter_output/permission-probe.json" \
  > "$counter_output/permission-probe.log" 2>&1
probe_status=$?
set -e
if [ "$probe_status" != 0 ] || grep -q 'ERR_NVGPUCTRPERM' "$counter_output/permission-probe.log"; then
  python - "$counter_output" "$probe_status" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);log=(root/'permission-probe.log').read_text()
reason='ERR_NVGPUCTRPERM' if 'ERR_NVGPUCTRPERM' in log else 'permission_or_compatibility_probe_failed'
(root/'status.json').write_text(json.dumps({'status':'counters_unavailable','reason':reason,
    'probe_exit_code':int(sys.argv[2]),'counter_capture_completed':False,
    'system_configuration_changed':False,'evidence_eligible':False},indent=2)+'\n')
print('Counter collection stopped cleanly:',reason)
PY
  exit 0
fi
"$ncu_bin" --import "$counter_output/permission-probe.ncu-rep" --csv --page raw \
  > "$counter_output/permission-probe.csv"
python - "$counter_output/permission-probe.csv" <<'PY'
import csv,math,pathlib,sys
rows=list(csv.reader(pathlib.Path(sys.argv[1]).read_text().splitlines()))
header=next(i for i,r in enumerate(rows) if 'Metric Name' in r and 'Metric Value' in r)
data=[dict(zip(rows[header],r)) for r in rows[header+1:] if len(r)==len(rows[header])]
metrics={r['Metric Name']:r['Metric Value'] for r in data}
for name in ['gpu__time_duration.sum','dram__bytes_write.sum']:
 assert name in metrics and math.isfinite(float(metrics[name].replace(',',''))),(name,metrics)
assert float(metrics['gpu__time_duration.sum'].replace(',',''))>0
PY
metrics='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active'
counter_command=(torchrun --standalone --nproc-per-node=2 --no-python "$ncu_bin"
  --replay-mode kernel --cache-control none --clock-control none --profile-from-start off
  --kernel-name-base function --kernel-name 'regex:^fused_moe_kernel$' --launch-count 8
  --metrics "$metrics" --export "$counter_output/counters-rank%q{RANK}.ncu-rep"
  python -m arrival_study.counter_profile --manifest "$counter_manifest" --banks-dir "$counter_banks"
  --weights-dir "$SKB_ROOT/models/qwen3-layer0" --output-dir "$counter_output/data")
printf '%q ' "${counter_command[@]}" > "$counter_output/provenance/command.txt"
printf '\n' >> "$counter_output/provenance/command.txt"
set +e
"${counter_command[@]}" > "$counter_output/collection.log" 2>&1
collection_status=$?
set -e
if [ "$collection_status" != 0 ]; then
  python - "$counter_output" "$collection_status" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
(root/'status.json').write_text(json.dumps({'status':'collection_failed','exit_code':int(sys.argv[2]),
    'counter_capture_completed':False,'evidence_eligible':False},indent=2)+'\n')
PY
  exit "$collection_status"
fi
for rank in 0 1; do
  "$ncu_bin" --import "$counter_output/counters-rank${rank}.ncu-rep" --csv --page raw \
    > "$counter_output/counters-rank${rank}.csv"
done
python - "$counter_output" <<'PY'
import csv,hashlib,json,math,pathlib,sys
from arrival_study.counter_profile import METRICS
root=pathlib.Path(sys.argv[1]);inventory={};counts={}
for rank in range(2):
 meta=json.loads((root/'data'/f'counter-metadata-rank{rank}.json').read_text())
 assert meta['completed'] is True and len(meta['profiles'])==4 and all(r['correctness_pass'] for r in meta['profiles'])
 parsed=list(csv.reader((root/f'counters-rank{rank}.csv').read_text().splitlines()))
 start=next(i for i,r in enumerate(parsed) if 'Metric Name' in r and 'Kernel Name' in r and 'ID' in r)
 rows=[dict(zip(parsed[start],r)) for r in parsed[start+1:] if len(r)==len(parsed[start])]
 launches={}
 for r in rows:
  if not r['ID'].isdigit():continue
  assert 'fused_moe_kernel' in r['Kernel Name'] and 'nccl' not in r['Kernel Name'].lower()
  launches.setdefault(r['ID'],{})[r['Metric Name']]=r['Metric Value']
 assert len(launches)==8,len(launches)
 for launch in launches.values():
  for metric in METRICS:
   assert metric in launch and math.isfinite(float(launch[metric].replace(',',''))),(metric,launch)
 counts[str(rank)]=len(launches)
for p in root.rglob('*'):
 if p.is_file():inventory[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'status.json').write_text(json.dumps({'status':'complete','counter_capture_completed':True,
 'selected_gemm_launches_by_rank':counts,'counter_latency_is_primary_performance':False,
 'evidence_eligible':False,'artifact_sha256':inventory},indent=2)+'\n')
PY
