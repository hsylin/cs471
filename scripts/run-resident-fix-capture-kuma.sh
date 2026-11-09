#!/usr/bin/env bash
set -euo pipefail
source scripts/cluster-env.sh
sources="${1:?private source manifest required}"
shift
splits=("$@")
if [ "${#splits[@]}" -eq 0 ]; then splits=(pilot final); fi
for split in "${splits[@]}"; do
  case "$split" in pilot|final) ;; *) exit 2 ;; esac
  if [ -e "results/resident-fix/capture-${split}-v1" ]; then
    echo "Fresh capture directory required: ${split}" >&2
    exit 2
  fi
done
python - <<'PY'
import json,os,socket,subprocess
expected=['GPU-204281f9-9c87-3c25-adfb-9edd6c972ec3','GPU-7d5c9009-305b-9341-957f-1deb96611ed5']
actual=sorted(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split())
assert socket.gethostname().split('.')[0]=='kh014'
assert actual==sorted(expected)
assert os.environ['SLURM_JOB_ID']=='4872248'
print(json.dumps(dict(capture_only=True,host='kh014',gpu_uuids=actual,allocation='4872248',step=os.environ.get('SLURM_STEP_ID'))),flush=True)
PY
for split in "${splits[@]}"; do
  output="results/resident-fix/capture-${split}-v1"
  python scripts/capture_resident_fix_inputs.py --sources "$sources" \
    --output-dir "$output" --cache-dir "$SKB_ROOT/cache/huggingface" \
    --device cuda:0 --split "$split" --local-files-only 2>&1 | tee "${output}.log"
  mkdir -p "$output/provenance"
  if [ -f SOURCE_COMMIT ]; then cp SOURCE_COMMIT "$output/source-commit.txt"; fi
  cp results/resident-fix/content-v1/source-manifest.json "$output/provenance/public-source-manifest.json"
  cp scripts/capture_resident_fix_inputs.py scripts/run-resident-fix-capture-kuma.sh scripts/cluster-env.sh "$output/provenance/"
  nvidia-smi -q > "$output/nvidia-smi.txt"
  python - "$sources" "$output" <<'PY'
import hashlib,json,os,socket,sys
from pathlib import Path
source=Path(sys.argv[1]);output=Path(sys.argv[2]);manifest=json.loads(source.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=list(manifest['preparation_source_files'])+['scripts/cluster-env.sh','scripts/run-resident-fix-capture-kuma.sh','scripts/capture_optimization_tapes.py']
base=Path('SOURCE_COMMIT')
provenance=dict(capture_only=True,allocation_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),hostname=socket.gethostname(),base_source_commit=base.read_text().strip() if base.exists() else None,source_commit_scope='Base label when supplied; exact file hashes identify code. A source-only snapshot may have no commit label.',private_manifest_file_sha256=sha(source),source_selection_sha256=manifest['content_sha256'],file_sha256={p:sha(p) for p in files})
(output/'provenance/capture-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
PY
done
