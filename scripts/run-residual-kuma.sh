#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster-env.sh
plan="${1:?plan}"; banks="${2:?banks}"; output="${3:?output}"
if [ -e "$output" ]; then echo 'Fresh output directory required' >&2; exit 2; fi
mkdir -p "$output/provenance"
cp "$plan" "$output/provenance/plan.json"
cp SOURCE_COMMIT "$output/provenance/source-commit.txt"
nvidia-smi -q > "$output/provenance/nvidia-smi.txt"
nvidia-smi topo -m > "$output/provenance/topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/provenance/slurm-job.txt"
python - "$plan" "$output" <<'PY'
import importlib.metadata as m,json,sys,tarfile
from pathlib import Path
from resident_residual.common import load_plan,source_identity
plan=load_plan(sys.argv[1]);out=Path(sys.argv[2])/'provenance'
source,files=source_identity()
(out/'environment-lock.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions() if d.metadata['Name']))+'\n')
with tarfile.open(out/'executed-source.tar.gz','w:gz') as archive:
    for rel in files:archive.add(rel,arcname=rel)
    for rel in ('docs/resident-residual/protocol-v1.md','configs/resident-residual-selected-v1.json'):
        archive.add(rel,arcname=rel)
(out/'source-inventory.json').write_text(json.dumps(dict(source_sha256=source,files=files),indent=2)+'\n')
PY
model="$SKB_ROOT/cache/huggingface/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
torchrun --standalone --nproc_per_node=2 -m resident_residual.runner \
 --plan "$plan" --banks-dir "$banks" --weights-dir "$SKB_ROOT/models/qwen3-layer0" \
 --model-path "$model" --output "$output" 2>&1 | tee "$output/run.log"
