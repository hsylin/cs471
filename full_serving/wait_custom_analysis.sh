#!/usr/bin/env bash
# Postprocess an existing frozen run. Never restart servers or change its protocol.
set -euo pipefail
umask 077
mode=${1:?mode required}
run_dir=${2:?absolute run directory required}
source_revision=${3:?analysis source revision required}
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
analysis="$run_dir/analysis"

if [[ "$mode" == analyze ]]; then
  cd "$code"
  source scripts/cluster-env.sh
  python -m full_serving.custom_report "$run_dir/results" --output "$analysis/resident-comparison.json"
  python -m full_serving.audit_final_statistics "$run_dir/results" \
    --report "$analysis/resident-comparison.json" --output "$analysis/independent-statistics-audit.json"
  python -m full_serving.render_custom_report "$analysis/resident-comparison.json" \
    --output-prefix "$analysis/resident-comparison"
  python - "$analysis" "$source_revision" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
from full_serving.render_custom_report import comparison_notes
p=Path(sys.argv[1]); report=json.loads((p/'resident-comparison.json').read_bytes())
notes={'scope':'Exploratory per-dataset full-model comparisons; no automatic superiority claim.',
       'datasets':[{'dataset':d['dataset'],'contrasts':{c:comparison_notes(d,c) for c in ('custom/off','custom/on')}}
                   for d in report['datasets']]}
(p/'resident-comparison-interpretation.json').write_text(json.dumps(notes,indent=2)+'\n')
receipt={'status':'remote_analysis_passed','utc':datetime.now(timezone.utc).isoformat(),
         'analysis_source_revision':sys.argv[2], 'local_transfer_closure_pending':True,
         'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()
                  if f.suffix in ('.json','.html','.csv') and f.is_file()}}
(p/'remote-analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
  exit 0
fi

[[ "$mode" == wait ]]
mkdir "$analysis"
trap 'result=$?; printf "analysis_exit=%s\n" "$result" > "$analysis/completed.txt"' EXIT
printf 'Waiting for the existing controller; no experiment mutation.\n'
deadline=$((SECONDS + 86400))
while [[ ! -f "$run_dir/completed.txt" ]]; do
  (( SECONDS < deadline )) || { echo 'Controller wait deadline reached.' >&2; exit 1; }
  sleep 45
done
[[ $(cat "$run_dir/completed.txt") == controller_exit=0 ]]
srun --jobid=4882284 --mpi=none --exact --nodes=1 --ntasks=1 --nodelist=kh074 \
  --cpus-per-task=4 --gpus=0 --mem=16G --input=none \
  --output="$analysis/analysis.log" --error="$analysis/analysis.err" \
  bash "$code/full_serving/wait_custom_analysis.sh" analyze "$run_dir" "$source_revision"
