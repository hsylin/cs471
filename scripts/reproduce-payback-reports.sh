#!/usr/bin/env bash
# Reanalyse archived payback observations without GPUs or model weights.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'Usage: PYTHON=python3 bash scripts/reproduce-payback-reports.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="$1"
case "$output_directory" in
  /*) ;;
  *) output_directory="$PWD/$output_directory" ;;
esac
if [ -e "$output_directory" ] || [ -L "$output_directory" ]; then
  echo "Refusing to overwrite: $output_directory" >&2
  exit 2
fi
python_executable="${PYTHON:-python3}"
"$python_executable" -c 'import numpy, matplotlib'
cd "$repo_root"
"$python_executable" - <<'PY'
import hashlib
from pathlib import Path
for line in Path('results/payback/SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    path = Path('results/payback') / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Archive checksum mismatch: {path}')
print('Payback archive checksums passed')
PY
mkdir -p -- "$output_directory"
export MPLCONFIGDIR="$output_directory/.matplotlib"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
for candidate in native one-copy; do
  "$python_executable" -m followup.payback_analysis report \
    --manifest "configs/frozen/payback-pilot-${candidate}-v1.json" \
    --records "results/payback/pilot-${candidate}/records.jsonl.gz" \
    --output-dir "$output_directory/pilot-${candidate}" --pilot-stage screening --no-plots
done
"$python_executable" -m followup.payback_analysis report \
  --manifest configs/frozen/payback-final-v1.json \
  --records results/payback/final/records.jsonl.gz \
  --output-dir "$output_directory/final"
"$python_executable" scripts/verify_payback_ratios.py \
  --manifest configs/frozen/payback-final-v1.json \
  --records results/payback/final/records.jsonl.gz \
  --report "$output_directory/final/summary.json" \
  --output "$output_directory/independent-ratio-audit.json"
"$python_executable" scripts/audit_payback_records.py \
  --input-dir results/payback/final --inputs-dir results/payback/inputs \
  --analysis-summary "$output_directory/final/summary.json" \
  --output-prefix "$output_directory/independent-record-audit"
"$python_executable" scripts/summarize_payback_mechanisms.py \
  --records results/payback/final/records.jsonl.gz \
  --output "$output_directory/mechanisms.json"
echo "Payback reports written to $output_directory"
