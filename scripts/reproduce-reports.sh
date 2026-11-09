#!/usr/bin/env bash
# Offline reports from the archived JSONL records; no GPU or model weights needed.
set -euo pipefail

if [ "$#" -ne 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo 'Usage: bash scripts/reproduce-reports.sh NEW_OUTPUT_DIRECTORY' >&2
  echo 'Requires Python with NumPy and Matplotlib. Set PYTHON to its executable if needed.' >&2
  if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    exit 0
  fi
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="$1"
case "$output_directory" in
  /*) ;;
  *) output_directory="$PWD/$output_directory" ;;
esac
if [ -e "$output_directory" ] || [ -L "$output_directory" ]; then
  echo "Refusing to overwrite existing output path: $output_directory" >&2
  exit 2
fi

manifest="$repo_root/configs/frozen/final.json"
records_archive="$repo_root/results/final/records-final.jsonl.gz"
branches_archive="$repo_root/results/final/branches.jsonl.gz"
for input in "$manifest" "$records_archive" "$branches_archive"; do
  if [ ! -f "$input" ]; then
    echo "Missing reproduction input: $input" >&2
    exit 2
  fi
done
gzip -t -- "$records_archive" "$branches_archive"
python_executable="${PYTHON:-python3}"
"$python_executable" -c 'import numpy, matplotlib'

mkdir -p -- "$(dirname -- "$output_directory")"
mkdir -- "$output_directory"
export MPLCONFIGDIR="$output_directory/.matplotlib"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$repo_root"
gzip -dc -- "$records_archive" > "$output_directory/records-final.jsonl"
gzip -dc -- "$branches_archive" > "$output_directory/branches.jsonl"

"$python_executable" -m skewbench.analysis report \
  --manifest "$manifest" --records "$output_directory/records-final.jsonl" \
  --output-dir "$output_directory/analysis"
"$python_executable" -m skewbench.analysis branch-report \
  --manifest "$manifest" --records "$output_directory/branches.jsonl" \
  --output-dir "$output_directory/analysis"
"$python_executable" "$repo_root/scripts/diagnostic_report.py" \
  --records "$output_directory/records-final.jsonl" \
  --branches "$output_directory/branches.jsonl" \
  --output-dir "$output_directory/diagnostics"
"$python_executable" "$repo_root/scripts/plot_primary_result.py" \
  --summary "$output_directory/analysis/summary.json" \
  --output-dir "$output_directory/analysis"

echo "Reports written to $output_directory/analysis and $output_directory/diagnostics"
