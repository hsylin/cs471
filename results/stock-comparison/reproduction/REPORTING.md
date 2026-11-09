# Final report generation

These two CPU-only scripts sit outside the frozen execution-source inventory. They do not run GPUs, change algorithms or thresholds, or compute new inferential tests. Run them only after all216 final trials and the raw-record, native-lifecycle and statistics audits pass. The builder rejects incomplete, pilot, invalid or mismatched evidence; existing output files require explicit `--overwrite`.

From the repository root, with final files at `results/stock-comparison/final-v1`:

```bash
.venv/bin/python results/stock-comparison/reproduction/stock_mechanisms.py \
  --records results/stock-comparison/final-v1 \
  --inputs-dir results/stock-comparison/final-v1/inputs \
  --validated-summary results/stock-comparison/final-v1/analysis.json \
  --output-dir results/stock-comparison/final-v1/mechanism

.venv/bin/python results/stock-comparison/reproduction/build_stock_report.py \
  --analysis results/stock-comparison/final-v1/analysis.json \
  --mechanism results/stock-comparison/final-v1/mechanism/mechanisms.json \
  --record-audit results/stock-comparison/final-v1/independent-audit.json \
  --lifecycle-audit results/stock-comparison/final-v1/lifecycle-audit.json \
  --statistics-audit results/stock-comparison/final-v1/independent-statistics.json \
  --markdown-output docs/stock-comparison/results.md \
  --html-output /Users/hsing-yulin/Documents/Codex/2026-11-08/us-2/outputs/stock-comparison/results.html
```

For another checkout, provide the desired local HTML output path explicitly. The standalone HTML embeds its CSS and has no network or JavaScript dependency. Review both generated files before publication.

The report separates the registered paired geometric inference from arithmetic descriptive trial means. It reports the hierarchy-closed secondary as a point estimate without a newly inferred interval. The64-position service decomposition is arithmetic association, not an estimate of isolated controller overhead. No text claims controller overhead dominates or causally explains the result without an ablation. Matched active-copy/tile totals can coexist with differing rank distribution; neither proves runtime benefit.

These tools generated the completed final report after all 216 trials and all required audits passed; the measured execution source remained unchanged.
