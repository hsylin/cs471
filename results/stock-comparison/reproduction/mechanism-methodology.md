# Descriptive stock comparison mechanism plan

The standalone `stock_mechanisms.py` script lives outside the closed execution source. It requires a complete, validated production summary and its matching raw records and saved input artifacts. It does not run GPUs, change policies, fit predictors, choose thresholds, or add inferential tests.

For every trial it recomputes response, queue, admission delay, dispatch delay and coordinator service from raw absolute clocks. It reports output completion, full observation extent, normal native completion tail, controller-inclusive drain and completed throughput separately. The 64 fixed positions63,127,…,4095 partition ours' service sum and permit paired arithmetic differences against the same positions in each stock control. Native asynchronous launch/commit positions and ±1 neighborhoods are labeled temporal associations, with overlapping categories disclosed; tail commits are excluded from per-job commit neighborhoods.

Actual saved vLLM logical routes and float32 weights must match preflight digests. The 4,112-entry resident diagnostics must match their recorded physical-assignment digest and collapse to those same logical routes; only4,096 measured jobs enter the work summaries. Stockoff routing uses one physical copy per logical expert over128slots. Ours uses the authenticated160-slot one-home assignments. Optional stockon work counts reconstruct the actual map active before each forward; a commit at step c changes forward c+1.

For each job report per-rank and summed active physical copies, routed tokens, nonempty M64 expert tiles `sum_e ceil(n_e/64)`, and padded expert rows. Report rank maxima as well as totals. Preserve each trial's actually resolved kernel configuration: M64 is a clearly named structural measure, not a claim that every CUDA block executes64 useful rows or that the runtime is proportional to the count. Stockoff and ours may have identical total active-copy/tile work but different distribution across ranks. Selector overhead and controller overhead cannot be separated causally from execution using these observations alone.

Output: per-trial/per-group JSON, paired CSV and a compact Markdown report. No new CIs, p-values, success rules or runtime predictions. Primary conclusions remain the separately frozen content-bundle analysis.

Usage after final validation:

```bash
.venv/bin/python ../stock_mechanisms.py \
  --records RESULTS_DIR \
  --inputs-dir RESULTS_DIR/inputs \
  --validated-summary RESULTS_DIR/summary.json \
  --output-dir ../stock-mechanisms-final
```

No final measurements have been inspected to write this method. The script is awaiting complete validated records.
