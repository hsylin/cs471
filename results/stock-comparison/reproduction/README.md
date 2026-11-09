# Independent statistical reconstruction

This standalone verifier imports only Python's standard library and NumPy. It does not import the production analysis, inference, timing or execution packages. It reconstructs mean/p95 response from all 4,096 raw immutable-arrival clock vectors, verifies the complete paired matrix and input identities, then recomputes all three contrasts. The production analysis is opened only after this reconstruction and used as a comparison target.

For final data it reproduces the frozen 100,000 stratified content-bundle draws, seed 471612, exact draw-table digest, primary, 14 guards, linked-calendar-date deletion checks, gated secondary and descriptive cells. Pilot mode verifies 72 trials descriptively and produces no confidence intervals. Integer counts and elapsed values must agree exactly; floating comparisons allow small numerical rounding differences. It rejects missing/duplicate trials and never filters slow or unsuccessful observations into a smaller complete matrix.

```bash
python results/stock-comparison/reproduction/verify_stock_final_statistics.py \
  --plan FINAL_PLAN.json \
  --records FINAL_RECORDS.jsonl.gz \
  --summary PRODUCTION_ANALYSIS.json \
  --protocol docs/stock-comparison/protocol-v1.md \
  --output NEW_INDEPENDENT_STATISTICS_AUDIT.json
```

The raw input may be uncompressed JSONL, gzip JSONL, or a run directory containing `trials/*.jsonl*`. The output must not already exist. The protocol bytes must match the plan's frozen protocol digest. This verifier checks raw/plan/source identities and the clock envelope needed for the statistical endpoints; the separate capture, physical routing, native-controller lifecycle and source-file execution audits remain required.

`synthetic-validation.json` records 20 focused analytic/mutation tests and a full 216-row synthetic integration comparison, including the gzip CLI. The integration driver separately evaluated the production inference helper as a comparison target. No experimental timing was used to develop these fixtures. The 5,714 compared scalar values agree within a maximum numerical difference of 1.11e-16, with the exact draw digest `0b20e84dbe760e5c0cca1895574655c48e13bafe9d5625e026e9156fc4442ab8`. These synthetic checks are software validation, not experimental evidence.

```bash
python -m pytest -q results/stock-comparison/reproduction/test_stock_statistics_verifier.py
```

This is an independent implementation within the same team, not an external organization or laboratory replication. Its intervals remain conditional on the selected trace material, constructed content groups and one hardware allocation; they do not estimate a production-arrival or hardware population. These archived files sit outside the frozen execution-source inventory.
