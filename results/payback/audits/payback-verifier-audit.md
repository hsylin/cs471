The standalone verifier `work/cs471/scripts/verify_payback_ratios.py` reconstructs the payback report from raw records using only Python's standard library. It imports neither `followup` nor `skewbench` and does not modify either frozen package. This is an independently written arithmetic and schema check by the same agent team, not external human reproduction or another experiment.

The verifier checks the manifest hash, full expected arm matrix, repetition schedule, record eligibility, complete horizons, matched source/candidate/capture/replay identities, prompt-document provenance, and allocation/host/process equality within each triple. It reconstructs every paired log ratio and seconds difference, averages technical repetitions within source blocks, and rebuilds both cell and equal-cell-weight aggregate estimates. Bootstrap indices are generated explicitly from the fixed Python RNG stream; the estimator uses `math.fsum` and an independently expressed interpolated percentile. It compares ordinary and family-adjusted intervals, family sizes, decision margins, criteria, and primary-horizon eligibility at absolute tolerance `1e-12`.

Validation completed before final results:

- Native pilot: all 96 records, 32 triples, 48 cell estimates, and 12 aggregate estimates matched exactly; no interval is expected from two blocks.
- One-copy pilot: the same counts matched, with maximum absolute difference `2.220446049250313e-16`.
- A temporary, explicitly synthetic 1,536-arm fixture with eight blocks, two candidate families, varying block effects, and two technical repetitions exercised the bootstrap path. All 96 cells and 24 aggregates matched with maximum difference `2.220446049250313e-16` over 1,000 test draws. Synthetic records were temporary and were not saved as measured evidence.
- Deliberately changing a reported interval by 0.001 was rejected. Removing one raw arm was rejected as an incomplete matrix.

The actual final report uses its frozen 20,000 bootstrap draws. It has not been verified by this audit yet. Run once it is complete:

```bash
python scripts/verify_payback_ratios.py \
  --manifest configs/frozen/payback-final-v1.json \
  --records /path/to/final/records.jsonl.gz \
  --report /path/to/final/analysis/summary.json \
  --output /path/to/fresh/independent-ratio-audit.json
```

The output file must be new. JSONL and gzip inputs are supported. A successful check validates arithmetic agreement and declared record structure, not hardware truth, interval coverage, population independence, or causal claims beyond the recorded experiment.
