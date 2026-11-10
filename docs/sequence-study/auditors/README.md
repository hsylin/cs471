# Independent sequence-study execution audit

Run from the repository root with the CPU NumPy environment. These files are
outside the frozen 116-file execution closure. They import only earlier
independent audit helpers in `results/resident-residual/auditors`, never the
production runtime or analyzers.

```bash
python -m pytest docs/sequence-study/auditors -q
python docs/sequence-study/auditors/audit_sequence.py \
  --plan configs/sequence-study-v2.json \
  --records results/sequence-study/final-v2 \
  --capture-root results/resident-residual/capture-final-v1/banks \
  --source-archive results/sequence-study/final-v2/provenance/executed-source.tar.gz \
  --binary-root results/resident-residual/compiled-artifacts \
  --output /fresh/path/independent-execution-audit.json
```

The output path must not exist. The complete run must contain all 54 raw records,
the nine input/routing artifact sets, `expert-probes.json`, and the production
`work/` artifacts. Raw JSONL and lossless `.jsonl.gz` files are accepted. The
captured banks are required; their bytes are authenticated against the plan.

Checks cover exact source/plan/calendars/order; full deterministic occurrence
reordering and collision repair; native float32 route artifacts; exact resident
assignments; raw clocks and completion; native observed windows, maps and normal
commits; actual compiled binary bytes; every structural work row; and all 512
rank/input expert-probe argument and padding reconstructions. Output includes
auditor/helper and artifact identities. Numeric GPU comparisons and full
checkpoint-weight equality remain authenticated recorded attestations, not a
new GPU execution by the CPU verifier.

This audit supplies no significance test. The four content bundles and arrival
calendars are reused. Expert CUDA-event intervals are instrumented stream
observations and are not pure GEMM residence, hardware FLOP counters, or an
isolated migration-payback experiment.
