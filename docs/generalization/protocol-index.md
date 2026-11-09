# Operative frozen protocol

This index summarizes the completed design; the immutable versioned specifications and manifests below are authoritative. It does not change an endpoint after measurement.

| Item | Frozen choice |
| --- | --- |
| Hardware | Allocation 4872248, kh014, the recorded two H100 94 GB UUIDs and NV6 topology |
| Computation | Pinned Qwen3-30B-A3B layer 0, BF16, hidden 2,048/intermediate 768, 128 logical experts, top eight, B128 |
| Residency / backend | Map `[0..127,0..31]`, 160 slots, 32 extra replicas; replicated inputs and output allreduce |
| Main arms | Native SKIP; unchanged-map shortcut/native dispatch; shortcut/frozen online resident dispatch |
| Pilot | Six content groups, 72 treatment trials; six eligible native calibrations |
| Final | Eighteen content groups, six per dataset, 216 trials; eighteen native calibrations |
| Content | WildChat first-user text, MBPP descriptions, GSM8K questions; 64 disjoint documents per group, full-text and captured-prefix duplicate filters |
| Job construction | Authentic captured causal-prefill records; 128 distinct records per job, 256 distinct measured job combinations, records may recur across jobs |
| External arrivals | One contiguous 256-timestamp code block and one conversation block per group; preserved absolute releases and ties, exact recorded scaling |
| Rate conditions | Per-content native-calibrated 0.85 load and common pilot-derived 1,059.009 jobs/s |
| Order / conditioning | All six arm permutations balanced separately within each rate; identical conditioning; all admitted jobs drained |
| Primary | Combined/shortcut calibrated mean-response geometric paired ratio; conditional two-sided 95% upper bound strictly below 0.95 |
| Guards | Six dataset/rate mean-response bounds and aggregate calibrated finite-trial p95; one-sided upper bounds at `1−0.05/7`, each at most 1.02 |
| Date sensitivity | Delete each shared actual calendar date jointly; all seven aggregate point ratios at most 1.02 |
| Secondary | Combined/native test opens only after primary, guards and date checks pass |
| Resampling | 100,000 PCG64 draws, seed 471512; whole content bundles stratified by dataset; conditional on selected calendars and one allocation |
| Diagnostics | Three pilot states × eight frozen conditions = 24 eligible v2 trials; 24 failed-attestation v1 trials retained separately |
| Total | 72 + 216 + 24 + 24 = 336 treatment/diagnostic executions; native calibration and untimed correctness/preparation are separate |
| Algorithm decision | Pilot did not justify a fallback; no dispatcher, placement or kernel tuning from final data |

The main timing boundary includes rank commands, online selection, expert execution, output collective, observation/planning, guarded normal completion and the both-rank ACK. Source capture, job materialization, full references and conditioning are excluded setup with recorded durations. Observed elapsed includes the complete offered interval and drain. This is layer-job response latency, not full-request execution.

The main execution source digest is `e2543561c11cf7b29c18bf429fa2833d8748decd6d73cb4b2422603c9c4fbb30`. The held-out plan digest is `84dc07a9e61a0ef2e0f601df51dfd3f187c71020120c676ece2609f3d387dddd`. Corrected diagnostic-v2 plan digest is `6c7e9240a48f91fca7f836f126f75ef578c4f2f3886cd4298040d7ba8c71ff18`; its change is a post-timing float-weight attestation correction, with the same inputs, orders and algorithms.

Read [v1](protocol-v1.md), the [source/inference amendment v2](protocol-v2.md), and the [within-rate order amendment v3](protocol-v3.md) together. The [diagnostic v1](diagnostic-protocol-v1.md) is superseded for eligible evidence by the [checksum amendment v2](diagnostic-protocol-v2.md). Frozen manifests are `configs/generalization-pilot-v2.json`, `configs/generalization-final-v1.json`, and `configs/generalization-diagnostics-v2.json`. The [attempt ledger](../../results/generalization/capture-attempts/ledger.json) preserves the corrections and failures. The [reproduction guide](reproduction.md) gives exact source, artifact and command checks.
