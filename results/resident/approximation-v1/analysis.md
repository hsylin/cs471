# Resident assignment approximation on old development inputs

Across all 384 original input batches, greedy adds a mean 7.383 routed tokens to the heavier rank relative to the offline DP tie-break optimum; the mean greedy/DP token-peak ratio is 1.004086. This includes the 16 conditioning and 32 measured inputs in each of eight cells.

Across the 256 unique measured inputs, greedy token-peak load exceeds the offline DP tie-break optimum on 218 inputs and equals it on 38. The mean excess is 7.457 routed tokens; the maximum is 56. The mean greedy/DP token-peak ratio is 1.004098, with maximum 1.025391.

Both assignments attain the same exact minimum maximum-active-copy count on every input. Total active copies and total padded rows are also identical. This does not make their runtimes equal: the assignment of token and padded work across ranks differs, and this CPU comparison does not measure the GPU selector's cost or completed layer latency.

| Old block | Domain | Batch | Unique measured inputs | Mean excess peak tokens | Maximum excess | Mean token-peak ratio | Mean padded-peak difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| opt-pilot-01 | 0 | 128 | 32 | 2.438 | 10 | 1.004761 | +0.0 |
| opt-pilot-01 | 0 | 1024 | 32 | 6.656 | 26 | 1.001625 | -4.0 |
| opt-pilot-01 | 1 | 128 | 32 | 1.406 | 11 | 1.002747 | +0.0 |
| opt-pilot-01 | 1 | 1024 | 32 | 6.250 | 18 | 1.001526 | -28.0 |
| opt-pilot-02 | 0 | 128 | 32 | 3.562 | 9 | 1.006914 | +0.0 |
| opt-pilot-02 | 0 | 1024 | 32 | 26.875 | 56 | 1.006561 | +0.0 |
| opt-pilot-02 | 1 | 128 | 32 | 3.281 | 13 | 1.006409 | +0.0 |
| opt-pilot-02 | 1 | 1024 | 32 | 9.188 | 25 | 1.002243 | -32.0 |

Greedy's maximum per-rank padded work is lower on 19 measured inputs, equal on 234, and higher on 3. DP optimizes its token-count tie-break, not padded work. The mean changed-route fraction is 9.241%; equal summary counts therefore need not imply identical physical assignments.

The JSON includes all 384 input-level records, both ranks' token/active/padded values and differences, and source/input hashes. Inputs were regenerated with the exact original input_indices function and checked against all eight existing pilot NPZ artifacts; every regenerated DP assignment matches the saved assignment. No new demand was selected, and no frozen execution source was changed.

These are descriptive comparisons over two reused source-document blocks. The eight cycles of each measured input are not independent observations. Neither a token gap nor an unchanged active-copy optimum establishes the sign or size of a runtime gap. The fully charged GPU validation remains the performance test; this result does not authorize changing the selector.

Reproduction:

```bash
python scripts/analyze_resident_approximation.py \
  --manifest configs/resident-online-validation-frozen-v1.json \
  --banks-dir /path/to/optimization-capture \
  --pilot-inputs /path/to/resident-results/pilot/inputs \
  --output /fresh/path/approximation.json \
  --markdown /fresh/path/approximation.md
```
