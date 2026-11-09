# Old-pilot phase association

Old six development content groups only; 24 matched ours/stockoff trials; no new final demand.

Service is coordinator finish minus start. These are descriptive paired associations, not a causal decomposition.

| Phase | Jobs per trial | Ours mean ms | Stock-off mean ms | Extra ms/job | Extra ms/trial |
|---|---:|---:|---:|---:|---:|
| nonrecording_ordinary | 3072 | 0.791366 | 0.719426 | 0.071940 | 220.999 |
| recording_ordinary | 960 | 0.794972 | 0.718777 | 0.076194 | 73.146 |
| boundary | 64 | 3.074768 | 0.719213 | 2.355555 | 150.755 |

All 24 ours records independently match the archived native observation indices: phases48–63 are recorded. The boundary class is separated from phases48–62.

- Recording phase is associated with elapsed service, not randomized independently of input, calendar, queue, or controller state.
- Phase differences do not isolate a kernel, synchronization, memory operation or Python component.
- The new matched control/selector ablation is needed for component attribution.
- Repeated jobs and arrival conditions are not independent workload samples.
