CPU analysis snapshot — CS471 partial critical-path diagnosis, 2026-11-13

This directory publishes reviewed analysis code and focused synthetic tests, not
GPU launchers, raw traces, request records, or the private deployment history.
Source snapshot: 13d0aa97f51595e6981562514978e19a4df9e3a8.
The manifest binds each exported file. Except for the input-root change below,
the eight analysis/dependency modules and six test files retain source bytes.

Run focused tests (standard library only):
  cd sources
  python3 run_tests.py

With the private trace/sidecar schema available, run from this directory:
  python3 -m full_serving.boundary_critical_attribution PRIVATE_CELL_DIRECTORY --output NEW_PRIVATE_OUTPUT
  python3 -m full_serving.boundary_critical_predecessor PRIVATE_CELL_DIRECTORY --windows onset early middle late --output NEW_PRIVATE_OUTPUT
Other exposure/readiness/barrier modules provide their own --help.
Outputs include private input hashes and must stay private until separately reviewed.
Raw inputs are intentionally not included, so public users cannot independently
reconstruct the reported event counts from this export alone.

Export change: boundary_critical_analysis.review now takes the private input root
from CS471_PRIVATE_INPUTS instead of an installation-specific absolute path.
The request-admission review() helper also requires client, data, and
resident_policy modules from the original experiment repository; it is not a
standalone admission tool. Trace analysis uses its rows/category helpers.
The legacy contract windows() function describes nominal wall-time scheduling;
it is not the v3 capture implementation. Actual v3 windows were rank-agreed
64 native dispatches (192 for proposal), with boundaries excluded.

Interpretation constraints:
- API attribution requires CPU thread identity and CPU annotation containment.
- A latest-rank claim must clear every other rank's empirical clock bounds.
- NCCL name/order pairing is a hypothesis, not a transport sequence identifier.
- Same-stream predecessor evidence leaves cross-stream dependencies unknown.
- Missing actual work is not zero-filled without explicit dummy evidence.
- Readiness false defers a map commit; it does not prove forward execution stops.
- Kernel unions, CPU spans, and no-observed-CUDA intervals are not removable time.
- These profiler runs do not measure optimization speedup or decompose the
  unprofiled B4/OFF latency difference causally.

Validation: 66 CPU tests passed in the private development checkout. The export
contains 29 focused tests for exposure, readiness, conservative barrier intervals,
thread correlation, clock ordering, and predecessor logic; these are run again
on the exported bytes. No GPU tests are performed by this snapshot.
