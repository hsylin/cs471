Selected source snapshot for the bounded capture / cross-stream event review.

Exact remote CPU-analysis source: 50ece6c26820fb72aec661a2b9f7aea1651658f2.
The accompanying manifest binds the exported source bytes. No remote Git history,
raw traces, request data, outputs, private input hashes or GPU launch scripts are included.

Files cover Python event-generation identity, read-only CUPTI stream identity queries,
CPU trace/event association, and conservative cross-stream dependency inspection.
boundary_critical_contract.py is the original dependency used by these analyzers;
its capture-window helpers are not the current continuous runtime protocol.
The successful fixed_idle runtime was 24f1ee916d877eebebb999a462f1f3b3b3e4a92e.
The later failed continuous fixed_charged runtime was 0c3fb034eb9b9a532db5dc3258464f2dd74ada59.
Do not interpret this source export as a GPU launch recipe or a passed charged admission.

Run the exported CPU tests from this directory with Python and pytest:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests

The tests use synthetic metadata and mocked stream queries; no CUDA device is required.
Real event records, traces and file-level evidence seals remain private.
Dependencies establish order, not exclusive critical-path delay or removable time.
