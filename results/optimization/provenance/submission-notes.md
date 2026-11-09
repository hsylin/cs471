# Submission history

Capture job 4867103 and tuning job 4867411 requested particular nodes and remained
pending for resources. Both were cancelled before starting; their elapsed time
is zero and no measurements were generated. Replacement jobs 4867142 and 4867433
used the same declared source/configuration with no hard node pin. An earlier
capture submission command also failed before creating a job because the wrapper
copy had not yet finished; it produced no allocation or experiment data.

The executed capture, tuning, validation/profile and final jobs all completed
with exit code zero. No failed or partial measured matrix was spliced into a
completed one. Shared-filesystem Python-import latency explains much of the
allocation startup time in the logs; the experimental timer starts only after
the declared setup, compilation and warmup. No package, kernel grid, selection
rule, threshold or trace family changed in response to final outcomes.

All submissions used the course cs471 account and two allocated H100 GPUs.
Scheduler estimates were CHF 0.00; no paid account was substituted.
