# RTX 3080 Unpermute research-candidate artifact bundle

Decision: rejected for promotion; benchmark-only research implementation retained.

See [central report](../../unpermute-sm86-candidate-eba8f02.md) for interpretation. Formal
benchmark evidence is under `benchmark/formal-run1`, `benchmark/formal-rerun`, and
`benchmark/chain`. Final validation is under `benchmark/validation`. Normalized profiler and
SASS evidence is under `profile`. Each formal directory also contains a 72-row
`throughput-comparison.csv` with p50/p95, CV, logical bytes, effective GB/s, tokens/s, and
elements/s; throughput fields are derived from the same unprofiled p50 and fixed work count.

Profiler duration is diagnostic only. Release comparisons are the promotion source of truth.
No `.ncu-rep`, `.nsys-rep`, SQLite, executable, library, or build artifact is included.
