# CUDA candidate promotion decision

- Decision: `promote`
- Baseline: `cuda_postlogit_integrated_latest`
- Candidate: `cuda_postlogit_graph_fixed_v1`
- Ratio-of-sums p50 speedup: 1.6205x
- Geometric mean speedup: 1.6205x
- Shape win fraction: 100.0%

## Reasons

- all correctness, stability, performance, and workspace gates passed

## Paired shapes

| Case | Baseline p50 (us) | Candidate p50 (us) | Speedup | p95 speedup |
|---|---:|---:|---:|---:|
| `graph.postlogit.steady.fixed` | 84.750 | 52.300 | 1.6205x | 1.9864x |
