# CUDA candidate promotion decision

- Decision: `promote`
- Baseline: `cuda_postroute_current_v1`
- Candidate: `cuda_postroute_graph_fixed_v1`
- Ratio-of-sums p50 speedup: 1.2336x
- Geometric mean speedup: 1.2338x
- Shape win fraction: 100.0%

## Reasons

- all correctness, stability, performance, and workspace gates passed

## Paired shapes

| Case | Baseline p50 (us) | Candidate p50 (us) | Speedup | p95 speedup |
|---|---:|---:|---:|---:|
| `graph.postroute.steady.fixed.topk248.top_k2` | 59.350 | 47.800 | 1.2416x | 1.4468x |
| `graph.postroute.steady.fixed.topk248.top_k4` | 86.400 | 71.300 | 1.2118x | 1.7855x |
| `graph.postroute.steady.fixed.topk248.top_k8` | 101.100 | 81.000 | 1.2481x | 1.3410x |
