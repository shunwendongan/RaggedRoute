# Top-K Gate performance record

## 2026-08-03 / RTX 3080 strict-FP32 candidate campaign

- Implementation commit: `1ce3398155a9f40bec383dc130ddc1787634d2f8`.
- Environment: RTX 3080 10 GiB, SM86, CUDA 13.3.73, driver 591.86,
  NCU 2026.2.1, NSYS 2026.1.3.
- Correctness: CTest 8/8; exhaustive `E=2..64` candidate matrix; ids exact,
  weights `atol=rtol=1e-6`; Compute Sanitizer memcheck/racecheck/initcheck/
  synccheck all clean.
- Release evidence: 13,370 records, 2,674 aggregate groups, 5 independent
  process runs/group, one GPU UUID, 30 samples/run, 100 launch repeats/sample.

Representative unprofiled L1 Release results:

| Shape | Variant | p50 us | p90 us | p95 us | CV | Mrows/s |
|---|---|---:|---:|---:|---:|---:|
| T2048/E64 | `cuda_naive` | 14.0544 | 16.2202 | 18.0859 | 0.103 | 145.719 |
| T2048/E64 | `cuda_warp_pair_top2_v1` | 9.6154 | 10.5902 | 10.9261 | 0.072 | 212.993 |
| T2048/E64 | `cuda_subwarp_pair_top2_v2` | 9.6358 | 10.4980 | 10.6803 | 0.070 | 212.540 |
| T2048/E64 | `cuda_vector_pair_top2_v3` | 9.9379 | 10.4090 | 10.5733 | 0.039 | 206.079 |
| T2048/E33 | `cuda_naive` | 10.3168 | 10.8554 | 10.9875 | 0.044 | 198.511 |
| T2048/E33 | `cuda_vector_pair_top2_v3` | 9.9072 | 10.6414 | 10.7827 | 0.058 | 206.718 |
| T2048/E8 | `cuda_naive` | 8.5606 | 9.0030 | 9.2058 | 0.054 | 239.234 |
| T2048/E8 | `cuda_vector_pair_top2_v3` | 8.7245 | 9.2170 | 9.3809 | 0.068 | 234.742 |

The T2048/E64 local speedup is 1.46x for v2 and 1.42x for v3, but the
reviewed dispatch rule is bucket-wide and continuous in T. No candidate passed
ratio-of-sums >=1.05, per-shape p50/p95 regression limits, and CV<=0.10 for
every larger shape at both L1 and L2. The aligned power-of-two v3 submatrix
also failed the same stability gates. `Auto` therefore remains
`cuda_naive`; this is a measured rejection, not an incomplete promotion.

Strict L1 pairing over the declared support sets:

| Candidate | Denominator | Paired shapes | Ratio-of-sums speedup | Worst regression | Result |
|---|---|---:|---:|---:|---|
| v2 | CUB BlockRadixSort | 195 | 2.221x | none (candidate always faster) | overall claim allowed |
| v3 | CUB BlockRadixSort | 195 | 2.234x | none (candidate always faster) | overall claim allowed |
| v2 | adapted vLLM | 78 | 0.925x | 41.3% | mixed |
| v3 | adapted vLLM | 78 | 0.944x | 13.0% | mixed |

NCU explains the local E64 gain: naive launches 8 blocks (0.0196 waves/SM,
26 registers/thread, 16.0% achieved occupancy), whereas v1/v2 launch 512
blocks (0.627 waves/SM, 20 registers/thread, about 49.5% occupancy). v3 uses
128 blocks and 0.314 waves/SM. Detailed collection reports zero local
load/store instructions for every variant. CUB reaches 1.88 waves/SM but
spends about 71% SM/Memory throughput on a complete radix sort; its detailed
trace also has substantially more short-scoreboard/barrier samples.

NSYS before/after-rejection chain traces both retain the naive Top-K kernel and
the same 11,456 ns diagnostic median. This confirms that no rejected candidate
entered Auto. These NSYS numbers are diagnostic and are not Release latency.

Full report: [RTX 3080 Top-K Gate campaign](../reports/rtx3080-topk-gate-1ce3398.md).
Normalized bundle:
[20260803T0705Z-1ce3398-topk-gate-v4](../reports/artifacts/20260803T0705Z-1ce3398-topk-gate-v4/).
