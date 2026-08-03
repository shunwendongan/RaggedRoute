# RTX 3080 Top-K Gate optimization report

## Outcome

Three exact CUDA candidates and two strong benchmark-only baselines were
implemented and validated. The candidates materially improve selected large-E
shapes—for example v2 reduces unprofiled L1 p50 at T2048/E64 from 14.0544 us
to 9.6358 us (1.46x) and increases throughput from 145.719 to 212.540
Mrows/s. However, no candidate satisfies the reviewed continuous bucket
promotion gates across all required T/E shapes. Public `Auto` remains
`cuda_naive`, and this PR is intentionally draft.

## Scope and semantics

- FP32 raw-logit deterministic Top-2, `2<=E<=64`, `top_k=2`.
- Lower expert id wins ties; NaN ranks as negative infinity.
- All-NaN fallback is ids 0/1 and weights 0.5/0.5.
- Selected-softmax normalizes only the selected logits.
- Caller stream, one launch, zero workspace, unchanged public API.
- Top-K+Histogram fusion is excluded.

## Correctness evidence

- CTest: 8/8.
- Persistent matrix: every E from 2 through 64, T={1,7,65}, three fixed
  seeds, aligned and 4-byte-only aligned inputs.
- Edge modes: duplicate maxima, all equal, signed zero, NaN/negative infinity,
  all NaN, single/multiple infinity, extreme gaps, tail rows.
- Explicit implementation ids, zero-token behavior, invalid ids, non-default
  stream, and guarded redzones.
- Compute Sanitizer: memcheck, racecheck, initcheck, and synccheck all report
  zero errors/hazards.

## Release protocol

The clean Release binary embeds commit `1ce3398155a9` and
`build_git_dirty=false`. The suite contains 13,370 raw records and 2,674
aggregate groups; every group has five independent process runs, 30 event
samples/run and 100 launch repeats/sample. All records use GPU UUID
`7c5e95c0e5a415d824a0c8c8b58d6f39`. Warm L1/L2, cold scrub, ties and
4-byte-only alignment cases are separated in case config.

All speedups below use unprofiled Release event timing. NCU/NSYS durations are
never substituted for Release latency.

## Representative Release results

| T/E | Variant | p50 us | p90 us | p95 us | CV | Mrows/s | Effective GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 32/64 | naive | 9.4771 | 10.0270 | 10.4858 | 0.053 | 3.377 | 0.918 |
| 32/64 | v1 | 8.4019 | 9.1146 | 9.5391 | 0.068 | 3.809 | 1.036 |
| 2048/8 | naive | 8.5606 | 9.0030 | 9.2058 | 0.054 | 239.234 | 11.483 |
| 2048/8 | v3 | 8.7245 | 9.2170 | 9.3809 | 0.068 | 234.742 | 11.268 |
| 2048/33 | naive | 10.3168 | 10.8554 | 10.9875 | 0.044 | 198.511 | 29.380 |
| 2048/33 | v3 fallback | 9.9072 | 10.6414 | 10.7827 | 0.058 | 206.718 | 30.594 |
| 2048/64 | naive | 14.0544 | 16.2202 | 18.0859 | 0.103 | 145.719 | 39.636 |
| 2048/64 | v1 | 9.6154 | 10.5902 | 10.9261 | 0.072 | 212.993 | 57.934 |
| 2048/64 | v2 | 9.6358 | 10.4980 | 10.6803 | 0.070 | 212.540 | 57.811 |
| 2048/64 | v3 | 9.9379 | 10.4090 | 10.5733 | 0.039 | 206.079 | 56.054 |

## Promotion decision

Promotion was evaluated for E=2, 3-8, 9-16, 17-32 and 33-64, plus the
aligned E={8,16,32,64} v3 path. For each candidate and T threshold, every
larger tested shape had to satisfy:

- p50 ratio-of-sums speedup >=1.05;
- per-shape p50 regression <=3% and p95 regression <=5%;
- all-sample CV <=0.10;
- five process runs, verified pairing, zero workspace growth and one launch.

No interval passed at both L1 and L2. The E33-64 candidates have strong
ratio-of-sums potential, but isolated E/T shapes breach regression and
stability limits. The complete per-bucket rejection table is in
`benchmark/REPORT.md` and `benchmark/promotion.json` in the bundle.

Relative to CUB, v2/v3 pass the overall comparison rule at 2.221x/2.234x
ratio-of-sums across 195 strict pairs with no candidate regression. Relative
to adapted vLLM, results are mixed: v2/v3 ratios are 0.925x/0.944x over 78
supported aligned pairs and have 41.3%/13.0% worst regressions. No blanket
“faster than vLLM” claim is made.

## NSYS and NCU diagnosis

NSYS was collected first with CUDA/NVTX tracing and CPU sampling/context
switch sampling disabled. At T2048/E64, diagnostic kernel medians are:
naive 11,200 ns; v1 4,800 ns; v2 4,800 ns; v3 3,520 ns; CUB 35,903 ns;
vLLM 2,528 ns. These include 20 warmups plus one validated invocation and
are diagnostic only.

| Variant | Grid/block | Waves/SM | Registers/thread | Achieved occupancy | SM % | Memory % |
|---|---|---:|---:|---:|---:|---:|
| naive | 8/256 | 0.0196 | 26 | 16.01% | 2.77 | 9.67 |
| v1 | 512/128 | 0.6275 | 20 | 49.50% | 37.76 | 15.78 |
| v2 | 512/128 | 0.6275 | 20 | 49.52% | 38.07 | 15.91 |
| v3 | 128/256 | 0.3137 | 20 | 29.45% | 16.62 | 7.13 |
| CUB | 2048/32 | 1.8824 | 32 | 29.19% | 71.07 | 71.07 |
| vLLM | 256/128 | 0.3137 | 23 | 26.65% | 5.20 | 9.32 |

Detailed NCU reports zero local load/store instructions for all six variants.
Naive/v1/v2/vLLM long-scoreboard samples are 44/13/20/12; CUB additionally
shows 719 short-scoreboard, 246 wait and 376 barrier samples. No source set was
collected because basic+detailed already explain launch underfill and radix
sort overhead.

Two independent `chain_from_logits` traces were collected before and after
the rejection decision. Both resolve Auto to `topk_gate_naive_kernel`, whose
diagnostic median is exactly 11,456 ns in both traces.

## Artifact policy

The committed bundle contains aggregate/comparison JSON/CSV, normalized
NSYS summaries, normalized NCU metrics, sanitizer logs, environment and
commands. Raw `.ncu-rep`, `.nsys-rep`, benchmark JSONL and full command
manifests remain under `out/`; their sizes and SHA256 hashes are recorded in
the bundle manifest. `SHA256SUMS` covers every committed artifact.

[Open the normalized artifact bundle](artifacts/20260803T0705Z-1ce3398-topk-gate-v4/).
