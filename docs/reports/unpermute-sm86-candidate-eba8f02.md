# Unpermute SM86 candidate evidence report

## Decision

`cuda_warp_token_vec4` is **not promoted**. It remains a benchmark-only research candidate;
public Unpermute dispatch stays naive-only and fails closed for optimized IDs.

The implementation preserves strict FP32 storage/accumulation, rank-0 then rank-1 reduction,
caller stream, zero workspace, token ownership, and no output atomics. Correctness passed, but
formal performance failed the stability, per-shape p50, and p95 gates.

## Setup

- Base: `main@927c585e031ba6a01e41c01d21db03cb0d1ca9d0`
- Candidate checkpoint: `eba8f025c718`
- GPU: NVIDIA GeForce RTX 3080, CC 8.6, 68 SMs, 10 GiB
- Driver/CUDA: 591.86 / CUDA 13.3.73
- NCU/NSYS: 2026.2.1 / 2026.1.3
- Release contract: `sm_86 -O3 -lineinfo`, five independent processes, 20 warmups,
  at least 30 samples, uniform and Zipf-1.4, L1/L2 warm plus selected cold-scrub cases
- Baselines: `cuda_naive` and adapted vLLM `finalizeMoeRoutingKernelLauncher`

## Unprofiled Release results

All ratios below are baseline latency divided by candidate latency. These are the only data used
for promotion decisions.

| Run | Warm L1 geomean | Warm L2 geomean | Candidate warm groups CV>0.10 |
|---|---:|---:|---:|
| Formal run 1 | 1.0084x | 1.0655x | 55/64 |
| Automatic rerun | 1.0876x | 1.0564x | 63/64 |

The averages do not constitute a promotion because the two runs disagree materially by shape.
Run 1 worst L1/L2 p50 speedups were `0.2641x/0.5415x`; rerun values were
`0.4461x/0.4924x`. Maximum p95 ratios were `7.1419/6.0640` and `3.6254/5.3753`, far above
the allowed `1.03`. High-noise pairs were rerun once as specified, then marked inconclusive.

The L3 replacement experiment produced `1.0016x`, p95 ratio `0.9466`, and
baseline/candidate CV `0.4348/0.3052`: no statistically credible regression and no credible
speedup.

## Profiler diagnosis

Profiler durations are diagnostic only.

- Final warp path (`T=1024,N=256`): grid 256 × block 128, 0.31 waves/SM, 34
  registers/thread, achieved occupancy 25.36%, SM throughput 2.61%, memory/DRAM throughput
  43.30% (`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`).
- Final CTA path (`T=64,N=1024`): grid 64 × block 256, 0.16 waves/SM, 34
  registers/thread, 16 B static shared memory, achieved occupancy 14.94%, SM throughput 1.62%,
  memory/DRAM throughput 21.72%.
- CTA NSYS median over 21 post-warmup launches was 1.792 us; adapted vLLM at the same large-N
  diagnostic shape previously measured 2.016 us. This does not override the unprofiled matrix.
- SASS contains `LDG.E.128` and `STG.E.128` on both aligned fast paths. `cuobjdump
  --dump-resource-usage` reports 34 registers/thread, `LOCAL:0`, and `STACK:0`; no `LDL/STL`
  mnemonic was found in the filtered candidate disassembly.
- Basic did not collect long-scoreboard, eligible-warp, or sectors/request metrics; these are
  `not_collected`, not zero. The launch/occupancy/throughput evidence is sufficient for the
  rejected-candidate record, so no expensive detailed/full replay was justified.

## Correctness and API result

- CTest: 8/8 passed.
- Compute Sanitizer: memcheck, initcheck, racecheck, synccheck all passed.
- Candidate matrix covers Top-K fallback, tails, pointer offset 1, uniform/Zipf/single-hot/
  round-robin, stream and redzones; smoke validates aligned warp/CTA plus unaligned and Top-4
  fallback paths.
- Public `select_kernel` and `unpermute` remain naive-only. The candidate is invoked directly by
  benchmark adapters and is labeled `in_tree_cuda_research`.

## Artifacts

The normalized bundle is at
[20260802T193000Z-eba8f025c718-unpermute-sm86-research-v1](artifacts/20260802T193000Z-eba8f025c718-unpermute-sm86-research-v1/).
It contains raw JSONL, aggregate/comparison data, manifests, sanitizer logs, NSYS/NCU CSV/JSON,
SASS evidence, and SHA256 checksums. Binary `.ncu-rep`, `.nsys-rep`, SQLite, and build products
are intentionally excluded.
