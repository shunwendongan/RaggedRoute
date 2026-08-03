# RTX 3080 Histogram `cuda_candidate` promotion report

## Outcome

The shape-dispatched `cuda_candidate` is promoted for SM86 Histogram `kAuto`. It keeps workspace at 0 and preserves explicit `cuda_naive` fallback. The shipping measurement revision is `cb02617af060`.

The strongest predeclared medium/large result is `R=1,048,576,E=64,uniform`: CUB p50 20.787 us versus candidate 16.947 us, a 1.227x speedup and 61.873 Gitems/s. Across the 12-case L2 matrix, candidate p50 is always below both naive and the case's declared strongest baseline; candidate p95 is also below the strongest baseline in every case.

## Setup and correctness

- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, 68 SMs, 10 GiB.
- Driver 591.86; CUDA compiler/runtime 13.3.73; NCU 2026.2.1; NSYS 2026.1.3.
- Release protocol: clean Git, Release `sm_86`, seed `20260729`, 5 independent processes, warmup 20, 30 samples/process, randomized case/variant order.
- Correctness: CTest 8/8; exact counts, sum invariant, old-count overwrite, zero route, input preservation, redzones, caller stream, local/unknown implementation IDs, and all dispatcher paths.
- Compute Sanitizer: memcheck, initcheck, racecheck, and synccheck all PASS.

## Dispatcher

| Route pairs | Shipping path | L2 reset |
|---:|---|---|
| `R<=4096` | one-CTA shared histogram | fused overwrite |
| `4097<=R<32768` | naive fallback | external async memset |
| `R>=32768` | block-private shared histogram, max 128 CTAs | external async memset |

The dispatcher uses only `R`; it performs no skew scan and hardcodes neither the distribution nor the RTX 3080 SM count.

## Profiler evidence

NSYS shows 21 single-CTA kernels and no per-invocation reset in the small trace. The large trace shows 21 block-private kernels with 21 corresponding L2 resets and no separate merge kernel.

NCU basic/detailed records:

- Small Zipf case: grid 1, block 256, 0.00245 waves/SM, 28 registers/thread, about 16.2% achieved occupancy. This is launch/underfill dominated.
- Large uniform/hot cases: grid 128, block 256, 0.3137 waves/SM, 21 registers/thread, no local load/store instructions, about 66% achieved occupancy, about 49% DRAM throughput, and long-scoreboard as the largest sampled stall.
- Both large distributions execute 32,768 warp-level shared atomic instructions. Shared-pipe activity is about 7.3% for uniform and 3.7% for hot; the evidence does not show shared-atomic-unit saturation. H4 warp-to-shared aggregation was therefore not implemented.

Profiler durations are diagnostic only. All speedup claims come from unprofiled release A/B.

## Evidence

The [artifact bundle](artifacts/20260803T0708Z-cb02617-histogram-candidate-v1/) contains raw JSONL, aggregate JSON/CSV, comparison output, normalized NSYS CSV, normalized NCU JSON/CSV, environment, sanitizer logs, manifest, and `SHA256SUMS`. Version-bound `.ncu-rep`, `.nsys-rep`, and SQLite files are inventoried by hash but intentionally not committed.

The complete 12-case table and rejected H1/H4 rationale are in the [Histogram performance record](../histogram/performance-record.md).
