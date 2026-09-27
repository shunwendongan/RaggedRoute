# Dense narrow portfolio: full region Release result

## Decision

**Do not promote `cuda_narrow_portfolio_v4`.** Restore and retain the original
`cuda_register_tiled_v3_64x32_async` kernel selection, including in existing L3
chains. The broad M>=256,N<=64,K<=256 rule proposed in `0b21ab8` is not supported
by the completed Release matrix. Earlier cross-session claims of a 17% gain
are withdrawn. This is a negative result, not a completed optimization.

`ea400b9` restored the original v3 launcher and isolated the selector in a
benchmark-only adapter. It does not add a public implementation ID or change
Auto. The candidate logs both selected variant and actual tail-fallback path.
Registered candidate absence is explicitly tested (red test reproduced), then
all eight CTest targets passed with the implementation and in the clean build.

## Reproduction and hardware

- GPU: one RTX 3080, sm_86, 68 SMs, UUID
  `7c5e95c0-e5a4-15d8-24a0-c8c8b58d6f39`; desktop WDDM load present.
- Clean Release binary built at `ea400b9`, CUDA 13.3.73 / MSVC 19.44.
- Strict FP32 GPU math, FP64 CPU reference, unchanged tolerance, seed 20260926.
- Configuration: `configs/project/benchmark/rtx3080_dense_portfolio_region.json`.
- Thirty shapes, L1 and L2 separately, two variants, five processes each:
  **600 records**, all validated. 20 warmups, 30 samples, 30 repeats, warm cache.
- Runner randomizes execution order in each process round. Pairing uses case,
  level and process_run, not adjacent launches in the same process.
- No profiler or sanitizer overlap with the unprofiled run. No outliers removed.
- Zero workspace for both variants.

```powershell
$env:RAGGEDROUTE_FETCH_REFERENCES='ON'
cmd /c scripts\configure_windows.bat rtx3080-sm86-release
cmd /c scripts\build_windows.bat rtx3080-sm86-release
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts/run_benchmarks.py --binary out/build/rtx3080-sm86-release/raggedroute_benchmark.exe --config configs/project/benchmark/rtx3080_dense_portfolio_region.json --output out/dense-portfolio-release.jsonl
```

Use a fresh output name and a clean worktree; do not weaken the runner's SHA or
dirty-worktree gates. Evidence for the run described here is under
`out/experiments/20260927-dense-portfolio/`, including `release.jsonl`, its
manifest, `aggregate.json`, `aggregate.csv`, `comparison.json`, `compare.py`,
and clean build/test logs. Raw local output is ignored by Git.

## Region results

The selector changes 25 shapes (including M=320 control-labelled case, which
is aligned and therefore selected); five shapes retain exactly the same kernel
path. Reported p50 is the median of five process medians. Speedup is baseline /
candidate. P95 ratios use medians of the five per-process P95s.

| Region | Level | Geometric speedup | Ratio of sums | Improved shapes | Shapes winning >=3/5 pairs | Worst latency ratio | Worst P95 ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| changed, 25 shapes | L1 | 0.9034 | 0.9001 | 2/25 | 1/25 | 1.2870 | 1.7330 |
| changed, 25 shapes | L2 | 0.9125 | 0.9084 | 2/25 | 5/25 | 1.2908 | 1.3267 |
| unchanged, 5 shapes | L1 | 1.0277 | 1.0257 | 4/5 | 4/5 | 1.0107 | 0.9866 |
| unchanged, 5 shapes | L2 | 0.9743 | 0.9766 | 1/5 | 1/5 | 1.0897 | 1.3729 |

The changed region fails the 1.03 speedup gate, 5% maximum shape-regression gate,
3% P95-regression gate and broad-improvement requirement. Unchanged controls
also show system noise, so tiny wins must not be used to cherry-pick a policy.

Selected L2 medians:

| M,N,K | retained v3 us | candidate us | Speedup | Paired wins |
|---|---:|---:|---:|---:|
| 256,64,128 | 9.506 | 12.271 | 0.775 | 0/5 |
| 512,64,128 | 9.677 | 11.605 | 0.834 | 0/5 |
| 2048,64,16 | 9.967 | 9.284 | 1.074 | 4/5 |
| 4096,64,256 | 18.603 | 22.460 | 0.828 | 0/5 |
| 512,512,512 (same path) | 28.467 | 28.928 | 0.984 | 1/5 |

Do not infer a new dispatch condition from the single winning row without an
independent holdout. At 512,64,128 the effective L2 throughput is about 0.867
TFLOP/s for retained v3 versus 0.723 TFLOP/s for the candidate.

## Kernel diagnosis is distinct from the Release score

NSYS on 512,64,128 confirms the exact v3 and v2 async kernel names. Across 21
profiled launches, median device duration was 5.248 us for v3 and 4.000 us for
v2. **These are diagnostic durations, not release speedups.** A faster isolated
device kernel is not sufficient to improve the CUDA-event span under the real
host submission and WDDM scheduling behavior.

NCU basic filtered one post-warmup launch for each:

| Metric | v3 | v2 selected by portfolio |
|---|---:|---:|
| launch grid blocks | 16 | 32 |
| registers/thread | 85 | 70 |
| shared bytes/block (reported total) | 15360 | 10240 |
| achieved occupancy | 8.30% | 8.32% |
| SM throughput | 4.57% | 10.65% |
| memory throughput | 6.80% | 10.65% |
| profiler duration | 8.832 us | 8.640 us |

Both grids are below the 68-SM count. Doubling the grid improves coverage but
still underfills; this is a useful mechanism observation, not promotion proof.
DRAM-throughput and stall-long-scoreboard counters were not collected by basic.
Do not interpret missing counters as zero. Kernel versus host/scheduling causes
of the Release reversal remain unresolved, not attributed solely to WDDM.

Reports: `out/experiments/20260927-dense-v3-system/`, `...dense-v4-system/`,
`...dense-v3-basic/`, `...dense-v4-basic/` each retain commands and profiler files.

## Next mechanisms

Keep original v3 while investigating shorter-instruction-path Top-K reductions
and the dominant Grouped GEMM stage of L3. No precision relaxation, untimed
input-dependent metadata computation or profiler-derived speed claim is allowed.
