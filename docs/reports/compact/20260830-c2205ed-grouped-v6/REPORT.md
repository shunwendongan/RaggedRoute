# RTX 3080 Grouped GEMM v5/v6 evidence report

## Outcome

`cuda_grouped_sm86_fp32_v6_balanced_32x128` is retained as a zero-workspace, benchmark-only SM86 research candidate. It improves the v5 scheduling/tile portfolio but does not pass the full-matrix promotion gate and does not change `KernelFamily::kAuto`.

The formal evidence is `c2205ed1ba1063fccce3cd417fd671798dbfb66f`, clean Release, RTX 3080/SM86, strict FP32, 10 predeclared shapes, five processes, 20 warmups, 30 samples/process, 50 repeats and seed `20260829`. All 250 records passed the CPU oracle. The maximum per-process CV is `0.4968`; 44/50 groups exceed the 0.10 risk marker, but none exceeds the 0.50 portfolio evidence ceiling. No outlier was removed.

## Release matrix

| Comparison | Ratio-of-sums | Shape geomean | p50 wins | Process-pair wins | Worst p50 / p95 regression | Decision |
|---|---:|---:|---:|---:|---:|---|
| v6 vs v2 | `1.0444x` | `1.0313x` | 4/10 | 27/50 | 4.14% / 45.10% | reject: coverage and p95 |
| v6 vs v5 | `1.0116x` | `1.0134x` | 9/10 | 35/50 | 0.09% / 27.96% | reject: p95 |
| v6 vs CUTLASS | `1.0652x` | `1.1388x` | 6/10 | 30/50 | 32.74% / 37.15% | reject: counterexample |
| v6 vs cuBLAS per-expert | `18.0552x` | `10.2548x` | 9/10 | 46/50 | 27.63% / 86.90% | not a headline; host-loop launch amplification |
| v6 vs fastest CUTLASS/cuBLAS envelope | `0.9946x` | `1.0352x` | 5/10 | 26/50 | 32.74% / 86.90% | reject |

The strongest interview-safe local shapes belong to the callable v6 hybrid portfolio:

- Uniform `T512/E64/K128/N128`: v6 `18.4627 us` versus CUTLASS `22.6304 us`, or `1.2257x`; all 5/5 process pairs agree.
- Single-hot `T512/E64/K128/N128`: v6 `12.4006 us` versus the fastest library result, cuBLAS per-expert `22.0262 us`, or `1.7762x`; all 5/5 process pairs agree.
- Many-empty `T16/E64/K64/N64`: v6 `10.2912 us` versus CUTLASS `12.8410 us`, or `1.2478x`; all 5/5 process pairs agree.

Attribution matters: none of those three shapes enters the `32x128` wide kernel. Uniform T512/E64 fails the `ceil-average >= 32` condition; single-hot fails the skew condition; many-empty fails the average-row condition. They execute the v5/v2 fallback chain. The wide kernel itself is selected for uniform T512/E16/N256 (`1.0120x` versus CUTLASS) and uniform T2048/E64/N128 (`0.7534x`). Therefore the local numbers support a shape-aware hybrid portfolio claim, not a claim that the `32x128` kernel beats CUTLASS.

The required counterexamples are uniform `T2048/E64/K128/N128` (`0.7534x` versus CUTLASS), non-aligned `K127/N129` (`0.8380x`), and the single-expert cuBLAS shape (`0.7835x`). Therefore neither a single local winner nor the CUTLASS-only aggregate is presented as an overall library win.

## Design progression

- v5 keeps the v2 `16x32x16` strict-FP32 `cp.async` mainloop but replaces prefix/binary-search persistent scheduling with a direct `(column,row,expert)` grid for balanced workloads. It removes serial queue traversal and reduces registers/thread from the v2 profile's 86 to 68.
- v6 changes only the balanced path to a `32x128x16` tile with 256 threads and a per-thread `4x4` outer-product micro-tile. Aligned `float4` stores, Ampere two-stage `cp.async`, and a selector (`ceil-average >= 32`, max rows <= 2x average, aligned `K%16/N%128`) preserve a v5 fallback for skew, tiny and non-aligned cases.
- An exploratory v7 `64x128` path was screened on a dirty three-process smoke only and was removed. At uniform T2048 it measured `32.8909 us` versus v6 `30.8634 us` (6.6% slower), while its isolated T512/E16 gain did not justify the longer accumulator live range and tail-row synchronization path. These smoke numbers are diagnostic, not formal claims.

## NCU mechanism evidence

At uniform T2048, v6 reduces v5 global load requests from 94,144 to 53,296 (`-43.4%`), store requests from 16,752 to 4,096 (`-75.5%`) and DRAM writes from 3.06 MB to 0.53 MB (`-82.7%`). This confirms the large-tile request-amplification hypothesis. Clean NSYS reports a 21-launch median of 31.104 us; profiler duration is diagnostic only.

The tradeoff is visible in the same NCU launch: v6 has 72 registers/thread, 22,528 B shared, 0.941 waves/SM and 31.60% achieved occupancy, versus v5's 68 registers, 7,680 B shared, 3.227 waves/SM and 48.71% occupancy. v6 issue active is 31.28%, still below CUTLASS's 53.65%; its SM active-cycle minimum is 54.17% below the mean. CUTLASS therefore remains faster at uniform T2048 despite v6's lower request count. The dominant remaining problem is underfill/work imbalance and instruction-issue efficiency, not spills: all three detailed profiles report zero local load/store instructions.

## Resume framing

A defensible bullet is: “Designed an SM86 shape-aware Grouped GEMM portfolio with direct-grid and `32x128x16` `cp.async` paths; its v5/v2 fallback regions reached `1.226x` over CUTLASS at uniform T512 and `1.776x` over the fastest library baseline on single-hot routing, while five-process/10-shape gating exposed a `0.995x` full-envelope ratio and prevented unsafe default promotion. NCU separately verified 43% fewer global load requests in the wide path and identified T2048 underfill/issue efficiency as the remaining bottleneck.”

Artifacts: [v6 summary](v6-matrix-summary.json), [v6 rows](v6-matrix-rows.csv), [v5 ablation](v5-matrix-summary.json), [profiler metrics](profiler-metrics.json), [environment](environment.json), [commands](commands.md), and evaluator decisions in this directory.
