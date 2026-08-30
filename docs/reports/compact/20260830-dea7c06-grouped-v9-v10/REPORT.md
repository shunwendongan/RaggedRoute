# RTX 3080 Grouped GEMM V9/V10 evidence report

## Outcome

`cuda_grouped_sm86_fp32_v9_balanced_32x64` is the strongest measured in-tree strict-FP32 Grouped GEMM candidate on this declared matrix. It keeps the V6 `32x128` row tile and two-stage Ampere `cp.async` pipeline, halves tile-N to 64, changes each thread's accumulator from `4x4` to `4x2`, uses 256 threads, requires no workspace, and retains the complete V6→V5→V2 fallback chain.

V10 is a selector-only experiment that restricts V9 to CTA windows derived from the RTX 3080's 68 SMs. The clean Release result does not validate that selector: V10 is slower than direct V9 in aggregate, so V10 is retained as a failed hardware-aware dispatch experiment and does not replace V9 as the strongest candidate. Neither variant changes `KernelFamily::kAuto` or the public runtime API.

## Formal Release evidence

The evidence is clean SHA `dea7c066a83a5df700aa60c03fd51446c6b4c5e5`, RTX 3080/SM86, strict FP32, 15 predeclared shapes, five independent processes, 20 warmups, 30 samples/process, 50 repeats and seed `20260830`. All 375 records match the CPU oracle, use one GPU UUID and one clean build SHA, and report zero workspace.

The Windows/WDDM policy discloses `CV>0.10` without automatic downgrade and uses `CV>0.50` as the evidence ceiling. Sixty-two of 75 case/variant groups exceed 0.10. One CUTLASS tail process reaches `CV=0.5041`; it remains in the data and makes the all-shape aggregate a research trend rather than a promotion-grade matrix. No sample or process was removed. The resume-safe local V9 headlines below stay under the 0.50 ceiling and have 5/5 process direction agreement.

| Comparison | Ratio-of-sums | Shape geomean | p50 wins | Process-pair wins | Worst p50 / p95 latency regression | Decision |
|---|---:|---:|---:|---:|---:|---|
| V9 vs V6 portfolio | `1.0486x` | `1.0386x` | 12/15 | 49/75 | 6.77% / 29.69% | V9 is the stronger custom candidate; p95 still blocks promotion |
| V9 vs fastest CUTLASS/cuBLAS envelope | `1.0916x` | `1.1397x` | 11/15 | 53/75 | 33.20% / 83.19% | positive aggregate research trend; counterexamples block Auto |
| V10 vs V9 | `0.9775x` | `0.9895x` | 6/15 | 36/75 | 17.47% / 21.88% | reject selector-only experiment |
| V10 vs fastest CUTLASS/cuBLAS envelope | `1.0671x` | `1.1277x` | 11/15 | 51/75 | 33.20% / 84.20% | weaker than direct V9 |

The fastest comparable library in all 15 cases is CUTLASS Grouped SIMT FP32; cuBLAS per-active-expert remains in the envelope but loses because its host loop amplifies launch overhead. This is a strict-FP32 comparison and does not mix in TF32/Tensor Core math.

## Resume-safe local results and counterexamples

| Shape | V9 p50 | CUTLASS p50 | p50 speedup | p95 speedup | Process direction |
|---|---:|---:|---:|---:|---:|
| uniform `T512/E32/K128/N64` | 12.48 us | 23.49 us | `1.8819x` | `1.7239x` | 5/5 |
| Zipf1.4 `T2048/E64/K128/N64` | 26.50 us | 38.07 us | `1.4366x` | `1.3018x` | 5/5 |
| uniform `T4096/E64/K128/N64` | 28.67 us | 39.17 us | `1.3661x` | `1.2665x` | 5/5 |

Direct V9 really launches the `32x64` kernel for those balanced/aligned N64 shapes, so the gains may be attributed to the new kernel rather than a fallback. This differs from V10's small-grid `T512/E32` result, which falls back to V6 and must not be attributed to V9.

Required counterexamples are uniform `T2048/E64/K256/N64` (`0.9072x`), uniform `T2048/E64/K128/N128` (`0.7736x` p50 and `0.5459x` p95), and non-aligned `K127/N129` (`0.7508x`). These cases show that the `32x64` kernel is a narrow-N, moderate-K portfolio component, not a universal CUTLASS replacement.

## NSYS and NCU mechanism evidence

NSYS on the clean `T4096/E64/K128/N64` shape identified the actual kernels before NCU filtering. V6 cannot enter its N128-aligned wide path and executes the V5 `16x32` fallback: 21 launches average 32.117 us. V9 executes `grouped_gemm_sm86_balanced_wide_kernel<32,64>` at 29.470 us; CUTLASS averages 40.994 us. These durations only select and diagnose kernels; the speedups above come from unprofiled Release data.

| Metric | V5 fallback | V9 `32x64` | CUTLASS |
|---|---:|---:|---:|
| Grid CTAs | 1280 | 320 | 68 |
| Block threads | 128 | 256 | 256 |
| Waves/SM | 2.689 | 1.569 | 1.000 |
| Registers/thread | 68 | 78 | 144 |
| Shared memory/block | 7,680 B | 14,336 B | 17,680 B |
| Achieved occupancy | 48.44% | 41.54% | 16.66% |
| Global-load requests | 75,824 | 47,096 | not collected in basic |
| Global-store requests | 16,552 | 8,192 | not collected in basic |
| Local load/store instructions | 0 / 0 | 0 / 0 | not collected in basic |

V9 reduces global-load requests by 37.9%, global-store requests by 50.5%, and CTA count by 75% relative to the V5 fallback. Its achieved occupancy is lower, not higher, yet Release latency improves by about 10% on the representative shape. The evidence therefore supports reduced over-partitioning, repeated requests and scheduling/tail-wave cost as the main mechanism; it does not support an “occupancy is the objective” story. CUTLASS's generic `128x128` SIMT tile launches only one 68-CTA wave and is register-limited to about 16.7% occupancy on this narrow-N ragged case.

V9 reads a similar amount from DRAM (6.42 MB versus 6.31 MB) and has a lower L2 hit rate (64.94% versus 80.64%), so raw DRAM traffic was not reduced. The win is specifically fewer issued global requests and coarser useful work, not a fabricated bandwidth claim. Detailed stall values are PC-sampling counts rather than normalized ratios and are preserved in `ncu-detailed.json` without being used as a speedup.

## Correctness and decision

Targeted Compute Sanitizer covers V9 active N64/N128 and V10 small-grid, K256, large-grid and non-aligned fallback paths under memcheck, initcheck, racecheck and synccheck: 24/24 invocations pass. Release validation is 375/375.

Decision: retain V9 and V10 as explicit benchmark-only research paths; identify V9 as the strongest measured custom candidate; reject V10's current 68-SM selector; keep `KernelFamily::kAuto` unchanged. A future dispatch may route the proven N64/K<=128 region to V9 and library-favorable K256/N128/non-aligned regions elsewhere, but that policy requires a new held-out matrix and must not be inferred post hoc from this table.

## Resume framing

A defensible bullet is: “Designed a zero-workspace SM86 strict-FP32 ragged Grouped GEMM with a `32x64x16` tile, per-thread `4x2` outer products and double-buffered `cp.async`; on a clean five-process/15-shape RTX 3080 study it reached `1.882x` over CUTLASS at `T512/E32/K128/N64` and `1.366x` at `T4096/E64/K128/N64`. NCU showed 75% fewer CTAs and 38%/51% fewer global load/store requests than the prior fallback, while K256, N128 and non-aligned counterexamples prevented unsafe Auto promotion.”

Artifacts in this directory contain both V9 and V10 matrix rows, normalized NCU basic/detailed metrics, NSYS kernel summaries, environment, commands, sanitizer results, raw-artifact hashes and SHA-256 checksums.
