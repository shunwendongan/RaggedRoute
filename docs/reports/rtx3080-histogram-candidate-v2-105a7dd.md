# RTX 3080 Histogram `cuda_candidate` v2 report

## Outcome

Histogram v2 promotes one distribution-independent specialization: for the existing contract `E==1` and valid ids, it executes `counts[0]=route_pairs` with a single thread. It reads no expert id and needs neither atomic updates nor an external counts reset. Every `E>1` shape falls through to the proven v1 dispatcher unchanged. The Auto implementation ID is 102; v1 remains explicit ID 101.

H6 CTA-cap experiments (256/384/512) are rejected. The cap384 variant improved some large uniform smoke points but regressed `R=1M,E=64,Zipf-1.4` by 6.59%. Because dispatch may not scan unknown skew, there is no safe shape-only H6 region. H7 vector loads/RLE was therefore not implemented.

## Release result

- Source: clean `105a7ddddcd2`, Release `sm_86`; CUDA 13.3.73, driver 591.86.
- GPU: NVIDIA GeForce RTX 3080, compute capability 8.6, 68 SMs.
- Protocol: 5 independent processes; warmup 20; 30 event-batch samples/process; seed `20260729`; shared input/cache/repeats and randomized case/variant order.
- Correctness: CTest 8/8 plus memcheck, initcheck, racecheck and synccheck PASS. Exact CPU bincount validation passed for every raw benchmark record. Workspace is 0.

| L2 `E==1` case | strongest baseline | baseline p50 (us) | v2 p50 / p95 (us) | speedup | Gitems/s |
|---|---|---:|---:|---:|---:|
| `R=128,single-hot` | v1 | 8.151 | 7.649 / 8.615 | 1.066x | 0.017 |
| `R=4096,single-hot` | v1 | 10.004 | 7.823 / 8.715 | 1.279x | 0.524 |
| `R=65536,single-hot` | v1 | 14.234 | 7.859 / 9.062 | 1.811x | 8.339 |
| `R=1M,single-hot` | v1 | 14.797 | 5.478 / 9.216 | 2.701x | 191.402 |

All four declared `E==1` points improved in all five processes. The L1→L2 absolute gap is retained in the evidence table; direct write includes the required counts store in L1, while L2 adds wrapper/dispatch. CUB has no comparable standalone L1 API and is reported only as full L2.

For `E>1`, IDs 101 and 102 dispatch the identical GPU body. WDDM event-batch p95 varies between randomly ordered independently launched processes; it is recorded, not silently filtered, and cannot represent a new algorithmic regression where no body changed.

## NSYS and NCU diagnostic evidence

NSYS collected v2, v1, naive and CUB for `R=4096,E=1` (small), `R=32768,E=64` (crossover), and `R=1M,E=64` (large). It shows v2 small contains 21 `histogram_single_bin_write_kernel` launches; CUB exposes `DeviceHistogramInitKernel` plus `DeviceHistogramSweepKernel`, so CUB L1 is intentionally N/A.

NCU basic, post-warmup and `--clock-control none`, reports:

| Case / variant | Hot kernel | Grid x block | waves/SM | achieved occupancy | registers/thread |
|---|---|---:|---:|---:|---:|
| small v2 | `histogram_single_bin_write_kernel` | 1 x 1 | 0.000919 | 2.08% | 16 |
| small v1 | `histogram_single_cta_shared_kernel` | 1 x 256 | 0.002451 | 16.08% | 28 |
| large v2 | `histogram_block_private_kernel` | 128 x 256 | 0.313725 | 30.78% | 21 |
| large v1 | `histogram_block_private_kernel` | 128 x 256 | 0.313725 | 30.79% | 21 |
| large CUB sweep | `DeviceHistogramSweepKernel` | 171 x 384 | 0.838235 | 54.95% | 54 |

Two independent signals support H5: NSYS removes the input-scan/shared-histogram kernels from the single-bin path, and NCU shows the smaller one-thread launch with 16 rather than 28 registers. This is a launch/underfill specialization, not a claim that profiler duration equals release latency. Basic lacks PC-sampling stall metrics, so they are `not_collected`, not zero.

## References and reproducibility

- Dong and Pai's 2025 model treats shared-atomic bottlenecks as a counter-driven question; it motivated retaining H4/H7 only under supporting evidence: [arXiv:2503.17893](https://arxiv.org/abs/2503.17893).
- Maucher et al. show why atomic-contention rules must be measured rather than assumed: [PLOS '25](https://doi.org/10.1145/3764860.3768338).
- GPU Multisplit supplies the small-bin/warp-localization design space but not an SM86 performance claim: [Ashkiani et al.](https://arxiv.org/abs/1701.01189).
- CUDA 13.2 documents the natural-alignment requirement for 16-byte vector memory accesses, so H7 was not added without an H6 parent winner: [Programming Guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/index.html).

The [artifact bundle](artifacts/20260804T0410Z-105a7dd-histogram-candidate-v2-r3/) stores compact Git evidence, normalized NSYS/NCU metrics, environment, sanitizer logs, manifest and checksums. Raw JSONL, full aggregates, manifests, `.ncu-rep`, `.nsys-rep` and SQLite are archived in the immutable Release ZIP named by that manifest.
