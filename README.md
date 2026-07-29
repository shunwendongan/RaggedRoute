# RaggedRoute

CUDA primitives for MoE routing and ragged expert computation.

> Status: initial project skeleton. No performance results are claimed yet.

## Planned operator modules

- Dense GEMM
- Fused Top-K Gate
- Expert Histogram
- Exclusive Scan
- Token Permute
- Grouped GEMM
- Unpermute + Weighted Reduce

## Hardware scope

- Primary development and validation: NVIDIA RTX 3080 (Ampere, SM86)
- Optional migration and re-benchmarking: NVIDIA H100 (Hopper, SM90/SM90a)
- Blackwell remains a future validation plan and is not currently claimed as supported

## Repository status

This commit establishes the directory layout only. Build configuration, public
headers, CUDA kernels, tests, benchmark implementations, and performance reports
will be added incrementally after their API and measurement contracts are fixed.
