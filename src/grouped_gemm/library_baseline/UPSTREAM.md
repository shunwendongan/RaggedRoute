# Grouped GEMM library baseline provenance

- `cublas_grouped.cpp`: RaggedRoute-owned FP32 wrapper that issues one `cublasSgemm` per active
  expert. The CUDA Toolkit is discovered locally; no NVIDIA binary is committed.
- `cutlass_grouped.cu`: adapted from NVIDIA CUTLASS example 24, pinned by CMake to tag `v4.6.1`.
  It keeps only the SM80 SIMT FP32 row-major device-scheduled grouped path required by this
  benchmark.
- Upstream example:
  <https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/24_gemm_grouped/gemm_grouped.cu>
- License: CUTLASS BSD-3-Clause; the retained license is at
  `third_party/licenses/CUTLASS-BSD-3-Clause.txt`.
- Benchmark boundary: plan construction, problem metadata transfer, initialization, and workspace
  allocation occur before timing. Empty experts are omitted from library problem arrays but remain
  present in the logical case metadata.
