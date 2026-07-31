# Dense GEMM library baseline provenance

- APIs: `cublasLtMatmul` and `cublasSgemm` from the locally discovered CUDA Toolkit.
- Integration: RaggedRoute-owned row-major FP32 wrappers; no NVIDIA binary or cuBLAS source is
  stored in this repository.
- Math contract: `CUBLAS_COMPUTE_32F_PEDANTIC` / `CUBLAS_PEDANTIC_MATH`, FP32 accumulate and
  FP32 output.
- Benchmark boundary: handle, descriptors, heuristic selection, and workspace allocation are
  completed before timing; the selected algorithm and Toolkit version are recorded per result.
- Terms: [NVIDIA CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cublas/).
