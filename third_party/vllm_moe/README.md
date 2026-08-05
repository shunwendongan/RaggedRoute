# vLLM MoE source archive

This directory archives the vLLM MoE implementation used as the production-reference
semantic path in the RaggedRoute experiment. The files are copied from
`https://github.com/vllm-project/vllm` at commit
`66b3c0e61f1e477820212201adf1ed871df7ee98`; see `SOURCE_MANIFEST.json` and
`../licenses/vLLM-Apache-2.0.txt` for attribution and licensing.

The runnable adapter is `scripts/vllm_semantic_benchmark.py`. It deliberately does
not copy the vLLM server, model loader, distributed runtime, quantization paths,
FP8 paths, or CUDA-graph scheduler. The adapter keeps the seven RaggedRoute stages:

`router GEMM → vLLM Top-K/normalization → device alignment metadata → offsets
→ stable permutation metadata → vLLM Triton fused Grouped GEMM → weighted unpermute`.

All measurements use strict FP32/IEEE (`TRITON_F32_DEFAULT=ieee`) on RTX 3080/sm_86.
The adapter includes vLLM alignment padding and temporary metadata work in the L3
timing boundary, uses preallocated buffers, and records the `[E,K,N]` to `[E,N,K]`
weight-layout adaptation. These changes are marked source-faithful-adapted rather
than binary-identical vLLM reproduction.
