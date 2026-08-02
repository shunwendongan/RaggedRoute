# Top-K Gate benchmark baseline provenance

These sources are benchmark-only. They are not linked into public runtime
dispatch and do not change the installed API.

## vLLM row-packed Top-K

- Upstream: `vllm-project/vllm`,
  `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`.
- Pinned revision: `55c98e370aa058f567a9e682dc0652bdfba6b0bb`.
- License: Apache-2.0.
- Preserved structure: multiple rows per warp, 16-byte loads for E>=4,
  four warps per CTA, iterative argmax, and lower-index tie resolution.
- RaggedRoute adaptation: selection is performed on raw logits, only the two
  selected values are normalized, NaN is canonicalized to negative infinity,
  and the all-NaN fallback is ids 0/1 with weights 0.5/0.5.
- Supported pairing: E={2,4,8,16,32,64} and the required float2/float4 input
  alignment. Other shapes are rejected rather than silently paired.

## CUB BlockRadixSort Top-2

- Upstream: the CUB headers actually selected by the configured
  `RaggedRoute::cccl` target.
- License: Apache-2.0 with LLVM exception (CCCL).
- Algorithm: `cub::BlockRadixSort<uint64_t,32,2>`, one block per row.
- Composite key: ordered canonical FP32 bits in the high word and reversed
  expert id in the low word. NaN becomes negative infinity, signed zero is
  canonicalized, and padded ids are greater than every valid expert.
- Provenance fields report the compiled `CUB_VERSION` and `CCCL_VERSION`,
  plus the provider and requested Fetch tag. The requested tag is not presented
  as the compiled version when system Toolkit headers win discovery.

## Reproducibility

The exact source hashes, compiler command lines, Toolkit version, GPU UUID,
and normalized results are captured by the benchmark/profile manifests.
