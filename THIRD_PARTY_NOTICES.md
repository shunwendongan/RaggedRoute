# Third-party notices

## vLLM Top-K gate structure (benchmark-only adaptation)

- Project: vLLM
- Source revision: `55c98e370aa058f567a9e682dc0652bdfba6b0bb`
- License: Apache License 2.0
- Use: row-packed/vector-load Top-K benchmark baseline adapted to
  RaggedRoute's raw-logit selected-softmax and deterministic NaN/tie contract.
- File: `src/topk_gate/library_baseline/vllm_row_packed_top2.cu`

## NVIDIA CCCL / CUB (benchmark-only)

- Project: NVIDIA CCCL
- License: Apache License 2.0 with LLVM exception
- Use: `cub::BlockRadixSort` strict Top-2 benchmark baseline.
- Compiled version: recorded at runtime as `CUB_VERSION` and
  `CCCL_VERSION`; the configured Fetch tag is recorded separately.

RaggedRoute keeps third-party performance baselines separate from its own CUDA implementations.
The external implementations are benchmark-only evidence and are not presented as original
RaggedRoute kernels.

## Dependency policy

- CUDA Runtime, cuBLAS and cuBLASLt are discovered from the locally installed NVIDIA CUDA
  Toolkit and linked through CMake imported targets. CUDA Toolkit binaries are not committed to
  this repository.
- NVIDIA CCCL, including CUB, is a header-only open-source dependency. RaggedRoute discovers a
  system/Toolkit installation or fetches the pinned source revision selected by CMake.
- NVIDIA CUTLASS is an open-source template library. RaggedRoute discovers a system checkout or
  fetches the pinned source revision selected by CMake.
- Source adapted from another project must retain its original license header, identify the exact
  upstream repository and revision, and carry a prominent modification notice. Its operator
  directory also contains an `UPSTREAM.md` record.

## Dependencies currently referenced by the build

| Component | Upstream | Integration | License/source terms |
|---|---|---|---|
| NVIDIA CUDA Toolkit / cuBLAS | <https://developer.nvidia.com/cuda-toolkit> | System dependency | NVIDIA CUDA Toolkit terms |
| NVIDIA CCCL / CUB | <https://github.com/NVIDIA/cccl> | Header-only, system or pinned FetchContent | Apache-2.0 with LLVM exceptions |
| NVIDIA CUTLASS | <https://github.com/NVIDIA/cutlass> | Header-only/templates, system or pinned FetchContent | BSD-3-Clause |
| vLLM Top-K gate adaptation | <https://github.com/vllm-project/vllm/tree/55c98e370aa058f567a9e682dc0652bdfba6b0bb> | Modified FP32 benchmark-only source in `src/topk_gate/library_baseline` | Apache-2.0; retained in `third_party/licenses/vLLM-Apache-2.0.txt` |
| vLLM permute/unpermute adaptation | <https://github.com/vllm-project/vllm/tree/837eae64580c885101ee95b073aafb27a485e7ce> | Modified FP32 benchmark-only source in `src/{permute,unpermute}/library_baseline` | Apache-2.0; retained in `third_party/licenses/vLLM-Apache-2.0.txt` |
| Triton teaching kernels and primitives | <https://github.com/triton-lang/triton/tree/4cf21fe8f40049dd95afa2beec02aae84598f1ac> | Modified benchmark-only Python references in `src/{dense_gemm,histogram,grouped_gemm}/triton` | MIT; retained in `third_party/licenses/Triton-MIT.txt` |
| FlagGems Top-K / scan adaptations | <https://github.com/flagos-ai/FlagGems/tree/43bf8524eb431cac4c891413566ebb1512c26099> | Modified benchmark-only Python references in `src/{topk_gate,scan}/triton` | Apache-2.0; retained in `third_party/licenses/vLLM-Apache-2.0.txt` |
| TransformerEngine permutation adaptations | <https://github.com/NVIDIA/TransformerEngine/tree/bffde8f4a0a4eea9036dc753e28269247e5de69d> | Modified benchmark-only Python references in `src/{permute,unpermute}/triton` | Apache-2.0; retained in `third_party/licenses/vLLM-Apache-2.0.txt` |
| vLLM weighted-reduction adaptation | <https://github.com/vllm-project/vllm/tree/c8602c79062440074a018c1d5f875a5571eb6881> | Modified benchmark-only Python reference in `src/unpermute/triton` | Apache-2.0; retained in `third_party/licenses/vLLM-Apache-2.0.txt` |

This file records provenance; it does not replace the license distributed by each dependency.
When a source snapshot or adaptation is added, its applicable license text is retained alongside
the source or in the corresponding operator provenance record.
