# Third-party notices

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

This file records provenance; it does not replace the license distributed by each dependency.
When a source snapshot or adaptation is added, its applicable license text is retained alongside
the source or in the corresponding operator provenance record.
