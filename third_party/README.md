# Third-party dependency directory

The default build does not vendor CUDA, cuBLAS, CCCL/CUB, or CUTLASS binaries here. CMake uses the
project's `AUTO|SYSTEM|FETCH|OFF` provider policy and records pinned revisions for fetched
header/template dependencies.

This directory is reserved for an explicitly selected local source checkout. Any copied or
adapted operator source belongs under the corresponding `src/<operator>/library_baseline/`
directory together with its `UPSTREAM.md` provenance record. See `THIRD_PARTY_NOTICES.md`.
