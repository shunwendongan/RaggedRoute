# Upstream provenance

- Project: Triton
- Repository: <https://github.com/triton-lang/triton>
- Revision: `4cf21fe8f40049dd95afa2beec02aae84598f1ac`
- Source: `python/tutorials/08-grouped-gemm.py`
- License: MIT
- Changes: removed TMA/Hopper paths, consumed contiguous expert weights and device offsets,
  added empty/ragged expert masks, and forced IEEE FP32 dot products.

This is a benchmark-only reference and is not part of RaggedRoute runtime dispatch.
