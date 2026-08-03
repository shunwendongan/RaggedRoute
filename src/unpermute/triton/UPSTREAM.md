# Upstream provenance

- Project: NVIDIA TransformerEngine
- Revision: `bffde8f4a0a4eea9036dc753e28269247e5de69d`
- Source: `transformer_engine/common/triton/permutation.py::_unpermute_kernel`
- Project: vLLM
- Revision: `c8602c79062440074a018c1d5f875a5571eb6881`
- Source: `vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py`
- Licenses: Apache-2.0
- Changes: use RaggedRoute `route_pos` indirection and fixed Top-K FP32 accumulation into a
  preallocated output.

This is a benchmark-only reference and is not part of RaggedRoute runtime dispatch.
