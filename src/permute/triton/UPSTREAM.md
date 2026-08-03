# Upstream provenance

- Project: NVIDIA TransformerEngine
- Repository: <https://github.com/NVIDIA/TransformerEngine>
- Revision: `bffde8f4a0a4eea9036dc753e28269247e5de69d`
- Source: `transformer_engine/common/triton/permutation.py`
- License: Apache-2.0
- Changes: consume Top-K IDs and expert offsets directly, build stable route maps with
  per-expert scans, and write RaggedRoute's optional inverse map.

This is a benchmark-only reference and is not part of RaggedRoute runtime dispatch.
