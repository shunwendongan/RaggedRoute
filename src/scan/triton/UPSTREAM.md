# Upstream provenance

- Project: FlagGems
- Repository: <https://github.com/flagos-ai/FlagGems>
- Revision: `43bf8524eb431cac4c891413566ebb1512c26099`
- Source: `src/flag_gems/ops/cumsum.py`
- License: Apache-2.0
- Changes: specialized the scan for `E<=64`, int32 counts, exclusive output, and the
  terminal route offset required by RaggedRoute.

This is a benchmark-only reference and is not part of RaggedRoute runtime dispatch.
