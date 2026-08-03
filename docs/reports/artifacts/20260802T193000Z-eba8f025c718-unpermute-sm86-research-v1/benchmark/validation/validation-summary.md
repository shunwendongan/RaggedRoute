# 最终验证摘要

- Release 构建：通过（`sm_86`、`-O3`、`-lineinfo`）
- CTest：8/8 通过
- Candidate smoke：aligned warp、aligned CTA、unaligned/tail fallback、Top-4 fallback 均通过
- Compute Sanitizer memcheck：通过
- Compute Sanitizer initcheck：通过
- Compute Sanitizer racecheck：通过
- Compute Sanitizer synccheck：通过
- 公共 optimized Unpermute ID：以 `kUnsupportedKernelVariant` 拒绝
- 公共 Auto/default Unpermute dispatch：保持 naive implementation
