# Final validation summary

- Release build: PASS (`sm_86`, `-O3`, `-lineinfo`)
- CTest: 8/8 PASS
- Candidate smoke: PASS for aligned warp, aligned CTA, unaligned/tail fallback, and Top-4 fallback
- Compute Sanitizer memcheck: PASS
- Compute Sanitizer initcheck: PASS
- Compute Sanitizer racecheck: PASS
- Compute Sanitizer synccheck: PASS
- Public optimized Unpermute ID: rejected as `kUnsupportedKernelVariant`
- Public Auto/default Unpermute dispatch: unchanged naive implementation
