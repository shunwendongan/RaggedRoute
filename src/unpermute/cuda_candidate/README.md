# Unpermute CUDA candidate status

`optimized.cu` contains the strongest retained Unpermute candidate on the
RTX 3080 / sm_86 research target: `cuda_warp_token_vec4`.

- Fast contract: strict FP32, `top_k=2`, 16-byte aligned input/output, and
  `N % 4 == 0`.
- Semantics: token-owned output, rank-0 then rank-1 accumulation, zero
  workspace, caller-owned stream, and no global atomics.
- Fallback: all non-vector, tail, or non-Top-2 cases call the validated naive
  implementation.
- Dispatch: this remains benchmark/research-only. Public `Auto` continues to
  select the naive path because the production promotion policy has not been
  satisfied on every measured shape.

The second-round grid and CTA experiments were rejected. Their compact,
reproducible evidence is recorded in
[`docs/unpermute/v2-evaluation.md`](../../../docs/unpermute/v2-evaluation.md)
and its linked artifact bundle. No rejected V2 source is retained here.
