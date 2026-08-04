# Unpermute SM86 v2 screening result

## Decision

The strongest retained implementation is the existing benchmark-only
`cuda_warp_token_vec4` in
[`src/unpermute/cuda_candidate/optimized.cu`](../../src/unpermute/cuda_candidate/optimized.cu).
It remains source-of-truth for the strict-FP32 Top-2 vector path; `Auto` is
unchanged and still selects naive. No V2 source is retained or dispatched.

The second-round candidates were screened on an RTX 3080 (`sm_86`) against
both the existing V1 and the adapted vLLM
`finalizeMoeRoutingKernel`. The matrix used 3 independent processes, 20
warmups, 30 samples, `kernel_repeats=50`, identical seed and timing boundary,
and warm L1/L2 measurements. It is a rejection screen, not a five-process
promotion measurement.

| Candidate | Single changed mechanism | L1 vs V1 | L2 vs V1 | Decision |
|---|---|---:|---:|---|
| v2a | token/feature-tile grid; 256 threads and shared metadata | 1.0152x | 0.9967x | reject |
| v2b | v2a geometry with 128 threads | 0.9944x | 1.0200x | reject |
| v2c | v2b geometry with warp-shuffle metadata | 1.0057x | 0.9873x | reject |
| v2d | one token-owning warp per CTA for `N<512` | 0.9834x | 0.9801x | reject |

V2d is the best V2 result against vLLM (`1.0705x` L1 and `1.0786x` L2), but
it is not a valid promotion because it loses to V1 overall. Only 33.3% of the
screened shapes improved over V1; its worst p50 speedup was `0.7972x` (L1)
and `0.8026x` (L2), its largest p95 ratio was `1.1919`, and all-samples CV
reached `0.1366`. Restricting it to the only promising bucket (`T=64`) still
gave merely `1.0182x` L1 and `0.9949x` L2 relative to V1. It cannot satisfy
the required 1.05x dual-baseline geometric mean or the per-shape regression
gate.

## Profile interpretation

NCU basic at `T=64, N=256` confirmed the hypothesis that motivated the final
launch-geometry test: V1 launched 16 blocks across 68 SMs and achieved 8.21%
occupancy. The adapted vLLM kernel launched 64 blocks and achieved 7.88%
occupancy. Increasing the number of CTAs alone did not create a stable
end-to-end benefit, so it is not retained. NCU and NSYS durations are used
only for this diagnosis, never as release speedups.

All experimental candidates passed the project CTest correctness matrix before
screening. After removing every V2 code-path, the restored V1 tree was rebuilt
and CTest again passed 8/8.

## Reproducible evidence

The compact evidence bundle is
[`20260804T034147Z-728b9f36a6ad-unpermute-v2-screen`](../reports/artifacts/20260804T034147Z-728b9f36a6ad-unpermute-v2-screen/).
It inventories comparison/CSV data and normalized NSYS/NCU output with SHA-256
checksums. Raw JSONL, full aggregate JSON, `.ncu-rep`, `.nsys-rep`, and SQLite
remain local and are deliberately excluded from Git.

The earlier formal V1 evidence remains available in
[`performance-record.md`](performance-record.md) and its immutable artifact
bundle; it is not overwritten by this rejection screen.
