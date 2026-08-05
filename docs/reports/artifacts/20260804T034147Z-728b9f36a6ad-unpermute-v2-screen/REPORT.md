# Unpermute SM86 v2 screening evidence

- Run ID: `20260804T034147Z-728b9f36a6ad-unpermute-v2-screen`
- Device: NVIDIA GeForce RTX 3080, compute capability 8.6, 68 SMs
- Build base: `728b9f36a6ad`; CUDA 13.3.73; driver/runtime 13.10; NCU 2026.2.1;
  NSYS 2026.1.3
- Contract: FP32 input/output and accumulator; Top-2 fast-path retains rank
  order; zero workspace; caller stream; no global atomics.
- Decision: `rejected_not_promoted`. V1 remains the strongest retained source;
  all V2 code was removed after screening.

`benchmark/v2d-screen.comparison.json` and
`benchmark/v2d-screen.aggregate.csv` contain the complete compact summary for
the final V2D three-process rejection screen. `profile/` contains normalized
NSYS and NCU results for the `T=64,N=256` underfill diagnosis. Raw profiler
databases/reports and raw benchmark records are intentionally not tracked.
