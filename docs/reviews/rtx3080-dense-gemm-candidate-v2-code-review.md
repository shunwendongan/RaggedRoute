# RTX 3080 dense GEMM candidate v2 code review handoff

## Review state

- Branch: `codex/dense-gemm-candidate-v2`
- Base: `b940477`
- Scope: handwritten CUDA candidates under `src/dense_gemm/cuda_candidate`
- Runtime state: not built, not executed, not benchmarked, and not profiled yet, following the requested code-review gate
- Promotion state: research-only explicit IDs; `kAuto` and optimized ID 0 are unchanged

## Candidate implementations

| Variant | ID | Kernel | Isolated mechanism |
|---|---:|---|---|
| `cuda_register_tiled_v2_sync` | 5 | `dense_gemm_register_tiled_v2_sync_kernel` | 32×32×16 CTA/warp/thread register blocking with synchronous `float4` staging |
| `cuda_register_tiled_v2_async` | 6 | `dense_gemm_register_tiled_v2_async_kernel` | Same compute mapping, replacing staging with a two-stage sm_86 `cp.async` pipeline |

Both kernels live in `src/dense_gemm/cuda_candidate/optimized_v2.cu`. The adapter and registry remain in their existing benchmark-layer directories.

## Mapping to review

- 128 threads / 4 warps per CTA.
- Four warps form a 2×2 layout of 16×16 warp tiles.
- Each lane owns a 4×2 output microtile and eight independent FP32 accumulators.
- 128 × 8 = 1024 unique output coordinates, exactly one 32×32 CTA tile.
- Each K stage covers all 512 A elements and all 512 B elements exactly once through `float4` copies.
- A shared layout is `[32][20]`; B is `[16][32]`.
- Every accumulator is updated in ascending K order using `__fmaf_rn`.
- Aligned output pairs use `float2` stores.

A host-side enumeration performed during static review found:

```text
OUTPUT_UNIQUE=1024 EXPECTED=1024
A_STAGE_UNIQUE=512 EXPECTED=512
B_STAGE_UNIQUE=512 EXPECTED=512
SHARED_READ_CONFLICT_CASES=0
```

This is only a mapping proof. It is not CUDA execution evidence and does not replace CTest, sanitizer, NCU bank-conflict metrics, or numerical validation.

## Fast path and fallback

The v2 fast path requires:

- `M % 32 == 0`, `N % 32 == 0`, `K % 16 == 0`, and `K > 0`;
- non-null A/B/C pointers aligned to 16 bytes;
- explicit optimized implementation ID 5 or 6.

Otherwise dispatch uses:

1. existing `cuda_tiled_vector` when its 16×16/alignment contract is satisfied;
2. existing `cuda_tiled_scalar` for edge, unaligned, or K=0 cases.

The direct v2 launchers independently reject unsupported arguments so an internal caller cannot silently bypass their preconditions.

## Async protocol to review closely

Each thread issues one 16-byte A copy and one 16-byte B copy, then commits one `cp.async` group.

1. Prologue commits tile 0 into stage 0.
2. When a next tile exists, the loop commits it into the other stage.
3. `wait_group 1` keeps the newest group in flight while guaranteeing the current stage is complete.
4. `__syncthreads()` establishes block-wide visibility before shared-memory consumption.
5. A second barrier before the next iteration prevents stage overwrite while any warp still reads it.
6. The final iteration uses `wait_group 0` and omits the unnecessary trailing barrier.

This protocol must pass `racecheck` and `synccheck` before any timing is accepted.

## Resource intent

- Sync user shared memory: 4608 bytes/CTA.
- Async user shared memory: 9216 bytes/CTA.
- `__launch_bounds__(128, 4)` prevents an unconstrained extreme register allocation but does not force an 80-register cap.
- The measured soft target is at most 80 registers/thread.
- Any local load/store or spill rejects the candidate regardless of p50.
- Lower occupancy is acceptable only if unprofiled latency and issue/eligible-warp evidence improve.

## Integration changes

- `CMakeLists.txt` compiles `optimized_v2.cu` into the existing candidate target.
- `optimized_internal.h` registers IDs 5 and 6.
- `optimized.cu` owns v2 fast-path selection and fallback.
- `dense_gemm_adapter.cpp` exposes L1/L2 variants and complete tile/pipeline metadata.
- `registry.cpp` exposes stable experiment names.
- API tests cover explicit dispatch IDs, K=0, and unaligned fallback.
- correctness tests include both variants for tiny/edge shapes and the 256³ fast path.
- versioned Release and profile suites cover 256³, 512³, and 1024³.

## Deferred until approval after review

No build or runtime command has been executed on this branch yet. After approval, the required order is:

1. configure/build Release sm_86 with line info;
2. full CTest and CPU oracle;
3. Compute Sanitizer memcheck/racecheck/initcheck/synccheck;
4. unprofiled L1/L2 Release A/B suite;
5. NSYS hotspot discovery;
6. NCU basic, then detailed for the 1024³ candidates;
7. keep/reject against p50, p95, CV, register, spill, issue-active, MIO, long-scoreboard, barrier, and bank-conflict gates.

Profiler duration will be diagnostic only. Promotion decisions will use the unprofiled Release suite.
