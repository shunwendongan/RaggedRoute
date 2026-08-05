# RTX 3080 Token Permute SM86 v2 evidence report

## Decision

This PR retains the strongest measured **explicit** SM86 v2 candidate in
`src/permute/cuda_candidate`: a shape dispatcher that sends large, 16-byte
aligned Top-2 rows (`T>=1024`, `K>=128`, `K%4==0`) to the tile4-direct kernel
and uses the existing token-owned Top-2 implementation everywhere else. The
full-from-ids benchmark path additionally fuses counts, exclusive scan, and
cursor reset into one CTA launch. Public API, caller-stream semantics, strict
FP32 semantics, and external workspace contract are unchanged. `kAuto` remains
on `cuda_naive`.

The candidate is retained as an explicit/research-facing choice, not promoted
to automatic dispatch: the pure-permute WDDM suite did not meet its stability,
coverage, or worst-shape gate. This distinction is deliberate and is recorded
in the committed evidence rather than hidden by the favourable full-path result.

## Validation

Evidence source SHA is `bdc77c276878ea2d0198e69a1fbcb4f9bd4355b1`, clean before
the experiment. Target was an NVIDIA GeForce RTX 3080 (GA102, CC 8.6, 68 SMs,
10 GiB), CUDA 13.3.73, driver 591.86, Nsight Systems 2026.1.3, and Nsight
Compute 2026.2.1. The code path in this PR is functionally identical to that
tested SHA except for making the v2 selector the `cuda_candidate` alias.

CTest passed 8/8. Compute Sanitizer memcheck, initcheck, racecheck, and
synccheck all passed. The suite covers tile tails, duplicate experts, optional
inverse mapping, unaligned scalar fallback, generic Top-K fallback, redzones,
and caller streams. Sanitizer logs are in the artifact bundle.

## Unprofiled release benchmarks

The release protocol used five independent processes, 50 warmups, and 50
samples; warm cases use 100 repeats and cold cases one repeat. Inputs, seed,
math mode, GPU UUID, cache mode, timing boundary, and workspace were paired.

| Boundary | Result | Status |
|---|---:|---|
| Pure permute, v2 vs token-owned ID4 | 1.0755x ratio-of-sums; 18/28 faster; worst 0.2487x | Not promotion-safe: WDDM CV > 0.10 for every pair |
| Full-from-ids, v2 vs old candidate | 1.7302x ratio-of-sums | Faster across the five reported cases |
| Full-from-ids, v2 vs pinned vLLM | 1.4110x ratio-of-sums | Faster across the five reported cases |

The v2 full-from-ids p50 sequence versus pinned vLLM is: anchor/uniform
64.881 vs 99.830 us, anchor/Zipf-1.4 48.005 vs 69.315 us, large/single-hot
64.154 vs 65.833 us, large/uniform 48.671 vs 78.735 us, and tail 42.281 vs
64.430 us. Candidate CV remains 0.160–0.281, so the values locate performance
on this WDDM workstation and must not be claimed as low-noise production
latency.

## Diagnostic profile

For `T=2048,E=64,top_k=2,K=256,uniform`, NSYS median diagnostic kernel times
were 10.400 us for ID4 and 9.568 us for tile4-direct. The v2 full path had a
10.432 us fused prepare launch and a 9.632 us copy launch. Pinned vLLM showed
7.680 us radix sort, 2.944 us expert-offset computation, and 9.632 us expand.
These durations are diagnostic only and are not used in the release claims.

NCU detailed reports 34 registers/thread and no local loads/stores for either
copy kernel. Tile4-direct uses 0.627 waves/SM, 41.7% achieved occupancy, and
48.6% DRAM throughput; ID4 uses 2.510 waves/SM, 63.6%, and 66.3%, respectively.
The evidence supports reduced copy-kernel time in the selected large aligned
case, while underfill and Windows scheduling explain why this is not a universal
pure-permute win. No Hopper/Blackwell-only mechanism is used.

## Artifacts and limits

The companion bundle contains compact benchmark comparisons and aggregates,
environment capture, NSYS CSV summaries, normalized NCU metrics, and sanitizer
logs with SHA-256 checksums. Its immutable raw archive contains JSONL,
`.ncu-rep`, `.nsys-rep`, and SQLite; the manifest records the asset URL, size,
and SHA-256. The raw binary artifacts remain out of Git history.

Implementation references: [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [warp-aggregated atomics](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/), [vLLM permute API](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/moe_permute_unpermute/), [ScatterMoE](https://openreview.net/forum?id=YDZ7GeFLxq), and [MoEBlaze](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html).
