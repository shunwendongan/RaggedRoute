# RaggedRoute

CUDA primitives and an auditable benchmark pipeline for single-GPU MoE routing and ragged expert computation.

> Current milestone: a two-layer P0 API, seven FP32 teaching/reference CUDA baselines, and auditable L1/L2/L3 measurement boundaries are implemented and revalidated on RTX 3080 / SM86. This is a reproducible resume project, not a production-ready operator library.

## Operators

- Dense GEMM
- Fused Top-2 Gate with selected-softmax
- Expert Histogram
- Exclusive Scan
- Token Permute
- Grouped GEMM
- Unpermute + Weighted Reduce

The registry exposes exactly seven operator adapters. Two separately named L3 suites prevent an ambiguous end-to-end number:

- `chain_from_tokens`: all seven operators, including router projection;
- `chain_from_logits`: the six operators after precomputed router logits.

## P0 API layers

The project deliberately keeps two interfaces rather than hiding an L1 result inside an end-to-end wrapper:

- `raggedroute::ops::launch_*_naive`: low-level Kernel Entry for L1 Kernel Body measurement. Its reset/workspace preconditions are explicit.
- `raggedroute::{dense_gemm, topk_gate, histogram, exclusive_scan, token_permute, grouped_gemm, unpermute}`: public Operator Wrapper for L2/L3. It validates the P0 contract, uses the caller's stream, performs no hot-path allocation or forced synchronization, and includes mandatory device-side resets.

`RuntimeContext` carries the stream, caller-preallocated workspace, and a cached architecture. P0 currently dispatches only contiguous row-major FP32 operators on SM86 to `cuda_naive`; unsupported architectures, dtypes, layouts, and variants fail explicitly rather than being treated as validated fallbacks. `histogram` clears its counts internally, while `token_permute` requires `E * sizeof(int32_t)` bytes of caller workspace for its cursors and clears that workspace internally.

## Benchmark design

One common runner owns CUDA Event timing, warmup, sampling, cache policy, environment capture, and versioned JSONL. Each adapter owns its typed parameters, reset semantics, reference implementation, validation tolerance, logical work, and operator-specific metrics.

Correctness, release performance, and profiling are separate flows:

| Flow | Purpose | Produces performance claims? |
|---|---|---|
| Two correctness executables / CTest | Adapter/reference plus dtype, failure-artifact roundtrip, redzone, edge, randomized, and stream-contract validation | No |
| `benchmark_smoke.json` | Fast executable/schema smoke | No |
| `benchmark_rtx3080_release.json` | Clean-Git, Release, 3-process raw measurement | Baseline latency only |
| `profile_benchmarks.py` | Nsight Compute diagnosis | No; profiler duration is not a score |

See [Benchmark architecture](docs/benchmark-architecture.md), the [correctness framework](docs/correctness-framework.md), [implementation status](docs/implementation-status.md), the [RTX 3080 naive baseline report](docs/reports/rtx3080-naive-baseline-e37c132.md), and the [full technical design](docs/RaggedRoute-最终产品技术文档.md).

## Build profiles

The repository uses CMake presets with isolated output directories under `out/build/`. CUDA is optional: CPU-only presets run host-side schema tests and never enable the CUDA language; all CUDA operators, benchmark targets, and install exports are unavailable in that mode.

| Preset family | Architecture | Intended use |
|---|---|---|
| `cpu-debug`, `cpu-release` | none | Validate host tooling without NVCC |
| `rtx3080-sm86-debug`, `rtx3080-sm86-release` | `86-real` | Local RTX 3080 development and measurement |
| `h100-sm90-debug`, `h100-sm90-release` | `90-real` | Portable H100 path; cross-compile locally, validate on H100 |
| `h100-sm90a-debug`, `h100-sm90a-release` | `90a-real` | Separate Hopper accelerated path; no support claim before H100 validation |

On Windows, install CMake 3.24+, Ninja, Python 3, CUDA Toolkit, and Visual Studio 2022 Build Tools. The wrapper prefers `vswhere`, then uses an existing `VSDEVCMD` value, and finally checks the standard VS 2022 Build Tools location.

```powershell
# Defaults to rtx3080-sm86-release.
cmd /c scripts\configure_windows.bat
cmd /c scripts\build_windows.bat
ctest --preset test-rtx3080-sm86-release

# Device-debug build (-G); never use it for a performance result.
cmd /c scripts\configure_windows.bat rtx3080-sm86-debug
cmd /c scripts\build_windows.bat rtx3080-sm86-debug
ctest --preset test-rtx3080-sm86-debug

# CPU-only configuration: no NVCC probe or CUDA target.
cmake --preset cpu-debug
cmake --build --preset build-cpu-debug
ctest --preset test-cpu-debug
```

Release CUDA builds use `-lineinfo` and deliberately omit `-G`; Debug CUDA builds use `-G`. `CMAKE_CUDA_ARCHITECTURES` remains overrideable for an explicitly separate build directory.

## Dependencies

`CUDAToolkit` is required only when `RAGGEDROUTE_ENABLE_CUDA=ON`; CUDA runtime is linked through CMake's imported targets. cuBLAS is detected for future baseline targets. CCCL and CUTLASS are independent optional dependencies controlled per library:

```powershell
# AUTO (default): use Toolkit/system headers when present, without network access.
cmake --preset rtx3080-sm86-release

# SYSTEM: require an existing installation or checkout.
cmake --preset rtx3080-sm86-release `
  -DRAGGEDROUTE_CUTLASS_PROVIDER=SYSTEM `
  -DRAGGEDROUTE_CUTLASS_ROOT=D:\deps\cutlass

# FETCH: download the pinned tag configured in CMake cache.
cmake --preset rtx3080-sm86-release `
  -DRAGGEDROUTE_CUTLASS_PROVIDER=FETCH
```

Valid provider values are `AUTO`, `SYSTEM`, `FETCH`, and `OFF`. `AUTO` discovers CUDA 13's bundled CCCL headers before falling back to an external checkout; it leaves absent optional libraries disabled with a clear configure message. The current pinned tags are CCCL `v3.4.0` and CUTLASS `v4.6.1`.

## Install and consume

CUDA-enabled builds export `RaggedRoute::runtime` (the public operator API),
`RaggedRoute::baseline_ops` (low-level L1 entries), and the separate
`RaggedRoute::correctness_framework` test-support library:

```powershell
cmake --install out\build\rtx3080-sm86-release
```

An external CMake project can then use:

```cmake
find_package(RaggedRoute CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE
  RaggedRoute::runtime)
```

The package config discovers the consumer's CUDA Toolkit and propagates the C++17 requirement from the public headers.
On Windows, consume a Release installation from a Release consumer (or install both configurations into the same prefix before selecting a Debug consumer), because the static CUDA libraries use the matching MSVC runtime.

## Run the smoke suite

```powershell
python scripts\run_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\benchmark_smoke.json `
  --output reports\runs\smoke.jsonl

python scripts\aggregate_results.py reports\runs\smoke.jsonl `
  --json reports\runs\smoke.aggregate.json `
  --csv reports\runs\smoke.aggregate.csv
```

List targets or run one adapter directly:

```powershell
out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --list
out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --operator histogram --level l2 --protocol smoke `
  --param T=2048 --param E=64 --param top_k=2 `
  --param distribution=zipf --param zipf_s=1.4
```

## Evidence boundary

Implemented now: caller-stream naive launchers, an SM86-only public runtime/dispatch layer, typed adapters, CPU oracles, raw samples, p50/p90/p95 of batch means, explicit excluded steps, L1/L2 cost boundaries, and both L3 chains. L2/L3 call the public wrappers, so histogram counts reset and permute cursor reset are included there while L1 keeps them as explicit preconditions.

Not implemented yet: executable failure replay, FP16/Tensor Core optimized variants, cuBLAS/CUTLASS/CUB performance baselines, automatic promotion evaluation, default shape dispatch, or H100/Blackwell validation. The promotion policy file is a roadmap draft only. Those capabilities must be implemented and measured under the same contract before reporting speedup or support.
