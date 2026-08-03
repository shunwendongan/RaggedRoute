# RaggedRoute

CUDA primitives and an auditable benchmark pipeline for single-GPU MoE routing and ragged expert computation.

> Current milestone: the source-breaking v0.2 public API, seven FP32 teaching/reference CUDA baselines, benchmark-only library/production variants, and auditable L1/L2/L3 measurement boundaries are implemented and revalidated on RTX 3080 / SM86. This is a reproducible resume project, not a production-ready operator library.

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

## v0.2 API layers

The project deliberately keeps two interfaces rather than hiding an L1 result inside an end-to-end wrapper:

- `raggedroute::ops::launch_*_naive`: low-level Kernel Entry for L1 Kernel Body measurement. Its reset/workspace preconditions are explicit.
- `raggedroute::{dense_gemm, topk_gate, histogram, exclusive_scan, token_permute, grouped_gemm, unpermute}`: public Operator Wrapper for L2/L3. It validates the v0.2 contract, uses the caller's stream, performs no hot-path allocation or forced synchronization, and includes mandatory device-side resets.

Floating payloads use `ConstTensorView`/`MutableTensorView`, whose `TensorSpec` records storage dtype, layout, and element strides. Integer routing metadata remains strongly typed. Runtime and correctness share one `ScalarType` vocabulary; FP8/FP6/FP4 enum values describe reference storage only and do not imply operator support. `KernelSelection` separates the stable `Auto`/`CudaNaive`/`CudaOptimized` family from an operator-local implementation id.

`RuntimeContext` carries the stream, caller-preallocated workspace, and a cached architecture. v0.2 currently dispatches only zero-stride contiguous row-major, all-FP32 operators on SM86 to `cuda_naive`; FP16/BF16 signatures are representable but explicitly unsupported until real kernels exist. SM90 is classified separately but remains unvalidated and unsupported. `histogram` clears its counts internally, while `token_permute` requires `E * sizeof(int32_t)` bytes of caller workspace for its cursors and clears that workspace internally.

## Benchmark design

One common runner owns CUDA Event timing, warmup, sampling, cache policy, environment capture, and versioned JSONL. Each adapter owns its typed parameters, reset semantics, reference implementation, validation tolerance, logical work, and operator-specific metrics.

Suite v2 groups multiple implementations under one logical case and one promotion baseline. The registry supplies standard implementation provenance, aggregate v2 retains every strict pairing key, and comparison v1 fails closed before computing speedup when hardware, build, semantics, seed, level, cache, repeats, or excluded steps differ. Suite v1 and existing benchmark v1 raw records remain readable.

Benchmark-only variants never enter public runtime dispatch. Dense GEMM has cuBLASLt/cuBLAS; Histogram and Scan use CUB; Grouped GEMM has a per-expert cuBLAS loop and an optional CUTLASS grouped implementation; Permute/Unpermute retain adapted vLLM FP32 references. The Top-K external baseline is deliberately unavailable because the locally available CUB APIs do not prove the project's tie/NaN/selected-softmax contract. Per-operator source provenance is recorded under `src/<operator>/library_baseline/`.

Correctness, release performance, and profiling are separate flows:

| Flow | Purpose | Produces performance claims? |
|---|---|---|
| Two correctness executables / CTest | Adapter/reference plus dtype, failure-artifact roundtrip, redzone, edge, randomized, and stream-contract validation | No |
| `benchmark_smoke.json` | Fast executable/schema smoke | No |
| `benchmark_rtx3080_release.json` | Clean-Git, Release, 3-process raw measurement | Baseline latency only |
| `benchmark_rtx3080_library_release.json` | Clean-Git, strict-FP32 library/reference pairing | Contract-matched comparison only |
| `profile_benchmarks.py` | NSYS system trace plus seven filtered NCU cases | No; profiler duration is not a score |

See [Benchmark architecture](docs/benchmark-architecture.md), the [correctness framework](docs/correctness-framework.md), [implementation status](docs/implementation-status.md), the [current RTX 3080 benchmark/Nsight report](docs/reports/rtx3080-naive-profile-a9489ab.md), and the [full technical design](docs/RaggedRoute-最终产品技术文档.md).

## Build profiles

The repository uses CMake presets with isolated output directories under `out/build/`. CUDA is optional: CPU-only presets run host-side schema tests and never enable the CUDA language; all CUDA operators, benchmark targets, and install exports are unavailable in that mode.

| Preset family | Architecture | Intended use |
|---|---|---|
| `cpu-debug`, `cpu-release` | none | Validate host tooling without NVCC |
| `rtx3080-sm86-debug`, `rtx3080-sm86-release` | `86-real` | Local RTX 3080 development and measurement |
| `h100-sm90-debug`, `h100-sm90-release` | `90-real` | Portable H100 path; cross-compile locally, validate on H100 |
| `h100-sm90a-debug`, `h100-sm90a-release` | `90a-real` | Separate Hopper accelerated path; no support claim before H100 validation |

On Windows, install CMake 3.24+, Ninja, Python 3, CUDA Toolkit, and Visual Studio 2022 Build Tools. Both wrappers call one shared environment resolver: it accepts a usable initialized Developer Shell, honors explicit `RAGGEDROUTE_VSDEVCMD`/`VSDEVCMD` overrides, checks `RAGGEDROUTE_VS_INSTALL_ROOT`, `VSINSTALLDIR`, and `VS2022_HOME`, discovers the latest x64 C++ workload through `vswhere`, checks standard VS 2022 editions as a fallback, and verifies that `cl.exe` is usable. This avoids pinning a machine-specific compiler path. Configure always uses a fresh CMake cache and explicitly passes the resolved MSVC/NVCC paths so an earlier MSYS2 compiler cannot win Ninja discovery. Set `RAGGEDROUTE_FETCH_REFERENCES=ON` when the Release evidence run must fetch the pinned CCCL/CUTLASS references. The build wrapper automatically reconfigures if an existing cache names a missing or different compiler. Run raw `cmake --build` only from an initialized Developer Shell; from an ordinary PowerShell use the wrappers below.

```powershell
# Defaults to rtx3080-sm86-release.
$env:RAGGEDROUTE_FETCH_REFERENCES = "ON" # Required for the pinned CUTLASS evidence variant.
cmd /c scripts\configure_windows.bat
cmd /c scripts\build_windows.bat
ctest --preset test-rtx3080-sm86-release

# Device-debug build (-G); never use it for a performance result.
cmd /c scripts\configure_windows.bat rtx3080-sm86-debug
cmd /c scripts\build_windows.bat rtx3080-sm86-debug
ctest --preset test-rtx3080-sm86-debug

# CPU-only configuration: no NVCC probe or CUDA target.
cmd /c scripts\configure_windows.bat cpu-debug
cmd /c scripts\build_windows.bat cpu-debug
ctest --preset test-cpu-debug
```

Release CUDA builds use `-lineinfo` and deliberately omit `-G`; Debug CUDA builds use `-G`. `CMAKE_CUDA_ARCHITECTURES` remains overrideable for an explicitly separate build directory.

## Dependencies

`CUDAToolkit` is required only when `RAGGEDROUTE_ENABLE_CUDA=ON`; CUDA runtime and cuBLAS/cuBLASLt are linked through CMake's imported targets for benchmark-only variants. CCCL and CUTLASS are independent optional dependencies controlled per library:

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

## Triton reference baselines

Each of the seven operator directories now has a benchmark-only Triton reference
under `src/<operator>/triton`. These implementations preserve the strict-FP32
and routing contracts, but remain outside the C++ runtime and promotion dispatch.
Use `scripts\setup_triton_windows.ps1` to create the native Windows Triton
environment, then run the cross-backend smoke/release suites in
[Triton reference baselines](docs/triton-baselines.md). Cross-backend ratios are
explicitly reference-only because NVCC and Triton/PyTorch use distinct toolchains.
The Triton evidence flow covers seven individual L1/L2 cases and one complete
seven-operator `chain_from_tokens` L3 case, with parsed NSYS/NCU evidence kept
separate from unprofiled release latency. See the [RTX 3080 Triton L1/L2/L3
report](docs/reports/artifacts/20260803T1315Z-d81f6b0-triton-three-levels-v2/REPORT.md).

## Install and consume

CUDA-enabled builds export `RaggedRoute::runtime` (the public operator API),
`RaggedRoute::baseline_ops` (low-level L1 entries), and the separate
`RaggedRoute::correctness_framework` test-support library:

```powershell
cmake --install out\build\rtx3080-sm86-release
```

An external CMake project can then use:

```cmake
find_package(RaggedRoute 0.2 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE
  RaggedRoute::runtime)
```

The package config discovers the consumer's CUDA Toolkit and propagates the C++17 requirement from the public headers.
Because v0.2 removes the v0.1 pointer-based API, the package deliberately rejects a 0.1 version request; downstream targets must rebuild against the v0.2 headers.
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

Implemented now: caller-stream naive launchers, the v0.2 self-describing tensor API, an SM86-only executable runtime/dispatch layer, typed adapters, per-operator CPU oracle implementations behind one correctness facade, benchmark-only cuBLAS/CUB/CUTLASS/vLLM variants, raw samples, p50/p90/p95 of batch means, explicit excluded steps, L1/L2 cost boundaries, and both L3 chains. L2/L3 call the public wrappers, so histogram counts reset and permute cursor reset are included there while L1 keeps them as explicit preconditions. A clean-Git RTX 3080 Release run now provides strict representative library pairings plus a full-chain NSYS trace, seven basic NCU captures, three detailed hotspot captures, and a checksummed text evidence bundle. Third-party dependency and source provenance rules are recorded in `THIRD_PARTY_NOTICES.md`.

Not implemented yet: executable failure replay, FP16/Tensor Core optimized variants, a Top-K external baseline with matching semantics, optimized-candidate shape sweeps, automatic promotion evaluation, default shape dispatch, or H100/Blackwell validation. `configs/benchmark_promotion_policy.json` currently expresses policy only; it is not an evaluator and cannot change dispatch. Those capabilities must be implemented and measured under the same contract before reporting an optimized-kernel speedup or support.

The remaining plan for promotion evidence, real trace/working-set workloads, plots, and optimized variants is tracked in [Development roadmap](docs/development-roadmap.md). Planned items are not current capabilities.
