# Correctness framework

The correctness framework is a separate library and CTest suite.  It never
depends on the benchmark harness; benchmark adapters only project its
post-measurement floating-point report into their lightweight
`ValidationResult`.  References, device-to-host copies, guards, and sanitizer
runs are outside the measured CUDA-event interval.

Each operator owns its CPU algorithm under
`src/<operator>/cpu_reference/reference.cpp`.  The installed correctness
header remains the single facade, so relocating the implementations does not
create seven competing public APIs.  Floating oracles compute in `double`,
integer metadata is checked exactly, and shared checked-arithmetic helpers
guard shape and offset conversion.

## Precision capability policy

| Storage type | Current framework status | RTX 3080 / SM86 | H100 / SM90a | Blackwell / SM100a |
|---|---|---|---|---|
| FP32 | reference, guarded GPU round-trip, baseline kernels | `runtime_verified` | no runtime claim | no runtime claim |
| FP16 | reference, guarded GPU round-trip | `runtime_verified` | no runtime claim | no runtime claim |
| BF16 | reference, guarded GPU round-trip | `runtime_verified` | no runtime claim | no runtime claim |
| FP8 E4M3 / E5M2 | traits, encode/decode, CPU reference | `reference_only` | `compile_only` until a real-card suite runs | no runtime claim |
| FP6 E2M3 / E3M2, FP4 E2M1 | traits, encode/decode, CPU reference | `reference_only` | `reference_only` | `compile_only` until a real-card suite runs |

TF32 is a math mode for FP32 storage, not a `ScalarType`.  The reference
rounds its mantissa with round-to-nearest-even before FP32 accumulation.  FP4
and FP6 traffic uses logical packed widths (four FP6 values are three logical
bytes; two FP4 values are one logical byte); wrapper object `sizeof` is never
used as a traffic claim.  Scaling metadata is represented separately from the
element type so a bare FP4 type is not mislabelled as NVFP4 or MXFP4.

`runtime_verified` is a statement about the test suite actually run on the
device, not merely about compiler support.  No H100 or Blackwell operator
support is claimed by this repository until the complete suite runs on that
hardware.

## Contracts checked

- All shape and byte products use checked arithmetic.  Host-invalid arguments
  fail synchronously; valid device ids and offsets are fast-path preconditions.
- Guarded device allocations have front/back redzones and poisoned outputs.
  The launch observer separately reports immediate launch and asynchronous
  stream errors.
- Floating references decode the quantized storage representation and compute
  in `double`.  GEMM uses a K- and `sum(abs(a*b))`-aware forward-error bound
  and normalized L2 diagnostic rather than one global tolerance.
- Top-2 is selected-softmax with lower expert id tie breaking.  NaN ranks as
  negative infinity; all-NaN returns ids `[0, 1]` and weights `[0.5, 0.5]`.
  One selected positive infinity receives weight one; equal selected
  infinities split equally.
- Histogram, scan, ids, counts, offsets, and mappings are exact checks.
  Permute uses `UnspecifiedWithinExpert`: it verifies segment membership,
  route-pair bijection, copied rows, and the optional inverse map instead of
  imposing a stable-sort order.
- Zero `T/R/M/N` is a successful no-op.  `K=0` dense/grouped GEMM writes the
  mathematical zero result (or `beta*C` in the CPU teaching reference).
  Top-K requires `E >= 2`; other routing operations require `E >= 1`.

Failure artifacts are JSON and contain the deterministic case descriptor,
dtype and math mode, seed, dimensions, CUDA errors, guard state, numerical
diagnostics, and the worst mismatch.  The framework validates case/failure
artifact serialization roundtrips, but it does not currently execute a saved
artifact as a replayed operator case.

## Test entry points

```powershell
cmd /c scripts\configure_windows.bat rtx3080-sm86-debug
cmd /c scripts\build_windows.bat rtx3080-sm86-debug
ctest --test-dir out/build/rtx3080-sm86-debug --output-on-failure
ctest --test-dir out/build/rtx3080-sm86-debug -L dtype --output-on-failure
ctest --test-dir out/build/rtx3080-sm86-debug -L correctness_edge --output-on-failure
```

The Windows wrappers dynamically resolve and validate the active MSVC x64
toolchain.  Direct `cmake --build` commands require an already initialized
Visual Studio Developer Shell.

The labels are `dtype`, `correctness_smoke`, `correctness_edge`,
`correctness_randomized`, `correctness_selftest`, and `stream_contract`.
The smoke test retains the seven baseline operators and the two chain suites.

To compile the future dtype declarations without adding fat-binary code to the
normal SM86 targets:

```powershell
cmake --preset rtx3080-sm86-debug -DRAGGEDROUTE_ENABLE_ARCH_COMPILE_TESTS=ON
cmd /c scripts\build_windows.bat rtx3080-sm86-debug raggedroute_arch_compile_tests
```

This creates separate `sm_90a` and `sm_100a` object probes only.  Passing the
probe is `compile_only`, not a hardware validation result.

For Compute Sanitizer, run small edge cases sequentially and retain a log per
tool:

```powershell
python scripts/run_sanitizers.py --build-dir out/build/rtx3080-sm86-debug `
  --output-dir reports/sanitizer
```

The script runs memcheck, initcheck, racecheck, and synccheck with a non-zero
error exit code.  Its default executable runs non-zero cases for all seven L2
operators and both L3 chains.  It does not run the framework self-test, which
intentionally corrupts a redzone.
