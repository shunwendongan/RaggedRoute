# Triton reference baselines

The seven `src/<operator>/triton` directories contain benchmark-only Python
references. They deliberately do not modify the v0.2 C++ API, runtime dispatch,
or CMake install surface.

## Contract

- Payload tensors are contiguous CUDA FP32; routing metadata is contiguous int32.
- Dense and grouped GEMM call `tl.dot(input_precision="ieee")`; TF32 is not a
  valid comparison mode for this strict-FP32 suite.
- Top-K is fixed to deterministic Top-2 selected-softmax: lower-id tie breaking,
  NaN treated as `-Inf`, all-NaN `{0,1}`/`{0.5,0.5}`, and explicit infinity cases.
- The target is RTX 3080 / SM86. The portable, non-TMA grouped-GEMM path is used;
  no Hopper/Blackwell mechanisms are included.

Each implementation retains an SPDX header and a pinned `UPSTREAM.md` record.
The primary references are Triton Matmul/Grouped GEMM (MIT), FlagGems Top-K/Cumsum
(Apache-2.0), TransformerEngine Permutation (Apache-2.0), and vLLM weighted
reduction (Apache-2.0).

## Native Windows setup

```powershell
scripts\setup_triton_windows.ps1
```

This creates the ignored `.venv-triton` using Python 3.12, PyTorch
`2.12.1+cu130`, and `triton-windows 3.7.1.post27`. The kernels themselves import
the standard `triton` package and are also usable in a matching Linux environment.

## Correctness and comparison

```powershell
.\.venv-triton\Scripts\python.exe -m unittest -v tests.test_triton_baselines

.\.venv-triton\Scripts\python.exe scripts\run_cross_backend_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --python .\.venv-triton\Scripts\python.exe `
  --config configs\benchmark_rtx3080_cross_backend_smoke.json `
  --output out\benchmark\triton-cross-smoke.jsonl

.\.venv-triton\Scripts\python.exe scripts\compare_cross_backend.py `
  out\benchmark\triton-cross-smoke.jsonl `
  --manifest out\benchmark\triton-cross-smoke.jsonl.manifest.json `
  --output out\benchmark\triton-cross-smoke.comparison.json
```

`raggedroute.cross_backend_comparison.v1` proves shared GPU, driver, Git,
shape, dtype, timing boundary, and math contract. It records the distinct
NVCC/PyTorch/Triton toolchains and always emits `promotion_eligible: false`.
Use it as an engineering reference when adding a future CUDA candidate; retain
the existing fail-closed `compare_results.py` for promotion evidence.

For release evidence, use `configs/benchmark_rtx3080_cross_backend_release.json`
on a clean worktree. It requires three independent processes, 20 warmups, and
30 samples. Capture NSYS before filtering a post-warmup emitted Triton kernel
with NCU `basic`; profiler duration is diagnostic only.
