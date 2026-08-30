# Reproduction commands

All Release speedups come from the clean binary built at `dea7c066a83a5df700aa60c03fd51446c6b4c5e5`. Profiler durations are diagnostic only.

```powershell
python scripts\run_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\operators\grouped_gemm\benchmark\candidate_v10_release.json `
  --output out\grouped-v10-wave-aware\release-dea7c06.jsonl `
  --expected-git-sha dea7c066a83a5df700aa60c03fd51446c6b4c5e5

python scripts\summarize_candidate_matrix.py `
  --input out\grouped-v10-wave-aware\release-dea7c06.jsonl `
  --candidate cuda_grouped_sm86_fp32_v9_balanced_32x64 `
  --baseline cuda_grouped_sm86_fp32_v6_balanced_32x128 `
  --library cutlass_grouped --library cublas_per_expert `
  --output-json out\grouped-v10-wave-aware\summary-v9.json `
  --output-csv out\grouped-v10-wave-aware\summary-v9.csv

python scripts\summarize_candidate_matrix.py `
  --input out\grouped-v10-wave-aware\release-dea7c06.jsonl `
  --candidate cuda_grouped_sm86_fp32_v10_wave_aware_portfolio `
  --baseline cuda_grouped_sm86_fp32_v6_balanced_32x128 `
  --baseline cuda_grouped_sm86_fp32_v9_balanced_32x64 `
  --library cutlass_grouped --library cublas_per_expert `
  --output-json out\grouped-v10-wave-aware\summary.json `
  --output-csv out\grouped-v10-wave-aware\summary.csv
```

Targeted sanitizer coverage ran `memcheck`, `initcheck`, `racecheck`, and `synccheck` for V9 active N64/N128 paths and V10 small-grid/K256/large-grid/non-aligned fallbacks. Every invocation used `--profile-once --warmup 1`, the same seed, and the same L2 oracle-backed adapter.

```powershell
compute-sanitizer --tool <tool> --error-exitcode 99 `
  out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --operator grouped_gemm --variant <variant> --level l2 --profile-once --warmup 1 `
  --seed 20260830 --param T=<T> --param E=<E> --param top_k=2 `
  --param K=<K> --param N=<N> --param distribution=<distribution> --param zipf_s=<s>
```

NSYS ran first for V6, V9, and CUTLASS on `T4096/E64/K128/N64`:

```powershell
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none `
  --output=<new-run-path> out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --operator grouped_gemm --variant <v6|v9|cutlass> --level l2 --profile-once `
  --warmup 20 --seed 20260830 --param T=4096 --param E=64 --param top_k=2 `
  --param K=128 --param N=64 --param distribution=uniform --param zipf_s=0.0
```

NCU then captured one filtered post-warmup launch per baseline/candidate, followed by detailed only for the V5 fallback and V9 hotspots:

```powershell
python scripts\profile_benchmarks.py compute `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\operators\grouped_gemm\profile\candidate_v10.json `
  --run-dir out\grouped-v10-wave-aware\profile-ncu-dea7c06

python scripts\profile_benchmarks.py compute `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\operators\grouped_gemm\profile\candidate_v10.json `
  --run-dir out\grouped-v10-wave-aware\profile-ncu-detailed-dea7c06 `
  --set detailed `
  --case grouped.v10.v6_portfolio_fallback.t4096_e64_k128_n64 `
  --case grouped.v10.v9_32x64.t4096_e64_k128_n64
```
