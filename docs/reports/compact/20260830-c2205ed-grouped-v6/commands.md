# Reproduction commands

The formal speedups come only from the first unprofiled Release command. NSYS and NCU commands are diagnostic.

```powershell
python scripts\run_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\operators\grouped_gemm\benchmark\candidate_v6_release.json `
  --output out\grouped-v6-balanced-wide\release-c2205ed1.jsonl

python scripts\summarize_candidate_matrix.py `
  --input out\grouped-v6-balanced-wide\release-c2205ed1.jsonl `
  --candidate cuda_grouped_sm86_fp32_v6_balanced_32x128 `
  --baseline cuda_grouped_sm86_fp32_v2 `
  --baseline cuda_grouped_sm86_fp32_v5_balanced_direct `
  --baseline cutlass_grouped `
  --baseline cublas_per_expert `
  --library cutlass_grouped `
  --library cublas_per_expert `
  --output-json v6-matrix-summary.json `
  --output-csv v6-matrix-rows.csv

python D:\CodexStorage\.codex\skills\ncu-report-skill\scripts\cuda_profile.py system `
  --run-dir out\grouped-v6-balanced-wide\profile-c2205ed1\nsys-v6-t2048-clean -- `
  out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --operator grouped_gemm --variant cuda_grouped_sm86_fp32_v6_balanced_32x128 `
  --level l2 --profile-once --warmup 20 --seed 20260829 `
  --param T=2048 --param E=64 --param top_k=2 --param K=128 --param N=128 `
  --param distribution=uniform --param zipf_s=0.0

python D:\CodexStorage\.codex\skills\ncu-report-skill\scripts\cuda_profile.py compute `
  --run-dir out\grouped-v6-balanced-wide\profile-c2205ed1\ncu-v6-t2048-detailed-clean `
  --kernel 'grouped_gemm_sm86_balanced_wide_kernel.*32' --set detailed `
  --launch-skip 20 --launch-count 1 --cache-control none -- `
  out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --operator grouped_gemm --variant cuda_grouped_sm86_fp32_v6_balanced_32x128 `
  --level l2 --profile-once --warmup 20 --seed 20260829 `
  --param T=2048 --param E=64 --param top_k=2 --param K=128 --param N=128 `
  --param distribution=uniform --param zipf_s=0.0
```

Validation gates:

```powershell
ctest --preset test-rtx3080-sm86-release --output-on-failure
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts\repository_checks.py --check all
python scripts\run_sanitizers.py `
  --build-dir out\build\rtx3080-sm86-release `
  --output-dir out\sanitizer\grouped-v6-c2205ed1
```
