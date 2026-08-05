# Reproduction commands

The candidate was built from `86cfbe2a805d8880b73896dece2c77031768245d` with the RTX 3080/sm_86 Release preset.

```powershell
cmd /c scripts\configure_windows.bat rtx3080-sm86-release
cmd /c scripts\build_windows.bat rtx3080-sm86-release
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts\run_sanitizers.py --build-dir out\build\rtx3080-sm86-release --output-dir out\research\topk_gate\sanitizer-v4
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\topk_gate\benchmark\release.json --output out\research\topk_gate\topk-v4-release.jsonl
python scripts\aggregate_results.py out\research\topk_gate\topk-v4-release.jsonl --manifest out\research\topk_gate\topk-v4-release.jsonl.manifest.json --json out\research\topk_gate\topk-v4-release.aggregate-v2.json --csv out\research\topk_gate\topk-v4-release.aggregate-v2.csv
python scripts\compare_results.py out\research\topk_gate\topk-v4-release.aggregate-v2.json --output out\research\topk_gate\topk-v4-release.comparison.json
python scripts\analyze_topk_gate_results.py --aggregate out\research\topk_gate\topk-v4-release.aggregate-v2.json --comparison out\research\topk_gate\topk-v4-release.comparison.json --raw out\research\topk_gate\topk-v4-release.jsonl --output-dir out\research\topk_gate\promotion-v4
```

NSYS was run before NCU for the selected small-T/E8/E33/E64 cases. NCU used the `basic` set on one stable post-warmup launch. Profiler duration is diagnostic only; Release A/B data in `aggregate/` controls the dispatch decision.
