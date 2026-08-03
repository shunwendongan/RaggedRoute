## Summary

Describe the user-visible or research-facing change and the exact scope of this PR.

## Correctness contract

- [ ] Semantics, shapes, dtypes, tolerances, seeds, and public API/ABI impact are stated.
- [ ] Correctness oracle and commands are listed; all applicable CTest/Python/Triton/smoke tests pass.
- [ ] CUDA launch errors and applicable Compute Sanitizer results are recorded.

## Performance evidence

- [ ] Not applicable, with a reason stated below; or the measurement boundary and cache mode are fixed.
- [ ] Baseline is the strongest contract-equivalent implementation, not only `cuda_naive`.
- [ ] Release A/B timing is unprofiled and uses independent processes with p50, p95, and CV.
- [ ] NSYS selects the hotspot before filtered, post-warmup single-launch NCU collection.
- [ ] Raw evidence is stored in an immutable Release asset; summary, manifest, and checksums stay in Git.
- [ ] Profiler duration is not used as benchmark speedup.

## Dispatch decision

- [ ] Promotion policy result is explicit per shape and includes regressions/counterexamples.
- [ ] Default dispatch is unchanged unless every configured promotion gate passes.
- [ ] Rejected candidates retain their reason and evidence.

## Evidence and commands

List exact commands, config paths, run/source SHA, clean state, hardware/toolchain, bundle path, and SHA256 verification result.

## Merge gate

This repository currently cannot enforce branch protection on its private plan. Maintainers must treat successful CI and this completed checklist as the merge gate until the repository is public or the plan supports required checks.
