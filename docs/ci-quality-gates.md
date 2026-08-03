# CI and merge quality gates

Pull requests run dependency-free repository checks, Python tests on Windows and Linux, and CUDA-off CMake/CTest builds on both platforms. The repository check fails on invalid JSON, broken relative Markdown links, evidence checksum/inventory errors, legacy source or flat-config layout, meaningless tracked `.gitkeep`, developer-machine paths in public evidence manifests, and tracked files larger than 5 MiB.

Run the same policy locally:

```powershell
python scripts/repository_checks.py
python -m unittest discover -s tests -p "test_*.py"
cmake --preset cpu-release
cmake --build --preset build-cpu-release --parallel
ctest --preset test-cpu-release
```

`GPU SM86 validation` is a manually dispatched workflow for the Windows RTX 3080 self-hosted runner. It performs the Release build, CTest, seven-operator plus two-chain smoke suite, and Compute Sanitizer. NSYS and NCU are optional inputs; when enabled, NSYS runs first and NCU defaults to the `basic` set. Profiler durations remain diagnostic and never substitute for unprofiled Release A/B timing.

The repository's current private GitHub plan does not allow the desired required-check branch protection. Until the repository is public or the plan is upgraded, maintainers must require a green CI run and completed pull-request checklist before merge. Do not bypass that procedural gate for performance or dispatch changes.
