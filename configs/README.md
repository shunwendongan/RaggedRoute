# Active benchmark and profiler configurations

Only active, runnable configuration files live under this tree. Historical
copies frozen inside `docs/reports/artifacts/<run-id>/` keep the paths and
contents used by the original run.

| Directory | Purpose |
|---|---|
| `project/benchmark/` | Seven-operator smoke, baseline release, and library pairing suites |
| `project/profile/` | Representative whole-project NSYS/NCU cases |
| `operators/<operator>/benchmark/` | Operator-specific smoke, release, selection, and rejected-candidate suites |
| `operators/<operator>/profile/` | Operator-specific profiler cases |
| `cross_backend/` | CUDA/Triton reference comparisons; never promotion eligible |
| `policies/` | Versioned default and operator-specific promotion thresholds |

File names describe the role within their directory, so an operator prefix is
not repeated. `smoke` proves execution and schema validity only. `release`
requires a clean optimized build and independent processes. Profiler duration
is diagnostic and must not be used as release latency.

Transient results belong under ignored `out/`. Published summaries and
manifests belong under `docs/reports/artifacts/`; large raw evidence is stored
as a checksummed GitHub Release asset.
