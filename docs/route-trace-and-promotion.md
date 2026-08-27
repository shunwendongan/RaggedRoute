# Route trace and three-state CUDA promotion

RaggedRoute accepts a versioned, anonymized routing working set without changing the public operator API. This research input exists so Grouped GEMM, Token Permute, and the six-stage post-logit chain can be evaluated on captured expert assignments rather than only generated uniform or Zipf distributions.

## Trace contract

The source JSON follows `raggedroute.route_trace.v1` and records `T`, `E`, `top_k`, a provenance class, and one or more frames of flattened per-token expert IDs. `source_kind` is one of `production`, `captured`, or `synthetic_fixture`; IDs must be in range and each token must route to distinct experts.

```powershell
python scripts\validate_route_trace.py configs\workloads\synthetic_fixture.route_trace.json `
  --output-normalized configs\workloads\synthetic_fixture.rrtrace
```

The normalized `.rrtrace` file is the only format read by the C++ benchmark adapters. `run_benchmarks.py` expands selected frames into paired cases and records the trace SHA-256 in the run manifest. Token Permute, Grouped GEMM, and `chain_from_logits` consume the exact frame; `chain_from_tokens` rejects traces because its measured router projection owns route generation.

The tracked fixture is intentionally tiny and marked `synthetic_fixture`. It proves parser, expansion, dispatch, and correctness behavior only. It is not production-performance evidence and must not be described as such.

## Three-state evaluator

`evaluate_promotion.py` emits one of:

- `promote`: all correctness, provenance, stability, performance, coverage, and workspace gates passed;
- `reject`: complete stable evidence exists, but correctness or a performance/workspace threshold failed;
- `insufficient_evidence`: the run is dirty, non-Release, incomplete, unstable, mismatched, or lacks a required captured/production trace.

Example for the six-stage v3 chain:

```powershell
python scripts\evaluate_promotion.py `
  --input out\research\six-ops-v3\six-ops-v3-release.jsonl `
  --baseline cuda_postlogit_integrated_latest `
  --candidate cuda_postlogit_research_v3 `
  --policy configs\policies\cuda_v3_promotion.json `
  --output out\research\six-ops-v3\l3-v3-decision.json `
  --markdown out\research\six-ops-v3\l3-v3-decision.md
```

For real-trace claims, use `cuda_v3_real_trace_promotion.json`. A synthetic fixture deliberately produces `insufficient_evidence`, even if its timings are favorable. Profiler duration is never an evaluator input; only clean, uninstrumented Release records can decide promotion.
