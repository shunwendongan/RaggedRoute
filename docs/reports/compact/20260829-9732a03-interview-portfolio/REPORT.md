# RTX 3080 / SM86 interview portfolio evidence

Release evidence commit: `9732a0343c60f869fc4166a0cc3cabba2fd67bbb`. All speedups below use clean, unprofiled, five-process Release data. NCU/NSYS durations are diagnostic only.

| Operator study | Candidate | Strongest comparable baseline | Ratio-of-sums | Shape geomean | Coverage | Best shape | Max CV | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| `dense_v3_vs_library_envelope` | `cuda_register_tiled_v3_64x32_async` | `cublaslt/cublas` | 0.8716x | 0.9376x | 1/3 | 1.0095x | 0.2999 | `local_winner_only` |
| `topk_v4_vs_exact_naive` | `cuda_local_pair_two_reduce_top2_v4` | `cuda_naive` | 1.0972x | 1.0859x | 21/30 | 1.6934x | 0.1783 | `local_winner_only` |
| `histogram_v2_vs_reference_envelope` | `cuda_candidate_v2` | `cuda_candidate_v1/cub_device_histogram/cuda_naive` | 1.1016x | 1.1588x | 9/15 | 4.3026x | 0.4503 | `matrix_winner_research_only` |
| `scan_cub_block_vs_naive` | `cub_block_scan` | `cuda_naive` | 1.0249x | 1.0250x | 4/5 | 1.0765x | 0.1581 | `library_reference_wins` |
| `permute_full_v2_vs_vllm` | `cuda_candidate_v2_from_ids` | `vllm_moe_permute` | 1.5671x | 1.5885x | 5/5 | 1.8501x | 0.1619 | `matrix_winner_research_only` |
| `grouped_v2_vs_external_envelope` | `cuda_grouped_sm86_fp32_v2` | `cutlass_grouped/cublas_per_expert` | 0.8697x | 0.9472x | 3/10 | 1.6411x | 0.2241 | `local_winner_only` |
| `unpermute_vec4_vs_reference_envelope` | `cuda_warp_token_vec4` | `vllm_finalize_routing/cuda_naive` | 1.0154x | 1.0172x | 12/32 | 1.2892x | 0.2520 | `local_winner_only` |

## Evidence policy

This portfolio policy uses `CV <= 0.50` as the evidence ceiling. Values above 0.10 remain visible stability risks but do not automatically become `insufficient_evidence`. Matrix claims still require five complete processes, matched semantics and timing boundaries, ratio-of-sums, geomean, coverage, maximum regression, and cross-process direction checks.

A `matrix_winner_research_only` label does not change `KernelFamily::kAuto`; `local_winner_only` must be described with its winning shape and full-matrix counterexample. External Top-K results apply only to the declared random-input subdomain.

## Files

- `summary.json` and `operator_summary.csv`: matrix-level decisions.
- `release_pairs.json` and `release_pairs.csv`: compact per-shape Release evidence.
- `ncu_metrics.json/csv`: selected normalized profiler metrics with explicit missing statuses.
- `nsys_hotspots.json/csv`: uniform and Zipf L3 hotspot summaries.
- `environment-and-commands.json`: provenance and executable command boundary.
- `SHA256SUMS`: integrity hashes for every compact file in this bundle.
