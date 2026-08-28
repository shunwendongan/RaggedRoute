# RaggedRoute SM86 aggressive v4 evaluation

## Scope and provenance

- GPU: NVIDIA GeForce RTX 3080, 68 SMs, compute capability 8.6; driver 616.56; CUDA 13.3.73; NCU 2026.2.1; NSYS 2026.1.3.
- Kernel experiments and initial clean Release runs: `9dbf1fb`; post-route and CUDA Graph Release runs: `b3429c2`. The only source changes between these clean build SHAs are campaign/configuration corrections; no public API or `KernelFamily::kAuto` dispatch was changed.
- All release records use five independent processes, 20 warmups, 30 samples, strict FP32, CPU-oracle validation, and `build_git_dirty:false`. The original raw JSONL and profiler reports remain in this ignored `out/research/six-ops-v4/` directory.
- Gates passed before timing: repository checks; 57 Python tests; 8 CTest tests; and Compute Sanitizer memcheck, initcheck, racecheck, and synccheck.

## Decisions

| Scope | Candidate comparison | Result | Evidence summary |
|---|---|---|---|
| Grouped descriptor v4A | best queue-1024 vs v2 | `insufficient_evidence`, reject direction | ratio-of-sums 0.9225x; geomean 0.8448x; 30% shape wins; worst p50/p95 regressions 55.6%/264.3%. |
| Route-prep R1 | shared-rank t1024 vs v3 | `insufficient_evidence` | ratio-of-sums 1.0439x and geomean 1.0078x, but only 40.3% shape wins and worst p50/p95 regression 55.9%/111.9%. |
| Token-owned R2 | Top-4 token-owned vs shared-rank t256 | `insufficient_evidence` | ratio-of-sums 1.0170x; geomean 1.0359x; 72.9% wins; no workspace growth, but CV evidence is insufficient. |
| Gather fusion R3 | post-route gather vs current | `insufficient_evidence`, reject direction | ratio-of-sums 0.8191x; geomean 0.9575x; 51.7% wins; +4,195,072 B workspace. |
| CUDA Graph fixed | postlogit fixed replay | `insufficient_evidence` for broad promotion | fixed-shape host p50 52.3 us vs 84.8 us, 1.6205x; CV rules still prevent a broad dispatch promotion. |
| CUDA Graph fixed | post-route Top-K 2/4/8 fixed replay | `insufficient_evidence` for broad promotion | event-span ratio-of-sums 1.2336x; host fixed-replay speedups 1.24x / 1.21x / 1.25x. |

No candidate is promoted and no default dispatch is modified. `insufficient_evidence` is intentional: WDDM timing produces multi-modal samples on several short shapes, and the aggressive policy requires CV <= 0.10 in every paired process record.

## CUDA Graph host boundary

Graph performance statements use host time-to-solution and include parameter update or graph executable update where applicable. Capture, instantiate, and upload are excluded from steady replay, then reported through amortization.

| Workload | Mode | Host p50 speedup | Setup | Break-even |
|---|---|---:|---:|---:|
| postlogit | fixed graph | 1.6205x | 262.7 us | 9 replays |
| postlogit | parameter update | 1.4611x | 258.3 us | 10 replays |
| postlogit | exec update | 1.4585x | 245.8 us | 10 replays |
| postlogit | LRU cache hit | 1.6298x | 258.9 us | 8 replays |
| post-route Top-K 2/4/8 | fixed graph | 1.2416x / 1.2118x / 1.2481x | 253.9 / 242.9 / 274.0 us | 22 / 17 / 14 replays |

The 17-key LRU cache-miss traces are explicitly `not_a_speed_claim`: they measure setup and eviction behavior rather than a stable fixed-shape replay.

## Profiler diagnosis (not release timing)

NSYS CUDA/NVTX traces used CPU sampling disabled. At T=512, E=64, Top-K=4, K=N=128, the retained post-route chain spends 70.0% (uniform) and 71.3% (Zipf-1.4) of traced GPU time in Grouped GEMM v2. Its token permutation is 11.7%/11.2%, fused histogram+scan 11.1%/10.3%, and unpermute 7.2%/7.2%.

The rejected gather path changes the profile rather than removing work: its descriptor mainloop is 61.5%/64.5%, single-CTA route-prep is 26.7%/23.0%, descriptor prepass is 6.2%/6.4%, and unpermute is 5.6%/6.1%. This is consistent with the release regression, but NSYS durations are diagnostic only.

Targeted NCU detailed for v4A static-t256 reports:

- `grouped_gemm_descriptor_prepass_kernel`: grid 1, block 256, 0.00245 waves/SM, 38 registers/thread, 0.12% SM throughput. It is structurally underfilled; adding threads cannot use more than one SM.
- `grouped_gemm_descriptor_kernel`: grid 340, block 128, 84 registers/thread, 7,692 B shared memory/block, 41.67% theoretical and 35.79% achieved occupancy; L2 hit rate 80.65%, but `smsp__pcsamp_warps_issue_stalled_long_scoreboard=400`, `...barrier=377`, and `...mio_throttle=280` samples.
- The detailed collection reports `global_load_sectors`, `global_store_sectors`, and eligible-warps-per-scheduler as `not_collected`; it reports zero local load/store instructions, so there is no observed register spill in this case.

The highest-confidence next hypothesis is therefore not more descriptor-prepass threads: it is to retain v2 scheduling and reduce the 84-register / barrier-heavy descriptor mainloop or redesign the metadata path so it does not add a serial one-CTA phase.

## Artifacts

- Compact SHA-256 bundle: `compact-evidence.zip` and `compact-evidence.zip.sha256`.
- Promotion table and per-shape SVG heatmap: `compact-evidence/REPORT.md`, `promotion_summary.csv`, and `shape_heatmap.svg`.
- NSYS raw reports and CSV summaries: `profiles/nsys-*`.
- NCU basic and descriptor-only detailed reports: `profiles/ncu-basic` and `profiles/ncu-detailed-descriptor`.
- The first post-route run was stopped by the invalid `cold-scrub` spelling and preserved as `postroute-v4-release-aborted-invalid-cache-mode.jsonl`; it is excluded from every decision.
