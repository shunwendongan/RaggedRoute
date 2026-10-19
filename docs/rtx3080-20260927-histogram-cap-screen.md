# RTX 3080 Histogram CTA-cap screen (2026-09-27)

Status: v3 (256 CTAs) and v4 (64 CTAs) rejected for promotion. Both remain
explicit benchmark-only controls; public runtime and Auto remain unchanged.

## Hypothesis and contract

Parent is `cuda_candidate_v2`, with a 128-CTA cap, 256 threads per CTA and
8 input items per thread per grid-stride tile. Only the CTA cap changes.
The E=1 direct-write specialization, small single-CTA overwrite path, middle
naive fallback, integer semantics, workspace (zero) and L2 reset are identical.
The adapter records the candidate's actual CTA cap and global-atomic upper bound.

NCU basic of R=1048576/E=64 parent: SM throughput 14.51%, memory throughput
49.57%, achieved occupancy 30.89%, 21 registers/thread. These counters suggested
testing coverage; they do not establish causation or predict a faster cap.

Measurement: local single RTX 3080, sm_86, seed 20260926, warm cache,
20 warmups, 30 samples, 30 repeats per sample, five independent processes per
variant/shape. Variant order alternated each round. T=R, top_k=1, E=64,
uniform or Zipf(s=1.4). The two experiments were separate paired batches;
their baseline values must not be mixed for ratios. Protocol was **smoke**
with dirty experimental binaries; these are rejection screens, not release
promotion evidence. No overlapping profiler/sanitizer ran during timing.

## Median of five process medians (microseconds)

| R | Distribution | Parent in v3 batch | v3 cap=256 | Parent in v4 batch | v4 cap=64 |
|---:|---|---:|---:|---:|---:|
| 32768 | uniform | 18.09 | 17.97 | 18.12 | 18.38 |
| 32768 | Zipf | 18.76 | 18.62 | 18.02 | 18.60 |
| 262144 | uniform | 18.53 | 18.14 | 18.14 | 19.10 |
| 262144 | Zipf | 18.14 | 18.01 | 18.06 | 19.35 |
| 1048576 | uniform | 18.43 | 21.78 | 18.57 | 31.32 |
| 1048576 | Zipf | 18.31 | 21.44 | 18.65 | 31.06 |

For R<=262144 the v3 experiment has exactly the same grid as its parent;
small differences there are controls for measurement noise, not optimization.
For R=1M v3 is about 17-18% slower and v4 about 67-69% slower. All 120 timed
processes validated exact counts. P95 also regressed at R=1M in both experiments.

## Evidence and limitations

Raw files (local ignored artifacts):

- `out/experiments/20260927-sm86-hist-v3/paired.jsonl` — v2/v3, 60 records.
- `out/experiments/20260927-sm86-hist-v3/paired64.jsonl` — v2/v4, 60 records.
- `out/experiments/20260927-sm86-hist-v3/ctest4.log` — 8/8 tests passing.
- `out/experiments/20260927-sm86-hist-r1m/reports/ncu_basic.ncu-rep`.

The proposed causes (less latency hiding at 64 CTAs, more merge atomics at
256 CTAs) remain hypotheses until matched counter/SASS analysis distinguishes
them. Low achieved occupancy alone is not a reason to maximize grid size.
The benchmark-only code is retained to reproduce these negative results.

## Campaign status correction

Update: [the completed Dense Release matrix](rtx3080-20260927-dense-portfolio-release.md)
rejects the broad selector; `ea400b9` restores original v3 and isolates the
candidate. The paragraph below records the earlier provisional state.

Dense's committed narrow selector in `0b21ab8` is experimental, **not a completed
release promotion**. Earlier cross-session smoke comparisons and three-process
medians do not prove the claimed 17% gain over its entire M>=256,N<=64,K<=256
region. A separate retained-v3 control, accurate selected-path metadata and a
paired release region matrix are still required. The change does not change
public Dense Auto (which remains naive), but does affect explicit v3 and chains
requesting it. Do not describe it as a validated default-path improvement.

Top-K two-warp screens likewise did not establish a reliable gain. The original
v4 source was restored. The observed timing distributions were bimodal; neither
one low run nor the automatic profiler's estimated speedup is promotion proof.
