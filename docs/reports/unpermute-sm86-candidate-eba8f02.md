# Unpermute SM86 candidate 正式证据报告

## 审阅结论

`cuda_warp_token_vec4` **不晋升为生产实现**，保留为 benchmark-only research candidate；
公开 Unpermute dispatch 继续只选择 naive，对 optimized ID fail closed。

实现保持 strict FP32 storage/accumulate、rank 0→1 累加顺序、caller stream、zero workspace、
token-owned 和无输出 atomic。正确性通过，但正式性能未通过稳定性、逐 shape p50 和 p95 门槛。

## 环境与测量合同

- 基线：`main@927c585e031ba6a01e41c01d21db03cb0d1ca9d0`
- 正式 benchmark 构建 SHA：`eba8f025c718`
- 最终接口收口 commit：`ce5a86b674b4652a00ed499201fdc2f0987d7803`
- GPU：NVIDIA GeForce RTX 3080，CC 8.6，68 SM，10 GiB
- Driver/CUDA：591.86 / CUDA 13.3.73
- NCU/NSYS：2026.2.1 / 2026.1.3
- Release 合同：`sm_86 -O3 -lineinfo`，5 个独立进程，20 warmups，至少 30 samples，
  uniform 与 Zipf-1.4，L1/L2 warm 和选定 cold-scrub case
- 对比对象：`cuda_naive` 与适配的 vLLM `finalizeMoeRoutingKernelLauncher`

## 未插桩 Release 结果

下表 ratio 为 baseline latency / candidate latency，是晋升决策唯一使用的 speedup 来源。

| Run | warm L1 几何平均 | warm L2 几何平均 | candidate warm groups 中 CV>0.10 |
|---|---:|---:|---:|
| 正式首轮 | 1.0084x | 1.0655x | 55/64 |
| 自动复测 | 1.0876x | 1.0564x | 63/64 |

两轮平均值按 shape 差异明显，不能据此晋升。首轮最差 L1/L2 p50 speedup 为
`0.2641x/0.5415x`，复测为 `0.4461x/0.4924x`；最大 p95 ratio 分别为
`7.1419/6.0640` 与 `3.6254/5.3753`，远超允许的 `1.03`。按规则复测一次后仍然
高噪声的 pair 标记为 inconclusive。

L3 只替换 Unpermute 的链路结果为 `1.0016x`，p95 ratio `0.9466`，baseline/candidate
CV 为 `0.4348/0.3052`：没有统计可信的回退，也没有统计可信的加速。

每个正式 pair 的 `throughput-comparison.csv` 记录 p50/p95 latency、CV、logical bytes、
effective GB/s、tokens/s 和 output elements/s。吞吐由固定工作量和同一个未插桩 p50
计算得到，因此吞吐 ratio 与 latency speedup 相同，不作为独立证据。两轮表各包含 72 个
candidate-vLLM pair，覆盖 warm 和选定 cold-scrub 的 L1/L2 case。

## Profiler 诊断

Profiler duration 仅用于归因，不用于生产 speedup。

- warp path（`T=1024,N=256`）：grid 256、block 128、0.31 waves/SM、34 registers/thread、
  achieved occupancy 25.36%、SM throughput 2.61%、memory/DRAM throughput 43.30%
  （`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`）。
- CTA path（`T=64,N=1024`）：grid 64、block 256、0.16 waves/SM、34 registers/thread、
  16 B static shared memory、achieved occupancy 14.94%、SM throughput 1.62%、
  memory/DRAM throughput 21.72%。
- CTA NSYS 在 21 次 post-warmup launch 上的 median 为 1.792 us；同一大 N 诊断 shape 的
  adapted vLLM 之前为 2.016 us。该数据不覆盖 Release 矩阵结论。
- 两条对齐 fast path 的 SASS 均含 `LDG.E.128` 和 `STG.E.128`；`cuobjdump
  --dump-resource-usage` 显示 34 registers/thread、`LOCAL:0`、`STACK:0`，筛选结果没有
  `LDL/STL`。
- basic profile 未采集 long-scoreboard、eligible-warp、sectors/request；这些字段是
  `not_collected`，不是零。现有 launch/occupancy/throughput 证据足以记录拒绝原因，
  因此没有进行昂贵的 detailed/full replay。

## Correctness 与 API 结果

- CTest：8/8 通过。
- Compute Sanitizer：memcheck、initcheck、racecheck、synccheck 全部通过。
- Candidate matrix 覆盖 Top-K fallback、tail、pointer offset 1、uniform/Zipf/single-hot/
  round-robin、stream 和 redzone；smoke 覆盖 aligned warp/CTA、unaligned 和 Top-4 fallback。
- 公开 `select_kernel` 与 `unpermute` 保持 naive-only；candidate 仅由 benchmark adapter 直接
  调用，并标记为 `in_tree_cuda_research`。

## 证据包

完整证据包见
[20260802T193000Z-eba8f025c718-unpermute-sm86-research-v1](artifacts/20260802T193000Z-eba8f025c718-unpermute-sm86-research-v1/)。
其中包含 raw JSONL、aggregate/comparison、归一化 latency/throughput 表、manifest、sanitizer
日志、NSYS/NCU CSV/JSON、SASS 证据和 SHA256。`.ncu-rep`、`.nsys-rep`、SQLite 与构建产物
均已排除。
