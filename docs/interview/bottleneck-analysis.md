# RTX 3080 / SM86 瓶颈分析

## 结论先行

端到端价值最高的热点是 Grouped GEMM，而不是某个最容易写的 metadata kernel。统一 portfolio 的 uniform/Zipf L3 中它占 GPU kernel time `71.6%/86.0%`，因此后续实验聚焦该算子。v5/v6 先证明 direct grid 和大 tile可降低调度与 request amplification；最新 V9 `32x64` 再在 narrow-N shape 对 CUTLASS 达 `1.8819x/1.4366x/1.3661x`，代表点相对 V5 fallback 将 CTA 降低 75%、global load/store requests 降低 37.9%/50.5%。但 K256、N128、non-aligned 仍显著回退，V10 wave-aware selector 也被正式拒绝，所以 V9 是有边界的 research candidate，不是通用 Auto。Permute 是高 DRAM 压力，Scan/Fused Histogram→Scan 则是单 CTA underfill。

## 环境与证据边界

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`，Release、clean、`-lineinfo`。
- GPU：RTX 3080，CC 8.6，68 SM，10 GiB；CUDA 13.3.73，driver 616.56，NCU 2026.2.1，NSYS 2026.1.3。
- NSYS：`chain_from_logits`，T512/E64/K128/N128，20 warmup 后采集一次 postlogit path；分别使用 uniform 和 Zipf s=1.4。
- NCU：七个语义算子加融合 primitive 的 candidate/strong-reference 各一个 post-warmup basic launch，共 16 个 basic；只有当时的 Grouped v3/CUTLASS 因 basic 证据不足升级 detailed。这里的 profile variant 是 `9732a03` 冻结时的诊断选择，不保证等于每项 primary Release winner：Top-K strong-reference profile 使用有限子域 vLLM、Permute profile 使用 v3 pure path、Grouped profile 使用 v3，而正式矩阵结论分别来自 exact naive、v2 full-from-ids 和 v2/library envelope。后续 strongest Grouped V9 另有 CUTLASS/V5/V9 basic 与 V5/V9 detailed，不能把旧 v3 profile 冒充 V9 归因。
- Profiler duration 只用于机制诊断；所有性能倍率来自未插桩五进程 Release。

Grouped-only follow-up 另固定在 clean SHA `c2205ed1ba1063fccce3cd417fd671798dbfb66f`，沿用 RTX 3080 / strict FP32 合同，对 v2/v5/v6/CUTLASS/cuBLAS 执行 10 shape、5 process、20 warmup、30 samples/process 的未插桩 Release 复测。它只更新 Grouped GEMM，不改写其他六算子的 `9732a03` 统一矩阵。完整证据见 [v5/v6 compact report](../reports/compact/20260830-c2205ed-grouped-v6/REPORT.md)。

最新 V9/V10 follow-up 固定在 clean SHA `dea7c066a83a5df700aa60c03fd51446c6b4c5e5`：15 shape、5 process、375/375 validation 通过。一个 CUTLASS tail process `CV=0.5041` 越过 0.50 ceiling，原样保留，因此完整 aggregate 只作为 research trend；下方 V9 headline shape 均 5/5 process pairs 同向且未越过 ceiling。完整证据见 [V9/V10 compact report](../reports/compact/20260830-dea7c06-grouped-v9-v10/REPORT.md)。

## V9 32x64：减少 over-partitioning，而不是追 occupancy

| Metric | V5 16x32 fallback | V9 32x64 | CUTLASS Grouped | 解释 |
|---|---:|---:|---:|---|
| Grid / block / waves per SM | 1280 / 128 / 2.689 | 320 / 256 / 1.569 | 68 / 256 / 1.000 | V9 合并过细工作；CUTLASS 在窄 N 下只有一 wave |
| Registers/thread | 68 | 78 | 144 | V9 增加 accumulator state，但无极端 register pressure |
| Shared memory/block | 7,680 B | 14,336 B | 17,680 B | 双缓冲 `32x64x16` staging 成本可控 |
| Achieved occupancy | 48.44% | 41.54% | 16.66% | V9 occupancy 更低仍更快，否定 occupancy 单指标优化 |
| Global load requests | 75,824 | 47,096 | basic 未采集 | V9 `-37.9%` |
| Global store requests | 16,552 | 8,192 | basic 未采集 | V9 `-50.5%` |
| DRAM read/write | 6.31 / 2.94 MB | 6.42 / 2.61 MB | basic 未采集 | raw DRAM read 未下降，不能包装成带宽减少 |
| L2 hit rate | 80.64% | 64.94% | basic 未采集 | V9 更快不是因为更高 cache hit |
| Local load/store | 0 / 0 | 0 / 0 | basic 未采集 | 没有 spill 证据 |

两组独立信号支持机制判断：第一，V9 将 CTA 与 global requests 大幅减少；第二，它的 occupancy 和 L2 hit 反而更低、DRAM read 近似不降，却在未插桩 Release 中把代表 T4096/N64 从 CUTLASS 的 39.1680 us 降到 28.6720 us，并相对 V6/V5 portfolio 获得 `1.0999x`。因此主收益是减少 `16x32` fallback 的 over-partitioning、重复请求和 scheduling/tail cost。CUTLASS 的 `128x128` 通用 tile 只产生 68 CTA，144 registers/thread 把理论/实际 occupancy 限在约 16.7%，解释了其窄 N underfill。

V9 的边界同样清楚：K256/N64 为 `0.9072x`，T2048/N128 为 `0.7736x` p50 / `0.5459x` p95，non-aligned 为 `0.7508x`。V10 仅根据 68-SM CTA window 在 V9/V6 间切换，但对 V9 ratio-of-sums `0.9775x`，说明 dirty screen 得到的区间不能直接当发布 selector；下一次 dispatch 必须使用新的 held-out matrix，而不是事后移动阈值。

## Grouped v6 follow-up：修复 request amplification 后的新瓶颈

| Metric | v5 16x32 direct-grid | v6 32x128 hybrid | CUTLASS Grouped | 解释 |
|---|---:|---:|---:|---|
| Grid / block / waves per SM | 1536 / 128 / 3.227 | 192 / 256 / 0.941 | 64 / 256 / 0.941 | v6 减少 CTA 数，但更容易 underfill |
| Registers/thread | 68 | 72 | 144 | v6 没有出现极端 register 膨胀 |
| Shared memory/block | 7,680 B | 22,528 B | 17,680 B | 大 tile 增加 staging 资源 |
| Achieved occupancy | 48.71% | 31.60% | 17.17% | occupancy 仍不能单独决定胜负 |
| Issue active | 40.97% | 31.28% | 53.65% | v6 仍难以持续发射 |
| Global load requests | 94,144 | 53,296 | 57,200 | v6 比 v5 下降 43.4% |
| Global store requests | 16,752 | 4,096 | 16,384 | v6 比 v5 下降 75.5% |
| DRAM writes | 3,058,560 B | 527,616 B | 2,293,376 B | v6 比 v5 下降 82.7% |
| Local load/store | 0 / 0 | 0 / 0 | 0 / 0 | 没有 spill 证据 |

两组独立信号支持该诊断：第一，v6 明显减少 global request 与 DRAM write，证明 `32x128` tile 修复了 v5 的 request amplification；第二，它仍只有 0.941 waves/SM、31.28% issue active，且 SM active-cycle minimum 比均值低 54.17%，同时 T2048 Release 对 CUTLASS 只有 `0.7534x`。所以下一轮不应继续盲目放大 tile，而要限定 v6 selector 适用区间并减少最后 partial wave/跨 SM 不均。

归因审计进一步收紧了结论：v6 的 `ceil-average >= 32`、skew 和对齐 selector 会让 uniform T512/E64、single-hot、many-empty 分别回退到 v5/v2，它们的 `1.2257x/1.7762x/1.2478x` 是 hybrid portfolio 的收益。wide kernel 直接激活的两个正式 shape 是 uniform T512/E16/N256（`1.0120x`）和 uniform T2048/E64/N128（`0.7534x`）。这说明 selector/fallback 本身是设计的一部分，但不能用 variant 外层名称替代实际 kernel 归因。

## V8 16x128：underfill 假设的反证

V8 保持 v6 的 N128、K16、256 threads、两级 `cp.async`、direct grid、selector、strict FP32、zero workspace 和 fallback 不变，只把 tile-M 32→16、thread micro-tile `4x4`→`2x4`。在四个预声明 balanced shape 的三进程 diagnostic screen 中，48/48 validation 通过，最大 CV `0.3278`、0 个超过 0.50；v8 对 v5/v6/CUTLASS 的 ratio-of-sums 约为 `1.06x/0.93x/0.81x`。它只在 T1024 对 v6 为 `1.17x`，T2048/N128、T2048/N256、T4096 分别为 `0.99x/0.92x/0.81x`，关键退化均 3/3 进程同向。

因此“增加 CTA 数量”不是充分条件。M tile 减半会让每个新 row tile 重复加载相同的 `Kx128` weight tile；在更大 T/N 上，这个 traffic/reuse 代价大于更高 wave coverage。V8 是 dirty-tree screening，只用于拒绝假设，不进入正式 speedup headline，也不需要用 profiler duration包装结论。

## NSYS 关键路径

| Kernel stage | Uniform share / median | Zipf share / median | 判断 |
|---|---:|---:|---|
| Grouped GEMM v2 | 71.6% / 23.744 us | 86.0% / 22.111 us | 绝对热点；Zipf 含一次系统长尾 |
| Token Permute token-owned | 8.3% / 2.752 us | 4.1% / 2.720 us | 次热点，但总占比远低于 Grouped |
| Top-K v4 | 7.0% / 2.304 us | 3.6% / 2.368 us | 大 E 有局部优化空间 |
| Fused Histogram→Scan | 6.6% / 2.176 us | 3.3% / 2.176 us | 单 CTA underfill；绝对时长小 |
| Unpermute vec4 | 6.5% / 2.176 us | 3.0% / 1.920 us | 窄 N 局部收益，不是首要链路热点 |

## NCU headline

| Kernel | Waves/SM | Achieved occupancy | Registers | SM / Memory / DRAM | 主要判断 |
|---|---:|---:|---:|---:|---|
| Dense v3 1024³ | 1.506 | 34.63% | 85 | 73.26% / 74.50% / 15.06% | 有计算与 cache reuse，但仍落后 cuBLAS 调度/数据复用 |
| Top-K v4 T2048/E64 | 0.314 | 27.70% | 18 | basic 低于峰值 | 行内归约/小 grid，收益来自算法工作量而非 occupancy |
| Histogram v2 R1M/E64 | 0.314 | 30.31% | 21 | DRAM 38.64% | 中等并行度；E=1 fast path 是不同机制 |
| Scan naive E64 | 0.000919 | 2.08% | 20 | 很低 | 单 block/单 thread launch-bound |
| Fused Hist+Scan | 0.002451 | 15.91% | 22 | 很低 | 单 CTA underfill |
| Permute v3 T4096/K1024 | 1.882 | 58.76% | 34 | DRAM 86.85% | 明确 bandwidth-bound |
| Grouped v3 Zipf | 1.000 | 30.26% | 96 | 35.39% / 49.76% / 25.66% | issue/stall 受限，不是 occupancy 不足 |
| Unpermute vec4 T1024/N256 | 0.314 | 26.90% | 34 | Memory/DRAM 45.72% | 中等 memory pressure + grid underfill |

## Grouped GEMM detailed 对照

| Metric | v3 16x64 candidate | CUTLASS Grouped | 解释 |
|---|---:|---:|---|
| Grid / block | 136 / 256 | 68 / 256 | 两者都是 one wave/SM 量级 |
| Registers/thread | 96 | 144 | CUTLASS registers 更多但仍更快 |
| Shared memory/block | 12,048 B | 17,680 B | 更少 shared 也未自动带来高效发射 |
| Theoretical / achieved occupancy | 33.33% / 30.26% | 16.67% / 16.98% | candidate occupancy 更高，不是胜负决定项 |
| SM / DRAM throughput | 35.39% / 25.66% | 34.79% / 26.00% | 峰值接近，需看 scheduler/stalls |
| L2 hit rate | 85.45% | 50.71% | 高 hit rate 没有抵消发射与同步损失 |
| Issue active | 23.87% | 34.79% | candidate 更难持续发射 eligible instruction |
| MIO throttle samples | 716 | 62 | shared/memory instruction pipeline 压力显著 |
| Barrier samples | 452 | 60 | CTA 内同步或不均衡更重 |
| Long scoreboard samples | 404 | 138 | 仍存在依赖等待/长延迟暴露 |
| Local load/store | 0 / 0 | 0 / 0 | 没有 register spill，不能把问题归因于 local memory |

根因判断至少由两组独立信号支持：v3 的 occupancy 和 L2 hit 都高于 CUTLASS，但 issue active 明显更低，且 MIO/barrier/long-scoreboard 同时更高；因此主要损失是 16x64 tile 带来的资源生命周期、同步与发射效率，以及 ragged task 的负载均衡，而不是“occupancy 太低”或“发生 spill”。

## 被证据否定的假设

1. “更高 occupancy 一定更快”：Grouped v3 的 30.26% 明显高于 CUTLASS 的 16.98%，Release 仍失败。
2. “更宽 tile + 更多 `cp.async` 能摊薄开销”：v3 16x64 比 v2 16x32 更弱，96 registers 与 12,048 B shared 扩大了 live state，issue activity 下降。
3. “descriptor queue 能自动解决 ragged imbalance”：v4A queue-1024 对 external envelope 仅 `0.7799x`，且 prepass 本身单 CTA underfill。
4. “继续放大到 64x128 能进一步摊薄 request”：v7 dirty smoke 在 uniform T2048 比 v6 慢约 6.6%，accumulator live range 和 tail-row 同步代价抵消了局部 T512/E16 收益，因此已撤回。
5. “把 tile-M 减半就能用更多 CTA 修复 underfill”：v8 `16x128` 对 v6/CUTLASS aggregate 只有约 `0.93x/0.81x`；额外 row tiles 重复加载大 weight tile，T4096 对 v6 退化到 `0.81x`。
6. “Pure Permute copy 越宽越能赢”：v2/v3 pure matrix 都低于 retained token-owned；真正的 `1.567x` 来自 full-from-ids 边界减少中间准备开销。
7. “Scan 应该直接用 DeviceScan”：E<=64 时 DeviceScan 只有 naive 的 `0.331x`，workspace 和额外 launch 无法摊销。
8. “Profiler duration 可以直接证明 speedup”：NCU replay/serialization 和 NSYS tracing 会改变时序，正式结论必须回到未插桩 Release。

## 下一轮最多三个实验

### 1. Grouped V9 held-out dispatch validation

- 固定：V9/V6/CUTLASS 实现、strict FP32、zero workspace、L2 边界与五进程 protocol；不再修改 kernel。
- 唯一变量：预先声明 dispatch region。候选区域是 aligned `N=64,K<=128` 的 V9，K256/N128/non-aligned 使用强 library reference 或保留 fallback；不得用当前 15 shape 的事后 candidate envelope充当验证。
- 新证据：围绕 T/E/K/N 边界生成未参与 V10 selector 的 held-out uniform/Zipf/empty/skew shape，记录实际 kernel path；任何 `CV>0.50` group 保留并降级对应结论。
- Keep 条件：held-out ratio-of-sums >=1.03、所有关键反例不超过预声明回退、至少 80% shape 和多数 process pairs 获益。满足前仍只做 benchmark-only hybrid；公共 Auto 需要真实 trace 与 L3 复测。

### 2. Top-K E64 / large-T 明确分区

- 固定：v4 kernel 和完整 tie/NaN/selected-softmax 合同。
- 唯一变量：只评估预声明的 E64、T>=512 显式 dispatch 区间，不用事后 envelope。
- 风险：小 T 启动开销、E bucket 边界和跨进程方向漂移。
- Keep 条件：区间内五进程全 shape 获益，区间外显式回退 naive；本轮仍不直接改 Auto，先形成可 review 的 dispatch 证据。

### 3. Full-from-ids Permute 的 L3 归因

- 固定：v2 fused route preparation + tile4 copy，与 adapted vLLM 相同输入/输出合同。
- 唯一变量：在 postroute L3 中替换 Permute stage，分开记录 metadata bytes、payload bytes、launch 数和 downstream Grouped 输入一致性。
- 目标：确认 L2 `1.567x` 能否转化为链级收益，而不是被 Grouped 热点吞没。
- Keep 条件：L3 ratio-of-sums >=1.03、无 downstream 语义变化、workspace 增长在预声明范围内。

这三个方向均使用 Ampere 可用机制：warp shuffle、shared memory、aligned vector access、`cp.async` 和传统 work scheduling；不使用 Hopper TMA/WGMMA 或 Blackwell TMEM/tcgen05。
