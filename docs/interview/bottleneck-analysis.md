# RTX 3080 / SM86 瓶颈分析

## 结论先行

端到端价值最高的热点是 Grouped GEMM，而不是某个最容易写的 metadata kernel。uniform L3 中它占 GPU kernel time `71.6%`，Zipf L3 中占 `86.0%`；Zipf 的 21 次 launch 中有一次 752.849 us 系统长尾，但 median 仍为 22.111 us，因此不能用该 profiler 长尾替代 Release latency。Permute 已表现为高 DRAM 压力，Scan/Fused Histogram→Scan 则是单 CTA underfill。下一轮优化应优先隔离 Grouped v2 的调度与负载均衡变量。

## 环境与证据边界

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`，Release、clean、`-lineinfo`。
- GPU：RTX 3080，CC 8.6，68 SM，10 GiB；CUDA 13.3.73，driver 616.56，NCU 2026.2.1，NSYS 2026.1.3。
- NSYS：`chain_from_logits`，T512/E64/K128/N128，20 warmup 后采集一次 postlogit path；分别使用 uniform 和 Zipf s=1.4。
- NCU：七算子 candidate/baseline 各一个 post-warmup basic launch；只有 Grouped v3/CUTLASS 因 basic 证据不足升级 detailed。
- Profiler duration 只用于机制诊断；所有性能倍率来自未插桩五进程 Release。

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
4. “Pure Permute copy 越宽越能赢”：v2/v3 pure matrix 都低于 retained token-owned；真正的 `1.567x` 来自 full-from-ids 边界减少中间准备开销。
5. “Scan 应该直接用 DeviceScan”：E<=64 时 DeviceScan 只有 naive 的 `0.331x`，workspace 和额外 launch 无法摊销。
6. “Profiler duration 可以直接证明 speedup”：NCU replay/serialization 和 NSYS tracing 会改变时序，正式结论必须回到未插桩 Release。

## 下一轮最多三个实验

### 1. Grouped v2 scheduling-only

- 固定：v2 16x32x16 strict-FP32 mainloop、寄存器/shared 布局、zero workspace 和计时边界。
- 唯一变量：expert tile 到 CTA 的静态映射或轻量 task stealing，目标是减少 empty/skew/tail 的 late wave；不叠加 16x64 tile或 descriptor prepass。
- 预声明矩阵：沿用十 shape，重点观察 uniform T2048、non-aligned、single-hot、many-empty。
- Keep 条件：对 CUTLASS/cuBLAS envelope ratio-of-sums >=1.01、coverage >=60%、最大回退 <=10%，且 Grouped 的 L3 share/absolute Release latency 同向改善。

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
