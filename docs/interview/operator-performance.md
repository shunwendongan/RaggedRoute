# 七算子性能卡片

## 统一口径

- Evidence SHA：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`，clean Release。
- Hardware：NVIDIA GeForce RTX 3080，compute capability 8.6，68 SM，10 GiB；CUDA compiler 13.3.73，driver 616.56。
- Protocol：5 个独立进程，20 warmup，30 samples/process，seed `20260828`，warm cache；不同算子只使用 suite 预声明的 kernel repeats。
- 正式速度来自未插桩 CUDA Event；NSYS/NCU duration 仅诊断。`CV<=0.50` 是本次 Windows/WDDM 作品集证据上限，`CV>0.10` 仍披露为风险。
- ratio-of-sums 表示把每个 shape 的 baseline p50 求和后除以 candidate p50 求和；geomean 表示 shape 等权；coverage、最大回退和五进程方向必须同时阅读。

所有逐 shape 数值见 [release_pairs.csv](../reports/compact/20260829-9732a03-interview-portfolio/release_pairs.csv)，聚合见 [operator_summary.csv](../reports/compact/20260829-9732a03-interview-portfolio/operator_summary.csv)。

## 1. Dense GEMM

- 语义：row-major `C = A x B`，strict FP32，`alpha=1,beta=0`；对应 router projection。
- 当前 Auto：`cuda_naive`。
- 最强树内候选：`cuda_register_tiled_v3_64x32_async`，64x32 register tile、Ampere `cp.async` 两级 global-to-shared staging。
- 强基线：每个 shape 选择更快的 cuBLASLt/cuBLAS strict-FP32 路径；不拿 TF32/Tensor Core 合同偷换分母。
- 矩阵：256³/512³/1024³ ratio-of-sums `0.8716x`，geomean `0.9376x`，1/3 获益；分别为 `1.0095x/0.9588x/0.8514x`。
- Profiler：1024³ candidate 为 1.506 waves/SM、85 registers/thread、34.63% achieved occupancy、SM 73.26%、memory 74.50%、DRAM 15.06%；cuBLAS SGEMM 为 126 registers/thread、24.10% occupancy。较高 occupancy 没有转化为库级调度和数据复用效率。
- 决策：局部 winner，不晋级。简历可讲 `cp.async` pipeline 和公平 cuBLAS 反例，禁止写“整体超过 cuBLAS”。

## 2. Top-2 Gate

- 语义：每个 token 在 E 个 router logits 中选择 Top-2，lower expert id 决定 tie，NaN 按合同处理，并对选中二项做 softmax。
- 当前 Auto：`cuda_naive`。
- 最强候选：`cuda_local_pair_two_reduce_top2_v4`；寄存器局部 pair、float2/float4 row packing、两个 subgroup reduction。
- 严格基线：exact-contract naive，`T={32,128,512,2048,4096}`、`E={2,4,8,16,32,64}`，L2 完整 operator。
- 严格矩阵：ratio-of-sums `1.0972x`，geomean `1.0859x`，21/30 shape 获益；五进程方向 109/150 次获益。E64/T4096 `1.6934x`，E64/T512 `1.6188x`，E64/T2048 `1.3663x`；E32/T32 最大回退 10.88%。
- 外部参考：在有限 random-input L1 子域，对最快 vLLM/CUB envelope 为 `1.0146x` ratio-of-sums、`1.0144x` geomean、18/30 获益；该结果不覆盖完整 tie/NaN/normalization 合同。
- Profiler：T2048/E64 v4 为 0.314 waves/SM、27.70% occupancy、18 registers/thread；当前大 E 收益来自更有效的行内并行和归约，不是高 resident-wave 数量。
- 决策：高收益 local/matrix trend，但按最大回退门禁不改 Auto。简历 headline 必须同时给出 E64 大 T 适用区间。

## 3. Expert Histogram

- 语义：统计每个 expert 的 route 数，exact int32；L2 包含必要 counts reset。
- 当前 Auto：shape-dispatched shipping `cuda_candidate`（v1）。
- 最强显式候选：`cuda_candidate_v2`。`E=1` 时 route IDs 不影响结果，直接写 `counts[0]=R`；其余 shape 复用 single-CTA shared / block-private dispatcher。
- 强基线：每 shape 最快的 shipping v1、CUB DeviceHistogram、naive。
- 15-case 矩阵：ratio-of-sums `1.1016x`，geomean `1.1588x`，9/15 获益，最大回退 5.56%，最大 CV 0.4503。对 shipping v1 为 `1.1070x`、10/15 获益。
- 最大收益：`R=1M,E=1` 为 `4.3026x`，`R=65536,E=1` 为 `1.7069x`，`R=4096,E=1` 为 `1.3024x`，`R=128,E=1` 为 `1.1496x`。
- Profiler：代表 `R=1M,E=64` block-private kernel 为 0.314 waves/SM、30.31% occupancy、21 registers/thread、DRAM 38.64%；`E=1` 收益属于语义退化 fast path，不应由该 E64 profile 外推。
- 决策：完整矩阵 research winner；本轮按“不改 Auto/实现”约束只保留显式 strongest candidate。

## 4. Exclusive Scan

- 语义：把 `E<=64` 个 expert counts 转为 offsets；exact int32 tiny metadata。
- 当前 Auto / retained candidate：`cuda_naive`；没有保留的自研 CUDA candidate。
- 参考矩阵：E={1,31,32,33,64}。CUB BlockScan 对 naive ratio-of-sums `1.0249x`、4/5 获益；WarpScan 在 E<=32 子域为 `1.0290x`、3/3 获益；DeviceScan 只有 `0.3310x`。
- Profiler：naive 单 block/单 thread、0.000919 waves/SM、2.08% occupancy；这是 launch/underfill 问题，不是大数组 scan throughput 问题。
- 决策：使用简单 naive 或库 warp/block reference 都只带来微小差距，不值得为 E<=64 引入复杂 standalone 自研路径。Histogram→Scan fusion 作为独立 primitive 讨论。

## 5. Token Permute

- 语义：按 expert offsets/rank 重排 token payload，并输出 route mapping。必须区分“已有 mapping 的 pure copy”和“从 expert IDs 开始的 full-from-ids”。
- 当前 Auto：`cuda_naive`。
- 最强简历候选：`cuda_candidate_v2_from_ids`；fused counts/exclusive-scan/cursor preparation 后进入 tile4 Top-2 copy。v3 使用 tile2，完整矩阵略弱。
- full-from-ids 强基线：adapted `vllm_moe_permute`，5 case。v2 ratio-of-sums `1.5671x`、geomean `1.5885x`、5/5 获益、五进程方向 25/25，一致区间 `1.2896x–1.8501x`。v3 为 `1.5089x`。
- pure-permute 消融：v2 对 retained token-owned 为 `0.9841x`、11/28 获益；v3 为 `0.9892x`、11/28 获益。不能把 full operator 收益写成 copy kernel 普遍更快。
- Profiler：代表 v3 tile2 为 1.882 waves/SM、58.76% occupancy、34 registers/thread、DRAM 86.85%，明确 bandwidth-bound；继续增加 occupancy 不是主方向。
- 决策：full-from-ids boundary 是完整矩阵 research winner；pure path 不晋级，Auto 不变。

## 6. Grouped GEMM

- 语义：对每个 expert 的 ragged token segment 执行 strict-FP32 GEMM；覆盖 tail、empty、uniform、Zipf、single-hot、non-aligned 与较大 shape。
- 当前 Auto：`cuda_naive`；所有 SM86 candidate 都是 benchmark-only research path。
- 证据演进：统一七算子 clean `9732a0343c60f869fc4166a0cc3cabba2fd67bbb` 中 v2 对 library envelope 为 `0.8697x`；clean `c2205ed1ba1063fccce3cd417fd671798dbfb66f` 的 v5/v6 follow-up 将其推进到 `0.9946x`；最新 clean `dea7c066a83a5df700aa60c03fd51446c6b4c5e5` 固定 15 shape、5 process 评估 V9/V10，不改写其他六算子矩阵。
- 最强树内候选：`cuda_grouped_sm86_fp32_v9_balanced_32x64`。它保持 V6 的 tile-M=32、K16、256 threads、direct grid、两级 `cp.async`、strict FP32 和 zero-workspace fallback chain，只把 tile-N 128→64、每线程 outer product `4x4`→`4x2`。相比被否定的 V8 `16x128`，V9 从 N 方向增加 CTA，不为每个额外 M tile 重复加载完整 `Kx128` weight tile。
- 强基线：每 shape 最快 CUTLASS Grouped / cuBLAS per-active-expert envelope；15 个 shape 实际均由 CUTLASS SIMT FP32 胜出。V9 对该 envelope 的 ratio-of-sums `1.0916x`、geomean `1.1397x`、11/15 p50 获益、53/75 process pairs 获益；对 V6 portfolio 为 `1.0486x`、12/15 获益。一个 CUTLASS tail process `CV=0.5041` 超过 0.50 ceiling，因此全矩阵只写 research trend，不写 promotion-grade overall win。
- 简历安全的 V9 kernel 结果：uniform `T512/E32/K128/N64` 对 CUTLASS `1.8819x`，Zipf1.4 `T2048/E64/K128/N64` 为 `1.4366x`，uniform `T4096/E64/K128/N64` 为 `1.3661x`；三者均真正执行 `32x64` kernel、5/5 process pairs 同向且未越过 CV ceiling。
- 反例：K256/N64 为 `0.9072x`，T2048/N128 为 `0.7736x` p50 / `0.5459x` p95，non-aligned K127/N129 为 `0.7508x`。因此 V9 是 narrow-N、moderate-K 组件，不是 CUTLASS 通用替代。
- Profiler：代表 T4096/N64 中 V6 实际回退 V5 `16x32`。V9 将 CTA 从 1280 降到 320，global load/store requests 从 75,824/16,552 降到 47,096/8,192（`-37.9%/-50.5%`），local load/store 均为 0。Achieved occupancy 从 48.44% 降到 41.54% 仍更快，证明机制是减少 over-partitioning、重复 request 和调度/尾波成本，而不是追 occupancy；CUTLASS 仅 68 CTA、约 16.66% occupancy，暴露通用 `128x128` tile 在窄 N ragged shape 的 underfill。
- 失败消融：v3 `16x64` 资源/发射代价过高；v4A descriptor queue 增加 prepass/mainloop；v7 `64x128` T2048 退化约 6.6%；v8 `16x128` 对 v6/CUTLASS 约 `0.93x/0.81x`；V10 `cuda_grouped_sm86_fp32_v10_wave_aware_portfolio` 只改 68-SM CTA-window selector，但对直接 V9 只有 `0.9775x` ratio-of-sums，正式拒绝。V10 的 T512/E32 成绩走 V6 fallback，不能归因给 V9。
- 决策：V9 是当前 strongest measured in-tree Grouped candidate；V10 selector 保留为失败 STAR。两者均 benchmark-only，`Auto` 与公共 API 不变。证据见 [V9/V10 compact report](../reports/compact/20260830-dea7c06-grouped-v9-v10/REPORT.md)、[v5/v6 report](../reports/compact/20260830-c2205ed-grouped-v6/REPORT.md) 与 [Grouped 性能记录](../grouped_gemm/performance-record.md)。

## 7. Unpermute

- 语义：依据 route mapping 将 expert 输出还原到 token 顺序，并按 Top-2 gate weights 加权归约。
- 当前 Auto：`cuda_naive`。
- 最强 retained candidate：`cuda_warp_token_vec4`，一 warp/token 的对齐 vec4 路径并保留 tail fallback。
- 强基线：每 shape 最快 adapted vLLM / naive envelope，`T={64,512,1024,4096}`、`N={64,128,256,1024}`、uniform/Zipf。
- 32-case envelope：ratio-of-sums `1.0154x`、geomean `1.0172x`、12/32 获益、最大回退 8.03%。对 vLLM 单独为 `1.0369x`、`1.0458x`、15/32 获益。
- 局部 winner：Zipf T4096/N128 对 envelope `1.2892x`，uniform T4096/N128 `1.2487x`，Zipf T4096/N64 `1.2405x`；优势集中在中大型 T、窄 N。
- Profiler：candidate 为 0.314 waves/SM、26.90% occupancy、34 registers/thread、Memory/DRAM 45.72%；vLLM reference 为 2.51 waves/SM、31.56% occupancy、40 registers/thread。candidate 不是靠更高 occupancy，而是依赖窄 N 下的 token-owned vector traffic。
- 决策：局部 winner，不足以支撑全矩阵 Auto dispatch；保留 benchmark-only candidate。

## 外部基线边界

- cuBLAS/cuBLASLt：只比较 strict FP32 与相同 row-major 语义；不把 TF32/Tensor Core 结果当等价 baseline。
- CUTLASS：Grouped GEMM 使用 v4.6.1 strict-FP32 Grouped；cuBLAS per-expert 同时作为另一种强 reference。
- CUB：Histogram 和 Scan 的库 primitive；DeviceScan 在 tiny E 上的 launch/workspace 开销是结果的一部分。
- vLLM：源码适配后只在声明的 mapping、normalization 和 input 子域可比。Top-K external 不具完整合同等价性；Permute/Unpermute 明确写出完整边界。
- Triton：历史 L3 只作为跨 backend 诊断，不进入本轮七算子强基线排名。

## 不混入七算子排名的结果

- Histogram→Scan fusion：跨算子 metadata primitive；本轮 NCU 代表 launch 为 1 CTA、0.002451 waves/SM、15.91% occupancy，属于 underfill。历史性能结论单列在 [fusion report](../reports/rtx3080-histogram-scan-fused-sm86-v2.md)。
- CUDA Graph fixed replay：只报告 setup 完成后的 host time-to-solution，postlogit `1.6205x`、postroute `1.2336x`；不是 kernel speedup，也不进入 `KernelFamily::kAuto`。见 [Graph report](../reports/rtx3080-sm86-v4-graph-promotion.md)。
