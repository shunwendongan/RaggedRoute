# CUDA / MoE / Benchmark 高频追问题库

以下回答按现场口语长度压缩。回答倍率时一定先说边界，再说数字。

## 项目与合同

### 1. 这个项目最核心的贡献是什么？

不是某一个 kernel，而是一条七阶段 Top-2 MoE CUDA 性能工程闭环。我把公共 API、CPU oracle、外部强基线、五进程 Release、NSYS/NCU 和失败决策统一起来，既得到 Histogram、Top-K、full-from-ids Permute 的可复现收益，也能解释 Dense/Grouped 为什么没有全面超过库。

### 2. 为什么做七阶段流水线，不做七个独立 demo？

MoE 算子之间通过 counts、offsets、route mapping 和 expert segments 耦合。独立 kernel winner 可能增加准备、workspace 或 launch，放到 L3 反而变慢。两条 L3 入口让我能区分 router projection 和 postlogit route path，并用 NSYS按真实阶段占比排序优化价值。

### 3. 你的 public runtime 和 benchmark-only 路径有什么区别？

Public wrapper 检查 dtype/layout/workspace，使用 caller stream，hot path 不分配、不无条件同步；benchmark-only cuBLAS/CUTLASS/CUB/vLLM 和 research IDs 用于公平 A/B，不自动进入 `KernelFamily::kAuto`。显式可调用不等于默认晋级。

### 4. 为什么固定 strict FP32？

为了让自研 CUDA core kernel 和 cuBLAS/CUTLASS 的数值合同一致。TF32/Tensor Core 可以更快，但它改变乘法精度，不能作为 strict-FP32 的公平分母。项目未来可以另开 TF32/FP16 合同，而不能混在同一排名。

## Benchmark 设计

### 5. 为什么同时报告 ratio-of-sums 和 geomean？

Ratio-of-sums 更接近把矩阵中各 shape 延迟相加后的总时间，较大延迟 shape 权重更高；geomean 给每个 shape 等权，能防止一个大 shape 完全掩盖其余 case。我还同时报告 coverage 和最大回退，避免两种 aggregate 都隐藏尾部反例。

### 6. 为什么不用单次最快值？

单次 minimum 对调度抖动、缓存和时钟最敏感，也最容易被挑选。正式结果用五个独立进程，每进程 warmup 后采 30 samples，以 process median 和全样本 p50/p95 汇总，并保留 raw samples，不手工删离群点。

### 7. 为什么把 CV 阈值从 0.10 放宽到 0.50？

这是用户授权的 Windows/WDDM 作品集 evidence policy，不是生产 SLA。`CV>0.10` 仍公开为稳定性风险，但不再单独把所有结果降级；只有超过 0.50 才因稳定性判证据不足。结论仍要通过完整进程数、ratio-of-sums、geomean、coverage、最大回退和跨进程方向共同约束。

### 8. CV 高时为什么还能谈性能？

可以谈中心趋势和适用区间，但不能谈严格 tail SLA。比如 Histogram v2 最大 CV 0.4503，仍有五进程完整记录和完整矩阵 aggregate；我把它写成 research winner，并明确 WDDM 风险，不外推生产稳定性。

### 9. 为什么 NCU/NSYS duration 不能当 speedup？

NCU 会 replay、序列化并改变 cache/clock 行为，NSYS tracing 也有扰动。它们回答“时间花在哪里、为什么慢”，正式倍率必须来自未插桩 Release 的同边界 CUDA Event A/B。

### 10. fastest baseline envelope 会不会 cherry-pick？

Envelope 只在预声明的可比基线集合中逐 shape 选择最快者，是让 candidate 面对更强分母，不是帮 candidate 挑弱对手。它和单 shape candidate envelope不同；后者只能作为诊断，不能包装成一个真实可部署 variant。

## CUDA 与 SM86

### 11. RTX 3080 / SM86 对设计有什么影响？

实卡是 68 SM、10 GiB。短 kernel 要先看 grid 和 waves，ragged kernel 要看 skew/tail；可用 `cp.async`、warp shuffle、shared memory 和传统 CUDA Graph。不能使用 Hopper TMA/WGMMA 或 Blackwell TMEM/tcgen05。10 GiB 也限制大 working set，不能只追理论峰值。

### 12. Occupancy 越高越好吗？

不是。Occupancy 是隐藏延迟的容量，不是得分。Grouped v3 achieved occupancy 30.26%，高于 CUTLASS 的 16.98%，但 issue active 只有 23.87% 对 34.79%，MIO/barrier/long-scoreboard stalls 更高，所以仍更慢。

### 13. 什么时候使用 `cp.async`？

当 global-to-shared tile 有足够复用和计算能与 copy 重叠，并且对齐、wait-group 和 shared/register 成本都可证明时使用。Dense/Grouped 都说明 `cp.async` 不是自动收益：tile 太宽会增加 live state、降低有效发射，短 shape 也可能摊不掉 pipeline setup。

### 14. 怎么判断 memory-bound？

至少结合 memory/DRAM throughput、SM throughput、cache hit、访问合并和 stalls。Permute v3 的 DRAM 86.85%、58.76% occupancy 和 payload copy 访问模式共同支持 bandwidth-bound；不是只看到一个“memory”指标就下结论。

### 15. Scan 为什么不用并行库实现？

这里 E<=64，是 tiny metadata。naive 只有一个 block/一个 thread、0.000919 waves/SM，瓶颈主要是 launch；CUB Warp/Block 仅约 1.03x，而 DeviceScan 只有 0.331x，还带 workspace/额外工作。复杂度不值得进入默认路径。

## 算子结果

### 16. 你最强的性能结果是什么？

按完整声明矩阵，Token Permute v2 full-from-ids 对 adapted vLLM 是 `1.5671x` ratio-of-sums、5/5 shape 获益；Histogram v2 对最快 reference envelope 是 `1.1016x`，峰值 `4.3026x`。Top-K 的最大局部收益是 E64/T4096 的 `1.6934x`，但完整矩阵有 10.88% 回退，所以不称无条件 winner。

### 17. Histogram `E=1` 为什么能到 4.3x？

因为这是语义退化：所有 R 条 route 都属于唯一 expert，结果必然是 `counts[0]=R`，不需要读取 route IDs 或做 atomic。v2 直接 O(1) 写回；它是合法 fast path，但必须和 E>1 dispatcher 分开解释。

### 18. Permute 为什么 pure path 不快，full-from-ids 却快？

Pure path只比较已有 mapping 后的 payload copy，v2/v3 和 token-owned 基本持平或略慢。Full-from-ids 把 counts、scan、rank/cursor preparation 都计入后，v2 减少中间 materialization 和 launch，因此对 vLLM 达到 `1.567x`。收益属于完整 operator boundary，不是 copy kernel 本身。

### 19. Grouped GEMM 为什么最难？

它同时有小 expert、empty expert、Zipf skew、tail wave、non-aligned K/N 和严格 FP32 math。局部 single-hot 可以减少库调度开销，但 uniform 大 shape 更看重成熟 tile、发射和复用。v2 有 `1.641x` 局部 winner，完整 external envelope 只有 `0.870x`，说明不能用一个调度策略覆盖全部分布。

### 20. Dense 为什么不直接写“超过 cuBLAS”？

因为完整三 shape ratio-of-sums 只有 `0.8716x`，只有 256³ 局部 `1.0095x`。cuBLAS 在 512³/1024³ 更强。诚实结论是我实现并 profile 了 64x32 `cp.async` strict-FP32 kernel，理解其资源和 reuse 边界，而不是伪造库级领先。

### 21. Top-K 为什么不直接拿 vLLM 当严格 baseline？

项目合同包含 tie-break、NaN policy 和 selected-softmax；adapted vLLM/CUB 只在有限随机输入子域可比。完整合同必须对 exact naive，外部结果另表标注 subdomain，不能混成同一严格排名。

### 22. Unpermute 的优势在哪里？

`cuda_warp_token_vec4` 在中大型 T、窄 N 更有优势，例如 Zipf T4096/N128 对最快 vLLM/naive envelope 为 `1.2892x`。但 32-case ratio-of-sums 只有 `1.0154x`、12/32 获益，所以是局部形状优势，不足以做全局 Auto。

## Profiler、取舍与后续

### 23. NSYS 为什么先于 NCU？

NSYS 先回答端到端哪一段占比最高，避免对非热点做深 profile。本项目 uniform/Zipf 都指向 Grouped GEMM，因此只对其 candidate/CUTLASS 从 basic 升级 detailed，其他七算子 basic 已足够分类。

### 24. Grouped detailed 最关键的三个指标是什么？

Issue active：v3 23.87% 对 CUTLASS 34.79%；MIO throttle：716 对 62；barrier：452 对 60。再结合 long scoreboard 404 对 138，可以说明候选虽 occupancy/L2 hit 更高，但同步、memory instruction pipeline 和依赖等待阻碍持续发射。

### 25. 下一轮你会怎么优化？

优先回到 Grouped v2 16x32 mainloop，只改变 task mapping/load balance，沿用十 shape 和 external envelope；第二是预声明 Top-K E64/T>=512 区间；第三是把 full-from-ids Permute 放入 L3 验证收益能否穿透 Grouped 热点。每轮只改一个机制。

### 26. CUDA Graph 的 1.62x 能写成 kernel speedup 吗？

不能。它是固定 shape、固定 buffer/topology、capture/instantiate/upload 已完成后的 host time-to-solution replay 收益，主要减少 CPU submission/launch overhead。Graph cache miss、动态 shape 和 setup 都不在该数字里。

### 27. 为什么保留失败 candidate？

失败记录能证明优化不是凭感觉。它保留了假设、父版本、矩阵反例、NCU 根因和拒绝理由，防止团队重复走同一路径，也能在面试中展示我会用证据停止错误方向。

### 28. 如果换 H100，能沿用这些结论吗？

语义、oracle 和 benchmark 方法可以沿用，性能结论不能。H100 的 SM 数、memory hierarchy、Tensor Core、TMA/WGMMA 和调度行为都不同，必须重新做 correctness、Release 和 profiler。当前只有 SM90 交叉编译，不宣称 H100 实卡支持。
