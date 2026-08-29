# RaggedRoute CUDA / AI Infra 面试入口

这份目录是校招与实习面试的唯一入口。项目的核心价值不是“七个 kernel 都超过库”，而是完整展示一次可复现的 GPU 性能工程闭环：冻结语义与数值合同，建立 CPU oracle 和外部强基线，用未插桩 Release 数据做结论，再用 NSYS/NCU 解释成功或失败，最后只保留经得住反例的声明。

最新证据来自 `9732a0343c60f869fc4166a0cc3cabba2fd67bbb`：RTX 3080（SM86、68 SM、10 GiB）、strict FP32、clean Release、5 个独立进程、每进程 20 次 warmup 和 30 samples、seed `20260828`。完整 compact bundle 见 [20260829 evidence](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)。

## 可直接放入简历的三条 bullet

- 设计并实现单 GPU Top-2 MoE 七阶段 CUDA 流水线，统一 v0.2 C++ runtime、CPU oracle、八个 benchmark adapter 与 L1/L2/L3 计时边界；在 RTX 3080 上完成 5 进程 Release、CTest、四类 Compute Sanitizer、NSYS 与 NCU 可复现验证。
- 面向 Ampere SM86 设计 Top-K 两级子组归约、Histogram shape dispatcher / `E=1` fast path、Token Permute fused route preparation 等候选：Top-K v4 对严格 naive 的 30-shape ratio-of-sums 为 `1.0972x`、峰值 `1.6934x`；Histogram v2 对最强参考 envelope 为 `1.1016x`、峰值 `4.3026x`；Permute v2 full-from-ids 对 adapted vLLM 为 `1.5671x` 且 5/5 shape 获益。
- 用 NSYS 定位 Grouped GEMM 在 uniform/Zipf L3 中占 GPU kernel time `71.6%/86.0%`，再以 NCU 证明“更高 occupancy 不等于更快”：v3 虽有 30.26% achieved occupancy 和 85.45% L2 hit，却受 MIO throttle、barrier、long scoreboard 与低 issue activity 限制；最终保留 v2 的局部 `1.6411x` winner，同时如实记录完整矩阵仅为最快 CUTLASS/cuBLAS envelope 的 `0.8697x`。

简历空间紧张时优先保留第二、第三条。所有倍率都必须和比较对象、shape 范围、计时边界一起出现。

## 30 秒介绍

RaggedRoute 是我围绕 Top-2 MoE 路由做的 CUDA 性能工程项目，包含从 router GEMM、Top-K、Histogram/Scan、Token Permute、ragged Grouped GEMM 到 Unpermute 的七阶段流水线。我没有只展示最好看的单点，而是在 RTX 3080 上固定 strict FP32 合同，用五个独立 Release 进程公平对比 cuBLAS、CUTLASS、CUB 和 adapted vLLM。项目中可复现的强结果包括 Permute full-from-ids 对 vLLM `1.567x`、Histogram v2 完整矩阵 `1.102x` 和 Top-K E64 大 T 最高 `1.693x`；同时 Grouped GEMM 的整体失败通过 NSYS/NCU 解释清楚，体现了我从 correctness、benchmark 到瓶颈定位和取舍的完整能力。

## 3 分钟介绍

第一层是工程合同。七个算子不是孤立 demo，而是两个 L3 入口下的同一条 MoE 数据流。公共 wrapper 使用 caller stream，hot path 不分配显存、不做无条件同步；每个算子都有独立 CPU oracle、边界 case、redzone、失败记录和 workspace 约束。外部路径只用于 benchmark，不偷换 runtime API。

第二层是公平测量。正式结果固定在同一 RTX 3080、同一 clean SHA、strict FP32、相同 seed/cache/level/repeats 和 CUDA Event 边界；五个进程各自 warmup 后采 30 samples。`CV>0.10` 在 Windows WDDM 下只作为风险披露，不自动抹掉结果；作品集上限放宽为 `0.50`，但判断仍同时看 ratio-of-sums、shape geomean、获益覆盖率、最大回退和跨进程方向一致性。NCU/NSYS duration 只做诊断，绝不当正式 speedup。

第三层是算子设计。Top-K v4 用寄存器局部 pair 和两级 subgroup reduction，E64/T4096 达到 `1.693x`，但小 shape 最大回退 10.88%，因此不改 Auto。Histogram v2 对 `E=1` 直接写 `counts[0]=R`，其余 shape 复用 single-CTA / block-private dispatcher，完整 15-case ratio-of-sums `1.102x`。Permute 的 pure-copy v2/v3 没有全面超过保留路径，但把 route preparation 纳入 full-from-ids 边界后，v2 对 adapted vLLM 达到 `1.567x` 且五个 shape 全胜，说明收益来自消除中间准备开销，而不是只优化 copy kernel。

第四层是失败诊断。NSYS 显示 Grouped GEMM 是绝对热点。最新 Release 中 v2 在 single-hot 有 `1.641x`，但对最快 CUTLASS/cuBLAS envelope 的完整矩阵只有 `0.870x`；uniform T2048 是关键反例。NCU detailed 对 v3/CUTLASS 的对照显示 v3 96 registers/thread、12,048 B shared memory、issue active 23.87%，且 MIO/barrier/long-scoreboard stalls 明显更高。结论不是继续堆更宽 tile 或 descriptor queue，而是回到 v2 的 16x32 mainloop，下一轮只隔离调度和负载均衡变量。

## STAR 案例

### 成功：Histogram v2

- Situation：Histogram 同时存在 tiny metadata、crossover、大 R、uniform/Zipf/single-hot，单一 kernel 无法覆盖所有区域。
- Task：在不改公共 API 和 workspace 的前提下，找到完整矩阵可复现收益，而不是只赢一个热点 shape。
- Action：先冻结 naive、CUB 和 shipping v1 envelope；将 `E=1` 识别为无需读 route IDs 的退化语义，增加 O(1) 写回 fast path，其他区域复用已经验证的 single-CTA / block-private dispatcher；用 15 case、五进程 Release 复测。
- Result：对最快参考 envelope 的 ratio-of-sums `1.1016x`、geomean `1.1588x`、9/15 shape 获益、最大回退 5.56%，`R=1M,E=1` 达 `4.3026x`。v2 作为显式 strongest candidate 保留，本轮按约束不改 Auto。

### 成功：Token Permute full-from-ids

- Situation：pure permute 只测 payload copy，会忽略从 expert IDs 生成 counts/offsets/rank 的真实准备成本，也会造成与 vLLM 边界不公平。
- Task：在相同 full-from-ids 边界下比较自研路径与 adapted vLLM。
- Action：把 fused counts/scan/cursor preparation 和 payload movement 一起纳入 L2；同时保留 pure-permute 矩阵作为消融，防止把准备阶段收益误写成 copy kernel 提升。
- Result：v2 full-from-ids ratio-of-sums `1.5671x`、geomean `1.5885x`、5/5 shape 获益，区间 `1.2896x–1.8501x`；pure v2 只有 `0.9841x`，因此结论明确限定为端到端 operator boundary。

### 高收益但不晋级：Top-K v4

- Situation：naive 每行工作串行，E64 大 T 下并行度和归约效率不足；外部 CUB/vLLM 又不能覆盖完整 tie/NaN/selected-softmax 合同。
- Task：验证两级 subgroup reduction 是否能形成可发布区间。
- Action：用寄存器局部 Top-2 pair、vector row load 与两级归约，并分别对严格 naive 的完整合同和 external random-input 子域做矩阵评估。
- Result：严格 30-shape ratio-of-sums `1.0972x`、geomean `1.0859x`、21/30 获益，E64/T4096 为 `1.6934x`；但 E32/T32 回退 10.88%，所以保留显式 v4，不改 Auto。这个案例展示“找到最大收益，同时守住反例门禁”。

### 失败：Grouped GEMM v3 / v4A

- Situation：Grouped GEMM 占 L3 GPU kernel time 70% 以上，v2 在 skew/single-hot 有优势，但 uniform 和大 shape 失败。
- Task：通过更宽 16x64 tile、`cp.async` 双缓冲和 descriptor queue 改善吞吐与 ragged 调度。
- Action：保持 strict FP32 与零 workspace，逐版本复测十 shape；NSYS 先确认热点，NCU basic 后仅对 v3/CUTLASS 升级 detailed。
- Result：最新矩阵最强仍是 v2；其对 external envelope ratio-of-sums `0.8697x`，只有 3/10 shape 获益。v3 的 occupancy 比 CUTLASS 高，但 issue active 更低，MIO/barrier/long-scoreboard stalls 更重；v4A queue-1024 更差。候选不晋级，下一轮回到 v2，只测试调度变量。

### 失败：Dense GEMM v3

- Situation：尝试用 64x32 register tile、`cp.async` 双缓冲在 SM86 strict FP32 下逼近 cuBLAS。
- Action：固定 256³、512³、1024³，对每个 shape 取最快 cuBLASLt/cuBLAS 参考；NCU 检查 waves、寄存器、occupancy 和 SM/memory throughput。
- Result：256³ 局部 `1.0095x`，但完整矩阵 ratio-of-sums `0.8716x`，1024³ 仅 `0.8514x`。因此不宣称超过 cuBLAS；项目亮点是正确实现 Ampere pipeline 并用强库基线证明其边界。

## 能力边界

- 已实测：Windows、RTX 3080、SM86、68 SM、10 GiB、FP32 row-major、单 GPU、Top-2 主路径。
- 未实现：FP16/BF16 Tensor Core runtime、训练、多 GPU Expert Parallel / All-to-All、完整 MoE FFN、H100/Blackwell 实卡支持。
- Top-K 的 CUB/vLLM 只在有限随机输入子域可比；完整 tie/NaN/selected-softmax 合同的严格 oracle 是 naive。
- CUDA Graph 结果是 fixed-shape、fixed-buffer、setup 完成后的 host time-to-solution，不是 kernel speedup；Histogram→Scan 是独立跨算子 primitive，不混入七算子排名。
- `matrix_winner_research_only` 不等于修改 `KernelFamily::kAuto`。本轮按计划不改公共 API、Auto 或算子实现。

## 阅读顺序

1. [七算子性能卡片](operator-performance.md)
2. [瓶颈分析与下一轮实验](bottleneck-analysis.md)
3. [高频追问题库](question-bank.md)
4. [compact evidence](../reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)
5. [分支与冗余清理 review](../cleanup-review.md)
