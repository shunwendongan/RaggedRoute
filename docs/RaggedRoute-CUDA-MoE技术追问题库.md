# RaggedRoute CUDA / MoE 技术追问题库

> 目标岗位：AI Infra / CUDA 实习
> 事实基线：`main@354e1ff0a3aaf602830c8d989eb358cfa857a4c9`
> 使用边界：本文是公开的项目面试准备补充材料。当前本地环境为 macOS；文中性能数字全部回溯到仓库已有 RTX 3080 / SM86 证据，本轮没有运行 CUDA 构建、算子测试、benchmark、NSYS 或 NCU。

## 0. 使用方法与回答纪律

先练熟“必讲 15 题”，再按算子展开。30 秒回答要先给结论，再说机制和证据，最后主动收住边界。不要在没有被问到时连续堆指标，也不要把“源码存在”“局部更快”“交叉编译成功”说成“默认晋级”“端到端更快”“实卡支持”。

本文采用四种事实标签：

- `已证明`：`main@354e1ff` 的源码、测试或已归档 RTX 3080 证据支持。
- `职责口径`：你确认自己主导方向、方案选择、实验协议和验收；Codex 辅助实现与文档。
- `原理解释`：可用于解释 CUDA/MoE 机制，但不代表本项目已经实测该结论。
- `待补强`：拿到独占 CUDA 环境后还需重跑或实现，不能提前包装成成果。

遇到记不清的数字，优先说：

> “我不想凭记忆报错数字。结论是它没有通过门禁；仓库报告保留了 shape、进程数、raw/aggregate 和比较文件，我可以沿 SHA 回溯。”

## 1. 面试前必须讲熟的 15 题

| 优先级 | 题号 | 必须掌握的主线 |
|---:|---|---|
| 1 | Q01 | 一句话说明项目价值和边界 |
| 2 | Q02 | 七阶段数据流与八个 adapter 的区别 |
| 3 | Q04 | 个人 ownership 与 Codex 辅助边界 |
| 4 | Q05 | 为什么 runtime 只兑现 SM86 strict FP32 |
| 5 | Q06 | v0.2 API、caller stream、workspace、fallback |
| 6 | Q15 | Dense GEMM tiling、register blocking、`cp.async` |
| 7 | Q18 | Top-2 selected-softmax、tie 与 NaN 语义 |
| 8 | Q21 | Histogram 原子冲突与正式 promotion |
| 9 | Q23 | Histogram→Scan F2 为什么仍是技术债 |
| 10 | Q24 | Permute 的 route mapping、cursor 与访存 |
| 11 | Q27 | Grouped GEMM 的 ragged `M_e`、skew 与尾波 |
| 12 | Q29 | `0.805x` / `0.467x` 为什么导致拒绝 |
| 13 | Q32 | L1/L2/L3 为什么必须分层 |
| 14 | Q36 | CUDA Event、NSYS、NCU 各回答什么问题 |
| 15 | Q40 | 为什么这不是生产 MoE runtime，以及下一步做什么 |

---

## 2. 项目边界与个人职责

### Q01：这个项目到底解决了什么问题？

- **面试官意图**：判断你是否能从 MoE 工作负载出发，而不是只会罗列 kernel 名称。
- **30 秒回答**：RaggedRoute 把单 GPU Top-2 MoE 的 routing、按 expert 重排、ragged expert linear 和加权还原串成一条七阶段 strict-FP32 CUDA 链。项目核心不是宣称所有 kernel 最快，而是建立“语义合同—CPU oracle—L1/L2/L3 测量—Nsight 归因—晋级或拒绝”的性能工程闭环。当前只验证 RTX 3080 / SM86，不是完整生产推理框架。
- **可继续展开的技术细节**：输入 token 先做 router projection；Top-2 产生 expert id 与权重；Histogram/Scan 生成分段 offset；Permute 把相同 expert 的 route 连续化；Grouped GEMM 按动态 `M_e` 计算；Unpermute 按 route position 和 gate weight 还原。可补充为什么动态 `M_e` 带来 skew、empty expert 和尾波。
- **证据位置**：[README](../README.md)、[`operators.h`](../include/raggedroute/operators.h)、[`chain_adapter.cpp`](../benchmarks/adapters/chain_adapter.cpp)。
- **常见翻车点**：说成“完整 MoE 推理引擎”；遗漏只有一次 expert linear、没有完整 FFN/激活/Down projection、All-to-All、continuous batching。

### Q02：为什么说是七阶段数据流，却有八个 benchmark adapter？

- **面试官意图**：检查你是否真正理解语义算子、融合 primitive 和基准注册之间的边界。
- **30 秒回答**：七个语义阶段是 Dense GEMM、Top-2 Gate、Histogram、Exclusive Scan、Token Permute、Grouped GEMM、Unpermute。第八个 adapter 是可选的 `histogram_exclusive_scan` 融合 primitive，它替换 Histogram+Scan 两段，但不增加新的 MoE 语义阶段。`chain_from_tokens` 计七阶段，`chain_from_logits` 从已有 logits 开始，只计后六阶段。
- **可继续展开的技术细节**：解释 `OperatorKind` 有八个枚举；公开 API 同时暴露 standalone Histogram、Scan 与 fused primitive；L3 的阶段数按语义边界而非注册表数量计算。
- **证据位置**：[`types.h`](../include/raggedroute/types.h)、[`operators.h`](../include/raggedroute/operators.h)、[`registry.cpp`](../benchmarks/core/registry.cpp)、[`chain_adapter.cpp`](../benchmarks/adapters/chain_adapter.cpp)。
- **常见翻车点**：说“八算子 MoE 链”；把融合路径说成默认替换了所有 L3 chain；混淆 `chain_from_logits` 为七阶段。

### Q03：它为什么不是七个孤立 CUDA Demo？

- **面试官意图**：判断项目是否有系统边界、状态传递和端到端验证。
- **30 秒回答**：算子之间共享明确的数据合同：Top-K 输出 route id/weight，Histogram/Scan 把 route 变成 expert segment，Permute 产生 `route_pos` 和连续的 expert rows，Grouped GEMM 消费 offsets，Unpermute 再用 `route_pos` 与 weight 恢复 token 顺序。公共 wrapper、workspace、caller stream、CPU reference 和两条 L3 chain 让这些接口能被组合和验证，而不只是各自跑一个 microbenchmark。
- **可继续展开的技术细节**：画出 `T×H → T×E → T×2 → counts[E]/offsets[E+1] → (2T)×H → (2T)×N → T×N`；说明 metadata 是 `int32`，floating payload 是 tensor view。
- **证据位置**：[`operators.h`](../include/raggedroute/operators.h)、[`chain_adapter.cpp`](../benchmarks/adapters/chain_adapter.cpp)、[Benchmark 架构](../docs/benchmark-architecture.md)。
- **常见翻车点**：只背阶段名称，答不出张量 shape；把 route 数 `R` 说成 token 数 `T`，忘记 Top-2 下 `R=2T`。

### Q04：这么多提交是不是 AI 写的？你个人到底做了什么？

- **面试官意图**：核验 ownership、技术判断和你能否独立解释/修改核心路径。
- **30 秒回答**：项目明确使用 Codex 辅助代码实现和文档。我主导的是问题定义、范围取舍、语义合同、候选方案选择、benchmark 协议和结果验收；是否进入 `Auto` 由 correctness、跨 shape、稳定性和外部基线共同决定。我不会把每个 kernel 说成完全手写，面试中愿意沿 dispatch、adapter、Histogram 和 Grouped GEMM 失败路径讲到代码级。
- **可继续展开的技术细节**：举两个决策例子：Histogram 满足 12-case 五进程门禁后晋级；Grouped GEMM 虽局部胜出但十 shape ratio-of-sums 失败而拒绝。说明 AI 产出仍需做语义审计、公平比较、反例和错误声明清理。
- **证据位置**：[答辩手册“个人职责口径”](RaggedRoute-简历话术与面试答辩手册.md)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)、[Histogram 记录](../docs/histogram/performance-record.md)、[Grouped GEMM 记录](../docs/grouped_gemm/performance-record.md)。
- **常见翻车点**：含糊说“都是我写的”；反过来只说“AI 帮我做了”，却说不出自己决定了什么、拒绝了什么、如何验收。

### Q05：为什么只支持 SM86 strict FP32？API 里不是还有 FP16/BF16/SM90 吗？

- **面试官意图**：检查你是否区分“类型可表达”“编译可通过”“运行时已支持”“实卡已验证”。
- **30 秒回答**：v0.2 的类型系统提前保留了 FP16/BF16 等表示，dispatch 也能识别 SM90，但当前可执行 kernel 只对 SM86、全 FP32、zero-stride row-major 签名返回成功。低精度签名会显式返回 unsupported，SM90 只有交叉编译记录，没有实卡 correctness/performance，因此不能宣称 Tensor Core 或 H100 支持。
- **可继续展开的技术细节**：`is_valid_dtype_signature` 先检查角色组合，`is_all_fp32` 再收紧当前执行能力；`select_kernel` 要求 resolved architecture 为 SM86。说明 API vocabulary 与 capability decision 分离可以避免未来低精度扩展再次破坏公共接口。
- **证据位置**：[`types.h`](../include/raggedroute/types.h)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)、[实现状态](../docs/implementation-status.md)。
- **常见翻车点**：看到枚举就说“支持 FP8”；看到 `sm_90` cubin 就说“H100 已支持”；把 FP32 CUDA Core kernel 说成 Tensor Core kernel。

### Q06：v0.2 API 设计中最值得讲的工程点是什么？

- **面试官意图**：判断你是否只懂 kernel，还是理解可集成的 operator contract。
- **30 秒回答**：v0.2 把低层 L1 launcher 和完整 public wrapper 分开。浮点数据使用带 dtype/layout/element-strides 的 tensor view，metadata 保持 `int32`；wrapper 使用调用方 stream，不在 hot path 分配或无条件同步，并把必要 reset 纳入 L2 边界。Permute workspace 由调用方提供，架构在 benchmark setup 缓存，unsupported dtype/layout/architecture 明确失败。
- **可继续展开的技术细节**：解释 source-breaking 删除旧 `float*` shim 的理由；`KernelSelection` 将稳定 family 与 operator-local implementation id 分离；`max_expert_tokens` 避免为了读 device offset 而 host sync。
- **证据位置**：[`operators.h`](../include/raggedroute/operators.h)、[`types.h`](../include/raggedroute/types.h)、[`runtime.h`](../include/raggedroute/runtime.h)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)。
- **常见翻车点**：说“完全零开销”；忘记 wrapper 仍有参数检查和必要 `cudaMemsetAsync`；说 workspace 是内部动态分配。

### Q07：`Auto`、显式 candidate、library baseline 三者有什么区别？

- **面试官意图**：检查你是否会把研究代码包装成已发布能力。
- **30 秒回答**：`Auto` 是公开 runtime 的默认选择，必须有足够 correctness、稳定性、shape coverage 和回退证据；显式 candidate 是可复现的研究路径，存在不等于晋级；cuBLAS/CUB/CUTLASS/vLLM 路径只用于 benchmark reference，不进入 public runtime dispatch。七个语义算子中目前只有 Histogram 的优化路径有证据支持默认晋级。
- **可继续展开的技术细节**：解释 `KernelFamily::{Auto,CudaNaive,CudaOptimized}`；implementation id 是算子局部控制；Grouped GEMM/Unpermute candidate 甚至只保留在 adapter 中。
- **证据位置**：[`types.h`](../include/raggedroute/types.h)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)、[算子优化索引](../docs/operator-optimization-index.md)。
- **常见翻车点**：把“可显式调用”说成“默认使用”；说所有库路径都能由 runtime 选择；忽略 F2 是一个特殊技术债。

### Q08：为什么 library baseline 不直接进入 public dispatch？

- **面试官意图**：考察 API 边界、第三方依赖、语义等价和 benchmark 公平性。
- **30 秒回答**：外部库路径的主要作用是提供强参考，而不是扩大 v0.2 runtime 的稳定承诺。不同库可能需要 descriptor、host offsets、临时 workspace、预处理或不同语义；只有在 adapter 中明确计时边界才能公平比较。保持 benchmark-only 也让 public runtime 的依赖、fallback 和 API 合同更简单。
- **可继续展开的技术细节**：Dense 有 cuBLAS/cuBLASLt，Histogram/Scan 有 CUB，Grouped 有 CUTLASS/cuBLAS-per-expert，Permute/Unpermute 有 adapted vLLM；Top-K 因 selected-softmax/tie/NaN 合同没有严格等价的通用库 promotion baseline。
- **证据位置**：[`library_baselines.h`](../include/raggedroute/benchmark/library_baselines.h)、各 `src/*/library_baseline/UPSTREAM.md`、[`registry.cpp`](../benchmarks/core/registry.cpp)。
- **常见翻车点**：把 benchmark-only 当成 runtime fallback；忽略第三方语义差异；说“用了 vLLM 所以性能等同 vLLM”。

---

## 3. CUDA 执行模型、内存层次与资源

### Q09：block、warp、thread 在这个项目里分别怎么映射工作？

- **面试官意图**：验证 CUDA 基础是否能落到具体 kernel。
- **30 秒回答**：映射取决于工作粒度。naive Dense GEMM 一线程计算一个输出元素；naive Top-K 一线程处理一个 token row；naive Permute 一 CTA 处理一条 route 并由线程协作复制 hidden row；naive Grouped GEMM 用 `grid.z` 选择 expert、二维 block 覆盖 row/column；Unpermute 一线程处理一个 token-output 元素。优化候选再用 warp/subwarp reduction、row tiling 或 persistent task map 改善复用和负载均衡。
- **可继续展开的技术细节**：说明 SIMT 中 warp 是实际调度单位；不同映射会影响 coalescing、并行度、分支发散、barrier 和尾波。用一个 kernel 例子画出 block 与数据坐标。
- **证据位置**：各 `src/*/cuda_naive/baseline.cu`，尤其 [`grouped_gemm baseline`](../src/grouped_gemm/cuda_naive/baseline.cu) 与 [`permute baseline`](../src/permute/cuda_naive/baseline.cu)。
- **常见翻车点**：把 block 当成硬件固定执行单元；说不同 block 可以用 `__syncthreads()` 同步；解释不出 `grid.z=expert` 的含义。

### Q10：什么是合并访存？RaggedRoute 哪些阶段最容易出现访存不规则？

- **面试官意图**：检查你是否能从 lane 地址模式判断内存效率。
- **30 秒回答**：同一 warp 的线程访问连续且对齐的地址时，global memory 请求更容易合并成较少 transaction。Dense tile load 和连续 row copy 容易设计成合并访问；routing metadata、按 expert 分桶和通过 `route_pos` 间接访问会引入不规则性。Permute 的目标 row 由 atomic cursor 决定，但一旦得到 destination，CTA/warp 对 hidden 维的连续 copy 仍可合并；Unpermute 的 source row 间接，但 column 维可以保持连续。
- **可继续展开的技术细节**：解释 `float4` 需要 16-byte 对齐和 hidden 可整除；vectorization 减少指令数但不能修复随机 destination；区分 L2 cache 命中与真正 coalescing。
- **证据位置**：[`permute candidate`](../src/permute/cuda_candidate/optimized.cu)、[`unpermute candidate`](../src/unpermute/cuda_candidate/optimized.cu)、[`dense v3`](../src/dense_gemm/cuda_candidate/optimized_v3.cu)。
- **常见翻车点**：说“使用 float4 就一定带宽满”；把相邻 thread 访问相邻 row 误当连续；忽略对齐和尾部条件。

### Q11：global、shared、register 在 Dense GEMM 优化中分别承担什么角色？

- **面试官意图**：考察内存层次与数据复用。
- **30 秒回答**：A/B 原始矩阵在 global memory；CTA 把 K tile 搬到 shared memory，让同一 tile 被多线程复用；每个线程在 register 中保留多个输出 accumulator，形成 register blocking，减少重复 load/store。代价是 shared 容量、barrier 和 register live range 会限制 resident blocks 与 occupancy，所以 tile 越大不一定越快。
- **可继续展开的技术细节**：v2 使用 32×32 CTA tile，v3 扩为 64×32；两者 K-depth 为 16，并使用 shared padding，异步版本使用双 stage；每线程持有多个 accumulator。解释 arithmetic intensity 随复用上升，但边界 shape、同步和 load mapping 仍可能成为瓶颈。
- **证据位置**：[`optimized_v2.cu`](../src/dense_gemm/cuda_candidate/optimized_v2.cu)、[`optimized_v3.cu`](../src/dense_gemm/cuda_candidate/optimized_v3.cu)、[Dense GEMM 记录](../docs/dense_gemm/performance-record.md)。
- **常见翻车点**：说 shared 比 register 快所以应该全放 shared；忽略 register 不可寻址共享；把 cache、shared 和 register 的作用混为一谈。

### Q12：shared-memory bank conflict 是什么？项目里如何规避？

- **面试官意图**：检查你能否从 shared 地址布局解释吞吐下降。
- **30 秒回答**：shared memory 被分成 banks，同一 warp 对不同地址但落到同一 bank 的访问会被串行化；相同地址广播是例外。Dense v2 把 A tile 的逻辑 K stride 从 16 padding 到 20，让 warp 的多行读取分散到不同 bank，减少冲突。padding 会增加 shared 占用，因此仍需和 occupancy、tile shape 一起评估。
- **可继续展开的技术细节**：用 `bank=(byte_address/4) mod 32` 的 FP32 近似模型解释 stride 16 为什么周期性碰撞、stride 20 如何打散；说明 bank conflict 是 shared 层问题，不等于 global uncoalesced access。
- **证据位置**：[`optimized_v2.cu`](../src/dense_gemm/cuda_candidate/optimized_v2.cu) 中 `kSharedAStride=20` 的注释与访问代码。
- **常见翻车点**：把所有同 bank 访问都说成 32 路冲突；忽略 broadcast；声称 padding 必然加速而没有 profile/benchmark。

### Q13：occupancy 是什么？为什么不能把 100% occupancy 当优化目标？

- **面试官意图**：判断你能否正确解释 profiler 指标而不是机械追高。
- **30 秒回答**：occupancy 是活跃 warp 相对硬件上限的比例，受 registers/thread、shared/block、threads/block 和架构限制。它只代表隐藏延迟的潜力，不等于执行单元利用率或性能。提高 occupancy 可能通过缩小 tile、减少 accumulator 损失数据复用；反过来，较低 occupancy 若 ILP 和数据复用好也可能更快。项目中 106 registers/thread 是 Grouped candidate 的风险信号，但必须与 waves、stall 和实际 latency 一起看。
- **可继续展开的技术细节**：区分 theoretical occupancy 与 achieved occupancy；说明一 wave/SM 会放大 tail/imbalance；讨论 `launch_bounds`、register cap 可能导致 spill。
- **证据位置**：[L3 三线路报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)、[Grouped GEMM 记录](../docs/grouped_gemm/performance-record.md)。
- **常见翻车点**：把 achieved occupancy 26.62% 直接等同“SM 只用了 26.62%”；建议无脑 `--maxrregcount`；不检查 local-memory spill。

### Q14：原子操作为什么既保证正确性又可能成为瓶颈？

- **面试官意图**：考察并发更新、冲突分布与优化策略。
- **30 秒回答**：Histogram 和 Permute cursor 都有多个 route 更新同一 expert 计数的竞争，`atomicAdd` 保证每次增量不丢失；当 expert 分布热点化时，同一地址原子会序列化并形成 contention。优化方向不是简单删除原子，而是降低全局原子次数，例如 shared privatization、warp 聚合相同 key，最后再合并；但小 R 时额外初始化/合并可能比原子本身更贵。
- **可继续展开的技术细节**：对比 uniform、single-hot、Zipf 分布；说明 shared atomic 仍有冲突，只是作用域/代价不同；route position 还要求每个 route 获得唯一 local rank。
- **证据位置**：[`histogram baseline`](../src/histogram/cuda_naive/baseline.cu)、[`histogram candidate`](../src/histogram/cuda_candidate/optimized.cu)、[`permute candidate`](../src/permute/cuda_candidate/optimized.cu)。
- **常见翻车点**：声称 atomic 一定慢；用非原子 load/add/store 导致丢更新；只测 uniform 分布就下普遍结论。

---

## 4. Dense GEMM

### Q15：Dense GEMM 从 naive 到 tiled/register-blocked 的核心变化是什么？

- **面试官意图**：判断你是否能讲清经典 GEMM 优化链，而不是只说“用了 shared memory”。
- **30 秒回答**：naive 是一线程一个 `C[m,n]`，每个输出独立从 global 读取整行 A 和整列 B，几乎没有跨线程显式复用。候选把输出按 CTA tile 分块，A/B 的 K-slice 协作搬到 shared，再由每线程维护多个 register accumulator。这样提高数据复用和 arithmetic intensity，但同时引入 shared 容量、barrier、边界 shape 和寄存器压力，必须按 L2 与 cuBLAS 证据判断。
- **可继续展开的技术细节**：说明 block tiling、warp tile、thread tile 三层映射；每个 K tile 的 global load 被多个 FMA 复用；v2/v3 的 shared padding、`float4` load、双 stage 和 64×32 映射。
- **证据位置**：[`dense naive`](../src/dense_gemm/cuda_naive/baseline.cu)、[`optimized_v2.cu`](../src/dense_gemm/cuda_candidate/optimized_v2.cu)、[`optimized_v3.cu`](../src/dense_gemm/cuda_candidate/optimized_v3.cu)。
- **常见翻车点**：把 register blocking 说成“每个 block 一个寄存器”；答不出谁复用 A、谁复用 B；忽略非整 tile shape 的适用条件或 fallback。

### Q16：`cp.async` 做了什么？为什么用了仍可能比 cuBLAS 慢？

- **面试官意图**：检查你是否把架构指令当成万能加速开关。
- **30 秒回答**：在 Ampere 上，`cp.async` 可把 global 数据异步搬到 shared，并用 commit/wait group 组织 pipeline，理想情况下让下一 tile 的搬运与当前 tile 计算重叠。它只优化 staging，不会自动解决 tile shape、reuse、barrier、register live range、指令吞吐和边界覆盖。项目 v3 有 `cp.async` 和双缓冲，但 1024³ strict-FP32 L2 仍慢于 cuBLAS，所以没有进入 `Auto`。
- **可继续展开的技术细节**：解释 16-byte copy、shared destination address、`commit_group`、`wait_group<N>`、`__syncthreads()` 的职责；只有足够计算覆盖搬运、pipeline 正确且 occupancy 不被资源拖垮时才有收益。
- **证据位置**：[`optimized_v3.cu`](../src/dense_gemm/cuda_candidate/optimized_v3.cu)、[Dense GEMM 记录](../docs/dense_gemm/performance-record.md)。
- **常见翻车点**：说 `cp.async` 完全不占带宽；漏掉 CTA 内可见性同步；把 Ampere FP32 `cp.async` 路径说成 Tensor Core。

### Q17：和 cuBLAS/cuBLASLt 比较时怎样保证公平？

- **面试官意图**：判断你是否理解 math mode、计时边界和强基线。
- **30 秒回答**：必须匹配 M/N/K、row-major 语义、输入输出、FP32 accumulation、alpha/beta、warmup、stream、workspace 和计时层级。尤其不能让 cuBLAS 默认使用 TF32 Tensor Core，再把结果称为 strict-FP32 同语义比较；也不能把候选只计 kernel body，而把 cuBLAS descriptor/必需工作全计入或反过来。项目把 cuBLAS/cuBLASLt 放在 benchmark adapter，并单独标注 reference 边界。
- **可继续展开的技术细节**：说明 latency speedup 为 baseline/candidate；L1 与 L2 不能混算；library handle/descriptor 的一次性初始化应在 setup，但每 invocation 必需的工作不能隐藏。
- **证据位置**：[`dense_gemm_adapter.cpp`](../benchmarks/adapters/dense_gemm_adapter.cpp)、[`cublas_gemm.cpp`](../src/dense_gemm/library_baseline/cublas_gemm.cpp)、[Benchmark 架构](../docs/benchmark-architecture.md)。
- **常见翻车点**：只说“同 shape 就公平”；忽略 TF32/math mode；拿最好一次 sample 对库的 median。

---

## 5. Top-2 Gate

### Q18：Top-2 Gate 为什么不只是“找最大的两个数”？

- **面试官意图**：考察算法语义、数值稳定性和 edge case。
- **30 秒回答**：项目合同不仅要返回两个 id，还固定 selected-softmax、相同 logit 时 lower expert-id 优先、NaN 按 `-Inf` 排序、全 NaN 行 fallback 到 `{0,1}` 且权重 `{0.5,0.5}`。softmax 只在选中的两个值上归一化，并用相对最大值计算避免溢出；`+Inf/-Inf` 还要避免 `inf-inf`。任何 candidate 或外部基线必须匹配这些语义才能严格晋级。
- **可继续展开的技术细节**：公式 `w0=1/(1+exp(v1-v0))`、`w1=1-w0`；tie 时稳定选择较小 id；all-NaN 与两个相等 finite/inf 值的处理区别。
- **证据位置**：[`types.h`](../include/raggedroute/types.h)、[`topk naive`](../src/topk_gate/cuda_naive/baseline.cu)、[`CPU reference`](../src/topk_gate/cpu_reference/reference.cpp)。
- **常见翻车点**：说成对全部 E 做 softmax 后取 Top-2；忽略 tie/NaN；对 `+Inf` 直接做 `exp(inf-inf)`。

### Q19：Top-K candidate 的 warp/subwarp reduction 思路是什么？

- **面试官意图**：考察 warp primitive、reduction 合并和 shape-dependent parallelism。
- **30 秒回答**：一个线程顺序扫 E 在 E 较大时并行度不足。candidate 让 lane 各自维护局部 top-2 pair，再通过 shuffle/reduction 合并 pair；根据 E 可用 subwarp 或 warp，v4 还做 row packing 和两轮 subgroup reduction。关键是 merge 操作必须保持 value/id 的全序、tie 和 NaN 语义。更强并行也会增加 shuffle、空 lane 和调度开销，所以小 E/小 T 不一定获益。
- **可继续展开的技术细节**：解释局部 pair 合并为何要保留两个不同 expert；active mask；subwarp 宽度与 E bucket；T 决定并行 row 数，E 决定每 row reduction 工作量。
- **证据位置**：[`topk candidate`](../src/topk_gate/cuda_candidate/optimized.cu)、[`optimized_internal.h`](../src/topk_gate/cuda_candidate/optimized_internal.h)、[Top-K 记录](../docs/topk_gate/performance-record.md)。
- **常见翻车点**：只归约最大值后再“随便找第二”；shuffle 未使用正确 mask；认为 warp 方案对所有 E 都更快。

### Q20：Top-K 单点局部更快，为什么仍没有 shape dispatcher？

- **面试官意图**：检查你是否尊重连续 coverage 和反例。
- **30 秒回答**：T2048/E64 的候选有约 1.46x 局部收益，但 promotion 不是给一个坐标写 `if`。v4 对 exact-E 和连续 T/E bucket 检查 p50、p95、CV 与多进程一致性，最终没有任何完整区间通过，因此 `Auto` 保持 naive。项目选择保留显式 candidate 和 rejection 证据，不把最好点外推成普遍结论。
- **可继续展开的技术细节**：说明 dispatcher 的边界会遇到邻近 shape discontinuity；过拟合当前 suite 会增加维护成本；没有真实 route distribution 时更应保守。
- **证据位置**：[`REJECTION.md`](../docs/reports/artifacts/20260804-86cfbe2-topk-v4/REJECTION.md)、[`promotion/REPORT.md`](../docs/reports/artifacts/20260804-86cfbe2-topk-v4/promotion/REPORT.md)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)。
- **常见翻车点**：说“Top-K 已优化 1.46x”；把显式 id 4 说成 `Auto`；只背 speedup 不知道 gate 为什么为零区间。

---

## 6. Histogram、Exclusive Scan 与融合

### Q21：Histogram candidate 为什么能晋级？

- **面试官意图**：要求你讲出本项目真正成功的默认优化与证据链。
- **30 秒回答**：naive 对每条 route 做 global atomic，不同 R 和 skew 的瓶颈不同。candidate 用 shape dispatcher 在 small/sparse/block-private 等路径间选择，保留正确 fallback；它通过 12-case、五个独立进程的 strict-FP32 门禁，所有 case 没有 naive p50 回退，candidate p95 也优于 strongest baseline，所以成为七个语义算子中唯一有证据支持的 optimized `Auto`。
- **可继续展开的技术细节**：小 R 关注 launch/underfill；热点分布关注 atomic contention；大 R 可用 shared privatization 后再合并。说明 exact-int32 correctness、reset 计入 L2、与 CUB strongest baseline 比较。
- **证据位置**：[Histogram 记录](../docs/histogram/performance-record.md)、[`histogram candidate`](../src/histogram/cuda_candidate/optimized.cu)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)。
- **常见翻车点**：把一个大 shape 的 1.227x 当全部证据；忘记五进程与 p95；说 Histogram→Scan F2 也同时通过了同一门禁。

### Q22：Exclusive Scan 为什么 naive 用单线程？为什么不能在多 CTA 中只用 `__syncthreads()`？

- **面试官意图**：考察 scan 依赖、同步作用域和 baseline 设计。
- **30 秒回答**：naive scan 用一个 thread 顺序生成 `offsets[e]` 和总 route 数，性能不强但语义简单、可作为正确 baseline，E=64 这类小规模下 launch 也可能主导。并行 scan 需要 CTA 内分层 reduction/downsweep；如果跨 CTA，就需要多 kernel、cooperative launch 或其他 grid-wide 机制。`__syncthreads()` 只同步同一 CTA，不能保证其他 CTA 已完成生产。
- **可继续展开的技术细节**：exclusive scan 定义 `offsets[0]=0`、`offsets[e+1]=offsets[e]+counts[e]`；当前 baseline 允许 counts/offsets alias，因为先读后写；CUB Device/Block/Warp 是不同边界的参考。
- **证据位置**：[`scan baseline`](../src/scan/cuda_naive/baseline.cu)、[`exclusive_scan_adapter.cpp`](../benchmarks/adapters/exclusive_scan_adapter.cpp)、[`cub_scan.cu`](../src/scan/library_baseline/cub_scan.cu)。
- **常见翻车点**：说 `__syncthreads()` 是全 GPU barrier；把 inclusive/exclusive scan 混淆；为了并行化小 E 忽略 launch/临时 workspace。

### Q23：Histogram→Scan F2 为什么机制合理，却仍是技术债？

- **面试官意图**：测试你能否同时解释优化原理与证据不足。
- **30 秒回答**：F2 对 `R<=4096` 使用单 CTA，在 shared 中完成 histogram，再由 subwarp scan 生成 counts/offsets，因此 CTA 内 `__syncthreads()` 足以建立生产者—消费者顺序，并省一次 launch 与中间 global 往返。问题是 archived release 运行受竞争 GPU workload 干扰，CV、fallback 和 L3 gate 都失败；虽然当前该 primitive 的 `Auto` 选择 F2，但它只能说是待独占 GPU 复测或 demote 的技术债，不能说稳定晋级。
- **可继续展开的技术细节**：解释单 CTA 限制为何决定 `R/E` coverage；更大 R 回退两阶段路径；融合收益来自 launch/HBM，代价可能是 under-parallelization、shared/resource 压力和覆盖收窄。
- **证据位置**：[`scan candidate`](../src/scan/cuda_candidate/optimized.cu)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)、[F2 报告](../docs/reports/rtx3080-histogram-scan-fused-sm86-v2.md)。
- **常见翻车点**：说 F2 已“生产晋级”；引用约 2.0045x 却不披露 45%–114% CV 和 L3 反例；建议在 Mac 上复测 CUDA。

---

## 7. Token Permute 与 Unpermute

### Q24：Token Permute 的 route mapping、cursor workspace 是怎么工作的？

- **面试官意图**：检查你能否讲清 metadata 状态变化和并发唯一性。
- **30 秒回答**：Histogram/Scan 给每个 expert 一个 segment 起点 `offsets[e]`。每条 route 读取自己的 expert，用 `atomicAdd(cursor[e],1)` 获得该 expert 内唯一 local rank，destination 就是 `offsets[e]+rank`；`route_pos[route]=destination`，可选 `sorted_route[destination]=route`。随后把源 token 的 hidden row 复制到 destination。cursor 是每次调用的临时状态，所以由调用方提供 `E*sizeof(int32)` workspace，并由 L2 wrapper 异步清零。
- **可继续展开的技术细节**：Top-2 下 route `r=t*2+k`，两个 route 复制同一 token row到不同 expert segment；route_pos 是 Unpermute 的逆向索引；atomic 返回旧值保证 segment 内不重复。
- **证据位置**：[`operators.h`](../include/raggedroute/operators.h)、[`permute baseline`](../src/permute/cuda_naive/baseline.cu)、[`permute operator`](../src/permute/operator.cpp)。
- **常见翻车点**：把 `route_pos` 说成 expert id；忘记清零 cursor 会跨 repeat 累积越界；说 atomic 决定了稳定排序——当前合同主要保证合法置换，不应擅自承诺 route 内稳定顺序。

### Q25：Permute candidate 做了哪些优化，为什么仍未晋级？

- **面试官意图**：考察 warp aggregation、vector copy 和 full-boundary benchmark。
- **30 秒回答**：候选包括 token-owned Top-2、tile4 direct、`float4` copy、`__match_any_sync` 聚合同 expert route 以减少 cursor atomic，还尝试把 counts/scan/cursor reset 融到 prepare。它在五个 full-from-ids case 的中心结果有优势，但 pure-permute 28-case 全部高 CV、只有 18/28 更快、最差 0.2487x；稳定性和 shape coverage 不够，所以只保留显式 research path。
- **可继续展开的技术细节**：说明 full-from-ids 与 prepared-mapping 是不同语义边界；`float4` 需要对齐/hidden 整除；同 token 的两个 route可共享源 load但写两个非连续 destination；warp aggregation 对 skew 更有效。
- **证据位置**：[`permute candidate`](../src/permute/cuda_candidate/optimized.cu)、[Permute 记录](../docs/permute/performance-record.md)、[`dispatch.cpp`](../src/runtime/dispatch.cpp)。
- **常见翻车点**：只引用 1.4110x 中心结果；把 pure copy 与 mapping preparation 混算；说 v2 已是默认。

### Q26：Unpermute 如何完成 weighted reduce？为什么可以避免原子？

- **面试官意图**：考察 gather/reduce 映射和写冲突。
- **30 秒回答**：每个输出线程负责唯一的 `(token,column)`，遍历该 token 的 Top-K route，通过 `route_pos` 找到 `y_permuted` source row，再乘对应 gate weight 累加并写回 `y[token,column]`。因为每个输出元素只有一个线程写，不需要 atomic；Top-2 时循环只有两项。代价是 source row 间接、两个 source 可能不连续，candidate 用 warp/CTA token mapping 与 `float4` 改善 column 维访问，但尾部和稳定性门禁失败，没有进入 runtime。
- **可继续展开的技术细节**：公式 `y[t,n]=Σ_k w[t,k]*yp[route_pos[t,k],n]`；weight 是 FP32；为什么一个 warp 处理一个 token 可复用 route metadata；output 小/大时 warp 与 CTA 映射的不同。
- **证据位置**：[`unpermute baseline`](../src/unpermute/cuda_naive/baseline.cu)、[`unpermute candidate`](../src/unpermute/cuda_candidate/optimized.cu)、[Unpermute 记录](../docs/unpermute/performance-record.md)。
- **常见翻车点**：说成 scatter-add 并需要 global atomic；忘记 gate weight；把 benchmark-only candidate 说成公开 `CudaOptimized` family。

---

## 8. Grouped GEMM、ragged 调度与 MoE

### Q27：Grouped GEMM 和普通 Dense GEMM 的根本区别是什么？

- **面试官意图**：判断你是否理解 MoE expert computation 的动态性。
- **30 秒回答**：Dense GEMM 只有一个固定 `M×K` 乘 `K×N`；Grouped GEMM 为每个 expert 执行 `M_e×K` 乘自己的 `K×N` 权重，其中 `M_e=offsets[e+1]-offsets[e]` 会随 routing 动态变化。总 route 数固定为 `R`，但工作分配可能 uniform、Zipf、single-hot，也可能出现 empty/tiny expert，所以调度要面对不均匀 tile、尾波和 task-map 开销，而不是只优化一个平均 M。
- **可继续展开的技术细节**：naive 用 `grid.z=expert` 和 `max_expert_tokens` 给出保守 grid.y；每个 CTA 仍从 device offsets 判定真实 begin/end。讨论 padding-to-max 会浪费多少工作，以及 ragged computation 为什么避免这种 padding。
- **证据位置**：[`GroupedGemmArgs`](../include/raggedroute/operators.h)、[`grouped baseline`](../src/grouped_gemm/cuda_naive/baseline.cu)、[Grouped GEMM 研究记录](../docs/grouped_gemm/research-notes.md)。
- **常见翻车点**：把 `M_e` 说成固定 batch size；忽略每个 expert 权重不同；说 `max_expert_tokens` 是从 GPU 同步读回的实际最大值。

### Q28：expert skew、empty expert 和尾波怎样影响 GPU 利用率？

- **面试官意图**：考察 load balance、waves 与 MoE workload distribution。
- **30 秒回答**：uniform routing 让 expert tile 数接近，CTA 比较均衡；skew 会让少数 hot expert 拥有大量 row，其他 expert empty/tiny。静态按 expert 分 grid 时，一部分 CTA 很快结束，hot expert 的 CTA 延长最后一波，导致 SM 间 active cycles 不均；如果总 CTA 数只够一 wave/SM，任何不均都会直接暴露成尾延迟。优化需要按 tile/task 而不是只按 expert 调度，但 task-map 本身也有成本。
- **可继续展开的技术细节**：解释 wave quantization、SM active-cycle max/min、L2 slice imbalance；对比 Split-K、persistent queue、prefix/task descriptor、work stealing 的适用条件。
- **证据位置**：[L3 三线路报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)、[Grouped GEMM 记录](../docs/grouped_gemm/performance-record.md)、[`grouped candidate`](../src/grouped_gemm/cuda_candidate/optimized.cu)。
- **常见翻车点**：说 skew 只影响 cache；认为 persistent kernel 自动均衡；只测 uniform shape 后宣称支持 MoE routing。

### Q29：Grouped GEMM candidate 为什么被拒绝？`0.805x` 和 `0.467x` 怎么解释？

- **面试官意图**：检查你能否用完整证据解释失败，而不是回避负结果。
- **30 秒回答**：正式 clean 三进程、十 shape strict-FP32 比较中，speedup 定义为 CUTLASS latency / candidate latency。ratio-of-sums 只有 `0.805x`，说明按整组总时长 candidate 比 CUTLASS 慢；最差 `T=2048,E=64,K=N=128,uniform` 只有 `0.467x`。即使 tiny/single-hot 有局部胜点，也不能覆盖大 T 反例，因此 candidate 保留为 benchmark research code，没有进入 runtime。
- **可继续展开的技术细节**：shape geometric mean 约 0.907x 与 ratio-of-sums 回答不同问题；后者让长耗时 shape 权重更高。NCU 的 106 registers/thread、one wave/SM、26.62% achieved occupancy 和 SM/L2 imbalance 是原因线索，不是单因果证明。
- **证据位置**：[Grouped GEMM 正式报告](../docs/reports/rtx3080-grouped-gemm-sm86-a5df6eb.md)、[性能记录](../docs/grouped_gemm/performance-record.md)、[L3 报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)。
- **常见翻车点**：把 0.805x 说成“快 80.5%”；只引用最高 2.208x；把 profiler 指标直接断言为唯一根因。

### Q30：为什么不直接给 Grouped GEMM 做 shape dispatch，把局部胜点放进去？

- **面试官意图**：考察 dispatcher 过拟合、coverage 与真实 workload 权重。
- **30 秒回答**：局部胜点还不足以定义稳定的连续区间，尤其当前缺少版本化真实 route trace、working-set/cache 语义和三态 promotion evaluator。写几个精确 shape 的 `if` 会过拟合 benchmark，并可能在邻近 shape 或 distribution 变化时回退。更合理的是先固定 trace 权重与反例，再一次验证一个 scheduling 或 register 假设，最后让 promote/retain/reject 有可审计规则。
- **可继续展开的技术细节**：讨论 cold/warm weight cache、expert weight working set、uniform/Zipf/single-hot 权重；dispatcher 的维护成本、fallback correctness 和边界 discontinuity。
- **证据位置**：[开发路线图](../docs/development-roadmap.md)、[Grouped GEMM 优化计划](../docs/grouped_gemm/optimization-plan.md)、[实现状态](../docs/implementation-status.md)。
- **常见翻车点**：为了保住亮点说“以后加个 if 就行”；把没有进入 `main` 的 realistic/vLLM-semantic 分支说成已有 trace 结论。

### Q31：persistent scheduling 为什么可能有效，也为什么可能失败？

- **面试官意图**：考察动态调度的收益模型与成本模型。
- **30 秒回答**：persistent kernel 让有限 CTA 反复领取 expert tile，理论上可减少静态 grid 的尾波并均衡 skew，还能减少逐 expert launch。代价是 task descriptor/prefix 查找、全局队列或原子、分支、weight 切换、tile 通用化和更长 register live range；工作很少或分布均匀时，这些开销可能超过收益。项目现有 candidate 的局部胜点和整体失败正说明调度机制必须按分布验证。
- **可继续展开的技术细节**：比较 host per-expert launch、static 3D grid、precomputed task map、persistent fetch；说明 task 粒度太小会增加调度开销，太大又失去均衡。
- **证据位置**：[`grouped candidate`](../src/grouped_gemm/cuda_candidate/optimized.cu)、[`optimized_register.cu`](../src/grouped_gemm/cuda_candidate/optimized_register.cu)、[Grouped GEMM 研究记录](../docs/grouped_gemm/research-notes.md)。
- **常见翻车点**：把 persistent 等同“kernel 一直不退出所以零开销”；没说退出条件/任务来源；同时更换 scheduler、tile 和 pipeline 导致无法归因。

---

## 9. Benchmark、公平性与证据

### Q32：为什么要分 L1、L2、L3？

- **面试官意图**：判断你是否会隐藏必要成本或混淆优化层级。
- **30 秒回答**：L1 只回答 kernel body 的机制是否更快，reset/workspace 可以是显式前置；L2 测完整 public operator，必须包含每次调用所需的 reset、mapping preparation 和 dispatch 合同；L3 测组合后的 MoE chain，workspace 可预分配，但每 invocation 必需工作不能排除。三层分开后，才能解释“copy kernel 变快但完整 Permute 变慢”，也不会把局部优化误报成端到端收益。
- **可继续展开的技术细节**：举 Histogram counts reset、Permute cursor reset/from-ids mapping、library host-offset 边界例子；`chain_from_tokens` 与 `chain_from_logits` 的不同入口。
- **证据位置**：[Benchmark 架构](../docs/benchmark-architecture.md)、[`runner.cpp`](../benchmarks/core/runner.cpp)、[`chain_adapter.cpp`](../benchmarks/adapters/chain_adapter.cpp)。
- **常见翻车点**：把 L1 speedup 写成项目 end-to-end speedup；把 input generation、H2D 或 CPU oracle错误计入 GPU operator；隐去 candidate 每次必须做的 preprocess。

### Q33：一次可信的 CUDA benchmark 需要哪些控制项？

- **面试官意图**：考察 warmup、重复测量、同步和环境噪声。
- **30 秒回答**：先冻结硬件/driver/toolkit/build、shape/dtype/layout/seed、math 与 cache policy；warmup 覆盖 lazy init、allocator、库 handle 和 cache；用同一 stream 与 CUDA Event 测同一边界，足够 repeats/samples，并用独立进程观察进程间噪声。记录 p50/p95/CV，保持时钟、电源和竞争 workload 稳定；validation 放计时外，但每次 invocation 必需工作必须在计时内。
- **可继续展开的技术细节**：区分 repeat 是一个 sample 内重复、sample 是统计分布、process 是独立启动；短 kernel 要防 event 分辨率、launch overhead 和状态污染；stateful operator 需 reset。
- **证据位置**：[`runner.cpp`](../benchmarks/core/runner.cpp)、[`run_benchmarks.py`](../scripts/run_benchmarks.py)、[Benchmark 架构](../docs/benchmark-architecture.md)。
- **常见翻车点**：只跑一次；用 CPU wall time但不同步；每轮输入或 state 不同；只报 best-of。

### Q34：p50、p95、CV 和多进程分别说明什么？

- **面试官意图**：判断你能否解释稳定性而非只背中心值。
- **30 秒回答**：p50 表示中心延迟，p95 暴露尾部；CV 是标准差/均值，用于判断相对波动，多进程则检查初始化、系统状态和外部干扰是否让方向改变。candidate p50 更快但 p95/CV 很差，不适合直接晋级；F2 就是中心看起来有收益但 CV 和 L3/fallback gate 失败的例子。
- **可继续展开的技术细节**：CV 对均值接近零的指标要谨慎；样本数不足时 p95 不稳定；跨进程可以报告 process center 的 CV，但不能用它替代原始 sample 分布。
- **证据位置**：[F2 报告](../docs/reports/rtx3080-histogram-scan-fused-sm86-v2.md)、[Histogram 记录](../docs/histogram/performance-record.md)、[`aggregate_results.py`](../scripts/aggregate_results.py)。
- **常见翻车点**：把 p95 当置信区间；认为 CV 越低一定越快；candidate 只赢 p50 就宣布 promotion。

### Q35：geometric mean、ratio-of-sums 和单 shape speedup 有什么区别？

- **面试官意图**：考察多 shape 汇总是否会掩盖回退。
- **30 秒回答**：单 shape speedup 只回答一个 workload。shape geometric mean 给每个 shape 近似相同乘法权重，适合看典型相对方向；ratio-of-sums 是 `Σbaseline_latency / Σcandidate_latency`，长耗时 shape 权重更大；trace-weighted ratio-of-sums 还应使用真实出现频率。无论汇总值多好，关键反例和 unmatched case 都不能被隐藏。
- **可继续展开的技术细节**：说明 latency speedup 大于 1 才是 candidate 快；0.805x 表示慢而不是快；为什么算术平均 speedup 容易被极端小 denominator 扭曲。
- **证据位置**：[`compare_results.py`](../scripts/compare_results.py)、[Grouped GEMM 记录](../docs/grouped_gemm/performance-record.md)、[`comparison schema`](../docs/benchmark-architecture.md)。
- **常见翻车点**：把 0.805x 解读为提升 80.5%；只报 geomean 不报最差 shape；对不匹配 case 静默丢弃。

### Q36：CUDA Event、NSYS、NCU 分别回答什么问题？

- **面试官意图**：检查工具选型与测量层级。
- **30 秒回答**：CUDA Event 在 unprofiled Release 中给 GPU stream 上的 latency，是发布数字主来源；NSYS 看系统 timeline、kernel 序列、launch gap、CPU API、copy/同步和各阶段占比；NCU 对少量代表 kernel 看 registers、occupancy、memory traffic、cache、指令和 stall。三者互补：Event 判断是否真的更快，NSYS 定位时间花在哪，NCU解释某个 kernel 为什么可能慢。
- **可继续展开的技术细节**：NVTX range 定义 capture scope；NCU replay 会扰动执行；NSYS kernel duration可用于阶段占比和方向解释，但不替代同协议 release event latency。
- **证据位置**：[L3 三线路报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)、[`profile_benchmarks.py`](../scripts/profile_benchmarks.py)、[RTX 3080 profiling 报告](../docs/reports/rtx3080-naive-profile-a9489ab.md)。
- **常见翻车点**：用 NCU/NSYS duration 算 speedup；对全套 workload 跑 NCU 再把 replay 总时长当真实延迟；看到一个 counter 就下结论。

### Q37：为什么 profiler duration 不能直接作为 speedup？

- **面试官意图**：考察观测扰动、采样范围和证据资格。
- **30 秒回答**：profiler 会注入 tracing、metric collection、serialization 或 kernel replay，采集范围也可能只覆盖某个 kernel/range；它改变了原执行环境。Profiler 数据适合解释 stage contribution、resource 和瓶颈方向，speedup 应来自未 profile 的、相同 release 协议与 CUDA Event 样本。不同 profiler、不同 capture filter 的 duration 更不能直接相除。
- **可继续展开的技术细节**：说明 NCU 多 pass replay、NSYS trace buffer/API tracing 开销、range capture 的 excluded work；为什么 profile 前先用 Event 建立可复现基线。
- **证据位置**：[RTX 3080 profiling 报告](../docs/reports/rtx3080-naive-profile-a9489ab.md)、[README“Measurement boundaries”](../README.md)、[L3 报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)。
- **常见翻车点**：把 NSYS summary 中百分比乘 release latency当精确算子延迟；跨工具相除；只 profile candidate 不 profile baseline 就归因。

### Q38：comparison 为什么要 fail closed？

- **面试官意图**：检查你如何防止自动脚本生成伪 speedup。
- **30 秒回答**：只有 GPU/build、语义、math、seed、measurement level、cache policy、repeats、samples 和 excluded steps 等 pairing fields 一致，比较才有资格。任何字段缺失或不匹配都拒绝计算，而不是“尽量 join”；因为一个看似相同的 case id 可能实际测了不同工作。suite v2 还要求一个 logical case 下恰有一个 promotion baseline。
- **可继续展开的技术细节**：raw JSONL → aggregate.v2 → comparison.v1；说明 provenance、variant id、baseline role、missing/unmatched cases；hash/manifest 用于回溯。
- **证据位置**：[`compare_results.py`](../scripts/compare_results.py)、[`aggregate_results.py`](../scripts/aggregate_results.py)、[Benchmark 架构](../docs/benchmark-architecture.md)。
- **常见翻车点**：根据文件名手工拼表；不匹配 case 静默 drop；一个 case 设多个 promotion baseline。

### Q39：最新 L3 的 `69.734 us` 和 `3.524x` 应该怎么安全表述？

- **面试官意图**：测试你会不会把诊断性跨后端结果夸大成产品性能。
- **30 秒回答**：这是 `main` 已归档的一个固定 strict-FP32 workload：`T=512,E=64,top_k=2,K=N=128`，selected CUDA research chain 三进程 aggregate p50 为 `69.734 us`；相对 Triton reference 的观察比值是 `3.524x`。但它跨 compiler/runtime stack，而且是 research chain 不是 public `Auto`，所以只能作为三线路诊断，不能写成生产 speedup或普遍超过 Triton。
- **可继续展开的技术细节**：library chain 还有 Top-K 与 host-offset boundary 差异；Triton p50 `245.760 us`；CUDA chain 中 Grouped GEMM 占 NSYS kernel time 64.4%。强调固定 shape、三进程和 boundary。
- **证据位置**：[L3 三线路报告](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)、[`report_data.json`](../docs/reports/l3_three_way_20260805/report_data.json)。
- **常见翻车点**：简历写“端到端比 Triton 快 3.524x”；省略 fixed shape；说这是 public `Auto` 或 production latency。

---

## 10. 项目反思、生产边界与后续路线

### Q40：为什么它不是生产 MoE runtime？如果继续做，下一步是什么？

- **面试官意图**：检查工程现实感、范围控制和优化优先级。
- **30 秒回答**：当前只覆盖单 GPU strict-FP32 routing、重排、一次 expert linear 和 weighted reduce；没有完整 FFN/激活/Down GEMM、FP16/BF16 Tensor Core、多 GPU EP/All-to-All、continuous batching、生产流量 trace 和 H100 实卡验证。下一步先在独占 RTX 3080 复测或 demote F2，再围绕 Grouped GEMM 一次验证一个 task-map 或 register 假设，同时补真实 route trace 与 cache/working-set 语义；低精度和 H100 必须等真实实现与实卡证据后再写简历。
- **可继续展开的技术细节**：若扩到真实 MoE：Gate-Up GEMM、activation、Down GEMM、capacity/drop policy、EP ownership、collective、quantization、graph capture、allocator/workspace、SLO/trace。说明 Mac 当前只能做文档/CPU host checks。
- **证据位置**：[README“Current limitations”](../README.md)、[开发路线图](../docs/development-roadmap.md)、[实现状态](../docs/implementation-status.md)。
- **常见翻车点**：把未来路线说成已实现；承诺 H100 只需重新编译；在无 CUDA 的 Mac 上制造新性能结论。

---

## 11. 八轮高压追问链

建议先遮住“优秀回答锚点”口述，每轮控制在 45–90 秒。后续问题故意沿你的上一轮答案收紧，目标是练习证据边界，不是把回答背成长稿。

### 第 1 轮：你说这是 MoE 项目，用 60 秒讲清输入、输出和价值

- **为什么这么问**：先确认你能否说清系统，而不是只会 CUDA 术语。
- **优秀回答锚点**：单 GPU、Top-2、七阶段；`T×H` token 到 logits、route metadata、按 expert 连续化、ragged linear、加权还原；价值是可审计性能工程闭环；明确不是完整 FFN/runtime。
- **容易被追穿的点**：张量 shape 说错；漏掉 `R=2T`；把 Histogram+Scan 融合算成第八语义阶段。
- **下一轮可能追问**：“这么完整的项目里，哪部分是你本人负责的？”

### 第 2 轮：这些大量提交是不是 AI 生成的？你现场能改哪部分？

- **为什么这么问**：核验 ownership 与真实性。
- **优秀回答锚点**：主动承认 Codex 辅助实现和文档；本人主导问题边界、语义、实验协议、promotion/rejection；能沿 `dispatch.cpp`、`runner.cpp`、Histogram promotion、Grouped rejection 讲清并做小改动；不声称所有 kernel 手写。
- **容易被追穿的点**：回答成“AI 只是补全”；无法举一个自己否决的方案；不知道 `Auto` 实际选择什么。
- **下一轮可能追问**：“那你解释一下 public wrapper 为什么需要 workspace 和 caller stream。”

### 第 3 轮：为什么不在 operator 内部 `cudaMalloc`，也不直接同步读 offsets？

- **为什么这么问**：判断你是否理解可集成 CUDA API 与异步执行。
- **优秀回答锚点**：hot path allocation 会引入延迟、隐式同步/allocator contention；caller workspace 便于复用、graph capture 和生命周期管理；caller stream 维持上层调度；Grouped 用 `max_expert_tokens` 作为保守 launch bound，避免 D2H/sync 读取 offsets。
- **容易被追穿的点**：宣称 wrapper 零成本；不知道 Permute workspace 是 cursor；说 `cudaMemsetAsync` 不计 L2。
- **下一轮可能追问**：“你固定了哪些 Top-K 语义，candidate 如何保持一致？”

### 第 4 轮：两个 logit 相等、出现 NaN 或 `+Inf` 时，你的 Top-2 输出是什么？

- **为什么这么问**：确认优化前是否有精确语义合同。
- **优秀回答锚点**：lower expert id tie-break；NaN 排为 `-Inf`；全 NaN fallback ids `{0,1}`、weights `{0.5,0.5}`；selected-softmax 只归一化两个选中值；`+Inf/-Inf` 避免 `inf-inf`；candidate pair merge 必须保留这个全序。
- **容易被追穿的点**：误答成 full softmax；不知道全 NaN；只说 CPU tolerance，不说 id 精确比较。
- **下一轮可能追问**：“那你怎么证明一个 candidate 不只是某个 shape 碰巧快？”

### 第 5 轮：说清一个 candidate 从源码到晋级需要过哪些 gate

- **为什么这么问**：核验 benchmark 与实验设计能力。
- **优秀回答锚点**：固定 SHA/环境/语义/shape/seed；CPU oracle 与边界/随机/redzone/sanitizer；warmup、repeats、samples、独立进程；L1→L2→代表分布→L3；p50/p95/CV；外部 strongest baseline；comparison pairing fail closed；反例/fallback；最后 promote/retain/reject。
- **容易被追穿的点**：只讲平均 speedup；不区分 L1/L2/L3；不知道 Histogram 五进程、Grouped 三进程。
- **下一轮可能追问**：“那为什么 Grouped GEMM 有 2.208x 的点仍然被拒绝？”

### 第 6 轮：局部最高 2.208x，为什么不写进简历并做 shape dispatch？

- **为什么这么问**：测试你是否会选择性汇报与过拟合。
- **优秀回答锚点**：正式十 shape CUTLASS 比较 ratio-of-sums `0.805x`，最差大 T uniform `0.467x`；局部胜点不能覆盖分布；缺真实 trace/cache 权重和连续区间；NCU 106 registers/thread、one wave/SM 与 imbalance 提供后续假设，不是为失败找借口。
- **容易被追穿的点**：把 0.805x 说成提升；无法说出 speedup 方向；用 occupancy 单指标断言根因。
- **下一轮可能追问**：“既然证据这么严格，为什么 F2 gate 失败却还在 `Auto`？”

### 第 7 轮：F2 失败还被 `Auto` 选择，是不是你的验收流程自相矛盾？

- **为什么这么问**：直击项目最大的公开技术债。
- **优秀回答锚点**：承认这是当前需要纠正的状态；源码路径在主分支且机制上合法，但 archived run 受竞争 workload 干扰，CV/fallback/L3 gate 均未通过；当前 README 明确警示；下一次独占 RTX 3080 首要动作是重跑，失败就 demote，不能把它包装成已晋级成果。
- **容易被追穿的点**：辩解“代码正确就够了”；只引用 2.0045x；不愿承认应该撤回 `Auto` 的可能性。
- **下一轮可能追问**：“这离生产系统还差什么？你会怎么排下一阶段？”

### 第 8 轮：如果让你把它接到真实 MoE 推理系统，前三项工作是什么？

- **为什么这么问**：考察迁移到工业 workload 的能力和优先级。
- **优秀回答锚点**：第一，补真实 route trace、expert skew、batch/token 分布和 weight cache/working-set 合同；第二，补完整 FFN/激活/Down GEMM、workspace/stream/graph-capture 与端到端 correctness；第三，再考虑 FP16/BF16 Tensor Core 和多 GPU EP/All-to-All，并在目标实卡重做 baseline/profile。当前先处理 F2 技术债和 Grouped 热点，不提前宣称 H100/生产性能。
- **容易被追穿的点**：直接回答“换成 FP16 就能上线”；忽略 collective、capacity、fallback、SLO 和可观测性；把交叉编译当实卡验证。
- **诚实收口**：如果具体生产流量未知，回答“我会先定义 workload contract，而不是先选 kernel；没有 trace 时只能提出设计和验证方式，不能给线上收益数字。”

---

## 12. 高频证据定位表

| 要回答的问题 | 首选证据 |
|---|---|
| 当前 runtime 支持什么 | [`src/runtime/dispatch.cpp`](../src/runtime/dispatch.cpp)、[`docs/implementation-status.md`](../docs/implementation-status.md) |
| 七阶段/八 adapter | [`include/raggedroute/operators.h`](../include/raggedroute/operators.h)、[`benchmarks/core/registry.cpp`](../benchmarks/core/registry.cpp) |
| 两条 L3 chain | [`benchmarks/adapters/chain_adapter.cpp`](../benchmarks/adapters/chain_adapter.cpp) |
| Top-K tie/NaN/softmax | [`src/topk_gate/cuda_naive/baseline.cu`](../src/topk_gate/cuda_naive/baseline.cu)、[`src/topk_gate/cpu_reference/reference.cpp`](../src/topk_gate/cpu_reference/reference.cpp) |
| Histogram promotion | [`docs/histogram/performance-record.md`](../docs/histogram/performance-record.md) |
| F2 技术债 | [`docs/reports/rtx3080-histogram-scan-fused-sm86-v2.md`](../docs/reports/rtx3080-histogram-scan-fused-sm86-v2.md) |
| Permute mapping/cursor | [`src/permute/cuda_naive/baseline.cu`](../src/permute/cuda_naive/baseline.cu)、[`src/permute/operator.cpp`](../src/permute/operator.cpp) |
| Grouped rejection | [`docs/reports/rtx3080-grouped-gemm-sm86-a5df6eb.md`](../docs/reports/rtx3080-grouped-gemm-sm86-a5df6eb.md) |
| L3 latency/hotspot/NCU | [`docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md`](../docs/reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md) |
| L1/L2/L3、公平配对 | [`docs/benchmark-architecture.md`](../docs/benchmark-architecture.md)、[`scripts/compare_results.py`](../scripts/compare_results.py) |
| 下一步与禁止声明 | [`README.md`](../README.md)、[`docs/development-roadmap.md`](../docs/development-roadmap.md) |

## 13. 面试当天的五条底线

1. 先说 workload、shape、dtype、边界，再说 speedup。
2. `baseline / candidate > 1` 才表示 candidate 对 latency 更快。
3. `Auto`、显式 candidate、benchmark-only baseline 和实验分支必须分开。
4. 性能数字来自已有 RTX 3080 证据；macOS 本轮没有产生任何 CUDA 新结论。
5. 不知道就回到源码、报告与验证方案，不用猜测补齐“漂亮答案”。
