# Grouped GEMM 研究说明与文献地图

## 1. 数值与布局合同

对 expert `e`，令

```text
M_e = offsets[e + 1] - offsets[e]
X_e = X_packed[offsets[e] : offsets[e + 1], 0:K]
Y_e = X_e @ W[e]
```

其中 `X_packed`、`W`、`Y_packed` 均为 row-major FP32，`W` 的逻辑形状为
`[E,K,N]`。本项目按一次 FMA 计 2 FLOPs，因此 useful work 为
`2 * R * K * N`，`R = sum_e M_e`。不考虑缓存重复读取时，活跃 expert 的最低逻辑流量近似为
`4 * (R*K + E_active*K*N + R*N + E+1)` bytes；benchmark 另报告固定 buffer footprint，
二者不能混用。输出和 accumulator 都是 FP32，不允许 TF32、FP16 或 BF16 改变数值合同。

`E<=64`，允许 `M_e=0/1`、单 active expert、非对齐 `K/N` 和 `hidden=0`。
`max_expert_tokens` 只是调用者提供的保守 launch bound，不等于真实 `max(M_e)`；调度器必须从
device offsets 得到真实 tile 数，不能按该上界为每个 expert 过量发射工作。

## 2. 性能问题分解

- **小 `M_e` 与空 expert**：算术强度低，per-expert launch、metadata 搜索和尾 tile 占比高。
- **倾斜分布**：single-hot 有较好的权重复用但并行 problem 数少；uniform 有更多独立 tile，
  也更容易让每个 CTA 处理不同 expert 的冷权重。Zipf 同时暴露两种效应。
- **尾波**：总 tile 数除以 resident CTA 数后的最后一波会直接影响短 kernel 延迟。
  persistent scheduler 能消除虚假 tile，却不能自动消除 tile 大小、register pressure 或真实负载不均。
- **metadata**：当前实验 kernel 在每个 CTA 中从 `offsets` 构造至多 65 项的 shared prefix，
  省去 workspace 和 host precompute，但 CTA 数增大时会重复付出 `O(E)` 成本。
- **缓存复用**：同一 expert 的连续 tile 可复用 `W_e` 的 L2 数据；tile 顺序和分布会改变复用距离。
  论文或其他 GPU 上的 cache-aware 顺序只能形成假设，不能直接当作本机结果。
- **SM86 资源边界**：RTX 3080 实测 68 SM。可用 CUDA core、shared memory、warp shuffle、
  conventional cache 和 `cp.async`；不可使用 Hopper TMA/WGMMA、warpgroup 或 Blackwell TMEM。
  strict FP32 也排除了 Tensor Core/TF32 换精度提速。

## 3. 本轮候选与可证伪假设

| ID | 单变量机制 | 预期 | 主要风险 |
|---|---|---|---|
| C0 `tiled16_sync_v0` | 复现历史 16x16 shared tile | 验证旧提交 `70327bb` | 256 threads、barrier 和小 tile 开销 |
| C1 `persistent16_v1` | device prefix + persistent round-robin | 去除按 `max_expert_tokens` 的虚假 tile | 每 CTA 重建 prefix；mainloop 未变 |
| C2 `register16x32_sync_v2` | 128 threads、每线程 2x2 输出 | 降低 shared 指令和 barrier 压力 | accumulator/live range 增加 |
| C3 `register16x32_async_v3` | aligned `float4` + `cp.async` 双缓冲 | 重叠 global-to-shared 与计算 | register/shared 增长降低 occupancy |
| V4 `async_full_v4` | occupancy API 给出的完整 resident CTA 数 | 修复 C3 人为 2 blocks/SM 上限 | 更多 CTA 重复 metadata 工作 |
| Final `sm86_fp32_v1` | tiny direct，否则 V4 | shape-aware 组合 | 一个静态阈值无法覆盖所有 `M_e` 分布 |

所有这些 ID 都是 benchmark-only。最终组合没有通过 CUTLASS 晋级门禁，public runtime 的
`KernelFamily::kCudaOptimized` 对 Grouped GEMM 仍明确拒绝，`kAuto` 仍选择 naive。

## 4. 文献与实现地图

下表只提炼机制与待验证假设；第三方性能数字不属于 RaggedRoute 的实测结果。

| 年份 / 类型 | 工作 | 核心机制 | 对本项目的启发 | 不可直接迁移的边界 |
|---|---|---|---|---|
| NVIDIA 官方文档 | [CUTLASS Grouped Kernel Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html) | persistent CTA、round-robin tile mapping、device-only/host-precompute schedule | 真实 tile prefix、固定 CTA 池和 scheduler 成本拆分 | 文档示例不等于本项目 shape；host precompute 会引入 workspace/通信 |
| Triton 教程 | [Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) | 固定 virtual SM 数，program id 跨 problem tile 前进 | 一维 persistent tile space 与问题映射 | 教程 dtype/后端不同；TMA 示例不是 SM86 能力 |
| 2025 / 工程文章 | [Persistent cache-aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) | persistent programs、grouped launch tile ordering、expert-weight locality | 比较 uniform/Zipf/single-hot，显式测 tile order 与 cache | BF16/H100/TMA 和其性能数字不能迁移到 strict FP32 SM86 |
| 2023 / MLSys | [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) | 将 dropless MoE 写成 block-sparse 运算 | 当 grouped dense tile 浪费严重时评估 block-sparse 替代路线 | 改变数据结构、训练/反向范围和 API，不是当前 kernel 的局部优化 |
| 2024 / COLM | [ScatterMoE](https://openreview.net/forum?id=YDZ7GeFLxq) | 避免 padding/过量 copy，融合 reordering 与 expert linear | 后续评估 Permute + Grouped GEMM fusion | 会改变测量边界和 API；本轮只做单算子 strict FP32 |
| 2023 / MLSys | [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html) | 动态并行与流水、通信/计算重叠 | `M_e`/capacity 应进入 workload 与调度决策 | 主要面向分布式训练和 A100 集群，不是单 GPU kernel 结论 |
| 2022 / PPoPP | [FasterMoE](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/20/FasterMoE-Modeling-and-Optimizing-Training-of-Large-Scale-Dynamic-Pre-Trained-Models) | 性能模型、动态 shadowing、细粒度调度 | 把负载失衡作为一等 benchmark 维度 | 系统级训练/通信优化不能替代本机 kernel A/B |
| 2026-07-28 / arXiv v2 | [Decoding the Skew](https://arxiv.org/abs/2607.23099) | distribution-aware benchmark 与 GPU-resident kernel dispatch | 增加可控 skew、effective-expert/真实 trace；验证无单一 kernel 全覆盖的假设 | 近期预印本；B200、低精度 fused-MoE 与 conditional graph 结果不适用于 SM86 strict FP32 |
| 2025 / arXiv | [SonicMoE](https://arxiv.org/abs/2512.14080) | IO/tile-aware 优化与 token rounding | 研究 tile padding 与物理/有效吞吐差异 | token rounding 会改变路由/模型语义，本项目当前禁止 |
| 2026 / arXiv | [Fine-grained Computation-Communication Overlap](https://arxiv.org/abs/2607.19539) | tile-level signaling，persistent compute/communication kernels | 后续多 GPU 可按 tile 暴露完成事件 | 面向 A100 多 GPU All-to-All，不属于本轮单 GPU operator |

## 5. 可复现检索记录

- 检索日期：2026-08-03。
- 关键词：`grouped GEMM persistent scheduler`、`MoE grouped GEMM cache-aware`、
  `dropless MoE block sparse`、`distribution-aware MoE kernel dispatch`。
- 优先来源：NVIDIA/Triton/PyTorch 官方文档、MLSys/PPoPP/OpenReview/arXiv 原文。
- 纳入条件：直接讨论 grouped/ragged expert GEMM、动态 expert 负载、数据重排融合或调度。
- 排除条件：只给模型质量、没有执行机制，或仅适用于 Hopper/Blackwell 且没有可转译目标的工作。

## 6. 后续最有价值的实验

1. 将 `T=2048` 反例拆成 scheduler metadata、K-loop 和 tile 尺寸三部分；优先降低 106 registers/thread。
2. 增加真实 route trace、Dirichlet/effective-expert sweep，而不是只用 uniform/Zipf/single-hot 三点。
3. 比较 per-CTA prefix 与一次 device precompute 的成本，但 workspace/API 变化必须另立版本。
4. 只有单算子重新达到 CUTLASS 门禁后，才评估 Permute + Grouped GEMM fusion 和 L3 chain。
