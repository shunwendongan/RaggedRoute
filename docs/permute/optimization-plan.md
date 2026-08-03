# Token Permute SM86 优化方案

## 1. 合同与目标

- 目标设备为 RTX 3080（GA102，SM86）；公开 `TokenPermuteArgs`、caller stream、strict FP32 和 `E * sizeof(int32_t)` cursor workspace 合同不变。
- 输入为 `X[T,K]`、flattened `expert_ids[T,top_k]` 和 exclusive `offsets[E+1]`。输出满足 `route_pos[route]=destination`，可选 `sorted_route[destination]=route`，且 expert 段内每行 bitwise 等于源 token 行。
- expert 段内顺序不属于公开合同；duplicate expert id 是两个独立 route pair。invalid expert id 和非法 offsets 仍为 caller precondition，不在热路径增加 device validation 或同步。
- 显式 implementation ID 用于研究。`KernelFamily::kAuto` 只有在 Release gate 全部通过后才允许切换；没有候选通过时继续使用 `cuda_naive`。

## 2. 单变量候选

| Variant | 相对父版本的唯一变化 | 假设 | 代价/回退 |
|---|---|---|---|
| `cuda_atomic_vectorized_128` | route-owned 128-thread CTA 的 scalar copy 攓为对齐 `float4` | 16-byte load/store 减少 copy 指令并改善 memory issue efficiency | `K%4!=0` 或任一 payload pointer 未 16B 对齐时走 scalar |
| `cuda_atomic_vectorized_64` | 只将 CTA 从 128 改为 64 threads | 小 K 下减少空闲线程并增加 waves | copy、atomic、mapping 与 fallback 均不变 |
| `cuda_atomic_vectorized_256` | 只将 CTA 从 128 改为 256 threads | 宽 K 下增加单行并行度 | 小 K 可能浪费线程或降低 resident blocks |
| `cuda_token_owned_top2` | 一个 CTA 处理 token 的两个 route | `top_k=2` 时 X 只读取一次并写两个 destination | `top_k!=2` 回退 128-thread variant；仍需两次 cursor atomic |
| `cuda_block_partial` | 256 routes/CTA 在 shared memory 求 local rank，每个 active expert/CTA 一次 global reserve，再单独 copy | single-hot/Zipf 热点下减少 global atomic serialization | 两个 kernel；shared `counts[64]`/`bases[64]`；不增加外部 workspace |

`cuda_candidate` 是经证据选择的别名；`cuda_candidate_from_ids` 明确包含 histogram、exclusive scan 和 candidate permute。L3 `cuda_permute_candidate` 只替换 chain 中的 permute，其他算子保持 naive。

本轮不做 shape/skew 自动调度、`cp.async`、Permute+GEMM fusion、新 dtype，也不采用 SM90/SM100 专属的 TMA/WGMMA。

## 3. Correctness 与安全性

- 精确检查 expert segment、route 唯一性、正逆 mapping、可选 `sorted_route` 和逐行 bitwise FP32 内容。
- 覆盖 `T={0,1,5,17}`、`E={1,4,8,64}`、`top_k={1,2,4,E}`、`K={0,1,3,4,19,255,256,257}`，并覆盖 uniform、Zipf、single-hot、round-robin、空 expert 和 duplicate id。
- 覆盖 sorted 开关、16B 对齐和故意未对齐 payload、workspace 缺失/不足/错位、redzone、非默认 stream 和 route-count overflow。
- 所有候选必须先通过 CTest，再通过 Compute Sanitizer memcheck、initcheck、racecheck、synccheck；性能结果不能替代 correctness。

## 4. Release 测量与晋升

正式配置为 `configs/benchmark_permute_candidate_release.json`：seed `20260729`，5 个独立进程，20 warmups、30 samples；L1 stateful kernel repeats 为 1，L2 为 10，cold scrub 为 1。覆盖 tiny/tail、sorted on/off、uniform/Zipf/single-hot、K tail、generic top-k、large/wide 和 64 MiB cold scrub。

报告 p50/p90/p95、mean、stddev、CV、per-shape speedup、ratio-of-sums、tokens/s、route-rows/s、logical effective GB/s、workspace 和 kernel launch 数。logical bytes 与 NCU physical DRAM/L2 bytes 分开报告。

候选只有同时满足以下条件才可晋升：

- correctness 与 sanitizer 全通过；CV `<=0.10`，超限只允许 GPU idle 后完整重跑一次；
- 至少 80% Release shapes 加速，ratio-of-sums speedup `>=1.03`；
- 任一 shape 回退不超过 5%，workspace 增长不超过 25%；
- GPU UUID、数学语义、输入和测量边界一致。

若多个候选通过且 ratio-of-sums 差异小于 1%，优先 kernel 更少、workspace 相同且实现更简单的版本。vLLM 仅作能力定位，默认 dispatch 晋升首先依据同合同 `cuda_naive` A/B。

## 5. Profiler 顺序

先用 NSYS（CUDA/NVTX，关闭 CPU sampling）确认 anchor L2 和两条 L3 chain 的 memset、placement、copy、library sort 与 launch 结构；再对 baseline 和候选跑 NCU basic。NCU detailed 仅保留最终候选与 baseline 的 `(512,64,2,256,Zipf-1.4)` 和 `(2048,64,2,1024,single-hot)`。

诊断关注 grid/block、waves/SM、register/shared memory、achieved occupancy、SM/memory/DRAM throughput、L1/L2 hit、sectors/request、atomic/serialization、long scoreboard、barrier/wait 和 local memory。`not_collected`、`unsupported_or_unknown` 原样保留；profiler duration 不用于 Release latency 声明。

## 6. 开源与论文依据

所有实现均在本仓库重新编写，不复制外部 kernel。外部项目仅用于核对 mapping 生命周期、数据流和优化机制；固定 revision 便于审计许可证与版本边界。

| 来源 | 固定 revision / 论文 | 本轮吸收的机制 | 不纳入范围 |
|---|---|---|---|
| [vLLM permute API](https://docs.vllm.ai/en/v0.15.0/api/vllm/model_executor/layers/fused_moe/moe_permute_unpermute/) | baseline `837eae64580c885101ee95b073aafb27a485e7ce` | expert offsets、正逆 mapping、16B row alignment、full-from-ids 与 prepared-mapping 边界 | 不更新正式库基线，不把 prepared mapping 当端到端结果 |
| [MegaBlocks](https://github.com/databricks/megablocks/tree/952db33d6eac334d22c61e47a0d5d41446298784) | MLSys 2023 | sort/histogram/bins/index 数据流和 dropless MoE 背景 | 不移植 block-sparse GEMM |
| [ScatterMoE](https://github.com/shawntan/scattermoe/tree/47b5e1502e5a10e82c8e5945d761b877849871e7) | COLM 2024 | `flatten_sort_count`、scatter-to-scatter layout、避免不必要 materialization 的动机 | fusion 只记为后续工作 |
| [Tutel](https://github.com/microsoft/tutel/tree/ac0e51b943bb7084d4a6a7243e44a4afe04b9911) | MLSys 2023 | fast encode/decode、dispatcher 生命周期和负载倾斜背景 | 不引入分布式通信路径 |
| [FastMoE](https://github.com/laekov/fastmoe/tree/55af4f98eee087cf5b3aac34318abf80c3bcbafd) | FasterMoE, PPoPP 2022 | routing/dispatch 系统边界和动态负载建模 | 不修改容量或调度语义 |
| [Megatron-LM dispatcher](https://github.com/NVIDIA/Megatron-LM/tree/9829b3f1dd16f5233ff7bf50e67f6443c527c2bc) | pinned source | mapping 生命周期、dispatcher 与 chain 边界 | 不引入多 GPU all-to-all |
| [MoEBlaze](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html) | MLSys 2026 | 新近 MoE kernel/system 设计作为后续研究参照 | 本 PR 不反向声明其硬件机制适用于 SM86 |

论文主线：[MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)、[ScatterMoE](https://openreview.net/forum?id=YDZ7GeFLxq)、[Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)、[FasterMoE](https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/20/FasterMoE-Modeling-and-Optimizing-Training-of-Large-Scale-Dynamic-Pre-Trained-Models) 和 [MoEBlaze](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html)。这些论文支持机制选择与边界分析，实际晋升仍完全由本仓库 SM86 correctness、Release A/B 和 profiler 证据决定。

## 7. 决策日志

| 日期 | 版本/唯一变化 | 当前状态 | 保留/淘汰依据 |
|---|---|---|---|
| 2026-08-03 | atomic vectorized 128 | reject；保留 explicit research ID | 高重复 selection 两轮均落后 token-owned Top-2 |
| 2026-08-03 | atomic vectorized 64 | reject；保留 explicit research ID | anchor occupancy 40.5%，两轮 gate 失败 |
| 2026-08-03 | atomic vectorized 256 | reject；保留 explicit research ID | Run 1 ratio 1.0373x，但 coverage 75%、worst 0.6300x，Run 2 也失败 |
| 2026-08-03 | token-owned Top-2 | selected as `cuda_candidate` | 高重复 L2 selection 两轮 ratio-of-sums 1.0456x/1.0463x，排名均为第一；generic top-k fallback 正确 |
| 2026-08-03 | block partial | reject；保留负面实验 | 第二个 launch 和低-wave placement 成本超过 atomic reduction 收益 |
