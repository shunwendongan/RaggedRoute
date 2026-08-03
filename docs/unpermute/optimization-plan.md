# Unpermute 优化方案

## 1. 当前范围

- 代码入口：`src/unpermute/cuda_naive/baseline.cu`、`src/unpermute/operator.cpp`。
- API：`UnpermuteArgs`；按 `route_pos` 从 packed expert output gather，并按 route weights 做 weighted reduce。
- 当前目标是 token-owned、无全局 atomic 的确定性合并；任何 fusion 方案都必须保持该语义。

## 2. 优化假设与顺序

1. **冻结 mapping 与权重语义**：确认每个 token 的 route 数、`route_pos` 方向、权重 dtype/accumulator 和输出布局。
2. **优化 gather 合并访问**：评估 token/route/feature 维度的工作划分、连续输出写入和 packed input 的访问合并。
3. **向量化 weighted reduce**：在 output 对齐时比较 vector width、FMA 指令和 register pressure；保留 tail path。
4. **降低重复读取**：评估 route metadata 缓存、warp-level reuse 和与 grouped GEMM 输出布局协同，避免只在单算子 warm cache 下获益。
5. **评估融合边界**：研究 Grouped GEMM→Unpermute 融合或 producer-friendly layout，但把融合版本和独立 unpermute 结果分开记录。

## 3. 文献到本轮实现的映射

| 来源 | 本轮采用 | 明确不采用 |
|---|---|---|
| [vLLM finalizeMoeRoutingKernel](https://github.com/vllm-project/vllm/blob/837eae64580c885101ee95b073aafb27a485e7ce/csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl) | 128-bit row access、token-owned k-way reduce、公平 production baseline | 不把单纯 `float4` 改写算作创新 |
| [ScatterMoE](https://openreview.net/forum?id=YDZ7GeFLxq) | 用于解释减少 permutation/materialization 的后续方向 | 本 PR 不融合 Grouped GEMM，也不包装 Triton 实现 |
| [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) | dropless mapping、动态 route 和完整 unpermutation 语义 | 不引入 block-sparse expert compute |
| [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)、[FastMoE](https://arxiv.org/abs/2103.13262) | uniform/Zipf 与 dispatch/combine 合同 | 不迁移多 GPU communication 优化 |
| [MoEBlaze](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html) | 后续减少中间 buffer 的研究依据 | 不进入本轮独立算子实现 |

## 4. `cuda_warp_token_vec4` 设计

- 保持 strict FP32、rank 0→1 累加顺序、zero workspace、caller stream、每个输出只写一次且无 atomic。
- `top_k=2`、输入/输出 16-byte 对齐且 `N%4==0` 时使用 128-bit fast path；其余合法 `top_k`、tail 或非对齐地址回退已验证的 naive kernel。
- `N<512`：一个 warp 负责一个 token，4 warps/CTA；lane 0 读取两个 `route_pos`/weight 并用 shuffle 广播。
- `N>=512`：一个 256-thread CTA 负责一个 token；thread 0 读取 metadata 到 shared memory，增加 grid 和行内并行度，避免 large-N 下单 warp/grid underfill。
- candidate 仅由 benchmark/research adapter 直接调用；由于正式 gate 未通过，公开
  `select_kernel`/`unpermute` 不接受该 optimized ID，`Auto` 和显式默认路径均保持 naive。

## 5. 单变量实验账本

筛选数据来自 dirty Release binary，只用于选候选，不能作为晋升数字：

| 版本 | 唯一变化 | 筛选结果 | 决策 |
|---|---|---|---|
| V1 warp scalar | 一 warp/token，lane-0 metadata broadcast | 相对 vLLM 三个 `T1024` N-shape 几何平均约 `1.143x` | 继续 vector 实验 |
| V2 vec4 i1 | 在 V1 上增加 128-bit access | 同一筛选约 `1.208x` | 保留为 small-N parent |
| V3 vec4 i2/i4 | 每 lane items 由 1 改为 2/4 | 约 `1.173x/1.134x` | 拒绝：额外 ILP 未带来收益 |
| V4 8 warps | V2 block 从 4 改为 8 warps | 争议 shape 复测几何平均约 `0.973x` | 拒绝 |
| Hybrid | `N>=512` 改为 CTA/token | 32-shape 筛选几何平均约 `1.088x`；剩一个噪声敏感回退组 | 进入 clean-release gate |

`T=64,N=1024` 的 NSYS median kernel duration 为 warp V2 `3.136 us`、vLLM `2.016 us`。NCU basic 显示 warp V2 只有 16 blocks、achieved occupancy `8.23%`、SM throughput `0.57%`；vLLM 为 64 blocks、`15.86%`、`1.15%`。Hybrid large-N CTA path由这些 underfill 信号驱动，而不是事后猜测。

## 6. 必测维度

- `T/E/N/top_k`、route distribution、output alignment、L1/L2/L3 chain。
- 指标：global load/store efficiency、L1/L2 hit、long scoreboard、register/thread、SM timeline、p95。
- 验收：token-owned CPU reference、权重累加容差、空/倾斜 expert、无 race/无 atomic 语义偏差。

## 7. 晋升门槛与决策记录

- 5 个独立 clean Release 进程；主矩阵 `T={64,512,1024,4096}`、`N={64,128,256,1024}`、uniform/Zipf-1.4、L1/L2。
- 相对 vLLM shape-balanced p50 几何平均至少 `1.05x`；任一主 shape p50 回退不超过 5%，p95 回退不超过 3%。
- correctness、API、stream、redzone、memcheck/initcheck/racecheck/synccheck 全部通过；workspace 保持 0；L3 无可信回退。
- profiler duration 只用于诊断，最终结论只取未插桩 release A/B。

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| 2026-08-03 | V1–V4 single-variable candidates | dirty screening + NSYS/NCU basic | 保留 Hybrid 进入正式 gate；其余拒绝 |
| 2026-08-03 | Hybrid formal gate | 两轮 5-process Release、L3、最终 NSYS/NCU/SASS、四类 sanitizer | **拒绝晋升**：平均收益信号存在，但 55/64 与 63/64 warm candidate groups 的 CV 超过 0.10，且逐 shape p50/p95 gate 均失败；保留 benchmark-only 研究实现和证据 |

正式首轮相对 vLLM 的 warm L1/L2 几何平均为 `1.008x/1.066x`，自动复测为
`1.088x/1.056x`。两轮方向和单 shape 结果不稳定，最差 p50 speedup 分别低至
`0.264x/0.542x` 与 `0.446x/0.492x`，p95 ratio 也远超 `1.03` 上限，因此不能用
几何平均掩盖回退。L3 为 `1.0016x`，baseline/candidate CV 为 `0.435/0.305`，只能判定
“无可信回退、也无可信加速”。完整报告见
[SM86 candidate evidence report](../reports/unpermute-sm86-candidate-eba8f02.md)。
