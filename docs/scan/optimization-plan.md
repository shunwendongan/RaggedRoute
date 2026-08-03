# Exclusive Scan：sm_86 深度优化与证据计划

## 1. 固定合同与范围

本轮只研究 `src/scan` 的独立 int32 exclusive scan，目标硬件为 RTX 3080 / GA102 / sm_86，shape 为 `1<=E<=64`。公共语义固定为：

- `offsets[0]=0`；
- `offsets[i+1]-offsets[i]=counts[i]`；
- `offsets[E]=sum(counts)`；
- 输入非负且总和不超过 `INT32_MAX`；
- caller stream、0 workspace、hot path 无分配和同步；
- 支持 4-byte 对齐的完全分离指针，以及 `counts==offsets` 原地扫描；不承诺其他部分重叠。

L2 operator latency 是 dispatch 决策指标，L1 kernel body 与 NSYS/NCU 只解释机制。Histogram→Scan 融合会改变测量边界，不属于本轮实现。

## 2. 为什么不直接采用经典大数组 Scan

Blelloch 的 work-efficient upsweep/downsweep给出了通用并行 scan 的算法基础；GPU Gems 进一步讨论了 shared-memory tree、bank conflict 和分层大数组 scan。它们对大输入非常重要，但 `E<=64` 的 RaggedRoute Scan 只有一个极小 CTA：更多 barrier、shared memory 或 device-wide 调度可能比 64 次整数加法本身更贵。

现有 SASS 与 profiler 证据支持将问题归类为 launch/underfill bound：naive 是 20 registers/thread 的单线程 LDG/IADD/STG 链；CUB WarpScan 是 5 级 shuffle；CUB BlockScan 使用 128 threads、block barrier 与 shared storage。RTX 3080 有 68 SM，而所有独立 Scan 候选都只有一个 block，`waves/SM=0.000919`。因此本轮优先验证“单 warp、多元素/线程、无 barrier”的形状特化，而不是多 block、`cp.async`、persistent scheduling 或更大 shared-memory tree。

## 3. 单变量候选

| 候选 | 唯一变化 | 预期收益 | 主要风险 | 结果 |
|---|---|---|---|---|
| C1 `cuda_warp_striped` | 32 lanes；`E>32` 分两半各做一次 warp scan | 标量访问完全合并 | 两条 shuffle 依赖链 | 拒绝：不如 C2 稳定，收益不一致 |
| C2 `cuda_warp_blocked_scalar` | 每 lane 相邻读取两个元素，pair sum 后只做一次 warp scan | shuffle 链减半 | stride-2 标量 sector amplification | 最快中心趋势，但稳定性/p95 门槛失败 |
| C3 `cuda_warp_blocked_vector` | C2 的 8-byte aligned 路径改用 `int2` | 用 64-bit LDG/STG 恢复合并事务 | 对齐分支、tail 和 terminal store 复杂度 | 拒绝：SASS 已向量化，但未获得足够收益 |

共同约束：全部输入必须先读完再写回以保证原地语义；显式使用正确 active mask 的同步 shuffle；禁止 local spill、shared memory、barrier、atomic、RDC 和运行时 autotune。候选只通过 benchmark adapter 显式选择，研究期间不修改 `kAuto`。

## 4. 测量与晋升门槛

主 sweep 覆盖 `E=1..64`、L1/L2、`R=4096`、uniform、warm cache；每个 variant 共享 seed、输入和计时边界。补充验证覆盖边界 shape、skew/热点分布、原地与分离指针、8-byte 与仅 4-byte 对齐、cold scrub 和单调用 tail。正式性能来自未插桩 Release A/B；NSYS 先确认 launch/API 边界，NCU basic 后仅在必要时升级 detailed/source。

Hybrid 只有同时满足以下条件才允许进入默认 dispatch：

- 全部 correctness 和 Compute Sanitizer 通过，workspace 仍为 0，SASS/NCU 无 local spill；
- 至少 3 个独立进程，稳定组 all-samples CV 不超过 10%；
- L2 trace ratio-of-sums 至少 `1.03x`，且至少 80% 的 candidate-used shapes 获益；
- 任一 shape p50 退化不超过 5%，batch-mean p95 退化不超过 3%；
- 阈值两侧相邻 shape 的方向跨进程一致；
- vector path 相对同 mapping scalar path 至少快 3%，且由 SASS 和 transaction 证据确认；
- 方差超限后扩展到 5 个进程、100 samples，仍超限则标记 `variance-limited`，不得晋升。

## 5. 本轮决策

研究提交 `bd68fd2`、`d26f8a6`、`32bf6c9` 完成了 C1/C2/C3、统一 exact-int32 benchmark metadata 和 5 进程稳定性复测。C2 在 `E=29..64` 的 ratio-of-sums 为 `1.0503x`，33/36 shapes 的 p50 获益；但所有 36 对的 baseline 与 candidate CV 都超过 10%，最坏 p95 ratio 达 `7.042x`，且 E=32 p50 退化 2.56%。C3 虽生成真实 `LDG.E.64/STG.E.64`，36 个 shape 中只有 6 个比 C2 快至少 3%，ratio-of-sums 反而为 `0.9936x`。

因此本轮没有赢家：最终树不保留 C1/C2/C3 源码，不注册 optimized implementation id，不改变 `kAuto`。保留的是 4-byte 对齐合同检查、加强后的 API/原地/redzone 测试、scan 专用 CUB 基准配置、报告与冻结证据。

## 6. 后续可执行方向

NSYS 中独立 Scan 占固定 L3 链 GPU kernel 时间 3.0%，高于 2% 观察阈值；但 standalone 候选未满足 `1.03x` 的稳定晋升合同。后续应单独提出 Histogram→Scan 融合 PR，重新冻结 L2/L3 边界，验证“减少一次 launch 和 metadata round trip”是否优于增加的 CTA 协作、shared memory 和寄存器生命周期。它不能复用本轮 standalone speedup 作为晋升证据。

## 7. 推荐文献与阅读顺序

检索范围：并行 prefix scan 的基础算法、GPU intra-warp/block/device 实现、单 pass 通信规避，以及 scan 与上游阶段融合。排除标准：只讨论 CPU SIMD、未给出 GPU 算法机制、或需要 Hopper/Blackwell 专有能力且无 sm_86 回退的工作。

| 层次 | 文献 | 对本项目的意义 | 不应直接照搬的部分 |
|---|---|---|---|
| 算法基础 | [Blelloch, *Prefix Sums and Their Applications* (1990)](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html) | exclusive scan 定义、work/depth、upsweep/downsweep | PRAM 模型不包含 GPU launch 成本 |
| GPU tree scan | [Harris, Sengupta, Owens, *GPU Gems 3 Chapter 39* (2007)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) | work-efficient tree、shared-memory bank conflict、分层 scan | 面向旧架构和大数组；tiny `E` 的 barrier 成本不同 |
| Warp 分解 | [Sengupta, Harris, Garland, *Efficient Parallel Scan Algorithms for GPUs* (2008)](https://research.nvidia.com/publication/2008-12_efficient-parallel-scan-algorithms-gpus) | 以 intra-warp scan 作为 block/device scan 基元，直接支撑 C1/C2 假设 | 论文吞吐问题规模大于本项目单 warp metadata |
| 单 pass device scan | [Merrill, Garland, *Single-pass Parallel Prefix Scan with Decoupled Look-back* (2016)](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) | 以约 `2n` 数据移动和通信规避理解 CUB DeviceScan | `E<=64` 不需要跨 CTA look-back |
| Shuffle/L2 协作 | [Liu, Aluru, *LightScan* (2016)](https://arxiv.org/abs/1604.04815) | warp shuffle 与 globally coherent L2 的组合设计 | K40c/PTX 路径和大数组跨 block 协作不可直接迁移 |
| 融合思想 | [Adinets, Merrill, *Onesweep* (2022)](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus) | 将 histogram/prefix/placement 的数据移动与 pass 数整体考虑 | radix sort 合同不同，只能借鉴融合评估方法 |
| 工业基线 | [CUB WarpScan](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpScan.html)、[BlockScan](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html)、[DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html) | 对照 warp/block/device 层级、TempStorage、completion kernel | 库 primitive 必须在相同 L1/L2 边界实测，不能凭 API 层级推断胜负 |

可复现搜索词：`GPU prefix scan warp shuffle`、`parallel prefix scan CUDA intra-warp`、`single-pass scan decoupled look-back`、`histogram scan fusion GPU`。完整实测结论见 [RTX 3080 Scan 研究报告](../reports/rtx3080-scan-sm86-research-32bf6c9.md)。
