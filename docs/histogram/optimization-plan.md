# RTX 3080 Expert Histogram 优化方案

## 合同与测量边界

- 目标为 RTX 3080（GA102，SM86），接口保持 `HistogramArgs`、caller stream、`1<=E<=64`、合法 expert id、exact `int32` counts 和 hot path 零 allocation。
- L1 只比较 in-tree kernel body；L2 包含得到完整新 counts 所需的 reset。CUB `DeviceHistogram` 仅作为 opaque library baseline。
- 性能结论只使用未插桩 Release A/B；NSYS/NCU duration 只用于解释机制。
- 晋升基准是每个 case 的 `min(cuda_naive, cub_device_histogram)`，workspace 必须保持 0。

## 已执行的假设

| 假设 | 实现 | 结论 |
|---|---|---|
| H1 warp 聚合 global atomic | `__match_any_sync` 后由 leader 更新 global bin | 热点分布改善明显，但 uniform 和小 `R` 的额外 warp 指令不稳定；拒绝进入 shipping code。 |
| H2 单 CTA shared histogram | 256 threads，shared 初始化、累加、覆盖写 counts | `R<=4096` 最稳定；融合 L2 reset，无外部 memset。 |
| H3 多 CTA block-private | 256 threads，8 items/thread 起步，shared 累加后非零 bin merge | `R>=32768` 的 winner；grid cap 128 CTA 降低 merge atomic 上界。 |
| H4 warp→shared 聚合 | 仅在 NCU 证明 shared atomic 饱和时实现 | NCU detailed 未证明 shared atomic 单元饱和，拒绝实现。 |
| H5 `E==1` direct write | 合法 id 合同下直接 `counts[0]=R`，不读 ids、无 atomic、无 reset | 晋升为 v2；在预声明的四个 `E==1` release case 全部 5/5 方向一致。 |
| H6 CTA cap 256/384/512 | 只调整 v1 block-private CTA 上限 | uniform 大 `R` 可获益，但 `R=1M,E=64,Zipf-1.4` 的 cap384 smoke 回退 6.59%；禁止按未知 skew dispatch，拒绝。 |
| H7 vector load + thread-local RLE | 仅基于 H6 winner 继续 | H6 没有 distribution-independent winner，未实施。 |

## Shape dispatcher

v2 候选优先使用 `E`，不读取运行时数据分布，也不硬编码 SM 数：

- `E==1`：一个 thread 覆盖写 `counts[0]=R`；不读取 `expert_ids`，该写入同时满足完整 L2 reset 语义。
- `R<=4096`：单 CTA shared histogram，覆盖写 counts。
- `4097<=R<32768`：交叉区间跨进程方向不稳定，回退 `cuda_naive`。
- `R>=32768`：多 CTA block-private shared histogram；grid 上限为 128 CTA，降低大 `R` 下的 merge atomic 上界。该值是 shape tuning 参数，不是 68 SM 的硬编码。

稳定 benchmark 名为 `cuda_candidate`，v1 implementation ID 为 `101`，v2 为 `102`。SM86 Histogram `kAuto` 指向 v2；显式 `cuda_candidate_v1` 和 `kCudaNaive/0` 保留为复现与 benchmark baseline。

## Correctness 与性能门禁

- correctness 轴：`E={1,2,8,16,31,32,33,64}`、`R={0,1,31,32,33,255,256,257,4096,8192,16384,32767,32768,65536,1M}`，覆盖 uniform、round-robin、Zipf `s={1.0,1.4,2.0}`、single-hot 和 top-2 dual-hot。
- 额外检查：旧 counts 覆盖、`R=0` 清零、输入不变、redzone、caller-stream ordering、unknown/跨算子 implementation ID、SM86/非 SM86 dispatch。
- sanitizer：memcheck、initcheck、racecheck、synccheck。
- release：5 个独立进程，warmup 20，每进程 30 samples，seed `20260729`，随机 case/variant 顺序；报告 process-median 的中位数、全 samples p95/CV、Gitems/s、logical/reset bytes、workspace、launch 与 in-tree atomic 上界。
- promotion：至少一个预声明中/大型 case 相对 strongest L2 baseline p50 提升 5%；无系统性 p95 回退超过 3%；任一 shape 相对 naive p50 回退不超过 5%；跨进程方向一致；workspace 0。

## 论文与设计依据

- Dong、Pai，[Modeling Utilization to Identify Shared-Memory Atomic Bottlenecks](https://arxiv.org/abs/2503.17893)：提供 Volta/Ampere shared atomic 的 counter-driven 模型，并直接比较 histogram kernel；用于决定是否值得实现 H4。
- Ashkiani 等，[GPU Multisplit](https://arxiv.org/abs/1701.01189)：小桶数场景使用 warp-synchronous communication 减少 divergence 和中间存储；用于提出 H1/H4 假设，不把 GTX 1080 的结果外推为 SM86 结论。
- Gale 等，[MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)：说明动态 MoE routing、padding/token dropping 与 GPU 效率之间的系统背景；本轮不做跨算子 fusion。
- Koppaka 等，[Fast Histograms using Adaptive CUDA Streams](https://arxiv.org/abs/1011.0235)：较早讨论 atomic 冲突和输入退化程度驱动的 histogram kernel 选择；本项目只采用离线 shape dispatch，不在 hot path 扫描 skew。
- Maucher 等，[Are Your GPU Atomics Secretly Contending?](https://doi.org/10.1145/3764860.3768338)：强调 atomic 经验规则必须由当前硬件微基准/计数器验证；支持拒绝 distribution-sensitive H6。
- NVIDIA [CUDA C++ Programming Guide 13.2](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/index.html) 与 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)：`int4` 访问需要自然 16-byte 对齐；SM86 的 occupancy/共享内存约束必须在实际 profile 中核验。H7 因 H6 未晋升而不引入该额外路径。

上述旧架构工作只用于提出可证伪假设。最终机制判断以本机 SM86 的 NSYS/NCU 和未插桩 A/B 为准。
