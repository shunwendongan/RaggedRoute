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
| H3 多 CTA block-private | 256 threads，8 items/thread 起步，shared 累加后非零 bin merge | 中大型 `R` 的候选；需以 clean release 和 profiler 决定最终 grid cap。 |
| H4 warp→shared 聚合 | 仅在 NCU 证明 shared atomic 饱和时实现 | 当前未实现；避免在没有 counter 证据时增加指令。 |

## Shape dispatcher

当前候选仅使用 `R`，不读取数据分布，也不硬编码 SM 数：

- `R<=4096`：单 CTA shared histogram，覆盖写 counts。
- `4097<=R<32768`：交叉区间跨进程方向不稳定，回退 `cuda_naive`。
- `R>=32768`：多 CTA block-private shared histogram；grid 上限为 128 CTA，降低大 `R` 下的 merge atomic 上界。该值是 shape tuning 参数，不是 68 SM 的硬编码。

稳定 benchmark 名为 `cuda_candidate`，Histogram-local implementation ID 为 `101`。在全部门禁完成前，`kAuto` 仍指向 `cuda_naive`。

## Correctness 与性能门禁

- correctness 轴：`E={1,8,16,31,32,33,64}`、`R={0,1,31,32,33,255,256,257,4096,65536}`，覆盖 uniform、round-robin、Zipf `s={1.0,1.4,2.0}`、single-hot 和 top-2 dual-hot。
- 额外检查：旧 counts 覆盖、`R=0` 清零、输入不变、redzone、caller-stream ordering、unknown/跨算子 implementation ID、SM86/非 SM86 dispatch。
- sanitizer：memcheck、initcheck、racecheck、synccheck。
- release：5 个独立进程，warmup 20，每进程 30 samples，seed `20260729`，随机 case/variant 顺序；报告 process-median 的中位数、全 samples p95/CV、Gitems/s、logical/reset bytes、workspace、launch 与 in-tree atomic 上界。
- promotion：至少一个预声明中/大型 case 相对 strongest L2 baseline p50 提升 5%；无系统性 p95 回退超过 3%；任一 shape 相对 naive p50 回退不超过 5%；跨进程方向一致；workspace 0。

## 论文与设计依据

- Dong、Pai，[Modeling Utilization to Identify Shared-Memory Atomic Bottlenecks](https://arxiv.org/abs/2503.17893)：提供 Volta/Ampere shared atomic 的 counter-driven 模型，并直接比较 histogram kernel；用于决定是否值得实现 H4。
- Ashkiani 等，[GPU Multisplit](https://arxiv.org/abs/1701.01189)：小桶数场景使用 warp-synchronous communication 减少 divergence 和中间存储；用于提出 H1/H4 假设，不把 GTX 1080 的结果外推为 SM86 结论。
- Gale 等，[MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)：说明动态 MoE routing、padding/token dropping 与 GPU 效率之间的系统背景；本轮不做跨算子 fusion。
- Koppaka 等，[Fast Histograms using Adaptive CUDA Streams](https://arxiv.org/abs/1011.0235)：较早讨论 atomic 冲突和输入退化程度驱动的 histogram kernel 选择；本项目只采用离线 shape dispatch，不在 hot path 扫描 skew。

上述旧架构工作只用于提出可证伪假设。最终机制判断以本机 SM86 的 NSYS/NCU 和未插桩 A/B 为准。
