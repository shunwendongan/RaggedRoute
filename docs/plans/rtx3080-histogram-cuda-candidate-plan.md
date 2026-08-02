# RTX 3080 Expert Histogram `cuda_candidate` 优化计划（待评审）

> 状态：**仅计划，尚未修改算子实现，也没有产生新的性能结论。**  
> 基线：`origin/main@927c585`；目标硬件为本机 NVIDIA GeForce RTX 3080（GA102、SM 8.6、68 SM、10 GiB）。  
> 工具可用性已核对：CUDA 13.3、Nsight Systems 2026.1.3、Nsight Compute 2026.2.1。

## 1. 目标、范围与非目标

本轮针对 `src/histogram` 的 Expert Histogram：输入 `R=T*top_k` 个 `int32` expert id，
其中 `id in [0,E)` 且 `1<=E<=64`；输出 `int32 counts[E]`：

```text
counts[e] = sum(id[r] == e), 0 <= r < R
sum(counts) = R
```

目标是在不改变公共语义、caller stream、SM86 支持边界和零 hot-path allocation 约束的前提下，
新增显式 `cuda_candidate` 算法族，比较：

- 当前 `cuda_naive`：每个 route pair 一次 global `atomicAdd`；
- `library_baseline/cub_histogram.cu`：`cub::DeviceHistogram::HistogramEven`；
- 各个单变量 CUDA candidate，以及最终只在证据支持时建立的 shape dispatch candidate。

主要输出指标为 L1/L2 p50、p95、CV、Gitems/s、有效 GB/s、workspace bytes；Nsight 指标只用于解释
launch/underfill、global/shared atomic、memory transactions、occupancy 和 stall，不用 profiler duration
宣称 release speedup。

本轮不做 Top-K→Histogram fusion、Histogram→Scan fusion、Hopper cluster/DSM、TMA、WGMMA、TMEM
或其他非 SM86 机制。融合有跨 CTA 全局依赖，必须作为独立后续课题，不能混入独立 Histogram 对比。

## 2. 当前实现与已有证据

### 2.1 源码与测量边界

| 位置 | 当前职责 | 本轮约束 |
|---|---|---|
| `src/histogram/baseline.cu` | 256 threads/CTA、grid-stride、每项 global atomic | 保持为不可变 oracle-side CUDA baseline |
| `src/histogram/operator.cpp` | 参数检查；L2 先 `cudaMemsetAsync(counts)` 再 launch | candidate 若覆盖写全部 counts，可安全跳过外部 reset；否则仍计入 L2 |
| `src/histogram/library_baseline/cub_histogram.cu` | CUB DeviceHistogram L2 reference | 不修改算法；固定 CCCL provenance |
| `src/histogram/cpu_reference/reference.cpp` | CPU bincount | exact correctness oracle |
| `benchmarks/adapters/histogram_adapter.cpp` | 数据分布、L1/L2、validation、work estimate | 增加 candidate 与更完整的分布/shape 元数据 |
| `benchmarks/core/registry.cpp` | variant 注册与 provenance | 每个实验 variant 独立可选，不用同名覆盖历史结果 |
| `src/runtime/dispatch.cpp` | 当前只允许 Dense GEMM optimized IDs | 增加 Histogram-local IDs，并拒绝跨算子 ID |

L1 `cuda_naive` 假定 counts 已清零，并在重复 launch 时累加；L2 public operator 每轮包含必要 reset。
CUB 只提供完整 L2 对比。对“单 CTA 覆盖写”候选，L1 自身包含 shared 初始化和最终 counts 写出，
不把这些必要工作排除；L2 不再额外执行无意义的 memset，但仍满足“任意旧 counts → 正确新 counts”。

### 2.2 已有基线不能外推的原因

仓库已有 RTX 3080 报告仅正式覆盖 `R=4096,E=64,Zipf s=1.4`：

| Variant / level | p50 (us) | p95 (us) | 备注 |
|---|---:|---:|---|
| `cuda_naive` L1 | 9.175 | 10.540 | 排除 counts reset |
| `cuda_naive` L2 | 17.500 | 18.305 | 包含 reset |
| CUB L2 | 20.275 | 23.695 | 同一完整 L2 边界 |

该 NCU capture 只有 16 blocks、0.039 waves/SM、15.7% achieved occupancy，SM/Memory 利用率约
0.1%/0.8%。这说明该点主要是 launch/underfill，不足以证明 global atomic 已成为瓶颈，也不足以否定
大 `R` 或高热点下的 privatization。新计划必须先扩展 shape/distribution，再决定算法。

## 3. 文献与实现资料如何映射到实验

检索范围为 GPU histogram、warp aggregation、shared-memory atomic，以及 MoE token-to-expert routing。
检索词包括 `GPU histogram privatization CUDA`、`warp aggregated atomics histogram`、
`shared-memory atomic Ampere histogram`、`MoE routing histogram expert counts`；优先论文、官方文档和
上游源码，不把博客测得的旧 GPU 数字外推到 SM86。

| 推荐顺序 | 资料 | 可用于本项目的结论 | 局限 |
|---:|---|---|---|
| 1 | Dong & Pai, [Modeling Utilization to Identify Shared-Memory Atomic Bottlenecks](https://arxiv.org/abs/2503.17893), 2025 | 直接覆盖 Volta/Ampere shared atomic；指导区分 shared atomic unit 瓶颈和其他瓶颈 | 是诊断模型，不替代本机 release A/B |
| 2 | Ashkiani et al., [GPU Multisplit: An Extended Study of a Parallel Algorithm](https://arxiv.org/abs/1701.01189), ACM TOPC 2017 | 小桶数的 warp-wide histogram、层次化局部处理，以及与 CUB 的逐桶数对比方式 | 主要硬件为 K40/GTX 1080；算法思想需在 GA102 复测 |
| 3 | Milic et al., [Parallelizing General Histogram Application for CUDA Architectures](https://doi.org/10.1109/SAMOS.2013.6621100), SAMOS 2013 | privatization 与 sort-search 的适用区间依赖输入规模、桶数和分布；支持做多维 sweep | 老硬件；本项目 `E<=64` 且 id 已离散，不需要通用 sort-search 作为首选 |
| 4 | Koppaka et al., [Fast Histograms using Adaptive CUDA Streams](https://arxiv.org/abs/1011.0235), 2010 | “输入退化/倾斜决定算法，必要时切换 kernel”的思路 | CPU/GPU stream overlap 不是本算子 steady-state 边界 |
| 5 | Shams & Kennedy, [Efficient Histogram Algorithms for NVIDIA CUDA Compatible Devices](https://users.cecs.anu.edu.au/~ramtin/ICSPCS/ICSPCS%2707/papers/131.pdf), ICSPCS 2007 | 历史上的 collision-free/private histogram 与归并思路 | 面向 compute capability 1.0、无现代 atomic，仅作历史背景 |
| 6 | Gale et al., [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html), MLSys 2023 | 说明 expert counts 是 token grouping、padding/topology 和后续 expert compute 的真实 metadata | 系统论文，不提供本项目独立 histogram 的性能上界 |
| 7 | NVIDIA [CCCL/CUB](https://github.com/NVIDIA/cccl)、[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 与 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) | API/源码基线、atomic/warp intrinsic 语义、SM86 资源边界 | 官方资料仍需结合当前 toolkit commit/version 和本机 profiler |

仓库文档列出的 NVIDIA shared-atomic、warp-vote、warp-aggregated atomic 博文保留为实现入口，但其性能图
不是论文证据，也不能替代当前编译器是否已自动聚合某种 atomic pattern 的 SASS/NCU 检查。

## 4. 冻结的任务合同

| 项目 | 决定 |
|---|---|
| Public API | `HistogramArgs` 与 `raggedroute::histogram` 不变 |
| 输入/输出 | device `int32` ids/counts；`0<=id<E` 是 v0.2 上游合同 |
| Shape | `route_pairs>=0`，`1<=E<=64`，计数必须能表示在 `int32` |
| 数值 | exact integer equality；无容差、无浮点 math mode |
| Stream | 只使用 caller stream；无 device-wide sync |
| Allocation | hot path 不分配；首轮候选 workspace=0 |
| Architecture | 仅在 SM86 验证和选择；其他架构继续返回 unsupported |
| L1 | 只计必要 kernel body；明确记录 reset/overwrite 前置条件 |
| L2 | 从任意旧 counts 得到新 counts；包含所有必要 reset/init/merge |
| Baselines | `cuda_naive` 与 CUB DeviceHistogram；输入、seed、cache、repeats、边界一致 |
| Release claim | 只采用未插桩、独立进程 A/B；NSYS/NCU duration 不用于 speedup |

## 5. 候选路线：一次只改变一个主要机制

### H0：重建并冻结基线

在 clean worktree 上重跑 build、correctness、sanitizer、扩展后的 `cuda_naive`/CUB release matrix。
若基线 correctness 失败或同配置与历史值发生无法解释的显著漂移，停止优化，先解决环境/合同问题。

### H1：warp-aggregated global atomic

每个 warp 用 `__match_any_sync` 对当前迭代的相同 expert id 分组；每组 leader 执行一次
`atomicAdd(counts[e], popc(peer_mask))`。

- 不引入 shared memory和 workspace；仍由 L2 wrapper memset；
- 高热点时全局 atomic 最多从每 lane 一次降到每 warp/key 一次；
- uniform `E=64` 可能因 `match_any/popc/ffs` 开销回退，因此必须保留 naive fallback；
- 检查 tail active mask、非法 id guard、Independent Thread Scheduling 语义和生成 SASS。

### H2：单 CTA shared histogram + 覆盖写输出

一个 256-thread CTA 初始化 `shared_counts[E]`，grid-stride 扫描全部 `R`，在 CTA barrier 后由前 `E`
个线程覆盖写 `counts[e]`。因为单 CTA 拥有完整结果，L2 不需要额外 `cudaMemsetAsync`。

- 目标是 tiny/small `R`：用一次 launch 完成 reset-equivalent + count；
- 优点是无 global atomic、无第二阶段 merge、workspace=0；
- 风险是只有一个 CTA，large `R` 严重 underfill；shared hotspot 也可能串行；
- 本实验先使用直接 shared atomic，不同时加入 warp aggregation。

### H3：多 CTA block-private shared histogram + global merge

每个 CTA 处理固定 items/thread 的 tile，在 shared 中累加，最后每个非零 expert 至多一次 global atomic
合并。输出仍由 L2 外部 memset。

- 初始设计使用 256 threads、8 items/thread，grid≈`ceil(R/2048)`，避免依赖硬编码 68 SM；
- global atomic 上界从 `R` 降为 `num_cta*min(E,active_bins)`；
- 风险是 shared 初始化、shared atomic、barrier 和 merge 在小 `R` 上得不偿失；
- 先独立测试，不能把 block-private 与 warp aggregation 一次性混成无法归因的候选。

### H4：仅在 H2/H3 的 NCU 证明 shared atomic 是主瓶颈时追加

在对应 shared candidate 内先做 warp same-key aggregation，再由 leader 更新 shared bin。该版本的 parent
必须是 H2 或 H3，只改变 shared atomic 请求数量。若 basic/detailed 证据已能否定 shared atomic 瓶颈，
不实现 H4。

### Final：证据驱动 shape dispatch

实验阶段保留 H1/H2/H3/H4 的显式 benchmark 名，以便复现 parent/child 结果；最终 PR 是否保留 rejected
candidate 的代码由本次 review 决定，但负结果、原始证据和 rejection reason 始终保留。`cuda_candidate`
只在 preflight/release 曲线出现稳定交叉点时按 `R`/`E` 选择 winner，否则直接指向单一 winner或回退
`cuda_naive`。不根据运行时未知的分布硬编码“Zipf dispatch”，也不为判断 skew 额外扫描 ids。

`KernelFamily::kAuto` 在所有 promotion gate 通过前保持 `cuda_naive`；实验 candidate 只通过显式
`KernelSelection{kCudaOptimized, histogram_local_id}` 调用。

## 6. 代码改动边界（计划）

新增：

- `src/histogram/cuda_candidate/optimized_internal.h`：Histogram-local implementation IDs、合法性检查、launch API；
- `src/histogram/cuda_candidate/optimized.cu`：H1/H2/H3（以及有条件的 H4）和最终 dispatcher；
- `configs/benchmark_histogram_candidate_smoke.json`；
- `configs/benchmark_histogram_candidate_release.json`；
- `configs/profile_histogram_candidate.json`；
- `docs/reports/rtx3080-histogram-candidate-<sha>.md` 和新的不可覆盖 artifact 目录。

修改：

- `CMakeLists.txt`：编译 candidate；
- `src/histogram/operator.cpp`：按 family/id 调用 naive 或 candidate，并处理 overwrite candidate 的 reset；
- `src/runtime/dispatch.cpp`：只接受 Histogram-local IDs，继续拒绝其他算子 optimized ID；
- `benchmarks/adapters/histogram_adapter.cpp`：variant、repeat policy、reset policy、work/atomic 元数据；
- `benchmarks/core/registry.cpp`：注册实验 variant 与 provenance；
- `tests/correctness_tests.cpp`、`tests/operator_api_tests.cpp`：candidate、边界、ID 隔离、redzone；
- `.codex/kernel-research.json`：切换为本轮 build/correctness/smoke/release/profile 合同；
- `docs/histogram/optimization-plan.md`、`docs/histogram/performance-record.md`：只写实测结果和保留/拒绝决定。

不修改 CUB wrapper 的算法，不删除 baseline，不把 rejected candidate 悄悄从历史中移除。

## 7. Correctness 与安全门

### 7.1 exact oracle cases

- `E={1,8,16,31,32,33,64}`，覆盖 warp 边界和最大 E；
- `R={0,1,31,32,33,255,256,257,4096,65536}`，覆盖 warp/CTA/tile tail；
- uniform、round-robin、Zipf `s={1.0,1.4,2.0}`、top-1 单热点、top-2 双热点；
- 空 expert、多数空 expert、所有 counts 预填非零值；
- exact counts、非负、`sum(counts)==R`、输入不变、输出两侧 redzone 不变；
- `route_pairs=0` 时仍把全部 counts 置零；null pointer 和非法维度的现有错误合同保持一致；
- 输入 id 越界属于上游违约，不增加 hot-path device validation；kernel 内 guard 行为不弱于 baseline。

### 7.2 dispatch/API cases

- 每个 Histogram implementation ID 在 SM86 可选，unknown ID 被拒绝；
- Histogram ID 用于 Dense GEMM 或反向使用必须被拒绝；
- `kAuto` 在 promotion 前仍解析为 naive；SM80/SM90 不冒充已验证支持；
- caller stream ordering、无强制同步、workspace query 与实际一致。

### 7.3 工具门

Release build 后运行 CTest，以及 Compute Sanitizer 的 memcheck、initcheck、racecheck、synccheck。
任何 candidate 在这些门失败时不得进入 benchmark；修复 correctness 时不同时继续做性能改动。

## 8. Benchmark 设计

### 8.1 preflight sweep

先用单进程 smoke 扫描候选曲线，重点覆盖：

- `E={1,8,16,31,32,33,64}`；
- `R={1,31,32,33,255,256,257,4096,65536,1048576}`；
- round-robin/uniform、Zipf-1.4/2.0、top-1 单热点；
- L1 和 L2；warm cache 为主，选取 small/large 两点补 cold-scrub。

preflight 只用于选择 release case 与候选阈值，不作为 PR speedup 结论。

### 8.2 固定 release matrix

至少保留以下代表点，所有 variant 共享相同输入和 seed：

| R | E | top_k / distribution | 目的 |
|---:|---:|---|---|
| 128 | 8 | 1 / uniform | tiny launch/reset 主导 |
| 4096 | 1 | 1 / single-hot | 极端 global/shared atomic 冲突 |
| 4096 | 64 | 2 / uniform | 与历史 uniform 接轨 |
| 4096 | 64 | 2 / Zipf-1.4 | 与历史正式基线接轨 |
| 4096 | 64 | 2 / round-robin | 确定性均衡分布 |
| 65536 | 16 | 1 / uniform | 小 E、中等 R |
| 65536 | 64 | 2 / uniform | 多 CTA 均衡吞吐 |
| 65536 | 64 | 2 / Zipf-1.4 | 多 CTA 现实 skew |
| 65536 | 64 | 1 / single-hot | 多 CTA 极端 skew |
| 1048576 | 64 | 2 / uniform | large-R throughput |
| 1048576 | 64 | 2 / Zipf-1.4 | large-R skew |
| 1048576 | 64 | 1 / single-hot | large-R atomic serialization |

Release 使用 clean Git、Release `sm_86` + `-lineinfo`、固定 seed `20260729`、随机化 variant/case 执行顺序，
至少 5 个独立进程、warmup 20、每进程至少 30 samples。`kernel_repeats` 按 case 固定并记录，既保证短
kernel 有足够 batch duration，也不得触发 L1 int32 累加溢出。

### 8.3 公平对比与输出

- L1：`cuda_naive` 与显式 CUDA candidates；明确 `counts_zeroed`、`accumulate` 或 `overwrite`；
- L2：`cuda_naive`、CUB、所有 candidates、最终 `cuda_candidate`，完整包含各自必要的初始化；
- 每 case 报告 process-median 的 median、全部 sample 的 p95/CV、原始 samples；
- `Gitems/s = R / p50_us / 1000`；有效 GB/s 使用 manifest 中明确的 logical bytes；
- 报告 workspace、kernel launch 数、理论 global atomic 上界、shared bytes/CTA；不伪装 CUB 内部策略；
- CUB 与 CUDA Toolkit/CCCL revision 固定在 manifest；CUB 不支持的 L1 显式记为 N/A。

## 9. NSYS / NCU 诊断顺序

1. 对 H0 baseline 及通过 smoke 的 candidate，先用 NSYS 捕获 exact release workload；关闭 CPU sampling，
   开 CUDA/NVTX trace，确认 memset、kernel/merge launch 数和 GPU-time rank。
2. 对 `R=4096,E=64,Zipf-1.4`、`R=1M,E=64,uniform`、`R=1M,E=1,single-hot` 各选择一个
   post-warmup launch，按实际 emitted kernel name 用 NCU `basic` 过滤。
3. 只有 basic 不能区分 global atomic、shared atomic、underfill 或 memory traffic 时，才对 baseline/candidate
   成对升级 `detailed`；source level 只用于仍无法定位的 source/SASS 问题。
4. 报告实际可用的 `gpu__time_duration.sum`、grid/block、waves/SM、achieved occupancy、SM/Memory/DRAM
   throughput、registers/thread、global/shared atomic sectors/wavefronts、L1/L2/DRAM traffic 与 dominant stalls。
   当前工具不支持的 metric 写 `not_collected` 或 `unsupported_or_unknown`，绝不写 0。
5. profiler 的 duration 只解释瓶颈；candidate 是否晋升重新回到未插桩 release A/B。

## 10. 晋升、回退与停止条件

最终 `cuda_candidate` 要同时满足：

1. 全部 exact correctness、API、CTest 和四类 sanitizer 通过；
2. workspace 不增长（首轮目标为 0）；
3. 至少一个预先声明的中/大 `R` case 相对 `min(cuda_naive, CUB)` 的 L2 p50 提升 `>=5%`；
4. release matrix 的总体 L2 process-median 不劣于逐 case 最强 baseline，且 p95 无 `>3%` 系统性回退；
5. shape dispatch 后任一 case 相对 `cuda_naive` 的 p50 回退不超过 `5%`，tiny case 优先 fallback；
6. 结果在至少 5 个独立进程中方向一致，且能由 NSYS/NCU 的机制证据解释；
7. 不以 profiler duration、单次最好值或不同 reset 边界作为 speedup。

若单个实验失败，记录 parent、hypothesis、raw artifact、指标和 rejection reason，再继续下一个独立机制。
若 H1/H2/H3 均不能在预声明矩阵上越过最强 baseline，则不改变默认 dispatch：保留研究代码与负结果
由评审决定，或只提交 benchmark/报告；不制造“candidate 已优化”的结论。

## 11. 执行批次与 review 检查点

| 批次 | 内容 | 提交 review 的产物 |
|---|---|---|
| D0 | 冻结配置、扩展 correctness/benchmark matrix、重建 H0 | clean baseline、原始 JSONL、环境与命令 |
| D1 | 实现 H1；correctness→release→NSYS→NCU basic | H1 保留/拒绝记录 |
| D2 | 实现 H2；重复同一门禁 | reset/launch 减少是否兑现 |
| D3 | 实现 H3；重复同一门禁 | privatization/merge 收益与代价 |
| D4 | 仅在证据要求时实现 H4 | shared atomic 瓶颈是否解除 |
| D5 | 建立或放弃 final shape dispatch；全量复验 | final A/B、p50/p95/Gitems/s、NCU/NSYS 报告 |
| D6 | 文档、artifact checksums、代码 review、PR | 可复现命令、负结果、PR 描述 |

每个批次只引入一个主要性能机制。D0 完成后若基线或合同发生变化，先更新本计划并再次 review；
最终 PR 不提交版本绑定的 `.ncu-rep/.nsys-rep/.sqlite` 二进制，但提交归一化文本/CSV、manifest、SHA256
和重现命令。

## 12. 本次 review 需要确认的决定

1. 是否接受 H1→H2→H3、H4 条件触发的顺序；
2. 是否接受把“单 CTA 覆盖写”视为合法 L2 reset fusion，同时在 L1 明确标注 overwrite 语义；
3. 是否接受上述 12 个 release cases 与 `>=5%` 收益、p95 `<=3%` 回退阈值；
4. 是否要求首个 PR 只保留最终 winner，还是也保留被拒绝但可复现的显式实验 variants。

收到批准前不进入 D0/D1 的实现与新 benchmark 阶段。
