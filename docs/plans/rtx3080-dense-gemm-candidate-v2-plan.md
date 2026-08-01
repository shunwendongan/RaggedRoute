# RTX 3080 dense GEMM `cuda_candidate` v2 优化方案（待评审）

## 1. 评审结论

本轮只优化 `src/dense_gemm/cuda_candidate` 中的手写 strict-FP32 CUDA kernel。建议的主方案不是先加 `cp.async`，而是先把当前“一个线程计算一个输出”的 16×16 kernel 改成 CTA/warp/thread 三层分块的 register-blocked kernel：

- CTA tile：`BM×BN×BK = 32×32×16`；
- 线程块：128 threads，即 4 warps；
- warp tile：`16×16`，4 个 warp 以 `2×2` 排列；
- thread microtile：`4×2`，每线程持有 8 个 FP32 accumulator；
- A/B 继续使用已验证的 16-byte 对齐 `float4` global-to-shared staging；
- A shared layout 使用 `[32][20]` padding，B 使用 `[16][32]`；
- 每个输出仍按 `k=0..K-1` 的顺序做 FP32 FMA，不使用 TF32、Tensor Core、WMMA 或降低精度的 fast math。

先实现同步 staging 的 `cuda_register_tiled_v2_sync`，用它单独验证 P0 的 register blocking、ILP 和 shared-memory reuse。只有同步版通过正确性和无 profiler 性能门禁后，才实现 `cuda_register_tiled_v2_async`，在相同 tile/mapping 上单独加入 Ampere `cp.async` 双缓冲。这样不会把 register blocking 的收益错误归因给异步流水线。

该方案是针对 RTX 3080、strict FP32、row-major 和当前主要规格得出的最优下一步实验，不声称是所有 shape/GPU 的全局最优 GEMM 配置。

## 2. 范围与不变量

### 2.1 修改范围

优化代码严格放在：

- `src/dense_gemm/cuda_candidate/optimized_v2.cu`：v2 kernel、fast-path launcher；
- `src/dense_gemm/cuda_candidate/optimized_internal.h`：新增实验 ID 和内部 launcher 声明；
- `src/dense_gemm/cuda_candidate/optimized.cu`：只增加 v2 dispatch 与已有 vector/scalar fallback 的连接。

为使候选可测，还需按现有工程职责修改：

- `benchmarks/adapters/dense_gemm_adapter.cpp`：L1/L2 benchmark 适配；
- `benchmarks/core/registry.cpp`：variant 注册与 metadata；
- `tests/`：API、oracle、edge/fallback 测试；
- `configs/`：版本化 benchmark/profile suite；
- `CMakeLists.txt`：只把 `optimized_v2.cu` 加入现有 CUDA target。

benchmark adapter 和 registry 是测量/注册层，继续放在原目录是合适的；它们不是 CUDA kernel，不放入 `cuda_candidate`。`library_baseline` 不移动、不修改。

### 2.2 保持不变

- 公共 `DenseGemmArgs`、`dense_gemm()` 签名和 workspace=0 合同不变；
- FP32 row-major，`alpha=1`、`beta=0`；
- cuBLASLt 使用 `CUBLAS_COMPUTE_32F_PEDANTIC`，cuBLAS 使用 pedantic/strict FP32；
- `kAuto` 继续选择当前 naive 路径；
- v2 只通过显式 `KernelFamily::kCudaOptimized` 实验 ID 调用；
- 不以 CUTLASS、cuBLAS、Triton 替换手写 candidate；它们只提供教材和强 baseline；
- 不引入 TF32/Tensor Core，不引入 Hopper TMA、thread-block cluster、WGMMA 或其他 sm_90+ 机制。

## 3. 现有证据与根因

现有最快候选 `cuda_tiled_vector` 是 16×16×16、256 threads/CTA、每线程一个输出、每个 K tile 两次 `__syncthreads()`。它的 `float4` 只是一次搬 4 个连续 FP32 到 shared memory，不代表每线程计算 4 个输出；真正缺少的是 register microtile。

### 3.1 无 profiler 结果

| Shape / L2 | p50 | 相对 strict cuBLAS |
|---|---:|---:|
| 256³ cuBLAS | 10.854 µs | 1.00× |
| 256³ tiled vector | 22.118 µs | 慢 2.04× |
| 512³ cuBLAS | 27.136 µs | 1.00× |
| 512³ tiled vector | 120.064 µs | 慢 4.42× |
| 1024³ cuBLAS | 155.034 µs | 1.00× |
| 1024³ tiled vector | 834.662 µs | 慢 5.38× |

256³ 的 cuBLAS CV 较高，不能单独用来判断小幅收益；1024³ 的 CV 为 0.008–0.010，证明差距不是短 kernel 计时误差造成的。L1/L2 差值远小于 kernel 差距，因此 public operator/dispatch 不是当前主瓶颈。

### 3.2 NCU 证据

1024³ L2：

| 项目 | tiled vector | strict cuBLAS |
|---|---:|---:|
| kernel | `dense_gemm_tiled_vector_kernel` | `ampere_sgemm_64x64_nn` |
| grid / block | 4096 / 256 | 512 / 64 |
| registers/thread | 40 | 126 |
| achieved occupancy | 96.5% | 24.1% |
| global load requests | 1.049 M | 1.082 M |
| issue active | 28.4% | 61.8% |
| long-scoreboard samples | 2352 | 358 |
| MIO-throttle samples | 20854 | 81 |
| barrier samples | 20691 | 194 |
| Tensor pipe | 0% | 0% |

这组信号共同说明：

1. vector staging 已把 global load request 数压到 cuBLAS 同一量级，所以继续只改 global vector width 的收益上限很低；
2. 96.5% occupancy 没有转化为 issue 效率，说明 occupancy 不是应继续最大化的目标；
3. 单 accumulator 形成长依赖链，shared loads/FMA 多，且每 16 个 K 步就经历两次 block barrier；
4. cuBLAS 用更多寄存器换取 register blocking 和 ILP，在较低 occupancy 下反而有更高 issue active；
5. Tensor pipe 为 0，现有 strict cuBLAS 比较是 SIMT FP32，不是拿 Tensor Core 路径与手写 CUDA core 路径混比。

256³ 的 cuBLAS 实际选择 `ampere_sgemm_32x32_sliced1x4_nn`，grid=64、block=128、86 registers/thread；这支持用 32×32/128-thread 作为小规格起点。64×64 CTA 在 256³ 只有 16 个 CTA，会让 68-SM RTX 3080 严重 underfill，因此不作为 v2 首选。

PC sampling 的 stall 数会随采样数和 kernel duration 变化。实施时比较 normalized share/warp-cycle 指标，不把上表的原始 sample count 直接当作可线性相除的性能比例。

## 4. 文献结论：GEMM-01～GEMM-14

| 编号 | 采用内容 | 本轮不采用或局限 |
|---|---|---|
| GEMM-01 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | 线程层次、tile kernel、shared-memory 可见性、pipeline/async-copy 正确同步。 | 文档同时覆盖更新架构；只采用 sm_86 可用语义，不把 cluster/TMA 等新机制移植到 3080。 |
| GEMM-02 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | APOD 迭代、coalescing、shared reuse、bank-conflict、occupancy 和有效带宽的测量方法。 | 它不是固定 GEMM 配方；tile 和 occupancy 必须由本项目实测选择。 |
| GEMM-03 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) | sm_86 每 SM 100 KiB shared、每 block 最多 99 KiB，以及硬件加速 global→shared async copy。 | 不套用 A100/sm_80 的 164 KiB shared 参数。 |
| GEMM-04 [Hopper Tuning Guide](https://docs.nvidia.com/cuda/pdf/Hopper_Tuning_Guide.pdf) | 只保留“按实际架构检查 occupancy/内存层次”的方法。 | TMA、cluster、distributed shared memory 和 H100 参数全部排除；目标是 sm_86。 |
| GEMM-05 [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) | 最关键依据：CTA/warp/thread 三层分块、register accumulator、global→shared→register mainloop 和 double buffering。 | CUTLASS 的示例 tile 不是本项目的直接答案；需要按 256/512/1024 和 68 SM 重新测。 |
| GEMM-06 [CUTLASS GEMM API 3.x](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) | 借鉴 mainloop/epilogue/launcher 分层，保持 kernel 与 adapter 职责清楚。 | 本轮不把 candidate 改写成复杂 CUTLASS collective/template。 |
| GEMM-07 [CuTe dense GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) | 用 layout 和 thread partition 检查“每个 lane 负责哪些 C 元素”和 shared partition 是否一一覆盖。 | v2 继续直接写 CUDA C++；CuTe 可在后续多 tile/autotune 阶段评估。 |
| GEMM-08 [CUTLASS repository](https://github.com/NVIDIA/cutlass) | 参考项目已固定的 CUTLASS `v4.6.1` 示例与 profiler 配置，学习而不复制未知架构参数。 | CUTLASS 不是本轮 promotion baseline，也不替换手写 kernel。 |
| GEMM-09 [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html) | 保持 cuBLAS/cuBLASLt pedantic FP32 强 baseline、warmup、固定 workspace 和同一 event timing 边界。 | heuristic 选择会随 shape/版本变，必须记录真实 kernel 名；profiler duration 不作加速比。 |
| GEMM-10 [Matrix Multiplication Background Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) | 用 arithmetic intensity、tile quantization 和 wave quantization解释 32×32 与 64×64 的取舍。 | 示例多来自数据中心 GPU/Tensor Core，不能替代 RTX 3080 实卡结果。 |
| GEMM-11 [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 借鉴 program-id 的二维/分组映射和受约束参数比较。 | 不引入 Triton、不把其 autotune 数字作为 CUDA baseline；此前 2D-only 实验已证明 `/`、`%` 不是 P0。 |
| GEMM-12 [Triton persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | 记录固定 CTA 数和 tile 调度思路，留给后续 small/irregular shape。 | 当前 256³ 已有 64-tile 强 baseline；persistent queue 增加调度状态，不能先于 register blocking。教程中的 TMA/warp-specialize 新架构路径不用于 3080。 |
| GEMM-13 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 仅学习流水线、调度和证据化工程方式。 | 其主要硬件/低精度合同面向更新架构和 FP8/FP4/BF16，不是 RTX 3080 strict-FP32 可运行 baseline。 |
| GEMM-14 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 用真实版本的 metric 名分析 launch、occupancy、issue、eligible warps、memory、stall、spill 和 source/SASS。 | metric 名随 NCU/架构变化；缺失项写 `not_collected` 或 `unsupported_or_unknown`，不填 0。 |

文献的共同结论不是“上更大的 tile”或“上 cp.async”本身，而是：先增加每线程独立 accumulator 和 shared→register reuse，使一个 warp 在等待某条依赖链时仍有可发射的独立 FFMA；然后才值得把下一 tile 的数据搬运与计算重叠。

## 5. v2 同步主设计

### 5.1 CTA/warp/thread 映射

```text
CTA C tile: 32 rows × 32 columns

             N: 0..15          N: 16..31
M: 0..15     warp 0            warp 1
M: 16..31    warp 2            warp 3

warp_id = threadIdx.x >> 5
lane_id = threadIdx.x & 31
warp_m  = warp_id >> 1
warp_n  = warp_id & 1
lane_m  = lane_id >> 3        // 0..3
lane_n  = lane_id & 7         // 0..7

row(r) = block_row*32 + warp_m*16 + lane_m + r*4, r=0..3
col(c) = block_col*32 + warp_n*16 + lane_n*2 + c, c=0..1
```

每线程的 `acc[4][2]` 恰好覆盖 8 个不同输出；128 threads × 8 = 1024 个输出，完整覆盖一个 32×32 C tile，无重叠也无缺口。

### 5.2 K mainloop

每个 `BK=16` stage：

1. 128 个线程各加载一个 A `float4` 和一个 B `float4`；
2. A 写入 padded `As[32][20]`，B 写入 `Bs[16][32]`；
3. `__syncthreads()` 后，按 `kk=0..15` 顺序执行：
   - 从 shared 取 4 个 A 值到 `areg[4]`；
   - 从 shared 取 2 个 B 值到 `breg[2]`；
   - 做 8 个相互独立的 `acc[r][c] = fmaf(areg[r], breg[c], acc[r][c])`；
4. 第二次 `__syncthreads()` 后复用 shared tile；
5. aligned fast path 用每行一个 `float2` 写回相邻两列，降低 epilogue store 指令；累加次序不变。

与当前 16×16 单输出 kernel 的理论差异：

| 项目 | 当前 vector | v2 sync | 方向 |
|---|---:|---:|---|
| outputs/thread | 1 | 8 | 8 条独立 accumulator 链 |
| shared loads/FMA | 2.00 | 0.75 | 降低 62.5% |
| 每 stage 的有效 global staging intensity | 约 4 FLOP/B | 约 8 FLOP/B | 约 2× |
| 1024³ CTA 数 | 4096 | 1024 | 减少重复 A/B tile staging |
| 每 CTA 每 stage barriers | 2 | 2 | 数量不变，但每次同步间计算量约 4× |
| static shared/CTA | 3 KiB | 4.5 KiB | 仍远低于 sm_86 上限 |

`As[32][20]` 的 4-float padding 既保持每行 16-byte 对齐，也避免 stride=16 时 4 个不同 A row 映射到重复 bank 的模式。B 的 warp load 由多个线程共享同一值，`Bs[16][32]` 可利用 broadcast；仍必须由 NCU shared wavefront/bank-conflict 指标验证，不能只靠纸面推断。

### 5.3 fast path 与 fallback

v2 fast path 条件：

- `M % 32 == 0`、`N % 32 == 0`、`K % 16 == 0` 且 `K > 0`；
- A/B/C 至少 16-byte aligned；
- sm_86 显式 optimized dispatch。

否则：

- 满足现有 16×16 vector 条件时回退 `cuda_tiled_vector`；
- edge、tail、仅 4-byte 对齐、`K=0` 回退已有 scalar tiled path；
- `M=0` 或 `N=0` 保持成功且不 launch。

因此 v2 不用在第一版同时承担 masked-tail 复杂度，所有合法公共输入仍有正确路径。fallback 本身必须在 benchmark metadata 和验证日志中可见，不能让 edge case 被误记成 v2 fast-path 性能。

### 5.4 资源预算

- sync shared memory：`32×20×4 + 16×32×4 = 4608 B/CTA`；
- async 双 stage：`9216 B/CTA`；
- accumulator：8 registers/thread；A/B microtile 至少 6 registers/thread；
- 推荐编译结果不超过 80 registers/thread；不使用全局 `--maxrregcount` 强压到 spill；
- 硬拒绝任何 local load/store 指令或 local-memory sectors；
- 目标是资源上至少允许 2 resident CTAs/SM。occupancy 不是分数：若寄存器超过软目标但无 spill，且 issue/latency/p50 全部明显改善，可保留供评审；
- 用 `ptxas -v`、NCU launch/resource metrics 和 SASS 三方核对，不只看源码变量数。

## 6. v2 async 二阶段实验

只有 `cuda_register_tiled_v2_sync` 通过门禁后，才在完全相同的 32×32×16、4×2 microtile 上增加：

- sm_86 16-byte global→shared `cp.async`；
- A/B 两套 shared stage，约 9 KiB/CTA；
- prologue 预取、steady-state 中“计算 stage t，同时发出 stage t+1”、epilogue drain；
- 每个线程固定负责相同的 A/B 16-byte copy，避免 producer/consumer 覆盖；
- 明确 commit/wait-group 和 block-wide visibility 协议。

不先承诺“删除所有 barrier”。`cp.async` 完成只解决 copy readiness，shared stage 的跨线程消费和 buffer reuse 仍需要正确同步。racecheck/synccheck 是硬门禁；若双缓冲增加寄存器/SMEM 后减少 resident CTA、出现 spill，或 Release p50/p95 没有同步改善，则保留 sync 版。

暂不做：

- 三/四 stage 深流水；
- TMA；
- warp-specialized producer/consumer；
- persistent CTA；
- 64×64 CTA；
- 大范围 autotune。

这些都增加变量或属于新架构，当前证据不足。

## 7. 实验顺序

### 实验 A：`cuda_register_tiled_v2_sync`

唯一主机制是层次化 register tiling；继续使用同步 `float4` staging。需要回答：

- 8 个独立 accumulator 是否提高 scheduler issue/eligible warp；
- shared loads/FMA 和 normalized MIO stall 是否下降；
- 32×32 tile 是否因 256³ 的 64 CTA underfill 抵消收益；
- padding 是否避免 shared bank conflict；
- 寄存器增加是否产生 spill。

### 实验 B：`cuda_register_tiled_v2_async`

父版本固定为实验 A，唯一主变量是 `cp.async` 双缓冲。需要回答：

- long-scoreboard/MIO 是否进一步下降；
- barrier/wait 是否只是转移而非消失；
- pipeline prologue/epilogue 是否在 256³ 太短，在 1024³ 才有收益；
- 9 KiB shared 和额外 pipeline state 是否降低可用并发。

### 可选受约束比较

只有 A/B 结果与假设矛盾时才做一次小型比较：

- `BK=8` 对 `BK=16`，或
- thread microtile `4×2` 对 `2×4`。

一次只改一个参数，不做无约束 autotune。单独去除 `/`、`%` 不再立项；v2 使用二维 grid 只是自然地址映射，不宣称为主要优化收益。

## 8. 正确性与安全门禁

每个候选在任何性能采集前必须完成：

1. Release 全量 CTest；
2. benchmark CPU oracle 且每条 `validation.ok=true`；
3. Compute Sanitizer：memcheck、racecheck、initcheck、synccheck；
4. launch error 和 redzone/canary 校验；
5. SASS/NCU 确认 strict FP32 SIMT、Tensor pipe=0、无 local load/store/spill。

覆盖 shape：

- `K=0`；
- `1×1×1`；
- `5×7×3`；
- `17×19×13`；
- `31×33×65`；
- `256×256×256`；
- `511×513×257` 之类非整除大形状；
- `512³`、`1024³`；
- A/B/C 只有 4-byte、不是 16-byte 对齐的指针；
- fast path 和每一种 fallback 都单独命中并记录。

数值合同以现有 CPU oracle 容差为准；另外对 fast-path 固定 seed 运行重复结果一致性检查。不得为了通过性能门禁放宽容差。

## 9. Release benchmark 合同

- GPU：当前检测到的 RTX 3080，UUID 固定，compute capability 8.6；
- Release、`sm_86`、优化编译、`-lineinfo`；不使用 `-G`；
- FP32 row-major、`alpha=1`、`beta=0`、warm cache；
- seed `20260729`；
- L1 kernel body 和 L2 operator steady 分开；
- 3 个独立进程、每进程 20 warmup、30 samples、每 sample 10 repeats；
- 同一累计 suite 比较 `cuda_tiled_vector`、v2 sync、v2 async、cuBLASLt、cuBLAS；
- 256³ 保持正式主规格；512³、1024³ 分开报告，作为降低短 kernel 方差和验证规模行为的补充，不能合并成一个总分；
- 只从无 profiler event timing 输出 p50、p95、CV、TFLOP/s 和倍数。

严格 pairing 字段：GPU UUID、Git SHA/dirty、shape、seed、dtype、layout、alpha/beta、math mode、cache mode、level、repeats、warmup 和 excluded steps。

若任一正式 variant `CV>0.10`，或同一 suite 的现有 vector/naive 漂移超过历史 p50 的 5%，确认无竞争 GPU workload 后整套用新 run ID 重跑一次；仍不稳定则标记 `variance-limited`，不晋升。

## 10. NSYS / NCU 采集

每个候选先对 1024³ L2 跑 NSYS CUDA/NVTX trace，确认真实 kernel 名和 GPU hotspot；再只过滤一个 post-warmup launch：

1. NCU `basic`：grid/block、waves、duration（仅诊断）、register、shared、occupancy、SM/memory/DRAM/L1/L2、Tensor pipe；
2. NCU `detailed`：eligible warps、issue active、global/shared requests/sectors、bank conflict/wavefront、long/short scoreboard、MIO、barrier、wait、math-pipe、local load/store；
3. 仅当 detailed 不能解释结果时，对对应候选用 `source` 或 `full`；
4. `--cache-control none --clock-control none`，不锁频、不改 power limit；
5. 缺失指标标记状态，不当作 0。

256³ 另采一次 basic，用来检查 64-CTA underfill、资源和 cuBLAS 实际 kernel；不依靠短 kernel 的 PC sampling 得出强 stall 结论。

每个根因结论至少需要两个独立信号，例如：

- register blocking 有效 = Release p50 改善 + issue active/eligible warps 改善；
- shared reuse 有效 = Release 改善 + shared loads/FMA 或 global requests/sectors 改善；
- async overlap 有效 = Release 改善 + normalized long-scoreboard/MIO 下降；
- register pressure 有害 = p95/latency 回退 + occupancy/resource 或 local spill 证据。

## 11. Keep / reject / promotion 标准

### 11.1 sync 相对当前 `cuda_tiled_vector`

必须同时满足：

- correctness 和四项 sanitizer 全通过；
- 256³ 主规格 L1、L2 的 p50 至少改善 5%；
- 1024³ L2 p50 至少改善 5%，作为非短 kernel 的独立确认；
- 任一报告 shape/level 的 p95 不回退超过 3%，CV≤0.10；
- 512³ 不出现超过 5% 的 p50 shape regression；
- workspace 仍为 0；
- local load/store/spill 为 0；
- 1024³ issue active 相对 28.4% 至少提高 5 个百分点，或 eligible warps/issue 指标给出等价的明确改善；
- normalized MIO、long-scoreboard、barrier 三项中至少两项改善，且没有新的 shared bank-conflict 瓶颈。

### 11.2 async 相对 sync

必须同时满足：

- correctness/sanitizer/无 spill；
- 1024³ L2 p50 至少改善 2%；
- 256³ 不回退超过 3%，所有 p95 不回退超过 3%，CV≤0.10；
- normalized MIO 和 long-scoreboard 均下降，其中至少一项相对下降 10%；
- issue active 不下降，shared-memory 增长没有把理论 resident CTA 降到 2 以下。

### 11.3 winner 与暂定接入

- 若 async 相对 sync 在 p50 的差异小于 1%，视为性能并列，依次按更低 p95、更低 CV、机制更简单选择，通常保留 sync；
- winner 还必须相对当前 vector 至少 1.05×，且不违反任一 shape 门禁；
- 通过后，显式 benchmark 名 `cuda_candidate_v2` 指向 winner；实验名继续保留以便复现；
- `kAuto` 仍保持 naive。只有在单独评审通过后，才允许 `KernelFamily::kCudaOptimized, implementation_id=0` 映射 winner；若没有候选过门禁，ID 0 继续 unsupported，不宣称 v2 晋升。

## 12. 计划产物

实施获批后新增版本化 suite，例如：

- `configs/benchmark_dense_gemm_candidate_v2_sync_release.json`；
- `configs/benchmark_dense_gemm_candidate_v2_async_release.json`；
- `configs/benchmark_dense_gemm_candidate_v2_final_release.json`；
- 对应 dense-only profile configs。

每轮保存：

- raw JSONL、manifest、aggregate CSV/JSON、strict comparison；
- correctness/CTest/sanitizer 摘要；
- environment、完整命令和 Git SHA/dirty；
- NSYS CSV/hotspot；
- NCU metrics CSV/JSON 和分析报告；
- `SHA256SUMS`；
- 二进制 `.ncu-rep/.nsys-rep/.sqlite` 留在忽略的 `profile/`，tracked manifest 记录名字、大小、SHA256 和 `committed=false`。

最终报告单列 L1/L2、256/512/1024，不混合样本；汇总 v1 vector、v2 sync、v2 async、strict cuBLASLt/cuBLAS 的 p50、p95、CV、TFLOP/s、差距、NCU 机制变化、winner/reject 原因和剩余瓶颈。

## 13. 实施前需要评审确认的决定

建议批准以下默认决定：

1. v2 首版采用 `32×32×16 / 128 threads / warp 16×16 / thread 4×2`；
2. sync register-tiled 先行，`cp.async` 必须作为第二个独立候选；
3. fast path 只覆盖 32/32/16 整除与 16-byte alignment，其余复用现有 fallback；
4. 256³ 是正式规格，1024³ 是强制的大计算量确认规格；
5. 通过门禁前不改 `kAuto`，不把任何 v2 性能结论写成已晋升。

评审通过后再创建独立优化分支、实现实验 A；在实验 A 的实测数据出来前，不预先假定实验 B 一定保留。
