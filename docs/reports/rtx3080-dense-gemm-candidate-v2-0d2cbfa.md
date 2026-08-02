# RTX 3080 Dense GEMM candidate v2：库基线与 profile 复核

## 范围与方法

- Commit：`0d2cbfa6bfe14c9fe87d5497d47d24cb4d068575`（clean Release / `sm_86` / `-O3` / `-lineinfo`）。
- GPU：GeForce RTX 3080，sm_86，68 SM，10 GiB；CUDA 13.3，Nsight Systems 2026.1.3，Nsight Compute 2026.2.1。
- 数值合同：row-major FP32，`alpha=1`、`beta=0`、ascending-K 的 strict FP32 CUDA-core FMA；未使用 TF32 或 Tensor Core。cuBLASLt 为 `CUBLAS_COMPUTE_32F_PEDANTIC`，cuBLAS 为 `CUBLAS_PEDANTIC_MATH`。
- Release bundle：warm cache，20 warmup，3 个独立进程，每进程 30 个 samples、每个 sample 10 次调用；每个单元共 90 raw samples，数值校验全部通过，strict pairing 通过。这里的 p50/p95 是 90 raw samples 的统计值。
- `L1_kernel_body` 已准备输入和前置状态；`L2_operator_steady` 纳入 steady-state device-side operator 的必要 reset/metadata，workspace 预分配。二者均不是 host-call 测量。

原始 Release evidence：`profile/dense-gemm-0d2cbfa-v2-release-rerun-20260801T165621Z/`。NCU evidence：`profile/dense-gemm-0d2cbfa-v2-ncu-20260801T170100Z/`。二进制 `.ncu-rep` / `.nsys-rep` 保持忽略，不纳入 Git；NCU/NSYS duration 只用于诊断，不作为下表性能结论。

## Release 性能（最佳 candidate 为 v2 async）

| Shape | Level | v2 async p50 / p95 / CV | cuBLAS p50 / p95 / CV | cuBLASLt p50 / p95 / CV | 最快库 / v2 async |
|---|---|---:|---:|---:|---:|
| 256³ | L1 | 14.746 / 15.529 us / 19.79% | 13.056 / 15.826 us / 16.41% | 10.957 / 16.031 us / 17.02% | 1.35x |
| 256³ | L2 | 14.797 / 15.314 us / 17.06% | 12.902 / 15.724 us / 16.15% | 11.469 / 15.667 us / 15.78% | 1.29x |
| 512³ | L1 | 41.165 / 41.426 us / 2.85% | 27.238 / 27.546 us / 0.60% | 27.238 / 28.524 us / 3.44% | 1.51x |
| 512³ | L2 | 41.216 / 41.984 us / 3.09% | 27.238 / 27.443 us / 2.28% | 27.238 / 27.500 us / 1.43% | 1.51x |
| 1024³ | L1 | 226.202 / 235.110 us / 1.98% | 156.416 / 158.572 us / 1.64% | 156.109 / 158.684 us / 1.08% | 1.45x |
| 1024³ | L2 | 226.611 / 234.404 us / 1.18% | 154.675 / 158.889 us / 1.35% | 155.955 / 158.218 us / 1.51% | 1.47x |

256³ 的所有相关 CV 都超过 10%，因此是 `variance-limited`：只能说明小矩阵存在 1.3x 左右的差距，不能据此作晋升或细粒度排序。512³、1024³ 稳定；v2 async 分别达到约 6.51、9.48 TFLOP/s，严格 FP32 cuBLAS 分别约 9.86、13.89 TFLOP/s。L1 与 L2 的差异在 512³/1024³ 均很小，说明目前主要差距在 kernel/device steady path，而非被 L2 特有 bookkeeping 放大。

作为 v2 内部对照，sync 的 L2 p50 为 41.677 us（512³）和 242.483 us（1024³）。所以 `cp.async` 在 1024³ 将 v2 延迟降低约 7.0%，但没有消除与库的约 1.47x 差距。

## NSYS：实际库内核

1024³、L2、20 warmup 后取 21 个 launch。cuBLAS 为 `ampere_sgemm_64x64_nn`，平均 144.820 us；cuBLASLt 同样选择该 kernel，平均 145.005 us。v2 async / sync 分别为 223.593 / 242.526 us，旧 16x16 tiled-vector 为 955.260 us。此处仅用于确认 dispatch 和 hotspot；Release 比值以上表为准。

## NCU full：1024³ L2 的瓶颈证据

| 指标 | v2 async | cuBLAS | 解读 |
|---|---:|---:|---|
| Grid / block | 1024 / 128 | 512 / 64 | v2 采用 32x32 CTA；库内核名为 64x64。 |
| Registers/thread；SMEM/block | 70；10,240 B | 126；9,472 B | v2 的较低寄存器数没有转化为更高 issue。 |
| Achieved occupancy | 49.89% | 24.14% | occupancy 不是根因；库以更低 occupancy 仍更快。 |
| Eligible warps/scheduler | 1.627 | 1.610 | 两者接近，不能把差距归因于没有可调度 warp。 |
| Issue active | 51.61% | 61.34% | v2 发射利用率仍低 9.72 percentage points。 |
| L2 hit rate；DRAM read | 96.04%；14.56 MB | 92.19%；13.24 MB | warm-cache 下两者都不是 DRAM 饱和；v2 不应以“再加全局向量加载”为首选。 |
| Global-load sectors | 9.44 M | 4.22 M | 总扇区约为库的 2.24x；不同 load 宽度使 sectors/request 不能直接横比。 |
| Global-sector theoretical excess | 10.62 M vs 8.52 M ideal（+24.6%） | 4.76 M，等于 ideal | v2 有可测的 transaction amplification；库内核没有该 excess。 |
| Shared wavefront excess | 21.76 M vs 18.87 M ideal（+15.3%） | 5.47 M vs 5.41 M（+1.2%） | v2 shared staging/read path 有额外工作。 |
| Local load / store | 0 / 0 | 0 / 0 | 两者均没有 profiler 证据表明发生 register spill。 |
| Tensor active | 0% | 0% | 比较的是 strict-FP32 CUDA-core 路径。 |
| MIO throttle / issue-active | 3.228 | 0.0666 | v2 的 shared/load-store 管线压力是主要限制。 |
| Barrier stall / issue-active | 2.825 | 0.113 | v2 的 CTA 同步等待显著。 |
| Long scoreboard / issue-active | 0.263 | 0.107 | 存在但不是 v2 的最大 stall；双缓冲已部分隐藏全局访存等待。 |

Stall 指标是“每个 active issue 对应的平均停顿 warp 数”，不是百分比，因此可以大于 1。v2 的 `MIO` 和 `barrier` 分别约为库的 48x、25x；同时它有更多 global/shared sectors，而 DRAM 读量只高约 10%。这两组独立信号共同支持如下判断：当前主要是 CTA 内共享内存/加载发射与同步开销、以及 tile reuse 不足造成的片上交易量，不是 DRAM 峰值带宽、spill 或低 occupancy。

v2 在 `optimized_v2.cu` 中对 K=1024 进行 64 个 K=16 tile。双缓冲循环保留了每 tile 的 consumer barrier，以及除最后一个 tile 外的 reuse-protection barrier（共 127 个 CTA barrier 点）。这与 NCU barrier 证据相符。A 的 padding 使显式 shared bank conflict 相对 shared-load wavefront 很小，故不把 bank conflict 列为首要瓶颈。

## 下一轮建议（不在本报告中实现）

1. **P0：扩大 CTA/warp 的输出 tile，同时保持严格 FP32。** 先做单变量 64x32 CTA（256 threads，保留 4x2 register microtile）实验，目标是减少 32x32 CTA 重复装入的 A tile，并让 grid 接近库内核的工作粒度。必须检查 registers、SMEM、occupancy、spill、global-sector excess 和 MIO；Release 1024³ 的 L1/L2 p50、p95、CV 都改善才保留。
2. **P1：增大 K stage 或重排 staging，减少同步/加载节奏。** 例如从 K=16 到 K=32 的独立实验可将 CTA barrier cadence 约减半，但会增加 SMEM 和可能降低 resident blocks。只有 MIO、barrier 和 Release p95 同时改善才保留。
3. **P2：再评估 `cp.async` pipeline 深度，而不是直接增加 vectorization。** 当前 long-scoreboard 已不是最大项，更多 stage 的收益不确定；若 P0/P1 后 long-scoreboard 仍高，才以 no-spill、CV<=0.10、p95 不退化为门槛测试三 stage。

不要把 TF32/Tensor Core 当作本 strict-FP32 合同下的等价优化，也不要从本组 256³ 的高方差数据宣称 winner 或改变 `kAuto` dispatch。
