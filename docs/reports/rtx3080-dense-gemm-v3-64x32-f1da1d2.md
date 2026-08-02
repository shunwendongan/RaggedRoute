# RTX 3080 Dense GEMM v3 64x32 CTA 实验

## 结论

`cuda_register_tiled_v3_64x32_async` 是目前在 512³、1024³ 上最快的显式 CUDA candidate。相对未改变的 v2 async，它在本次 clean Release run 的 L2 p50 分别快 18.5% 与 24.7%；1024³ 仍比 strict-FP32 cuBLAS 慢约 18.8%。

它**没有**取代 `kAuto` 或 optimized ID 0：256³ 所有相关 variant 的 CV 都超过 10%，且 v3 的 256³ L2 p50 在本 run 中比 v2 差。原始正式合同以 256³ 为主规格，不能用不稳定的小 shape 数据晋升默认 dispatch。ID 7 保留为显式研究 candidate，v2 ID 6 继续可用。

## 实现

- Commit：`f1da1d2557e1c83508ed157a8117b3f1c602dcc7`；RTX 3080 sm_86、CUDA 13.3、Release `-O3 -lineinfo`。
- 新 kernel：`src/dense_gemm/cuda_candidate/optimized_v3.cu`，显式 implementation ID 7。
- 机制组合：二维 CTA 映射、`64x32x16` CTA tile、四个 `32x16` warp tile、每线程 `8x2` register microtile、对齐 `float4` cooperative staging、Ampere `cp.async.cg` 两级 shared-memory 双缓冲。
- 这不是 `cp.sync`：`cp.async` 只负责 global-to-shared copy；`cp.async.wait_group` 与 CTA `__syncthreads()` 仍保证 consumer 读取和 stage 重用安全。
- FP32 row-major、`alpha=1`、`beta=0`、ascending-K FMA 合同不变；不使用 TF32/Tensor Core；外部 workspace 为 0。非整除或未 16-byte 对齐的输入继续回退到 vector/scalar path。

## 正确性

- Release/sm_86 clean rebuild 与 8/8 CTest 通过。
- 17x19x13 fallback、256³ fast path smoke 均 `validation.ok=true`。
- 对 256³ v3 fast path 的 Compute Sanitizer：memcheck、initcheck、racecheck、synccheck 均为零错误/零 hazard。
- 这是 targeted sanitizer 范围；没有把先前已知会超时的全算子 sanitizer 误报为本次完成。

## 无 profiler Release A/B

协议：warm cache、20 warmup、3 独立进程、每进程 30 samples、每 sample 10 calls、seed `20260729`。表中为 90 raw samples 的 p50 / p95（us）/ CV。

| Shape | Level | v2 async | v3 64x32 | cuBLAS | cuBLASLt | 结论 |
|---|---|---:|---:|---:|---:|---|
| 256³ | L1 | 13.312 / 15.053 / 16.82% | 11.981 / 18.488 / 18.76% | 15.053 / 15.770 / 12.99% | 15.155 / 15.724 / 12.85% | variance-limited |
| 256³ | L2 | 9.882 / 13.880 / 19.61% | 12.186 / 18.432 / 18.80% | 14.950 / 15.565 / 14.17% | 13.824 / 15.565 / 14.50% | variance-limited；不晋升 |
| 512³ | L1 | 29.594 / 31.642 / 3.63% | 25.037 / 28.467 / 6.39% | 24.115 / 27.238 / 6.21% | 23.859 / 27.443 / 6.61% | v3 比 v2 快 1.18x；比最快库慢 1.05x |
| 512³ | L2 | 29.491 / 31.447 / 2.19% | 24.883 / 28.570 / 6.74% | 27.443 / 29.798 / 3.53% | 23.859 / 28.063 / 7.47% | v3 比 v2 快 1.19x；比最快库慢 1.04x |
| 1024³ | L1 | 229.683 / 233.277 / 1.45% | 181.248 / 188.733 / 2.23% | 156.262 / 159.391 / 1.07% | 154.726 / 156.657 / 0.84% | v3 比 v2 快 1.27x；比最快库慢 1.17x |
| 1024³ | L2 | 227.686 / 232.320 / 1.38% | 182.630 / 189.041 / 2.02% | 153.754 / 155.034 / 0.94% | 155.034 / 158.070 / 1.04% | v3 比 v2 快 1.25x；比 cuBLAS 慢 1.19x |

运行 manifest 记录了 `douyin.exe` GPU 进程。因此 256³ 的高方差不能在“无竞争 GPU 工作”条件下重跑确认；本报告不输出小 shape 的确定性 winner，也不改变默认 dispatch。

## NSYS 与 NCU（1024³ / L2）

NSYS 发现的真实 v3 kernel 为 `dense_gemm_register_tiled_v3_64x32_async_kernel`；21 次 post-warmup launch 的平均诊断 duration 为 177.657 us。它只用于确认 dispatch/hotspot，不是 Release latency。

| NCU full 指标 | v3 64x32 | v2 async | cuBLAS |
|---|---:|---:|---:|
| Diagnostic duration | 177.280 us | 222.528 us | 144.256 us |
| Grid / block | 512 / 128 | 1024 / 128 | 512 / 64 |
| Registers/thread；SMEM/block | 85；15,360 B | 70；10,240 B | 126；9,472 B |
| Achieved occupancy | 34.62% | 49.87% | 24.14% |
| Eligible warps/scheduler | 1.561 | 1.632 | 1.611 |
| Issue active | 55.84% | 51.47% | 63.69% |
| Global-load sectors | 7.34 M | 9.44 M | 4.22 M |
| Local load / store | 0 / 0 | 0 / 0 | 0 / 0 |
| MIO throttle / issue-active | 1.272 | 3.235 | 0.066 |
| Barrier stall / issue-active | 1.236 | 2.791 | 0.115 |
| Long scoreboard / issue-active | 0.137 | 0.240 | 0.107 |

相对 v2，v3 的 global-load sectors 降低 22.2%，MIO stall 降低 60.7%，barrier stall 降低 55.7%，long-scoreboard 降低 42.9%，并把 issue active 提高 4.37 percentage points。较低 occupancy 没有阻止加速，且 `local load/store = 0` 排除了 spill 解释。

残余瓶颈仍清晰：v3 的 global-load sectors 是 cuBLAS 的 1.74x，MIO/barrier stall 约为 cuBLAS 的 19x/11x。`memory_l2_theoretical_sectors_global` 为 8.52 M、reported ideal 为 6.42 M（+32.6%）；v2 是 10.62 M/8.52 M（+24.6%），而 cuBLAS 无 reported excess。也就是说 v3 以更大 tile 降低了绝对 transaction 数，但没有消除其相对 transaction amplification；下一候选应优先隔离 K-stage/staging 映射或进一步的 CTA reuse，而不是增加 TF32 或盲目增加 pipeline stages。

## 证据

已版本化的文本 evidence bundle（raw benchmark、aggregate、严格 pairing comparison、sanitizer 日志、NCU analysis、NSYS CSV、SHA256SUMS）：

`docs/reports/artifacts/20260802T045524Z-f1da1d2-dense-gemm-v3-64x32/`

二进制 NCU report 保留在忽略的 `profile/dense-gemm-f1da1d2-v3-ncu-20260802T050000Z/reports/`，其文件名、大小、SHA256 与 `committed=false` 已记录在 evidence manifest。v3 NSYS `.nsys-rep`（62,506 B，SHA256 `c196f3c2240bc31d98d8018fa8d5c11f0b44b056f86085a4d7a488eba50f19dc`）和 `.sqlite`（266,240 B，SHA256 `66b5bfd55f250568b79f0469d1da6315d25cc87c5863cb8d90a03798ce81a9e4`）也仅保留在忽略的 profile 目录。
