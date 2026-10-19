# RaggedRoute 项目简历表述

## 推荐版（中文，3 条）

- 面向 RTX 3080（SM86）重写 RaggedRoute 的 Dense GEMM 与 Ragged Grouped GEMM CUDA candidate：新增 FP16/BF16 Tensor Core `mma.sync.aligned.m16n8k16` 路径，FP32 累加/输出，并实现同步 staging 与 16B `cp.async` 双缓冲版本；Grouped 路径直接读取 device offsets，保持零全局 workspace 和现有公共 API 不变。
- 针对 8 组 tile/K-stage/pipeline 配置建立有界筛选和 94 个正式 shape–dtype 组合的 Release benchmark；Grouped GEMM FP16 在 15 个正式 shape 上相对本次测得同精度库基线达到 **5.21× ratio-of-sums p50、2.40×几何平均**，相对原 strict-FP32 V9 入口及其 fallback 达到 **2.04× ratio-of-sums p50、1.84×几何平均**。
- 完成 518 case 数值正确性、CTest 9/9、memcheck/initcheck/racecheck/synccheck；通过 `cuobjdump` 的 HMMA、NSYS 实际 kernel、NCU Tensor pipe 非零（Dense 约 30%，Grouped 约 15%）确认候选真实使用 Ampere Tensor Core。Dense GEMM 在大矩阵和部分 shape 获益，小矩阵/窄 shape 保留回退。

## 英文版

- Implemented FP16/BF16 Tensor Core GEMM candidates for Dense and ragged Grouped GEMM on RTX 3080 SM86 using `mma.sync.aligned.m16n8k16`, FP32 accumulation/output, synchronous staging, and 16B `cp.async` double buffering while preserving the existing public API.
- Built bounded configuration screening and a 94 case–dtype Release matrix. Grouped GEMM FP16 achieved **5.21× ratio-of-sums p50 and 2.40× geometric-mean speedup** versus the measured same-precision library envelope, and **2.04×/1.84×** versus the strict-FP32 V9 parent path.
- Validated 518 correctness cases and all four Compute Sanitizer modes; confirmed real Tensor Core execution with HMMA SASS, Nsight Systems kernel traces, and non-zero Nsight Compute Tensor pipe activity. Dense gains were shape-dependent, with fallbacks retained for narrow/small workloads.

## 面试口径

把“库基线”说成“本次测得的 cuBLAS/cuBLASLt/CUTLASS 配置”，把 Grouped FP16 的 5.21×/2.40× 说明为 15 个正式 shape 的 ratio-of-sums/几何平均；不要说成超过 cuBLAS 或 CUTLASS 的所有算法。Dense 的整体几何平均没有提升，不要写成 Dense 全面领先。Tensor Core 使用由 HMMA、实际 kernel 和 Tensor pipe 三重证据确认。
