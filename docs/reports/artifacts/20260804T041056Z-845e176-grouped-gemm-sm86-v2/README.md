# Grouped GEMM SM86 v2：显式研究候选证据

本 artifact 对应实现 commit `845e176a9960c974eeab14bb012a6249db7bb8e4`，目标硬件为 RTX 3080 / SM86，CUDA 13.3.73，CUTLASS 4.6.1。

## 变更

- 新增 `cuda_grouped_sm86_fp32_v2`，保留 strict row-major FP32、FP32 accumulator、零 workspace、caller-stream 和现有 API 合同。
- v2 使用两 warp prefix metadata、`16x32` register tile、Ampere `cp.async` 双缓冲和 `__launch_bounds__(128,5)`。
- v2 只通过显式 `KernelFamily::kCudaOptimized` / benchmark variant 调用；`kAuto` 默认分派保持不变。
- 新 kernel 仅使用 SM86 支持的 CUDA core、shared memory、warp shuffle 和 `cp.async`，没有迁移 Hopper/Blackwell 专属 TMA/WGMMA/TMEM 机制。

## Clean Release A/B

协议：20 warmup、30 samples、5 kernel repeats、3 独立进程、seed `20260729`、warm cache、L2 operator steady、strict FP32。

Clean v2 相对 CUTLASS 的 ratio-of-sums 为 `0.9849x`，6/10 shape 获益，最差 shape 为 `T2048 uniform`（`0.5318x`）。因此它相对旧 v1 的改善不能等同于超过 CUTLASS；本 PR 明确将 v2 标记为显式 research candidate，而不是 `kAuto` 默认路径。

代表性 clean p50：

| Shape | v2 | CUTLASS |
|---|---:|---:|
| many-empty T16 | 9.01 us | 13.93 us |
| non-aligned T512 | 48.95 us | 42.39 us |
| single-hot T512 | 13.52 us | 22.73 us |
| uniform T2048 | 45.06 us | 23.96 us |
| Zipf1.4 T2048 | 42.19 us | 45.06 us |

历史同协议筛选中，v7 等价 kernel 相对旧 v1 ratio-of-sums 为 `1.121x`、8/10 获益；但 CUTLASS gate 仍失败。该限制在 PR 中保留，供 reviewer 决定是否接受研究代码。

## Profiler 诊断

NCU detailed（`v2.uniform.t2048`）：86 registers/thread、1.0 waves/SM、36.24% achieved occupancy、53.12% SM throughput、54.71% memory throughput、30.00% DRAM throughput。无 local spill；主要剩余问题是细粒度 tile/grid、metadata 复制和权重 tile reuse。

NSYS representative trace 的 kernel 为 `grouped_gemm_register16x32_sm86_v2_kernel`，total GPU kernel time `874,989 ns / 21 launches`；NSYS duration 只作诊断，正式 latency 来自上面的 unprofiled Release A/B。

## 验证

- Release build：通过；binary reports `build_git_dirty=false` and SHA `845e176a9960`。
- CTest：8/8 通过。
- v2 profile-once correctness：aligned T512 Zipf 和 non-aligned tail 均通过。
- 原基线 sanitizer：memcheck、initcheck、racecheck、synccheck 均通过。
- 原始 `.ncu-rep` / `.nsys-rep` 保留在本机 ignored `out/profile/grouped-gemm-v2-pr/`；仓库提交 normalized text/CSV/JSON、命令、manifest 和 `SHA256SUMS`。

参考边界：[CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)、[PyTorch cache-aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)、[Decoding the Skew](https://arxiv.org/abs/2607.23099)、[SonicMoE](https://arxiv.org/abs/2512.14080)。
