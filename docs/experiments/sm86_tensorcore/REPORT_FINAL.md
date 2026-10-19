# SM86 Tensor Core / MMA GEMM 实验结论

## 结论摘要

本轮在 RTX 3080（SM86）上完成了 Dense GEMM 与 Ragged Grouped GEMM 的 FP16/BF16 Tensor Core 候选实现、正确性验证、sanitizer、正式 Release 矩阵和 Nsight 证据采集。候选使用 `mma.sync.aligned.m16n8k16`，FP32 累加与 FP32 输出；异步版本使用 16B `cp.async` 双缓冲。

最明确的结果是 Grouped GEMM：冻结配置为 `m32_n64_k32_async`，相对本次测得的同精度库基线 envelope，FP16 ratio-of-sums p50 为 **5.2093×**、几何平均为 **2.3975×**，15 个正式 shape 全部获益；相对原 strict-FP32 V9 入口及其 fallback，FP16 ratio-of-sums p50 为 **2.0395×**，几何平均为 **1.8384×**。BF16 Grouped 的同精度库 ratio-of-sums p50 为 **5.5939×**、几何平均为 **2.4147×**，但少量 shape 的高 CV 使脚本将结论标为证据不足，不能把它写成稳定总体保证。

Dense GEMM 的收益集中在大矩阵和部分形状。相对原 strict-FP32 V3 入口及其 fallback，FP16/BF16 ratio-of-sums p50 分别为 **1.2458×/1.2796×**；相对同精度库 envelope 分别为 **1.0433×/1.0267×**。但 Dense 的几何平均同精度结果分别为 **0.9882×/0.9815×**，获益覆盖率为 **31.2%/34.4%**，因此不能声称 Dense 在整个正式矩阵上整体领先。

## 数值与正确性

- 518 个 Tensor Core 测试 case 通过，覆盖 FP16/BF16、tile 边界、非对齐输入、K=0、空 expert、many-empty、single-hot、uniform、Zipf、输入偏移、非默认 stream、redzone、输入/offset 不变性和异步错误检查。
- CTest 9/9 通过；原 strict-FP32 correctness 回归通过。
- Compute Sanitizer：memcheck、initcheck、racecheck、synccheck 均为 0 errors/hazards。
- 正式矩阵 2350 条进程记录，5 个独立 process run，各 470 条；候选和 parent validation 全部通过。4 条 CUTLASS unsupported shape 单独记录，没有用 SIMT 或虚构时间替代。
- 低精度实现正确性使用量化后输入的 FP64 oracle；相对原始 FP32 master 的误差另行记录，未通过放宽旧 FP32 容差掩盖精度变化。

## 硬件证据

- `cuobjdump --dump-sass` 确认 32 个 MMA kernel instantiation 均包含 `HMMA.16816.F32`；异步实例同时包含 `LDGSTS`；SASS 中 local `LDL/STL` 为 0。
- NSYS 成功观察到实际运行的 `dense_gemm_mma_kernel` 与 `grouped_gemm_mma_kernel`。
- NCU basic/targeted 成功采集四条最终路径，Tensor pipe 非零：Dense FP16 **30.016%**、Dense BF16 **30.080%**、Grouped FP16 **15.146%**、Grouped BF16 **15.108%**。
- 代表 Dense kernel：79 registers/thread、37888 B shared/block、15.52% achieved occupancy；代表 Grouped kernel：66 registers/thread、15360 B shared/block、约 35.25% achieved occupancy。

证据目录：

- 正式结果：[analysis/REPORT.md](../../../out/tensorcore/analysis/REPORT.md)、[summary.json](../../../out/tensorcore/analysis/summary.json)、[per_shape.csv](../../../out/tensorcore/analysis/per_shape.csv)
- 硬件：[profile-final-v2/analysis/summary.json](../../../out/tensorcore/profile-final-v2/analysis/summary.json)、[ncu_metrics.csv](../../../out/tensorcore/profile-final-v2/analysis/ncu_metrics.csv)、[instruction-evidence.json](../../../out/tensorcore/sass/instruction-evidence.json)
- 工具链：[toolchain.md](toolchain.md)

## 计时解释与边界

正式 Release 计时没有使用 profiler duration。native low precision 的输入已驻留设备；conversion 视图把两个 FP32→低精度转换和 GEMM 一起计入。Grouped cuBLAS 基线把 device offsets 的 D2H、pinned 缓冲和同步计入 L2；CUTLASS 基线把 device metadata preparation 计入。库基线是本次实际测得的配置，不代表穷举 cuBLASLt heuristic、CUTLASS tile 或 split-K 的全部可能性。

screening 中有 8 条 GPU timing CV>0.50 的原始记录。按用户要求保留并披露这些样本，冻结规则仍使用几何平均 p50；完整原始样本和覆盖原因保存在 `out/tensorcore/frozen.json`。因此 BF16 结论保留“方向和数值”，不升级为稳定总体承诺。

候选仍保持 `benchmark_only`，公共 API 和 `Auto` 分发没有改变。原 strict-FP32 V3/V9 结果继续保留。

## 可用于简历的事实边界

可以写入简历的重点是：在 RTX 3080 SM86 上实现并验证 FP16/BF16、FP32 accumulation/output 的 `mma.sync` Tensor Core Dense/Grouped GEMM candidate；Grouped GEMM FP16 在 15 个正式 shape 上相对本次测得同精度库 envelope 的 ratio-of-sums p50 为 5.21×、几何平均为 2.40×，相对原 strict-FP32 parent 为 2.04×；通过 518 case correctness 与四项 Compute Sanitizer，并用 HMMA、NSYS kernel、NCU Tensor pipe 三类证据确认实际走 Tensor Core。

Dense 不应写成“全矩阵整体提速”，应写成“在大矩阵和部分形状取得收益，窄/小形状存在回退”。库基线应写成“本次测得的 cuBLAS/cuBLASLt/CUTLASS 配置”，不要写成超过库的全部可能实现。
