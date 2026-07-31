# Dense GEMM 优化方案

## 1. 当前范围

- 代码入口：`src/dense_gemm/baseline.cu`、`src/dense_gemm/operator.cpp`。
- API：`DenseGemmArgs`，当前文档主线覆盖 row-major、FP32 语义；FP16/FP32 accumulate 作为后续优化版本单独记录。
- 对照基线：naive CUDA、cuBLAS/cuBLASLt；所有对比固定 `M/N/K`、dtype、layout、`alpha/beta` 和测量层级。

## 2. 优化假设与顺序

1. **先建立 tile 基线**：确认 global load 合并、shared-memory tile、register blocking 和边界路径正确。
2. **提升数据复用**：比较 CTA tile、warp tile、thread tile，以及 shared-memory bank conflict、寄存器占用和 occupancy。
3. **流水化访存**：在 SM86 上评估 `cp.async`/双缓冲、stage 数和同步开销；以 Nsight Compute 指标验证是否减少 long scoreboard。
4. **向量化与尾部路径**：只在对齐 shape 启用 16B load/store，非对齐 shape 保留明确 edge path，避免为了单点收益破坏通用路径。
5. **Tensor Core 路径**：在 FP16/BF16 输入、FP32 accumulate 下单独建立候选 variant，与 CUDA Core 路径严格配对。
6. **shape-aware dispatch**：按 `M/N/K`、对齐条件和 workspace/寄存器约束选择 variant，不用一个平均结果覆盖所有 shape。

## 3. 必测维度

- `M/N/K`：小矩阵、典型 router projection、非 8/16/64 对齐 shape。
- 指标：latency、effective TFLOP/s、SM throughput、Tensor Core utilization、occupancy、register/thread、L1/L2/DRAM throughput。
- 验收：数值误差、p50/p95、CV、L1/L2/L3 测量边界、相对 promotion baseline 的 shape-balanced speedup。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
