# Grouped GEMM 优化方案

## 1. 当前范围

- 代码入口：`src/grouped_gemm/baseline.cu`、`src/grouped_gemm/operator.cpp`。
- API：`GroupedGemmArgs`；输入为 packed tokens、expert weights 和 offsets，支持空 expert，使用 `max_expert_tokens` 作为保守 launch bound。
- 这是主算子，优化结论必须覆盖 ragged `M_e` 分布，不能只用均匀 expert workload。

## 2. 优化假设与顺序

1. **建立 per-expert GEMM 基线**：记录每个 expert 的 `M_e`、空 expert、总 `R` 和 launch bound，拆分算术与调度成本。
2. **改善 ragged 负载均衡**：比较 per-expert launch、grouped tile scheduler、persistent CTA/warp 调度，观察 tail effect 和 SM active variance。
3. **Tensor Core 路径**：在 FP16/BF16 输入、FP32 accumulate 下评估 tile、alignment、stage 和 accumulator register 压力。
4. **减少 metadata/launch overhead**：评估 grouped dispatch、workspace reuse 和与前后算子的融合边界，但保持 L1/L2/L3 结果可比。
5. **空 expert 与小 M 特化**：针对大量空 expert、`M_e=1/2/4` 和大 expert 头部建立 dispatch 分支。
6. **对比 cuBLAS per-expert/CUTLASS grouped**：所有 baseline/candidate 共享输入、weights、offsets、math mode 和测量协议。

## 3. 必测维度

- `T/E/K/N`、uniform/Zipf/single-hot、空 expert、最大 `M_e/min M_e` 比例。
- 指标：effective TFLOP/s、Tensor Core utilization、waves/SM、SM active timeline、long scoreboard、register/shared memory、launch overhead。
- 验收：逐 expert CPU GEMM、空 expert、L2 steady latency、L3 chain latency 和 shape-balanced promotion policy。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
