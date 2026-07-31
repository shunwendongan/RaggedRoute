# Unpermute 优化方案

## 1. 当前范围

- 代码入口：`src/unpermute/baseline.cu`、`src/unpermute/operator.cpp`。
- API：`UnpermuteArgs`；按 `route_pos` 从 packed expert output gather，并按 route weights 做 weighted reduce。
- 当前目标是 token-owned、无全局 atomic 的确定性合并；任何 fusion 方案都必须保持该语义。

## 2. 优化假设与顺序

1. **冻结 mapping 与权重语义**：确认每个 token 的 route 数、`route_pos` 方向、权重 dtype/accumulator 和输出布局。
2. **优化 gather 合并访问**：评估 token/route/feature 维度的工作划分、连续输出写入和 packed input 的访问合并。
3. **向量化 weighted reduce**：在 output 对齐时比较 vector width、FMA 指令和 register pressure；保留 tail path。
4. **降低重复读取**：评估 route metadata 缓存、warp-level reuse 和与 grouped GEMM 输出布局协同，避免只在单算子 warm cache 下获益。
5. **评估融合边界**：研究 Grouped GEMM→Unpermute 融合或 producer-friendly layout，但把融合版本和独立 unpermute 结果分开记录。

## 3. 必测维度

- `T/E/N/top_k`、route distribution、output alignment、L1/L2/L3 chain。
- 指标：global load/store efficiency、L1/L2 hit、long scoreboard、register/thread、SM timeline、p95。
- 验收：token-owned CPU reference、权重累加容差、空/倾斜 expert、无 race/无 atomic 语义偏差。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
