# Top-K Gate 优化方案

## 1. 当前范围

- 代码入口：`src/topk_gate/baseline.cu`、`src/topk_gate/operator.cpp`。
- API：`TopKGateArgs`；当前合同是 Top-2、selected-softmax、lower expert id tie-break，以及固定 NaN 策略。
- 正确性优先于吞吐：ids 必须精确一致，weights 按合同比较容差。

## 2. 优化假设与顺序

1. **确认访问与布局**：评估 token-major logits `[T,E]` 的 coalescing，以及 `E` 较小时一个 warp/block 处理一个 token 的代价。
2. **优化 Top-K selection**：比较寄存器局部候选、warp shuffle reduction、分层 selection，避免不必要的全量排序。
3. **融合归一化**：在不改变 selected-softmax、tie-break 和 NaN 行为的前提下融合 max/sum/weight 写回。
4. **处理小规模与长尾 shape**：为 `T=1/8/32`、`E=8/16/32/64` 建立专用 launch/config，避免固定大 block 的空转。
5. **评估确定性成本**：明确是否需要跨 block 确定性；任何 tie/NaN 改动都必须增加回归用例。

## 3. 必测维度

- `T/E`、uniform/热点/NaN/相同 logits 分布、不同 cache 模式。
- 指标：latency、instructions、warp divergence、register/thread、occupancy、memory throughput、branch/stall breakdown。
- 验收：ids 精确一致、weights 误差、selected-softmax 不变量、L2 operator latency 和 p95。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
