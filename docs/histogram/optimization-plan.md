# Expert Histogram 优化方案

## 1. 当前范围

- 代码入口：`src/histogram/baseline.cu`、`src/histogram/operator.cpp`。
- API：`HistogramArgs`；输入为 route expert ids，输出满足 `sum(counts)=route_pairs`。
- 重点是 route 分布下的计数冲突、reset 成本和小 `E` 延迟，而不是单纯追求 uniform 数据吞吐。

## 2. 优化假设与顺序

1. **建立分布基线**：比较 uniform、Zipf、single-hot、round-robin，确认 global atomic contention 的敏感区间。
2. **降低 atomic 冲突**：评估 warp/block-private histogram、shared-memory 累加后合并，以及按 expert 分片的代价。
3. **优化 reset**：比较 device memset、融合清零和 persistent workspace；reset 必须按 L1/L2 合同分别记录。
4. **针对小 E 特化**：`E=8/16/32/64` 分别评估 block 配置、共享内存占用和空转。
5. **对比 CUB/library baseline**：统一 route input、seed、cache、repeats 和 measurement boundary，不能用不同 reset 规则比较。

## 3. 必测维度

- `T/E/top_k`、route distribution、cold/warm/rotating cache、L1 kernel body 与 L2 operator。
- 指标：atomic throughput/serialization、shared-memory bank conflict、global memory throughput、occupancy、reset latency。
- 验收：counts 精确一致、总数不变量、p50/p95、分布覆盖和最坏热点退化。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
