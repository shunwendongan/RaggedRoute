# Exclusive Scan 优化方案

## 1. 当前范围

- 代码入口：`src/scan/baseline.cu`、`src/scan/operator.cpp`。
- API：`ExclusiveScanArgs`；输出 `offsets[0]=0`，相邻差分等于 counts，`offsets[E]=route_pairs`。
- 这是典型的小数组 metadata 算子，优先优化 launch latency 和同步层级，不把大数组吞吐假设直接套用到 `E<=64`。

## 2. 优化假设与顺序

1. **测量规模边界**：覆盖小 `E`、大 `E`、不同 route count，确认单 warp/block 的适用区间。
2. **减少同步**：比较 warp shuffle、block scan、分层 scan，记录 barrier 和 eligible warp 变化。
3. **选择 library baseline**：与 CUB device/block/warp scan 在相同 workspace、reset、调用边界下比较。
4. **融合可能性**：评估 Histogram→Scan 的 metadata 融合，但必须单独报告 fused variant，不能与独立 scan latency 混淆。
5. **workspace 与 stream**：确认 hot path 不分配、不隐式同步，并记录 workspace 对 operator 的影响。

## 3. 必测维度

- `E`、route distribution、L1 kernel body、L2 operator、完整 L3 chain。
- 指标：latency、barrier stall、instruction count、occupancy、launch overhead、workspace bytes。
- 验收：offsets 精确一致、首尾/差分不变量、p95 和 chain contribution。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
