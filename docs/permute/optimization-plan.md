# Token Permute 优化方案

## 1. 当前范围

- 代码入口：`src/permute/baseline.cu`、`src/permute/operator.cpp`。
- API：`TokenPermuteArgs`；根据 expert ids 和 offsets 生成 `x_permuted`、`route_pos`，可选 `sorted_route`。
- 需要同时关注 token 数据搬运和 cursor/metadata 的写入冲突，不能只看 copy bandwidth。

## 2. 优化假设与顺序

1. **冻结 mapping 语义**：验证 expert segment、`route_pos` 方向和 `sorted_route` 逆映射，先建立可审计 reference。
2. **比较 scatter/gather 布局**：评估 token-owned、route-owned 和 expert-segment-owned 工作划分，观察 global store 合并与写入冲突。
3. **优化 cursor 分配**：比较 global atomic、warp/block-private cursor、预计算 route positions，分别记录 metadata 成本。
4. **向量化 copy 与 tail path**：在 hidden 对齐时启用向量化，非对齐 shape 使用安全 edge path；检查 register/occupancy 变化。
5. **利用真实 route skew**：uniform、Zipf、热点和空 expert 必测，验证负载不均衡是否产生 tail effect。
6. **比较 prepared mapping variant**：如果提前准备 mapping，必须把 prepare 成本在 L2/L3 中按合同计入或明确排除。

## 3. 必测维度

- `T/E/K/top_k`、route distribution、`sorted_route` 开关、L1/L2/L3。
- 指标：global load/store sectors/request、L1/L2 hit、atomic/serialization、SM active timeline、tail effect。
- 验收：expert segment、双向 mapping、逐行内容和 empty expert 行为完全一致。

## 4. 决策记录

| 日期 | 版本/假设 | 证据 | 结论 |
|---|---|---|---|
| YYYY-MM-DD | `[待填写]` | 见 performance-record | `[保留/回退/继续]` |
