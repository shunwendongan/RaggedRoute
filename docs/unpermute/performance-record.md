# Unpermute 实际性能记录

## 测量环境

| 字段 | 值 |
|---|---|
| GPU / SM | `[待填写]` |
| CUDA / Driver / Compiler | `[待填写]` |
| Git revision | `[待填写]` |
| Benchmark config / seed | `[待填写]` |
| 测量层级 / Cache | `[L1/L2/L3/L4]` / `[warm/cold/rotating]` |

## 结果表

| Case | `(T,E,N,top_k)` | Distribution | Variant | p50 (us) | p95 (us) | CV | Speedup | Artifact |
|---|---|---|---|---:|---:|---:|---:|---|
| `[待填写]` | `[待填写]` | `[待填写]` | `[待填写]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[path]` |

## NCU/NSYS 摘要

- Gather/store sectors/request 与 cache hit：`[待实测]`
- weighted reduce 指令/寄存器/occupancy：`[待实测]`
- route skew、tail effect、chain contribution：`[待实测]`
- 是否存在 atomic/race 风险：`[待实测]`

## 结论与限制

- 保留/回退决定：`[待填写]`
- 受益 shape 与分布：`[待填写]`
- 下一步实验：`[待填写]`
