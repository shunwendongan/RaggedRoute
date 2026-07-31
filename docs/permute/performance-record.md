# Token Permute 实际性能记录

## 测量环境

| 字段 | 值 |
|---|---|
| GPU / SM | `[待填写]` |
| CUDA / Driver / Compiler | `[待填写]` |
| Git revision | `[待填写]` |
| Benchmark config / seed | `[待填写]` |
| 测量层级 / Cache | `[L1/L2/L3/L4]` / `[warm/cold/rotating]` |

## 结果表

| Case | `(T,E,K,top_k)` | Distribution | Mapping | Variant | p50 (us) | p95 (us) | Speedup | Artifact |
|---|---|---|---|---|---:|---:|---:|---|
| `[待填写]` | `[待填写]` | `[待填写]` | `[prepared/full]` | `[待填写]` | `[待实测]` | `[待实测]` | `[待实测]` | `[path]` |

## NCU/NSYS 摘要

- Cursor/reset 是否计入：`[是/否，依据]`
- Load/store sectors/request：`[待实测]`
- Atomic/serialization 与 tail effect：`[待实测]`
- Prepared mapping 的额外成本：`[待实测]`

## 结论与限制

- 保留/回退决定：`[待填写]`
- 受益分布与 shape：`[待填写]`
- 下一步实验：`[待填写]`
