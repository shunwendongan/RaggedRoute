# Exclusive Scan 实际性能记录

## 测量环境

| 字段 | 值 |
|---|---|
| GPU / SM | `[待填写]` |
| CUDA / Driver / Compiler | `[待填写]` |
| Git revision | `[待填写]` |
| Benchmark config / seed | `[待填写]` |
| 测量层级 / Cache | `[L1/L2/L3/L4]` / `[warm/cold/rotating]` |

## 结果表

| Case | `E` / route count | Variant | p50 (us) | p95 (us) | CV | Workspace (B) | Speedup | Artifact |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `[待填写]` | `[待填写]` | `[naive/CUB/candidate]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[path]` |

## 正确性与 profile 摘要

- `offsets[0]` / `offsets[E]`：`[通过/失败]`
- 相邻差分与 counts：`[通过/失败]`
- Barrier/synchronization signal：`[待实测]`
- 独立 scan 与 L3 chain contribution：`[待实测]`

## 结论与限制

- 保留/回退决定：`[待填写]`
- 适用 `E` 区间：`[待填写]`
- 下一步实验：`[待填写]`
