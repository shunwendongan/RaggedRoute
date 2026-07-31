# Grouped GEMM 实际性能记录

## 测量环境

| 字段 | 值 |
|---|---|
| GPU / SM | `[待填写]` |
| CUDA / Driver / Compiler | `[待填写]` |
| Git revision | `[待填写]` |
| Benchmark config / seed | `[待填写]` |
| 测量层级 / Cache | `[L1/L2/L3/L4]` / `[warm/cold/rotating]` |

## 结果表

| Case | `(T,E,K,N)` | Distribution | `M_e` stats | Variant | p50 (us) | p95 (us) | TFLOP/s | Speedup | Artifact |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `[待填写]` | `[待填写]` | `[待填写]` | `[min/mean/max]` | `[待填写]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[path]` |

## NCU/NSYS 摘要

- Tensor Core / SM throughput：`[待实测]`
- waves/SM、tail effect、active-cycle variance：`[待实测]`
- register/shared-memory/occupancy 限制：`[待实测]`
- launch/metadata/chain contribution：`[待实测]`

## 结论与限制

- 保留/回退决定：`[待填写]`
- 受益 `M_e` 分布：`[待填写]`
- 退化或不适用分布：`[待填写]`
- 下一步实验：`[待填写]`
