# Dense GEMM 实际性能记录

## 测量环境

| 字段 | 值 |
|---|---|
| GPU / SM | `[待填写]` |
| CUDA / Driver / Compiler | `[待填写]` |
| Git revision | `[待填写]` |
| Benchmark config / seed | `[待填写]` |
| 测量层级 | `L1/L2/L3/L4` |
| Cache 模式 | `warm/cold/rotating` |

## 结果表

| Case | Shape `(M,N,K)` | Variant | p50 (us) | p95 (us) | TFLOP/s | CV | Baseline speedup | Artifact |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `[待填写]` | `[待填写]` | `[baseline/candidate]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[待实测]` | `[path]` |

## NCU/NSYS 摘要

- Kernel duration：`[待实测]`
- SM throughput / Tensor Core utilization：`[待实测]`
- Occupancy / registers per thread：`[待实测]`
- 主要 stall 或 memory signal：`[待实测]`
- NSYS launch/host/chain 观察：`[待实测]`

## 结论与限制

- 保留/回退决定：`[待填写]`
- 受益 shape：`[待填写]`
- 退化 shape：`[待填写]`
- 下一步实验：`[待填写]`
