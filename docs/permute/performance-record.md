# Token Permute 实际性能记录

## 2026-07-31 / RTX 3080 strict-FP32 baseline

- Git：`a9489abce704`；case `T=512,E=64,K=256,top_k=2,Zipf s=1.4`；expert segment、`route_pos`、`sorted_route` 与逐行内容通过。

| Level / variant | p50 (us) | p95 (us) | CV | Mapping boundary |
|---|---:|---:|---:|---|
| L1 `cuda_naive` | 10.240 | 10.240 | 0.094 | cursor reset 排除，单 launch |
| L2 `cuda_naive` | 19.763 | 21.356 | 0.173 | 包含 cursor reset |
| L2 vLLM full-from-ids reference | 32.358 | 44.595 | 0.179 | histogram/sort/scan/expand |
| L2 `cuda_naive_from_ids` | 40.550 | 48.548 | 0.112 | histogram/scan/atomic permute |

严格 full-from-ids 配对中 vLLM p50 快 1.25×，但两方 CV 均超过 0.10，不能用于 promotion。NCU basic 对 kernel body 显示 1024 blocks、1.255 waves/SM、70.9% achieved occupancy、SM 10.3%、Memory/DRAM 19.6%，没有带宽饱和证据。

结论：保留 naive 基线；后续优先比较预计算位置与向量化 copy，但必须把 mapping preparation 在 L2/L3 中公平计入。

完整证据：[中央报告](../reports/rtx3080-naive-profile-a9489ab.md)；[artifact bundle](../reports/artifacts/20260731T115243Z-a9489abce704-rtx3080-naive-profile-v1/)。
