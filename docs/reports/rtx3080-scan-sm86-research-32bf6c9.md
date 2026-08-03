# RTX 3080 Exclusive Scan 单 warp优化研究

## 结论

本轮没有候选满足晋升合同，最终默认 dispatch 保持 `cuda_naive`。C2 `cuda_warp_blocked_scalar` 在 E=29..64 的 5 进程复测中得到 `1.0503x` ratio-of-sums 和 91.7% p50 获益覆盖，但 36/36 pairs 的 baseline 与 candidate all-samples CV 都超过 10%，最坏 p95 ratio 为 `7.042x`。这是硬门槛失败，不用中心趋势掩盖 Windows/WDDM tail。

最终 PR 因此只保留公共 4-byte 对齐检查、原地/redzone/边界 API 测试、exact-int32 benchmark metadata、CUB 对比配置、文档和可审计证据；C1/C2/C3 源码、optimized id 和 runtime dispatch 接口全部删除。

## 环境与合同

- 基线：`origin/main@927c585e031b`；研究提交：`bd68fd2`、`d26f8a6`、`32bf6c9`。
- GPU：NVIDIA GeForce RTX 3080，CC 8.6，68 SM，10 GB，UUID `GPU-7c5e95c0-e5a4-15d8-24a0-c8c8b58d6f39`。
- Driver 591.86；CUDA compiler/runtime 13.3/13.1；NCU 2026.2.1；NSYS 2026.1.3。
- 数据：int32 exact exclusive scan，`1<=E<=64`，`R=4096` 主 sweep，uniform、warm cache。
- API：caller stream、0 workspace、无 hot-path allocation/sync，支持分离与完全原地扫描。
- Release 决策数据不含 profiler；NCU/NSYS duration 仅用于机制诊断。

## 正确性

三种候选在研究分支通过：

- E=1..64 的 L1/L2 CPU oracle 对拍；
- E=1/31/32/33/63/64、全零/热点/随机/最大安全 int32 总和；
- 分离与 `counts==offsets`，8-byte 与仅 4-byte 对齐，redzone；
- null、E=0/65、非法 optimized id；
- CTest 8/8；Compute Sanitizer memcheck、initcheck、racecheck、synccheck，均为 `ERROR SUMMARY: 0 errors`。

最终树仍重新运行完整 correctness/sanitizer，研究通过不替代最终验证。

## Clean-main library 基线

E=64 L2，3 processes、30 samples/process、100 repeats/sample：

| Variant | p50 us | p95 us | mean us | CV | calls/s | items/s | effective GB/s | launches/workspace |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cuda_naive` | 8.586 | 9.239 | 8.577 | 0.051 | 116,465 | 7.45 M | 0.0598 | 1 / 0 B |
| CUB BlockScan | 8.768 | 10.822 | 9.138 | 0.112 | 114,051 | 7.30 M | 0.0497 | 1 / 0 B |
| CUB DeviceScan | 35.054 | 41.066 | 32.883 | 0.189 | 28,528 | 1.83 M | 0.0142 | scan+completion / 1023 B |

这些数字来自未修改的 clean main bundle。旧 registry 将所有通用 descriptor 标为 `strict_fp32`；研究提交 `d26f8a6` 将 Scan 的 naive/CUB/candidate metadata 统一为 `exact_int32`，实际整数语义没有变化。

## 候选结果

### C2 稳定性复测

E=29..64，5 independent processes、100 samples/process、1000 repeats/sample：

| E | naive p50 us | C2 p50 us | speedup |
|---:|---:|---:|---:|
| 29 | 8.387 | 8.480 | 0.989x |
| 32 | 8.371 | 8.585 | 0.975x |
| 33 | 8.877 | 8.560 | 1.037x |
| 48 | 9.262 | 8.625 | 1.074x |
| 64 | 8.829 | 8.536 | 1.034x |

汇总 ratio-of-sums `1.0503x`，geometric mean `1.0496x`，33/36 shapes 获益。最坏 p50 退化 2.56% 尚在 5% 门槛内，但所有 pair 的 CV 均超 10%；E=31 candidate p95 100.523 us，而 baseline p95 14.274 us，ratio `7.042x`，远超 1.03 上限。按预注册规则结论为 `variance-limited / reject`。

### C1 与 C3

- C1 为 E>32 执行两条连续 shuffle scan 链，registers/thread 为 18，无 spill；相较 C2 未形成稳定优势，拒绝。
- C2 registers/thread 为 14（cuobjdump resource report），无 local load/store；它验证了“减少 shuffle 链”有中心趋势价值，但没有通过 tail 稳定性。
- C3 registers/thread 为 12，SASS 确认 `LDG.E.64/STG.E.64`，无 spill；然而 E=29..64 相对 C2 ratio-of-sums 仅 `0.9936x`，只有 6/36 shapes 快至少 3%。事务宽度减少并未转化为 L2 latency 收益，拒绝。

## NSYS / NCU 机制证据

固定 L3 `chain_from_tokens(T=512,E=64,K=N=128)` 中，Scan 共 6 次调用、总 GPU kernel time 19.392 us，占 3.0%，平均 3.232 us。完整排序为 Grouped GEMM 71.7%、Top-K 10.4%、Dense 8.1%、Permute 3.1%、Scan 3.0%、Unpermute 2.4%、Histogram 1.3%。

| Kernel | Grid/block | waves/SM | achieved occupancy | registers/thread | local LD/ST |
|---|---|---:|---:|---:|---:|
| naive E64 | 1 / 1 | 0.000919 | 2.08% | 20 | not collected in basic；SASS resource 0 local |
| C1 E64 | 1 / 32 | 0.000919 | 2.18% | 18 | 0 / 0 |
| C2 E64 | 1 / 32 | 0.000919 | 2.22% | 14 resource / 16 NCU launch | 0 / 0 |
| C3 E64 | 1 / 32 | 0.000919 | 2.25% | 12 resource / 16 NCU launch | 0 / 0 |
| CUB BlockScan E64 | 1 / 128 | 0.001225 | 8.10% | 15 resource / 16 NCU launch | not collected in basic |

吞吐百分比在所有单 block Scan 上都很低，且只有一个极小 wave；这与 underfill/launch-bound 诊断一致。NCU replay duration 没有用于任何 speedup 计算。detailed 的 PC-sampling stall 计数为 0，短 kernel 样本不足，因此不把“0 stall”解释成不存在 stall。

## 晋升判定

| Gate | 要求 | C2 | 判定 |
|---|---:|---:|---|
| correctness/sanitizer | 全通过 | 全通过 | PASS |
| workspace/spill | 0 / 无 spill | 0 / 无 spill | PASS |
| ratio-of-sums | ≥1.03x | 1.0503x | PASS |
| shape coverage | ≥80% | 91.7% | PASS |
| max p50 regression | ≤5% | 2.56% | PASS |
| all-samples CV | ≤10% | 0/36 pairs 合格 | FAIL |
| p95 regression | ≤3% | 最坏 +604.2% | FAIL |

任一硬门槛失败即拒绝，因此没有阈值或 Hybrid dispatch 可以合法生成。

## 证据与复现

- [候选 raw/aggregate/comparison、NCU/NSYS CSV/JSON、sanitizer logs](artifacts/scan-sm86-research-32bf6c9/)
- [clean-main CUB/naive baseline](artifacts/scan-library-main-927c585/)
- 二进制 `.ncu-rep/.nsys-rep` 不提交；文件名、大小、SHA256 和工具命令记录在 manifests/SHA256SUMS 中。
- 最终可执行配置：`configs/operators/scan/benchmark/library_smoke.json`、`configs/operators/scan/benchmark/library_release.json`、`configs/operators/scan/profile/library.json` 和 `.codex/campaigns/scan.json`。

## 后续

由于 Scan 在固定链中仍占 3.0%，可另起 Histogram→Scan 融合研究。但新 PR 必须以减少 launch/metadata round trip 为单一机制，重新做 L2/L3 correctness、NSYS 和未插桩 A/B；不能把本轮被拒绝的 standalone C2 结果当作融合晋升证据。
