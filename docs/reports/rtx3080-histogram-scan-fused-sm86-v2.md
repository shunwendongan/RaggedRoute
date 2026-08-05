# RTX 3080 / SM86 Histogram→Scan F2 提交报告

## 结论

本提交保留单一最强实现：`cuda_fused_histogram_scan`（F2）。它在
`src/scan/cuda_candidate/optimized.cu` 中使用单 CTA shared histogram，随后由
16-lane、每 lane 4 个 int32 的 shuffle scan 直接生成独立的 `counts[E]` 和
`offsets[E+1]`。`R<=4096` 走 fused kernel，较大 `R` 回退现有 Histogram +
exclusive Scan 两阶段路径；workspace 始终为 0。

Standalone C2/S1/S2 和融合 F1 均从最终源码、registry 和 benchmark 配置删除。
它们的负结果仍保留在历史研究报告和本轮 raw archive 中。

本 PR 是按用户要求提交的最强版本候选；由于本机 Windows GPU 在正式测量期间
同时运行 TopK campaign 与 Intel Graphics Overlay，正式结果 CV 超出门禁，不能
宣称已经满足生产晋级门槛。PR 中的 dispatch 体现 F2 选择，发布前应在独占 GPU
窗口重新跑 `fused_v2_promoted.json` 和完整 L3 campaign。

## 语义与实现合同

- exact int32；`1<=E<=64`；`route_pairs>=0`。
- caller stream；无热路径分配、同步或额外 workspace。
- `expert_ids`、`counts`、`offsets` 必须 4-byte 对齐且彼此不重叠；`R=0` 可传空
  `expert_ids`。
- standalone `exclusive_scan` 保持默认 `cuda_naive` 和 `counts==offsets` 原地语义。
- F2 的 shared histogram 对非法 expert id 忽略；公开 wrapper 仍保持现有输入合同。

## 代码变更

| 位置 | 变更 |
|---|---|
| `src/scan/cuda_candidate/optimized.cu` | 仅保留 F2 single-CTA/subwarp kernel |
| `src/scan/cuda_candidate/optimized_internal.h` | 仅保留 fused implementation id 2 与 launcher |
| `src/scan/operator.cpp` | 合并公共 Histogram→Scan wrapper；自动 dispatch 选择 F2 |
| `include/raggedroute/operators.h` | `HistogramExclusiveScanArgs`、workspace query、wrapper |
| `include/raggedroute/types.h` | 追加 `kHistogramExclusiveScan` |
| `benchmarks/` | F2 组合 adapter、CUB 对照、L3 chain adapter |
| `configs/operators/scan/benchmark/fused_v2_promoted.json` | 5 进程、100 samples、100 repeats 正式合同 |

## 已运行验证

- Release 编译：CUDA 13.3.73，`-O3/-lineinfo/-arch=sm_86`，通过。
- CTest：8/8 通过。
- Python：39 passed，7 skipped，16 subtests passed。
- Compute Sanitizer（F2-only final source）：memcheck、initcheck、racecheck、synccheck
  均为 `ERROR SUMMARY: 0 errors`；日志保存在本地
  `out/sanitizer/scan-fused-sm86-v2-final/`。

## 性能证据

以下来自此前 5-process 文件，但受到另一个 GPU campaign 和 Overlay 干扰；只作
机制与候选排序记录，不作 release speedup：

| 对照 | F2 结果 | 门禁含义 |
|---|---:|---|
| L2 fused region（11 shapes） | ratio-of-sums `2.0045x`，11/11 更快，最小 `1.696x` | 中心趋势强，但 CV `45%–114%`，失败 |
| L2 fallback（`R=4097/8192`） | ratio-of-sums `0.9872x`，最坏 p95 ratio `1.162x` | fallback 回退，失败 |
| L3 chain aggregate | ratio-of-sums `0.9398x`，获益覆盖 `66.7%`，最坏 speedup `0.497x` | 非回退，失败 |
| NSYS 诊断 | separate Histogram+Scan 约 `4.651 us`；F2 kernel 约 `2.213 us` | 仅解释 launch/融合机制，不作生产结论 |

NCU basic：F2 约 22 registers/thread、achieved occupancy 14.87%、waves/SM
0.002451；短 kernel 的 PC-sampling stall 指标缺失时记为 `not_collected`。F2
仍属于 underfill/launch-bound metadata work，不适用 Tensor Core、TMA、WGMMA、
TMEM 或其他 Hopper/Blackwell 专属机制。

## 文献到实现映射

- [CUB DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html)：
  decoupled look-back 是大数组基线；本项目 `E<=64` 不照搬跨 CTA 协议。
- [Merrill & Garland, Decoupled Look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)：
  支持减少全局 pass 的分析框架。
- [GPU Multisplit](https://arxiv.org/abs/1701.01189) 与
  [Onesweep](https://arxiv.org/abs/2206.01784)：只吸收 histogram/prefix/placement
  融合、减少中间物化的思想。
- [Decoupled Fallback (SPAA 2025)](https://escholarship.org/uc/item/0bk9z4bt)：
  记录为大数组对照资料，不在单 CTA `E<=64` 路径实现。
- [MegaBlocks](https://arxiv.org/abs/2211.15841)、
  [ScatterMoE](https://arxiv.org/abs/2403.08245)、
  [Tutel](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html)：
  只吸收减少 routing 中间物化和 launch 边界的系统思想。

实际库基线来自 CUDA 13.3 bundled CCCL/CUB headers：`CUB_VERSION 300304`
（3.3.4）；CUB 变体保持 benchmark-only。

## 复现与证据位置

- 正式合同：[fused_v2_promoted.json](../../configs/operators/scan/benchmark/fused_v2_promoted.json)
- compact bundle：[scan-fused-sm86-v2](artifacts/scan-fused-sm86-v2/)
- 完整 raw JSONL、NSYS/NCU reports、SQLite 与 sanitizer 日志：本地
  `out/benchmark/`、`out/profile/`、`out/sanitizer/`，仅由 manifest/hash 引用。
