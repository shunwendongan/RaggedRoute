# RTX 3080 FP32 naive baseline（e37c132）

> 结论：正式 release 协议与三进程聚合链路执行成功，但这份结果只建立可复现的 naive baseline，不支持任何 speedup 或默认 variant 晋升结论。26 个聚合组中有 18 个 `all_samples_cv > 0.10`，当前 WDDM/桌面后台环境的抖动明显；后续优化对比应在更安静、锁频且可控的测量环境重跑。

## 1. 证据边界

- tested commit：`e37c132ae2879c4ce6a74be9823da980a1c9f717`
- run id：`20260729T134736Z-e37c132ae287-rtx3080_baseline_release_v1`
- suite：`configs/benchmark_rtx3080_release.json`
- GPU：NVIDIA GeForce RTX 3080，10 GiB，SM 8.6，68 SM
- driver：591.86；CUDA compiler：13.3.73；记录中的 CUDA runtime/driver API version 字段均为 `13010`
- build：Release、`sm_86`、clean Git；二进制内嵌 SHA 为 `e37c132ae287`
- protocol：3 个独立进程；每个 case 固定 seed `20260729`；warmup 20；每进程 30 个 raw samples；case 顺序按进程确定性打乱
- correctness gate：CTest 2/2；Compute Sanitizer memcheck/racecheck/synccheck 均为 0 error/hazard
- result integrity：78 条 JSONL，26 个聚合组，所有 post-measurement reference validation 通过；三个进程使用同一 GPU UUID、case config、variant config、build SHA 和 build type

原始 JSONL、run manifest、aggregate JSON/CSV 位于本机 `reports/runs/`，按仓库策略不提交生成物。它们可由本文末尾命令从 tested commit 重新生成。

## 2. 聚合结果

主值是三个进程各自 p50 的 median；`raw p95` 与 `CV` 使用三进程全部 90 个 raw batch-mean samples。单位均为微秒。`kernel_repeats > 1` 时，一个 raw sample 是 event batch elapsed 除以 repeats，不是单调用 tail latency。

| Operator / case | Level | Cache | Process-median p50 (us) | Raw p95 (us) | CV |
|---|---|---:|---:|---:|---:|
| Dense GEMM `m256_n256_k256` | L1 | warm | 27.750 | 30.572 | 0.041 |
| Dense GEMM `m256_n256_k256` | L2 | warm | 27.750 | 27.909 | 0.005 |
| Top-K Gate `t2048_e64` | L1 | warm | 14.868 | 20.011 | 0.230 |
| Top-K Gate `t2048_e64` | L2 | warm | 14.838 | 16.316 | 0.166 |
| Histogram uniform `t2048_e64` | L1 | warm | 9.155 | 12.012 | 0.206 |
| Histogram uniform `t2048_e64` | L2 | warm | 17.592 | 21.911 | 0.220 |
| Histogram Zipf-1.4 `t2048_e64` | L1 | warm | 8.837 | 12.061 | 0.347 |
| Histogram Zipf-1.4 `t2048_e64` | L2 | warm | 16.476 | 18.694 | 0.200 |
| Exclusive Scan `e64` | L1 | warm | 8.207 | 10.869 | 0.196 |
| Exclusive Scan `e64` | L2 | warm | 8.402 | 11.527 | 0.159 |
| Token Permute uniform `t512_k256` | L1 | warm | 10.240 | 70.656 | 1.017 |
| Token Permute uniform `t512_k256` | L2 | warm | 20.685 | 23.378 | 0.464 |
| Token Permute Zipf-1.4 `t512_k256` | L1 | warm | 10.240 | 69.632 | 0.998 |
| Token Permute Zipf-1.4 `t512_k256` | L2 | warm | 19.149 | 21.606 | 0.185 |
| Token Permute Zipf-1.4 `t512_k256` | L1 | cold scrub | 12.288 | 13.312 | 0.073 |
| Token Permute Zipf-1.4 `t512_k256` | L2 | cold scrub | 15.360 | 16.384 | 0.198 |
| Grouped GEMM uniform `t512_k128_n128` | L1 | warm | 42.803 | 43.735 | 0.013 |
| Grouped GEMM uniform `t512_k128_n128` | L2 | warm | 42.394 | 46.418 | 0.054 |
| Grouped GEMM Zipf-1.4 `t512_k128_n128` | L1 | warm | 53.043 | 53.453 | 0.048 |
| Grouped GEMM Zipf-1.4 `t512_k128_n128` | L2 | warm | 52.838 | 53.975 | 0.013 |
| Unpermute uniform `t1024_n256` | L1 | warm | 9.574 | 12.959 | 0.469 |
| Unpermute uniform `t1024_n256` | L2 | warm | 9.574 | 13.471 | 0.427 |
| Unpermute uniform `t1024_n256` | L1 | cold scrub | 13.312 | 15.360 | 0.350 |
| Unpermute uniform `t1024_n256` | L2 | cold scrub | 13.312 | 14.336 | 0.054 |
| `chain_from_tokens` full 7-op `t64_e16_k64_n64` | L3 | warm | 91.648 | 109.107 | 0.326 |
| `chain_from_logits` 6-op Zipf-1.4 `t128_e16_k64_n64` | L3 | warm | 45.056 | 104.448 | 0.417 |

## 3. 可解释结论与限制

1. Histogram 与 Token Permute 的 L2 包含各自必要的 device reset，因此不能用 L1 数字替代生产 operator 成本。其他无动态 reset 的 naive adapter，L1/L2 接近是预期行为。
2. Grouped GEMM 的 Zipf case 比相同总 route 数的 uniform case 更慢，符合当前 `grid-z-per-expert` naive scheduler 对负载不均衡敏感的预期；这是后续 scheduler 优化假设，不是 speedup 证据。
3. 两条 L3 suite 的入口与 shape 不同：`chain_from_tokens` 包含 7 个算子，`chain_from_logits` 只有后 6 个算子。两者不得直接相除或解释为 Dense GEMM 的边际成本。
4. 18/26 组超过 promotion policy 的 `CV <= 0.10` 门槛，最高 CV 为 1.017。运行期间存在 WDDM 桌面与多个后台图形进程，GPU 起始/结束状态也从 P8 切到 P0；因此本文保留原始波动并拒绝性能晋升结论。
5. 当前 variant 是 FP32 scalar `cuda_naive`。没有 cuBLAS/CUTLASS/CUB production baseline、FP16/Tensor Core candidate 或同语义候选数据，不能声称算子已经优化。

## 4. 复现命令

```powershell
cmd /c scripts\configure_windows.bat rtx3080-sm86-release
cmd /c scripts\build_windows.bat rtx3080-sm86-release
ctest --preset test-rtx3080-sm86-release

python scripts\run_benchmarks.py `
  --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe `
  --config configs\benchmark_rtx3080_release.json `
  --output reports\runs\rtx3080-release.jsonl

python scripts\aggregate_results.py reports\runs\rtx3080-release.jsonl `
  --json reports\runs\rtx3080-release.aggregate.json `
  --csv reports\runs\rtx3080-release.aggregate.csv
```

正式 candidate/baseline 对比必须重新在同一 tested commit lineage、同一 GPU UUID、同一 case/seed/math semantics 下运行，并由 `configs/benchmark_promotion_policy.json` 做晋升判定。
