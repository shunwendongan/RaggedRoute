# RTX 3080 Token Permute SM86 candidate report

## 1. 结论

本轮五个候选均通过 correctness。初始低重复 Release 无法稳定选优；后续增加 L2-only 高重复 selection suite 后，token-owned Top-2 在两轮中均取得最高 ratio-of-sums（1.0456x/1.0463x），因此 `cuda_candidate` 已绑定 implementation ID 4。默认 `KernelFamily::kAuto` 仍继续 dispatch 到 `cuda_naive`，explicit candidate 选择与默认 runtime promotion 分开。

原始低重复结果和负面实验仍作为历史证据提交；最终 alias 选择依据新增的 100 repeats warm-shape suite，而不是 profiler duration。

## 2. 环境与合同

- Source SHA：`5cb9bd49afba09edf8438416312257e9827cf4da`，clean Release build。
- GPU：NVIDIA GeForce RTX 3080，SM86，68 SM，GPU UUID `GPU-7c5e95c0-e5a4-15d8-24a0-c8c8b58d6f39`。
- CUDA compiler/runtime 13.3，NCU 2026.2.1，NSYS 2026.1.3，driver 591.86。
- strict FP32；caller stream；zero hot-path allocation；workspace 恒为 `4*E` bytes；数学、mapping 和输入边界配对一致。
- seed `20260729`；5 processes；20 warmups；30 samples；L1 repeats=1、L2 repeats=10、cold repeats=1。
- 当前为非独占桌面 GPU；采样时存在非 CUDA 的桌面图形上下文。没有擅自关闭用户应用，稳定性限制按原样计入 gate。

## 3. Correctness capability

- CTest：8/8 passed。
- Compute Sanitizer：memcheck、initcheck、racecheck、synccheck 全部 passed。
- 两轮 candidate Release 共 1920 records，library/chain Release 140 records；全部 `validation.ok=true`，均为单一 GPU UUID、单一 clean SHA。
- 覆盖 scalar/vector path、Top-2 与 generic Top-K fallback、duplicate expert route、optional inverse mapping、unaligned payload、redzone、caller stream 和 overflow/invalid dispatch。
- 五个 explicit implementation ID 可用；未知 ID 拒绝。Auto dispatch 未改变。

## 4. Initial low-repeat matrix（历史）

下表只使用未采 profiler 的 L2 paired medians。`faster` 是 16 个固定 shape 中 candidate p50 小于 naive 的数量；`worst` 是最差单 shape speedup。所有候选 workspace growth 均为 0%。

| Candidate | Run 1 faster | Run 1 ratio-of-sums | Run 1 worst | Run 2 faster | Run 2 ratio-of-sums | Run 2 worst | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| atomic vectorized 128 | 11/16 (68.8%) | 1.0145x | 0.6053x | 9/16 (56.2%) | 0.9858x | 0.1902x | reject |
| atomic vectorized 64 | 11/16 (68.8%) | 0.9675x | 0.5610x | 10/16 (62.5%) | 0.9314x | 0.1746x | reject |
| atomic vectorized 256 | 12/16 (75.0%) | 1.0373x | 0.6300x | 8/16 (50.0%) | 0.8171x | 0.1697x | reject |
| token-owned Top-2 | 10/16 (62.5%) | 1.0317x | 0.5631x | 10/16 (62.5%) | 1.2806x | 0.5674x | inconclusive |
| block partial | 0/16 (0.0%) | 0.6917x | 0.3333x | 3/16 (18.8%) | 0.6719x | 0.1431x | reject |

Run 1 的 16/16 naive L2 groups 和每个候选的 15–16/16 groups 超过 CV 0.10；按协议在 GPU 无 CUDA compute workload 时完整重跑。Run 2 仍为所有 baseline/candidate L2 groups 超限，因此标记 unstable，不从中挑选“赢家”。

### 4.1 High-repeat candidate selection

为完成 explicit `cuda_candidate` 选优，新增 `benchmark_permute_candidate_selection_release.json`：只测同合同 L2，5 个独立进程、50 warmups、50 samples，warm shape 每 sample 100 repeats；cold 保持单次。两轮排名一致：

| Candidate | Selection 1 faster | Ratio-of-sums | Worst shape | Selection 2 faster | Ratio-of-sums | Worst shape | Alias decision |
|---|---:|---:|---:|---:|---:|---:|---|
| token-owned Top-2 | 13/16 | 1.0456x | 0.9712x | 14/16 | 1.0463x | 0.9418x | selected |
| atomic vectorized 128 | 14/16 | 1.0394x | 0.9566x | 10/16 | 1.0327x | 0.9789x | reject |
| atomic vectorized 64 | 12/16 | 1.0372x | 0.9671x | 10/16 | 1.0302x | 0.9469x | reject |
| atomic vectorized 256 | 5/16 | 0.9911x | 0.9387x | 5/16 | 0.9881x | 0.9250x | reject |
| block partial | 0/16 | 0.7292x | 0.6178x | 0/16 | 0.7265x | 0.6159x | reject |

Token-owned Top-2 的主要收益来自 `T=2048,K=256`：selection 1 的 uniform/single-hot 分别为 1.2173x/1.1682x；wide/hot 为 1.0283x，anchor 基本持平。第二轮 aggregate 排名和约 4.6% 总体收益复现，但桌面环境重新出现多组 CV 超限；因此这里完成的是 explicit `cuda_candidate` alias 选择，默认 Auto promotion 仍保持独立。

代表 shape 的 Run 1 median 可说明能力范围，但不能作为 promotion 声明：

| Shape / variant | L2 p50 (us) | tokens/s | route rows/s | logical GB/s | all-sample CV |
|---|---:|---:|---:|---:|---:|
| anchor Zipf-1.4 naive | 17.664 | 28,985,507 | 57,971,013 | 119.43 | 0.221 |
| anchor Zipf-1.4 atomic-256 | 17.101 | 29,940,119 | 59,880,238 | 123.37 | 0.588 |
| anchor Zipf-1.4 token-owned | 17.664 | 28,985,508 | 57,971,016 | 119.43 | 0.412 |
| wide/hot naive | 47.514 | 43,103,449 | 86,206,897 | 707.25 | 0.571 |
| wide/hot atomic-128 | 46.182 | 44,345,898 | 88,691,796 | 727.63 | 0.473 |
| wide/hot atomic-256 | 45.773 | 44,742,729 | 89,485,459 | 734.14 | 0.511 |
| wide/hot token-owned | 45.670 | 44,843,050 | 89,686,100 | 735.79 | 0.976 |

## 5. Library baseline 与 L3 capability

以下均为同一 clean build 的 median-of-process-medians。所有 full-from-ids/L3 组也存在 CV 超限，只用于定位能力。

| Boundary / case | vLLM or naive (us) | cuda naive (us) | cuda candidate (us) | Candidate workspace |
|---|---:|---:|---:|---:|
| full-from-ids tail | vLLM 28.160 | 120.934 | 38.093 | 100 B |
| full-from-ids anchor uniform | vLLM 70.144 | 105.574 | 37.683 | 772 B |
| full-from-ids anchor Zipf-1.4 | vLLM 31.232 | 106.598 | 40.141 | 772 B |
| full-from-ids large uniform | vLLM 37.581 | 45.056 | 44.851 | 772 B |
| full-from-ids large single-hot | vLLM 72.602 | 54.579 | 142.438 | 772 B |
| L3 chain from tokens | naive 87.040 | n/a | permute-candidate 76.288 | 64 B |
| L3 chain from logits | naive 71.168 | n/a | permute-candidate 40.960 | 64 B |

Prepared-mapping 是 copy 上界诊断，不是 full-from-ids：tail/anchor/wide-hot 的 vLLM `expand_rows` p50 分别为 8.192/10.240/57.344 us，in-tree candidate 分别为 9.216/9.216/45.056 us。两者 `excluded_steps` 和 workspace 合同不同，严格比较器拒绝生成 promotion speedup；本报告保留原值而不伪造配对结论。

## 6. NSYS

NSYS 使用 CUDA/NVTX、`--sample=none --cpuctxsw=none`。Profiler duration 仅诊断：

- anchor L2 的 kernel median：naive 3.168 us，atomic-128 2.736 us；各 6 instances。
- chain-from-tokens 中 permute median：naive 1.728 us（kernel time 9.8%），candidate 1.696 us（9.6%）。
- chain-from-logits 中 permute median：naive 1.744 us（11.7%），candidate 1.712 us（11.5%）。
- 两条 chain 的 kernel 数不变；candidate 没有引入额外 chain launch。Block-partial 独立候选仍为 placement+copy 两个 kernel。

这些 trace 说明 atomic-128 的 launch 结构正确，但微小的 profiler 内核差异远小于 unprofiled Release 抖动，不能覆盖 promotion 失败。

## 7. NCU

NCU 使用 `--clock-control none`，先 basic，再只对 anchor 和 wide/hot 的 naive/atomic-128 做 detailed。Duration 不作为 Release latency。

| Kernel / case | Waves/SM | Registers | Achieved occ. | SM % | DRAM % |
|---|---:|---:|---:|---:|---:|
| naive anchor basic | 1.255 | 26 | 71.1% | 12.7% | 31.7% |
| atomic-64 anchor basic | 0.941 | 38 | 40.5% | 4.8% | 18.3% |
| atomic-128 anchor basic | 1.255 | 38 | 54.7% | 8.1% | 22.9% |
| atomic-256 anchor basic | 2.510 | 38 | 68.9% | 12.3% | 20.5% |
| token-owned anchor basic | 0.627 | 34 | 45.3% | 3.8% | 19.6% |
| block placement anchor basic | 0.010 | 16 | 16.5% | 0.1% | 0.1% |
| block copy anchor basic | 1.255 | 34 | 42.1% | 7.3% | 18.5% |

Detailed physical traffic：

| Case | Physical DRAM read/write | Logical bytes | L2 hit | Local load/store |
|---|---:|---:|---:|---:|
| anchor naive | 533,376 / 124,800 B | 2,109,440 B | 71.3% | 0 / 0 |
| anchor atomic-128 | 534,656 / 36,224 B | 2,109,440 B | 71.4% | 0 / 0 |
| wide/hot naive | 8,409,600 / 14,274,688 B | 33,603,584 B | 73.2% | 0 / 0 |
| wide/hot atomic-128 | 8,413,056 / 16,666,368 B | 33,603,584 B | 73.4% | 0 / 0 |

当前 NCU metric alias 对 global sector 总数报告 `not_collected`，但 global request、DRAM bytes、hit rate、stall 和 local memory 状态均保留在 `ncu_metrics.json/csv`。Vector path 把 registers/thread 从 26 提高到 38；anchor occupancy 降低，wide/hot 已接近高 occupancy 且 DRAM 约 86%，与“只靠 float4 不会普遍获胜”的 Release 结果一致。

## 8. Reproduction

```powershell
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts\run_sanitizers.py --build-dir out\build\rtx3080-sm86-release --output-dir out\sanitizer\rtx3080-permute-candidate
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\benchmark_permute_candidate_release.json --output out\benchmark\permute-candidate-release.jsonl
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\benchmark_permute_library_release.json --output out\benchmark\permute-library-release.jsonl
python scripts\profile_benchmarks.py compute --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\profile_permute_candidate.json --run-dir out\profile\permute-sm86\ncu
python scripts\profile_benchmarks.py analyze --run-dir out\profile\permute-sm86\ncu
```

Raw `.ncu-rep`、`.nsys-rep`、`.sqlite` 只保存在本地忽略目录；versioned evidence bundle 只提交 raw benchmark/text/JSON/CSV、sanitizer logs、profiler normalized outputs、raw report inventory hashes 和 `SHA256SUMS`。

Versioned evidence：[artifact bundle](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/)；[bundle manifest](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/manifest.json)；[SHA256SUMS](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/SHA256SUMS)。
