# RTX 3080 Token Permute SM86 candidate report

## 1. 结论

本轮五个候选均通过 correctness。初始低重复 Release 无法稳定选优；后续增加 L2-only 高重复 selection suite 后，token-owned Top-2 在两轮中均取得最高 ratio-of-sums（1.0456x/1.0463x），因此 `cuda_candidate` 已绑定 implementation ID 4。默认 `KernelFamily::kAuto` 仍继续 dispatch 到 `cuda_naive`，explicit candidate 选择与默认 runtime promotion 分开。

原始低重复结果和负面实验仍作为历史证据提交；最终 alias 选择依据新增的 100 repeats warm-shape suite，而不是 profiler duration。

## 2. 环境与合同

- Initial candidates SHA：`5cb9bd49afba09edf8438416312257e9827cf4da`；selection suite SHA：`cc559173155e2f584303bef54f43d61f685df516`；selected alias/library SHA：`af4947fcd665a899e72714122a0765340903d5c1`。全部为 clean Release build。
- GPU：NVIDIA GeForce RTX 3080，SM86，68 SM，GPU UUID `GPU-7c5e95c0-e5a4-15d8-24a0-c8c8b58d6f39`。
- CUDA compiler/runtime 13.3，NCU 2026.2.1，NSYS 2026.1.3，driver 591.86。
- strict FP32；caller stream；zero hot-path allocation；workspace 恒为 `4*E` bytes；数学、mapping 和输入边界配对一致。
- seed `20260729`；5 processes。初始套件为 20 warmups/30 samples/L2 repeats=10；最终 selection 为 50 warmups/50 samples/warm L2 repeats=100；最终 library 对照为 repeats=50。
- 当前为非独占桌面 GPU；采样时存在非 CUDA 的桌面图形上下文。没有擅自关闭用户应用，稳定性限制按原样计入 gate。

## 3. Correctness capability

- CTest：8/8 passed。
- Compute Sanitizer：memcheck、initcheck、racecheck、synccheck 全部 passed。
- 初始两轮 candidate Release 共 1920 records，初始 library/chain Release 140 records；新增 selection 两轮 960 records、selected library 75 records。全部 `validation.ok=true`，每个 suite 内均为单一 GPU UUID、单一 clean SHA。
- 覆盖 scalar/vector path、Top-2 与 generic Top-K fallback、duplicate expert route、optional inverse mapping、unaligned payload、redzone、caller stream 和 overflow/invalid dispatch。
- 五个 explicit implementation ID 可用；未知 ID 拒绝。`cuda_candidate`/`cuda_candidate_from_ids` 的记录均确认 `implementation_id=4`；Auto dispatch 未改变。

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

## 5. Selected candidate 与 library baseline

最终 full-from-ids 套件使用 50 warmups、50 samples、每 sample 50 calls；以下为 250 raw samples 的 median-of-process-medians。candidate 和 naive workspace 为 772 B（tail 100 B）；vLLM 为 16,928–66,080 B。

| Full-from-ids case | vLLM p50/CV | cuda naive p50/CV | selected candidate p50/CV | candidate vs naive | candidate vs vLLM |
|---|---:|---:|---:|---:|---:|
| tail | 26.061 us / 0.079 | 35.348 / 0.065 | 34.826 / 0.027 | 1.015x | 0.748x |
| anchor uniform | 30.812 / 0.028 | 41.277 / 0.067 | 40.274 / 0.055 | 1.025x | 0.765x |
| anchor Zipf-1.4 | 30.853 / 0.035 | 41.636 / 0.066 | 41.011 / 0.055 | 1.015x | 0.752x |
| large uniform | 33.403 / 0.139 | 42.988 / 0.047 | 43.141 / 0.045 | 0.996x | 0.774x |
| large single-hot | 32.625 / 0.108 | 47.923 / 0.049 | 46.182 / 0.045 | 1.038x | 0.706x |

Selected candidate 相对仓库 `cuda_naive_from_ids`：4/5 case 加速、ratio-of-sums 1.0182x；相对 vLLM full-from-ids：0/5 case 加速、ratio-of-sums 0.7484x，即 candidate aggregate latency 约高 33.6%。后者是明确保留的 library capability gap，不用 prepared-mapping 数字掩盖。

Prepared-mapping 仍只作为 copy 上界诊断；它与 full-from-ids 的 `excluded_steps` 和 workspace 合同不同，不参与上述 ratio。

## 6. NSYS

NSYS 使用 CUDA/NVTX、`--sample=none --cpuctxsw=none`。Profiler duration 仅诊断：

- prepared selected candidate 的 token-owned kernel median 为 2.736 us（6 instances）。
- full candidate-from-ids 的 histogram/scan/token-owned medians 为 1.456/3.072/2.736 us，共 3 个 kernel。
- vLLM full-from-ids 的 radix-sort/offset/expand medians为 7.328/2.848/3.248 us；initialize_source_rows 只出现一次。
- Block-partial 负面候选仍为 placement+copy 两个 kernel；selected candidate 没有增加该额外 launch。

这些 trace 用于核对 dispatch 和 pipeline 组成；Release latency 仍以上一节的未采 profiler 数据为准。

## 7. NCU

NCU 使用 `--clock-control none`。五候选 initial basic 仍保留；selected follow-up 对 anchor 和 wide/hot 的 naive/token-owned 做 detailed。Duration 不作为 Release latency。

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

| Case | Grid / waves | Physical DRAM read/write | L2 hit | Local load/store |
|---|---:|---:|---:|---:|
| anchor naive | 1024 / 1.255 | 533,376 / 209,792 B | 71.3% | 0 / 0 |
| anchor token-owned | 512 / 0.627 | 535,296 / 177,536 B | 69.7% | 0 / 0 |
| wide/hot naive | 4096 / 5.020 | 8,409,472 / 15,910,144 B | 74.1% | 0 / 0 |
| wide/hot token-owned | 2048 / 2.510 | 8,411,520 / 15,887,744 B | 67.2% | 0 / 0 |

Token-owned 将 Top-2 grid 减半，registers/thread 为 34（naive 26），没有 spill。Anchor underfill 下 NCU diagnostic duration 8.864 us，和 naive 8.768 us 基本相同；wide/hot 为 38.592 us，低于 naive 40.320 us，且 DRAM 约 85.2%。这解释了 selection 中 anchor 持平、large/wide 获益的形状分布。

## 8. Reproduction

```powershell
ctest --preset test-rtx3080-sm86-release --output-on-failure
python scripts\run_sanitizers.py --build-dir out\build\rtx3080-sm86-release --output-dir out\sanitizer\rtx3080-permute-candidate
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\permute\benchmark\candidate_release.json --output out\benchmark\permute-candidate-release.jsonl
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\permute\benchmark\candidate_selection_release.json --output out\benchmark\permute-candidate-selection-release.jsonl
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\permute\benchmark\library_release.json --output out\benchmark\permute-library-release.jsonl
python scripts\run_benchmarks.py --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\permute\benchmark\selected_library_release.json --output out\benchmark\permute-selected-library-release.jsonl
python scripts\profile_benchmarks.py compute --binary out\build\rtx3080-sm86-release\raggedroute_benchmark.exe --config configs\operators\permute\profile\candidate.json --run-dir out\profile\permute-sm86\ncu
python scripts\profile_benchmarks.py analyze --run-dir out\profile\permute-sm86\ncu
```

Raw `.ncu-rep`、`.nsys-rep`、`.sqlite` 只保存在本地忽略目录；versioned evidence bundle 只提交 raw benchmark/text/JSON/CSV、sanitizer logs、profiler normalized outputs、raw report inventory hashes 和 `SHA256SUMS`。

Versioned evidence：[artifact bundle](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/)；[bundle manifest](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/manifest.json)；[SHA256SUMS](artifacts/20260802T185236Z-5cb9bd4-permute-sm86-candidates-v1/SHA256SUMS)。

Selected token-owned follow-up evidence：[artifact bundle](artifacts/20260803T071455Z-af4947f-permute-selected-token-owned-v1/)；[manifest](artifacts/20260803T071455Z-af4947f-permute-selected-token-owned-v1/manifest.json)；[SHA256SUMS](artifacts/20260803T071455Z-af4947f-permute-selected-token-owned-v1/SHA256SUMS)。
