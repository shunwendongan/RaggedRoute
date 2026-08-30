# RaggedRoute 套餐完成审计

审计日期：2026-08-30（Asia/Shanghai）

仓库快照：PR #39 合并后的 `main@2366acec5d2ab6552e6112e5019ac948a51fbf34`

统一七算子证据：`9732a0343c60f869fc4166a0cc3cabba2fd67bbb`

Grouped follow-up：`c2205ed1ba1063fccce3cd417fd671798dbfb66f`、`dea7c066a83a5df700aa60c03fd51446c6b4c5e5`

## 结论

原套餐的第一阶段目标已经形成可审计闭环：远端分支治理、RTX 3080 / SM86 统一 Release 复测、uniform/Zipf NSYS、candidate/strong-reference NCU、Grouped V9 follow-up、七算子中文总表、英文事实同步、四份面试文档和七份 `performance-record.md` 回链均已完成。公共 API、`KernelFamily::kAuto` 和算子实现没有在文档治理阶段被改动。

当前最适合写入简历的主 kernel 是 Grouped GEMM V9 `cuda_grouped_sm86_fp32_v9_balanced_32x64`；最适合写成完整 operator 收益的是 Token Permute v2 full-from-ids。Dense、Scan 和 Unpermute 的价值主要是展示公平对比、瓶颈判断和拒绝错误优化，不能包装成全面超过库。

第二阶段冗余删除、真实 captured/production trace、held-out dispatch 和生产级 Auto 晋级仍明确保留为后续工作，不属于本次“已完成”。

## 套餐逐项验收

| 套餐项 | 状态 | 可核验结果 | 边界 |
|---|---|---|---|
| 远端分支治理 | 完成 | live 远端只剩 `main` 与 review-held `codex/grouped-gemm-sm86-v2`；4 个 evidence tag 保留 | 本地分支、5 个 worktree、1 个 stash 均未删除 |
| 合并后自动清理 | 完成 | GitHub `delete_branch_on_merge=true`；PR #38/#39 已合并且远端任务分支已删除 | 私有仓库套餐不支持 branch protection，继续使用 CI/PR 门禁 |
| 七算子统一 Release | 完成 | 3725/3725 records、745 aggregate groups、5 个独立进程、20 warmup、30 samples、固定 seed，validation 全过 | strict FP32、RTX 3080 / SM86、Windows/WDDM |
| CV 口径 | 完成 | 现行 `default_promotion` 与 portfolio policy 的 evidence ceiling 均为 `0.50` | `CV>0.10` 只披露风险；`CV>0.50` 才因 CV 自动降级，历史 decision 不回写 |
| NSYS 关键路径 | 完成 | uniform/Zipf 各一条 L3 trace；均抽取 5 个阶段热点，kernel time 占比各自合计 100% | profiler duration 不进入正式 speedup |
| NCU basic | 完成 | 七个语义算子加融合 primitive，candidate/strong-reference 各 8 个 basic，共 16 个 | profile variant 是诊断选择，不保证等于最终 Release winner |
| NCU detailed | 完成 | 统一证据对 Grouped v3/CUTLASS 各 1 个 detailed；V9 follow-up 对 V5/V9 各 1 个 detailed | 只升级实际热点；缺失概念保留状态而非填 0 |
| Grouped V9 复测 | 完成 | 15 shape、5 variant、5 process、375/375 oracle 通过；24/24 targeted Sanitizer 通过 | 一个 CUTLASS process `CV=0.5041`，完整矩阵只作 research trend |
| 中文 README / 面试文档 | 完成 | 七算子总表、设计机制、外部基线边界、反例、简历 bullet、STAR、题库均已建立 | 不能把 local winner 写成无条件 Auto 或生产 SLA |
| 七份算子记录回链 | 完成 | Dense、Top-K、Histogram、Scan、Permute、Grouped、Unpermute 全部回链统一 bundle；Grouped 另回链 V9 | 旧实验保留其当时 policy，不事后重写结果 |
| 第二阶段冗余清理 | 等待 review | 398 个历史 artifact 文件、6 个原始 Nsight 二进制、重复指标、超大 aggregate、旧长文已分类 | 未经用户逐项批准不删除 |

## Profiler coverage 与证据演进

统一 bundle 的 `ncu_metrics.json` 包含 18 条记录：candidate/basic 8 条、strong-reference/basic 8 条，以及 Grouped v3/CUTLASS detailed 各 1 条。basic 只负责 launch、occupancy、SOL 和工作分布，因此 cache hit、local instruction 与 warp-stall 等 detailed 概念显示为 `not_collected` 是预期行为，不是数值 0，也不代表 Release evidence 缺失。Grouped detailed 中仅 `eligible_warps_per_scheduler` 未被当前 NCU concept resolver 收集。

两条 `chain_from_logits` NSYS trace 都在 T512/E64/K128/N128、20 warmup 后采集：

| 分布 | Grouped GEMM | Permute | Top-K | Histogram→Scan | Unpermute |
|---|---:|---:|---:|---:|---:|
| uniform | 71.6% | 8.3% | 7.0% | 6.6% | 6.5% |
| Zipf 1.4 | 86.0% | 4.1% | 3.6% | 3.3% | 3.0% |

证据必须按时间读取，不能把不同 SHA 的 variant 淹成一个“最新 profile”：

1. `9732a03` 的统一 Release primary 中 Grouped 是 v2 对 CUTLASS/cuBLAS envelope；统一 NCU 为当时诊断用的 v3/CUTLASS。Top-K strong-reference NCU 使用有限随机子域 vLLM，正式 primary 分母仍是 exact-contract naive；Permute NCU 使用 v3 pure path，正式 headline 是 v2 full-from-ids。
2. `c2205ed` 单独把 Grouped 推进到 v5/v6，并证明减少 request amplification 不足以消除 T2048 underfill。
3. `dea7c06` 单独验证 V9/V10：basic 覆盖 CUTLASS、V5 fallback、V9；detailed 覆盖 V5 fallback 与 V9。该证据才支持 V9 的 CTA/request/occupancy 归因。

因此，“七算子 profile coverage 完整”表示预声明的诊断 suite 已完整采集，不表示同一份 NCU 文件覆盖后来出现的 V9，也不允许用 v3 profiler duration 替代 V9 的 Release speedup。

## 简历可用的最大收益与设计亮点

| 优先级 | 可写结论 | 高性能设计 | 必须同时披露 |
|---|---|---|---|
| P0 | Grouped V9 在 T512/E32/K128/N64、Zipf T2048/E64/K128/N64、T4096/E64/K128/N64 对 CUTLASS 为 `1.8819x/1.4366x/1.3661x`，均 5/5 process pairs 同向 | SM86 `32x64x16`、256 threads、每线程 `4x2` outer product、双缓冲 `cp.async`、direct grid、zero workspace fallback | K256/N128/non-aligned 为 `0.9072x/0.7736x/0.7508x`；V10 selector 对 V9 仅 `0.9775x`，不进 Auto |
| P0 | Permute v2 full-from-ids 对 adapted vLLM ratio-of-sums `1.5671x`，5/5 shape 获益 | 将 counts/scan/cursor preparation 纳入同一 L2 边界，减少中间准备与 launch | pure-copy v2 仅 `0.9841x`，不能写成 copy kernel 普遍更快 |
| P1 | Histogram v2 对 v1/CUB/naive envelope `1.1016x`，E1 局部 `4.3026x` | 利用 `E=1` 语义退化直接写结果，其余 shape 复用 dispatcher | 最大 CV `0.4503`，只属于当前 WDDM 作品集证据 |
| P1 | Top-K v4 对 exact naive ratio-of-sums `1.0972x`，E64/T4096 `1.6934x` | 寄存器 Top-2 pair、vector row load、两级 subgroup reduction | 小 shape 最大回退 10.88%，vLLM/CUB 不是完整 tie/NaN/selected-softmax 等价基线 |

V9 的 NCU 归因可直接用于面试：相对 V5 fallback，grid 从 1280 CTA 降到 320 CTA，global load/store requests 从 75,824/16,552 降到 47,096/8,192，即 `-37.9%/-50.5%`；achieved occupancy 从 48.44% 降到 41.54% 仍更快，local load/store 都为 0。它证明收益来自减少 over-partitioning、重复请求与尾波/调度成本，而不是简单追求更高 occupancy。

## 仍未完成或不应宣称

- 没有可发布的 optimized Grouped runtime，也没有修改 `KernelFamily::kAuto`。
- 没有 FP16/BF16 Tensor Core kernel、H100/Blackwell 实卡证据、多 GPU Expert Parallel 或完整 MoE FFN。
- 没有匿名 captured/production route trace；synthetic matrix 不能外推部署分布。
- V9 完整 15-shape aggregate 因一个 baseline CV ceiling violation 只能写 research trend；简历 headline 应使用三条未越过 ceiling、5/5 同向的 local shape。
- fixed CUDA Graph 的 `1.6205x/1.2336x` 是 setup 完成后的 host replay time-to-solution，不是 kernel speedup。
- 第二阶段清理必须按 [cleanup-review](cleanup-review.md) 逐项获得 `保留`、`归档后删除`、`直接删除` 或 `继续观察` 结论。

## 证据入口

- [面试唯一入口](interview/README.md)
- [七算子统一 compact report](reports/compact/20260829-9732a03-interview-portfolio/REPORT.md)
- [Grouped v5/v6 report](reports/compact/20260830-c2205ed-grouped-v6/REPORT.md)
- [Grouped V9/V10 report](reports/compact/20260830-dea7c06-grouped-v9-v10/REPORT.md)
- [实现状态与声明边界](implementation-status.md)
- [分支治理与第二阶段 review](cleanup-review.md)
