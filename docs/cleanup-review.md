# RaggedRoute 清理 Review 清单

初始快照时间：2026-08-28（Asia/Shanghai）
初始事实基线：`main@3597731de8ddbdf0238574cb99b58be90226c2ce`
最新 live 复核：2026-08-30（Asia/Shanghai），`main@9782740138998330e9ac3e3a1b8c46a05fa4f830`
治理分支：`codex/repo-governance-performance-docs`（本轮从 `dea7c066a83a5df700aa60c03fd51446c6b4c5e5` 创建）

执行结果：初轮 29 个 merged-PR 远端分支已按 live SHA 和 `merged_at` 复核后删除；后续 PR #36/#37 的任务分支由 `delete_branch_on_merge=true` 自动清理。2026-08-30 重新读取 `git ls-remote` 和全部 PR 后，远端仍只保留 `main` 与无关联 PR 的 `codex/grouped-gemm-sm86-v2`，因此本轮没有新的远端删除目标。4 个 evidence tag 不变。本地 44 个分支、5 个 worktree、1 个 stash 和未跟踪实验文件均未删除。

本文件是两阶段清理的审计账本。第一阶段只删除 GitHub 上已经合并、没有打开 PR、且 live tip 与本表一致的远端分支。任何本地分支、worktree、stash、未跟踪实验文件和文档/证据冗余都必须经过人工 review 后才能进入第二阶段。

## 0. 2026-08-30 live 复核

| Ref / 分支 | Live tip | PR 状态 | 相对 `main` 独有提交 | worktree | 当前动作 |
|---|---|---|---:|---|---|
| `main` | `978274013899` | 默认分支 | 0 | 无 | 保留 |
| `codex/grouped-gemm-sm86-v2` | `f9f5e7eb8a68` | 无关联 PR、无 open PR | 2 | 无 | **保留并人工 review** |
| `codex/repo-governance-performance-docs` | `dea7c066a83a`（提交文档前） | 本轮任务，尚未推送/建 PR | 2 | `RaggedRoute-six-ops-v2-eval` | 完成门禁后推送并建 PR；合并后自动删除远端 |

本地旧 `codex/repo-governance-performance-docs@a47169d19ba4` 对应已合并 PR #36，但 squash 历史使其不是当前 `main` 的 ancestor。为避免删除本地 ref，本轮将其保留为 `codex/cleanup-safety/repo-governance-performance-docs-pr36`，再创建新的治理分支。`git fetch --prune` 只清理了已不存在的 remote-tracking refs，没有删除本地分支。

当前非任务 worktree 中有 3 个 dirty 实验 worktree，内容与初始治理范围一致：Histogram 为 4 modified + 2 untracked，L3 为 5 untracked，vLLM industrial correction 为 2 modified + 1 untracked。当前任务 worktree 的文档/evidence 改动将在本 PR 提交，不计入第二阶段冗余清理；其余 dirty 内容原样保留。

## 1. 远端分支快照

快照时 GitHub 有 `main` 加 30 个其他分支。其中 29 个分支对应已合并 PR，计划在重新校验 tip SHA 与 PR 状态后删除；`codex/grouped-gemm-sm86-v2` 没有关联 PR，保留等待 review。`codex/six-ops-v2-unified-eval` 在执行前已经不再是 live ref，因此不纳入本轮删除计数。

“独有提交”是 `origin/main..<branch-tip>` 的提交数量，只用于暴露 squash/stacked-PR 历史，不能单独推翻 GitHub 的 merged 状态。删除依据必须同时满足：关联 PR 的 `merged_at` 非空、当前没有 open PR、live SHA 未变化。

| 远端分支 | Tip SHA | PR | PR base | 已合并 | 相对 main 独有提交 | 第一阶段动作 |
|---|---|---:|---|---:|---:|---|
| `codex/audit-truth-sync` | `98ec26156817` | 22 | `main` | 是 | 1 | 删除远端分支 |
| `codex/benchmark-harness` | `62cfc1b92a22` | 1 | `main` | 是 | 0 | 删除远端分支 |
| `codex/ci-quality-gates` | `7ad9aac7765e` | 25 | `codex/evidence-bundle-v2` | 是 | 0 | 删除远端分支 |
| `codex/dense-gemm-layout` | `b94047787589` | 13 | `main` | 是 | 3 | 删除远端分支 |
| `codex/dense-gemm-optimization` | `af5ff733c8d7` | 12 | `main` | 是 | 8 | 删除远端分支 |
| `codex/dense-gemm-v3-cta-tile` | `052f29f30633` | 14 | `main` | 是 | 9 | 删除远端分支 |
| `codex/evidence-bundle-v2` | `700a84eed58d` | 24 | `codex/layout-normalization` | 是 | 1 | 删除远端分支 |
| `codex/grouped-gemm-sm86-opt` | `db901a7a3adb` | 16 | `main` | 是 | 4 | 删除远端分支 |
| `codex/grouped-gemm-sm86-v2` | `f9f5e7eb8a68` | — | — | 否 | 2 | **保留并人工 review** |
| `codex/histogram-cuda-candidate` | `9e2eff98a5ef` | 20 | `main` | 是 | 13 | 删除远端分支 |
| `codex/l3-realistic-moe-eval` | `1bf8ece8f71f` | 31 | `codex/l3-three-way-eval` | 是 | 4 | 删除远端分支 |
| `codex/l3-three-way-eval` | `e783cc33b118` | 30 | `main` | 是 | 3 | 删除远端分支 |
| `codex/layout-normalization` | `5a1330f54697` | 23 | `codex/audit-truth-sync` | 是 | 1 | 删除远端分支 |
| `codex/operator-optimization-docs` | `21b25faf0950` | 10 | `main` | 是 | 0 | 删除远端分支 |
| `codex/p0-infrastructure` | `c9866cd40093` | 2 | `main` | 是 | 0 | 删除远端分支 |
| `codex/permute-candidate-sm86` | `67835a5ce374` | 15 | `main` | 是 | 10 | 删除远端分支 |
| `codex/permute-sm86-v2` | `0e130a43e41c` | 28 | `main` | 是 | 0 | 删除远端分支 |
| `codex/readme-truth-sync` | `99456d646389` | 33 | `main` | 是 | 0 | 删除远端分支 |
| `codex/readme-zh-cn` | `5ab4810fffa2` | 34 | `main` | 是 | 0 | 删除远端分支 |
| `codex/refine-development-roadmap` | `992e133f2a52` | 5 | `main` | 是 | 1 | 删除远端分支 |
| `codex/resume-project-fixes` | `f9364dfc80c6` | 4 | `main` | 是 | 1 | 删除远端分支 |
| `codex/rtx3080-naive-profile` | `a16d29c0c304` | 11 | `main` | 是 | 2 | 删除远端分支 |
| `codex/scan-fused-sm86-v2` | `8792d91a11fe` | 29 | `main` | 是 | 0 | 删除远端分支 |
| `codex/scan-sm86-hybrid` | `23fd331e8d73` | 18 | `main` | 是 | 2 | 删除远端分支 |
| `codex/topk-gate-optimization` | `d16bd526b816` | 19 | `main` | 是 | 3 | 删除远端分支 |
| `codex/topk-gate-sm86-v4-pr` | `0af82f9a75d8` | 26 | `main` | 是 | 0 | 删除远端分支 |
| `codex/triton-baselines` | `f32be4def977` | 21 | `main` | 是 | 4 | 删除远端分支 |
| `codex/unpermute-sm86-v2` | `befdb78b6ce8` | 27 | `main` | 是 | 0 | 删除远端分支 |
| `codex/unpermute-warp-vectorized` | `a51702f0aaba` | 17 | `main` | 是 | 5 | 删除远端分支 |
| `codex/vllm-semantic-moe-chain` | `052f32cdb971` | 32 | `codex/l3-realistic-moe-eval` | 是 | 3 | 删除远端分支 |

以下 evidence tag 必须保留：

- `evidence-2026-08-03-v1` → `dab5793057eb`
- `evidence-2026-08-04-permute-v2` → `f71929e6ec1b`
- `evidence-2026-08-04-topk-v4` → `e07f7c3f718c`
- `evidence-2026-08-04-topk-v4-v3` → `e07f7c3f718c`

## 2. 本地 worktree、clone 与 stash

第一阶段禁止删除或改写以下内容：

| 路径 / 对象 | 状态 | Review 建议 |
|---|---|---|
| `RaggedRoute` worktree | `codex/unpermute-sm86-v2`，clean，落后 `origin/main` 35 commits | 等当前 PR 合并后决定移除 worktree 或重新指向 main |
| `RaggedRoute-histogram-v2` worktree | 4 个 modified 文件、2 组未跟踪报告/制品 | 先查看 diff 与 evidence 完整性；可选择归档到独立分支或丢弃 |
| `RaggedRoute-l3-eval-8d4c281` worktree | 5 个未跟踪 L3/vLLM 报告目录 | 与 main 的 L3 报告去重后决定归档或删除 |
| `RaggedRoute-vllm-industrial-correction` worktree | 2 个 modified 脚本、1 个未跟踪同步脚本 | 需要代码 review，不得随 stacked branch 清理 |
| `RaggedRoute-six-ops-v2-eval` worktree | 当前治理/性能文档任务；V9/V10 compact evidence 与文档待提交 | 本轮 PR 正常提交；不得混入其他 worktree 的实验内容 |
| `RaggedRoute-unpermute-v2` 独立 clone | clean，旧 `codex/unpermute-sm86-v2` | 可在确认无独立 object/evidence 后删除候选 |
| `RaggedRoute-histogram-candidate` 普通目录 | 非 RaggedRoute Git worktree，含历史源码/报告/out | 先做文件级 manifest，再判断是否与 Git 历史重复 |
| `stash@{0}` | `histogram-v2 accidental shared-worktree edits` | 必须单独 review；禁止自动 drop |

第二阶段如果批准清理，必须先记录绝对路径、Git SHA、dirty diff 摘要和未跟踪文件 SHA-256；可保留内容先进入独立归档分支、tag 或不可覆盖的压缩证据包。

## 3. 文档与证据冗余候选

当前 `main` 的 `docs/reports/` 约包含：398 个 `artifacts` 文件（约 13.8 MB）、29 个 compact 文件、38 个 L3 文件和 17 个顶层报告。下列项目只进入 review，不在第一阶段删除：

| 类别 | 发现 | 建议 |
|---|---|---|
| 原始 profiler 二进制 | `l3_three_way_20260805` 中跟踪了 3 个 `.ncu-rep` 和 3 个 `.nsys-rep` | 校验已存在规范化 CSV/JSON 和外部资产后移出 Git |
| 超大聚合文件 | Top-K evidence 中有约 2.5 MB、2.4 MB、1.5 MB、1.0 MB 的 aggregate/decision 文件 | 保留 compact comparison 与 hash；raw aggregate 外移 |
| 完全重复指标 | Scan 两个 bundle 的 `ncu_metrics.json/csv/REPORT.md` blob 完全相同 | 保留一份规范来源，另一份改为链接或 manifest 引用 |
| 重复 sanitizer 日志 | Permute 的多组 sanitizer blob 完全相同 | 由 manifest/hash 引用，避免重复提交正文 |
| 旧版长文 | `RaggedRoute-最终产品技术文档.md` 仍混合计划、未实现路线和面试内容 | 新面试入口稳定后，review 是否标记 historical 或拆分 |
| 旧面试文档 | 现有简历手册、追问题库与新 `docs/interview/` 会暂时重叠 | review 后保留 canonical 版本，旧文件可转为短重定向页 |

## 4. 第二阶段批准格式

Review 时请按条目给出 `保留`、`归档后删除`、`直接删除` 或 `继续观察`。没有明确结论的条目一律按 `保留` 处理。
