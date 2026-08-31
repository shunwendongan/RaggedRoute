# RaggedRoute 清理 Review 清单

初始快照时间：2026-08-28（Asia/Shanghai）
初始事实基线：`main@3597731de8ddbdf0238574cb99b58be90226c2ce`
固定完成基线：套餐完成审计 [PR #39](https://github.com/shunwendongan/RaggedRoute/pull/39) `main@2366acec5d2ab6552e6112e5019ac948a51fbf34`
最终事实同步：[PR #40](https://github.com/shunwendongan/RaggedRoute/pull/40) `main@fc421be035576b1bce7b9d71f3d60578b5d5d0aa`
最新 live 复核：2026-08-31（Asia/Shanghai）；远端只有 `main` 与 review-held `codex/grouped-gemm-sm86-v2`，无 open PR，`delete_branch_on_merge=true`。
治理结果：`codex/repo-governance-performance-docs` 已通过 [PR #38](https://github.com/shunwendongan/RaggedRoute/pull/38) 合入；套餐完成审计通过 PR #39 合入；事实同步通过 PR #40 合入。三个远端任务分支均已自动删除。

执行结果：初轮 29 个 merged-PR 远端分支已按 live SHA 和 `merged_at` 复核后删除；后续 PR #36/#37/#38/#39/#40 的任务分支由 `delete_branch_on_merge=true` 自动清理。2026-08-31 再次读取 live refs、PR 状态和仓库设置，远端只保留 `main` 与无关联 PR 的 `codex/grouped-gemm-sm86-v2`。4 个 evidence tag 不变。本地分支、5 个 worktree、1 个 stash 和未跟踪实验文件均未删除。

本文件是两阶段清理的审计账本。第一阶段只删除 GitHub 上已经合并、没有打开 PR、且 live tip 与本表一致的远端分支。任何本地分支、worktree、stash、未跟踪实验文件和文档/证据冗余都必须经过人工 review 后才能进入第二阶段。

## 0. 2026-08-31 第二阶段精确审计与批准表

0.1 及后续小节保留批准前的事实快照与建议；实际执行状态以 0.0 为准。本文的目录树 SHA-256 定义为：按相对路径排序文件，以 Git for Windows `sha256sum` 生成逐文件 manifest，再对 manifest 原始字节计算 SHA-256；它用于确认归档前后文件集合没有被悄悄改变。

### 0.0 2026-08-31 用户批准后的执行结果

用户明确批准：`R1、B1、B2、B3、C1、E3、E4；W2/W3/W4/S1 归档后删除；其余继续观察`。执行时仍逐项复核 live SHA、PR、dirty 状态和归档 hash；结果如下：

- R1：已创建并推送 `archive/grouped-gemm-sm86-v2-f9f5e7e`，peeled target 为 `f9f5e7eb8a68987efa562b83024e19a333422853`；远端和本地同名 branch 已删除。远端 branch 现只剩 `main`。
- B1/B2/B3：14 + 14 + 7，共 35 个批准的旧本地 branch 已删除。B2 复核发现 `dense-gemm-optimization` 有 1 个、`topk-gate-optimization` 有 5 个 PR head 未覆盖的后续提交，因此先额外创建 `archive/dense-gemm-post-pr12-af5ff73-20260831` 和 `archive/topk-post-pr19-712dc89-20260831`，再删除 branch；没有丢弃未受保护的提交。
- W2：4 个 modified 与 51 个 non-ignored R3 evidence files 已提交为 `632286842d3b347c089eb18105cdeed4ba68b2a2`，推送 tag `archive/histogram-v2-w2-105a7dd-20260831` 后移除 worktree；ignored build/profile/benchmark 输出按批准删除。
- W3：stacked branch tip 已由 `archive/vllm-semantic-moe-chain-052f32c-20260831` 保留；`final2 + verified + raw profiler` 共 113 files / 9,473,572 uncompressed bytes 已归档为本地 ZIP，SHA-256 `94b3601dce2eb329fd657a2c64319968127fb8f267708396679542678c20fbe9`，随后移除 worktree及约 697.7 MB ignored 输出。
- W4：三个脚本已提交为 `8454ebd7f3cd4f74e75b426a889bd08eceb28766`，推送 tag `archive/vllm-industrial-correction-1bf8ece-20260831` 后移除 worktree。
- S1：tag `archive/histogram-h5-h6-stash-a5e64f2-20260831` 已精确指向原两父提交 stash `a5e64f206cdba2af7f4d58e0dbabecaac4a5df9e`；远端复核成功后已 drop，本地 stash 数量为 0。
- C1：clean 独立 clone `RaggedRoute-unpermute-v2` 已移入 Windows 回收站，仍可恢复。
- E3/E4：重复正文已改为 canonical-reference 文件；两个 evidence manifest 与 `SHA256SUMS` 同步更新。Scan canonical 位于 `scan-library-main-927c585`，Permute sanitizer canonical 位于 `20260803T071455Z-af4947f-permute-selected-token-owned-v1`。
- W1 已安全完成：主工作树已切回 `main@193de56`，旧本地 `codex/unpermute-sm86-v2` 已删除；主工作树目录保留，以避免破坏 Git linked-worktree 元数据。
- C2、E1、E2、D1/D2、B4、W5 及其他未获批准对象均保持不动。

W3 本地 ZIP 与七个远端 archive tag 的校验清单位于 `C:\Users\Administrator\Documents\Playground\RaggedRoute-local-archives\20260831-approved-cleanup\README.md`。

### 0.1 可直接回复的批准表

| ID | 精确对象 | 审计结论 | 建议动作 | 默认状态 |
|---|---|---|---|---|
| R1 | 远端/本地 `codex/grouped-gemm-sm86-v2@f9f5e7e` | 2 个独有提交、无 PR；V2 实现已被 `main` 后续 V5/V6/V9 链覆盖，但 2026-08-04 三进程 artifact 只存在该分支 | 创建 `archive/grouped-gemm-sm86-v2-f9f5e7e` 后删除远端和同名本地 branch | **归档后删除** |
| B1 | 14 个已完全包含于 `main`、未挂 worktree且不是 safety alias 的旧本地 branch | 无独有提交；远端均已删除 | 删除本地 branch | **直接删除** |
| B2 | 14 个已有 merged PR、相对 `main` 因 squash 历史仍显示独有提交、但未挂 worktree的 branch | GitHub PR 已保存 review 与 commit 对象 | 删除本地 branch；不额外制造 14 个 archive tag | **直接删除** |
| B3 | 7 个 `cleanup-safety/*` 本地安全别名 | 两个已完全包含于 `main`；其余指向 merged PR #15/#16/#18/#19 或 PR #36 的旧 tip | 删除本地 branch | **直接删除** |
| B4 | `dense-gemm-analysis`、`dense-gemm-candidate-v2`、`dense-gemm-v4-shuffle-pipeline`、`profiler-integration`、`scan-research-sm86` | 无直接 PR，分别保留 1–12 个独有实验提交 | 每条创建 `archive/...-<shortsha>` 后删除本地 branch | **归档后删除** |
| W1 | `C:\Users\Administrator\Documents\Playground\RaggedRoute` | clean；`codex/unpermute-sm86-v2@728b9f3` 已完全包含于 `main` | 移除 worktree，再删除本地 branch | **直接删除** |
| W2 | `C:\Users\Administrator\Documents\Playground\RaggedRoute-histogram-v2` | 4 modified、51 non-ignored untracked、527 ignored；含唯一 R3 evidence | 归档 commit/diff/R3 evidence；构建缓存不归档；再移除 worktree 与 branch | **归档后删除** |
| W3 | `C:\Users\Administrator\Documents\Playground\RaggedRoute-l3-eval-8d4c281` | 0 modified、221 non-ignored untracked、2,025 ignored；旧/`final`/`final2` 高度重复 | 归档 `final2` 与 `verified` 及原始 profiler；删除旧重复副本和 ignored build/research 输出；再移除 worktree | **归档后删除** |
| W4 | `C:\Users\Administrator\Documents\Playground\RaggedRoute-vllm-industrial-correction` | 2 modified + 1 untracked script；修正 digest pin、五进程和“component semantic adapter”口径 | 先形成独立 archive/代码 review，不直接丢弃；随后移除 worktree | **归档后删除** |
| W5 | 当前 `RaggedRoute-six-ops-v2-eval` / `codex/cleanup-phase2-review` | 本审计任务 worktree | PR 合并前保留；合并后可移除本地 worktree/branch | **继续观察** |
| C1 | `C:\Users\Administrator\Documents\Playground\RaggedRoute-unpermute-v2` | clean 独立 clone；PR #27 已合并；tip 已是 `main` ancestor；0 untracked；pack 4.48 MiB | 删除整个独立 clone | **直接删除** |
| C2 | `C:\Users\Administrator\Documents\Playground\RaggedRoute-histogram-candidate` | 无独立 `.git`；4,530 文件、117,109,441 B，`out/` 占 116,630,730 B；含 5 NCU + 4 NSYS | 先归档唯一源码、evidence 与 profiler manifest，再删除整个目录 | **归档后删除** |
| S1 | `stash@{0}@a5e64f2` | 6 文件、+120/-21；H5 single-bin/H6 grid-cap 实验被后续 main 部分吸收，但 patch 不完全相同 | 创建 archive branch 或 patch bundle 后 drop stash | **归档后删除** |
| E1 | tracked 3 `.ncu-rep` + 3 `.nsys-rep` | 共 866,334 B；每项已有 normalized CSV/JSON/TXT 和 manifest | 外移到不可覆盖资产并保留 hash，再从 Git 删除 | **归档后删除** |
| E2 | Top-K 六个 0.69–2.53 MB aggregate/comparison | 共 9,102,061 B；已有 compact comparison/决策 | 保留 compact 与 SHA，raw aggregate 外移 | **归档后删除** |
| E3 | Scan 两个 bundle 的 NCU JSON/CSV/REPORT | 三组逐 blob 完全相同，可减少 74,452 B | 保留 `scan-library-main-927c585` 为 canonical，另一份改 manifest 引用 | **直接删除重复副本** |
| E4 | Permute 两个 bundle的四项 sanitizer log | 四组逐 blob 完全相同，可减少 17,621 B | 保留 selected-token-owned bundle 为 canonical，另一份改 hash 引用 | **直接删除重复副本** |
| D1 | `RaggedRoute-最终产品技术文档.md` | 1,461 行/106,157 B；README 中英文仍显式链接，含历史规划 | 暂保留并标记 historical；不在本轮删 | **继续观察** |
| D2 | 旧简历手册与旧 CUDA/MoE 追问题库 | 726 行/80,330 B；与 `docs/interview/` 重叠，但旧长文仍回链 | 先移到 `docs/archive/` 或改为短重定向页并修复回链 | **归档后删除正文** |

用户批准时可直接回复，例如：`批准 R1、B1、B2、B3、C1、E3、E4；W2/W3/W4/S1 归档后删除；其余继续观察`。没有明确批准的 ID 一律按保留处理。

### 0.2 held 远端分支为何必须先归档

`origin/main..origin/codex/grouped-gemm-sm86-v2` 是：

- `845e176a9960`：新增 `cuda_grouped_sm86_fp32_v2`；
- `f9f5e7eb8a68`：提交 2026-08-04 SM86 V2 三进程 evidence。

当前 `main` 已包含同名 V2 kernel、registry、adapter，并由 V5/V6/V9 fallback 使用；更新的五进程 portfolio 对 V2 与 CUTLASS 的证据也已进入 compact bundle。该分支的旧 artifact 目录 `20260804T041056Z-845e176-grouped-gemm-sm86-v2` 没有进入 `main`，所以“直接删除”会丢失旧证据对象。archive tag 可同时保留两次提交、实现和旧 artifact，随后远端 branch 才适合删除。

### 0.3 dirty worktree 与 stash 的可复核快照

| 对象 | Tip / patch SHA-256 | tracked diff | non-ignored untracked | ignored 输出 | 建议保留边界 |
|---|---|---:|---:|---:|---|
| Histogram W2 | tip `105a7ddddcd2`；diff `31a188968e77a6a931bedad80baa0a75aa6ccaddfcd78ba11341733f34bfa27a` | 4 files，+51/-3 | 51 files，316,312 B；tree `3d534425345d4030511fd174b55d729d6fe048e6a85fe237d4e477afbee43c76`；与 `main` 精确哈希匹配 0 | 527 files，91,227,417 B；tree `db56b60c7e8df4a62f9b74eed1244a30e302b91ca2af172638a3fa120b4bb3f0` | 保留 branch、diff、51-file R3 evidence；`out/build` 60.35 MB、profile 17.40 MB、benchmark 7.12 MB 仅在归档证据后删除 |
| L3 W3 | tip `052f32cdb971`；tracked diff 为空 | 0 | 221 files，15,681,528 B；tree `7d694c625c4c9cc4f080f7103d88e83a6e7897768ee178ddf527d47d288a8266`；77 个重复哈希组覆盖 201 files | 2,025 entries，697,719,306 B；其中 `out/research/l3_three_way` 602.05 MB、build 64.98 MB | 保留 `final2`、`verified` 与 raw profiler；其余由 hash/relative-path 对比证明重复后清理 |
| vLLM W4 | tip `1bf8ece8f71f`；diff `dd83b458f21de1c9f2b88bafc9b35263883e871d7503b78656af7dd5e05ec7c4` | 2 files，+34/-19 | 1 file，3,384 B；tree `955ece500a88b24347cabd56d800263389b6ea694f07577e4c266b751b7df9a9` | 0 | 三个脚本作为一个 review 单元归档 |
| stash S1 | stash `a5e64f206cdb`；patch `18f35432b1b8d267818cdb88f2ee2451e2af914ee7af140fd1fb6abe850ac62b` | 6 files，+120/-21 | 不适用 | 不适用 | 保留 Histogram H5/H6 patch 与 base SHA `728b9f36a6ad` |

L3 五个未跟踪报告目录的物理文件（含 ignored raw profiler）如下：

| 目录 | 文件 / 字节 | tree SHA-256 | 相对最新目录的重复关系 |
|---|---:|---|---|
| `l3_realistic_moe_20260805` | 68 / 6,316,675 | `c3ce93b11a5859510c6b876a166346b68565c33d092c3d281fbc7dc7cb2f6bb6` | 对 `final2`：65 同路径同 hash，3 changed，0 only-old |
| `l3_realistic_moe_20260805_final` | 68 / 6,316,648 | `dc91a66f72047fedcb71b2e8ba766f951ef4f77b62f0bf15b2273fddcad294d4` | 对 `final2`：65 同路径同 hash，3 changed，0 only-old |
| `l3_realistic_moe_20260805_final2` | 69 / 6,322,485 | `c1635667b86dbd671feb9b0bab5ddb3df0b65002afd8cf94308861bf52ac7093` | canonical 候选；额外 `collection_manifest.json` |
| `vllm_semantic_moe_20260806` | 44 / 3,039,448 | `5111e24f015920da5e92007f9011ddaf9ed16f3e2d3577c6c902521d24bfda9c` | 对 `verified`：36 同路径同 hash，8 changed |
| `vllm_semantic_moe_20260806_verified` | 44 / 3,151,087 | `343e7d6ba3707fe5ca2b839a26398bbcabd362b72462cf50297ccc122084aced` | canonical 候选 |

特别注意：`codex/vllm-semantic-moe-chain@052f32c` 的 PR #32 是合入 stacked base `codex/l3-realistic-moe-eval`，当前相对 `main` 仍有 3 个独有提交；其中 `vllm_semantic_moe_20260806_final` 的 35 个 tracked 文件不在 `main`。因此 W3 的 branch 与 worktree 必须先归档，不能只凭 `PR #32 merged` 直接删除。

### 0.4 独立目录与 clone

- C1 clone：`codex/unpermute-sm86-v2@befdb78b6ce8` 对应 merged [PR #27](https://github.com/shunwendongan/RaggedRoute/pull/27)，`origin/main..befdb78` 为 0，clean、0 untracked；独立对象库 `size-pack=4.48 MiB`。这是可直接删除对象，但仍等待批准。
- C2 普通目录：没有自己的 `.git`，整树 digest 为 `10094da0413c96568941ef2ce9d6e80b59a693c88a714cde53704a3ccd4014a9`。其中 `out/evidence/histogram-candidate-cb02617` 为 11 files / 1,670,139 B；5 个 NCU raw 共 842,285 B；4 个 NSYS 路径中有两对内容完全重复。归档时应保留一份 raw、normalized report、命令和源码快照，构建树和重复 raw 不进入归档。

### 0.5 tracked evidence 精确冗余

6 个误提交的 raw Nsight binary：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `docs/reports/l3_three_way_20260805/artifacts/profile/cuda_candidate_ncu/reports/ncu_basic.ncu-rep` | 189,071 | `53e4f9389a60e5899e262a06e1c8c97443e4f1a0150886cd8cecef0b7beee7f4` |
| `docs/reports/l3_three_way_20260805/artifacts/profile/cuda_candidate_nsys/reports/system.nsys-rep` | 80,077 | `9ac3914159d91c50b1a55ef01ac38b597967bd00c255703929b63bd15d614945` |
| `docs/reports/l3_three_way_20260805/artifacts/profile/library_ncu/reports/ncu_basic.ncu-rep` | 345,699 | `f0a5bf1eec3d28ec9a5ba90604b341c9c96ee635919c106d37739caaee7d6e9b` |
| `docs/reports/l3_three_way_20260805/artifacts/profile/library_nsys/reports/system.nsys-rep` | 89,167 | `b37598d04ef8bc4eb1db19fb4113c92e893256697c41beb858cb9f5fc3ed1b67` |
| `docs/reports/l3_three_way_20260805/artifacts/profile/triton_ncu/ncu_basic.ncu-rep` | 115,625 | `80b8be3613c49e476a91b6ec2b9c531fadf501c55e42b2337daa6d017916699a` |
| `docs/reports/l3_three_way_20260805/artifacts/profile/triton_nsys_capture/system.nsys-rep` | 46,695 | `60e1053d8fcea8a2625ac86dc6c200b2abdd42cde7fee4fdbd735b7ac4d557ef` |

每个 NCU 项旁已有 `ncu_basic_raw.csv`、`supported_key_metrics.json`、details/manifest；每个 NSYS 项旁已有 kernel summary、hotspots/timeline CSV/JSON 与 manifest。只有在外部资产校验上述 hash 后，E1 才能执行。

Top-K 大文件 E2 的六项分别为 2,528,631、2,457,198、1,519,639、1,028,315、874,444、693,834 B，总计 9,102,061 B；它们位于 `20260804-86cfbe2-topk-v4` 与 `20260803T0705Z-1ce3398-topk-gate-v4` 的 aggregate/benchmark 目录。保留 compact comparison、promotion decision、manifest 与 SHA 即可复核 headline。

Scan E3 的重复 blob 为：`ncu_metrics.json` 41,747 B、`ncu_metrics.csv` 30,762 B、`REPORT.md` 1,943 B；`scan-library-main-927c585` 与 `scan-sm86-research-32bf6c9` 两份逐 SHA 完全一致。Permute E4 的 memcheck/initcheck/racecheck/synccheck 四份重复日志分别为 4,395/4,396/4,434/4,396 B。

### 0.6 47 个本地 branch 快照

`ahead/behind` 以 `origin/main...branch` 为准；`contained=yes` 表示 branch tip 已是 `main` ancestor。所有 upstream 中除 `origin/main` 与 held V2 外均已 gone。

| Branch | Tip | ahead / behind | PR | worktree | 建议 |
|---|---|---:|---|---|---|
| `codex/audit-truth-sync` | `b2fec8f4b73a` | 0 / 51 | #22 merged | — | 直接删除 |
| `codex/benchmark-harness` | `62cfc1b92a22` | 0 / 76 | #1 merged | — | 直接删除 |
| `codex/ci-quality-gates` | `7ad9aac7765e` | 0 / 47 | #25 merged | — | 直接删除 |
| `codex/cleanup-phase2-review` | `fc421be03557` | 0 / 0 | 当前任务 | W5 | 继续观察 |
| `codex/cleanup-safety/RaggedRoute-eval-8d4c281-20260825` | `8d4c281eb28b` | 0 / 36 | safety alias | — | 直接删除 |
| `codex/cleanup-safety/RaggedRoute-main-verify-20260825` | `4359b16a0b0c` | 0 / 53 | safety alias | — | 直接删除 |
| `codex/cleanup-safety/RaggedRoute-pr15-merge-20260825` | `67835a5ce374` | 10 / 55 | #15 tip alias | — | 直接删除 |
| `codex/cleanup-safety/RaggedRoute-pr16-merge-20260825` | `db901a7a3adb` | 4 / 56 | #16 tip alias | — | 直接删除 |
| `codex/cleanup-safety/RaggedRoute-pr18-merge-20260825` | `23fd331e8d73` | 2 / 57 | #18 tip alias | — | 直接删除 |
| `codex/cleanup-safety/RaggedRoute-pr19-merge-20260825` | `d16bd526b816` | 3 / 54 | #19 tip alias | — | 直接删除 |
| `codex/cleanup-safety/repo-governance-performance-docs-pr36` | `a47169d19ba4` | 2 / 14 | #36 safety alias | — | 直接删除 |
| `codex/dense-gemm-analysis` | `307ffe7d3fcc` | 2 / 62 | — | — | 归档后删除 |
| `codex/dense-gemm-candidate-v2` | `b0930616ce62` | 6 / 61 | — | — | 归档后删除 |
| `codex/dense-gemm-layout` | `b94047787589` | 3 / 61 | #13 merged | — | 直接删除 |
| `codex/dense-gemm-optimization` | `af5ff733c8d7` | 8 / 62 | #12 merged | — | 直接删除 |
| `codex/dense-gemm-v3-cta-tile` | `052f29f30633` | 9 / 60 | #14 merged | — | 直接删除 |
| `codex/dense-gemm-v4-shuffle-pipeline` | `1860420ad922` | 12 / 60 | — | — | 归档后删除 |
| `codex/evidence-bundle-v2` | `9e8ed1136de0` | 0 / 49 | #24 merged | — | 直接删除 |
| `codex/grouped-gemm-sm86-opt` | `fb10b86e9c02` | 3 / 59 | #16 merged | — | 直接删除 |
| `codex/grouped-gemm-sm86-v2` | `f9f5e7eb8a68` | 2 / 44 | — | — | R1 归档后删除 |
| `codex/grouped-v2-scheduling-portfolio` | `b0202fa6396f` | 0 / 10 | #37 merged | — | 直接删除 |
| `codex/grouped-v8-16x128` | `dea7c066a83a` | 0 / 7 | contained | — | 直接删除 |
| `codex/histogram-candidate-v2` | `105a7ddddcd2` | 3 / 44 | — | W2 dirty | 归档后删除 |
| `codex/histogram-cuda-candidate` | `9e2eff98a5ef` | 13 / 58 | #20 merged | — | 直接删除 |
| `codex/l3-realistic-moe-eval` | `69eba763c564` | 2 / 33 | #31 merged | — | 直接删除 |
| `codex/l3-three-way-eval` | `8dc2382e6874` | 0 / 33 | #30 merged | — | 直接删除 |
| `codex/layout-normalization` | `e36bcfe6b0d4` | 0 / 50 | #23 merged | — | 直接删除 |
| `codex/library-baselines` | `5cf466d87339` | 1 / 66 | #9 merged | — | 直接删除 |
| `codex/permute-candidate-sm86` | `f49ba5777c6d` | 8 / 57 | #15 merged | — | 直接删除 |
| `codex/permute-sm86-v2` | `0e130a43e41c` | 0 / 20 | #28 merged | — | 直接删除 |
| `codex/portfolio-completion-audit` | `1b64e47420a2` | 0 / 3 | #39 merged | — | 直接删除 |
| `codex/portfolio-truth-sync` | `e1c3add7c469` | 0 / 1 | #40 merged | — | 直接删除 |
| `codex/profiler-integration` | `74b7e0697542` | 3 / 66 | — | — | 归档后删除 |
| `codex/refine-development-roadmap` | `992e133f2a52` | 1 / 70 | #5 merged | — | 直接删除 |
| `codex/repo-governance-performance-docs` | `e0e92fb5be38` | 0 / 5 | #36/#38 merged | — | 直接删除 |
| `codex/resume-project-fixes` | `f9364dfc80c6` | 1 / 72 | #3/#4 merged | — | 直接删除 |
| `codex/scan-fused-sm86-v2` | `8792d91a11fe` | 0 / 41 | #29 merged | — | 直接删除 |
| `codex/scan-research-sm86` | `d9e745a3a129` | 4 / 59 | — | — | 归档后删除 |
| `codex/scan-sm86-hybrid` | `6935f73dd8f1` | 1 / 59 | #18 merged | — | 直接删除 |
| `codex/topk-gate-optimization` | `712dc8936007` | 7 / 59 | #19 merged | — | 直接删除 |
| `codex/topk-gate-sm86-v4-pr` | `0af82f9a75d8` | 0 / 41 | #26 merged | — | 直接删除 |
| `codex/triton-baselines` | `f32be4def977` | 4 / 53 | #21 merged | — | 直接删除 |
| `codex/unpermute-sm86-v2` | `728b9f36a6ad` | 0 / 44 | #27 merged | W1 clean | 直接删除 |
| `codex/unpermute-warp-vectorized` | `a51702f0aaba` | 5 / 59 | #17 merged | — | 直接删除 |
| `codex/vllm-moe-industrial-correction` | `1bf8ece8f71f` | 4 / 33 | — | W4 dirty | 归档后删除 |
| `codex/vllm-semantic-moe-chain` | `052f32cdb971` | 3 / 33 | #32 merged to stacked base | W3 | 归档后删除 |
| `main` | `3597731de8dd` | 0 / 14 | default local ref | — | 保留并快进 |

## 0. 2026-08-30 live 复核

| Ref / 分支 | Live tip | PR 状态 | 相对 `main` 独有提交 | worktree | 当前动作 |
|---|---|---|---:|---|---|
| `main` | `2366acec5d2a` | 默认分支；PR #39 已合并 | 0 | 无 | 保留 |
| `codex/grouped-gemm-sm86-v2` | `f9f5e7eb8a68` | 无关联 PR、无 open PR | 2 | 无 | **保留并人工 review** |

本地旧 `codex/repo-governance-performance-docs@a47169d19ba4` 对应已合并 PR #36，但 squash 历史使其不是当前 `main` 的 ancestor。为避免删除本地 ref，本轮将其保留为 `codex/cleanup-safety/repo-governance-performance-docs-pr36`，再创建新的治理分支。`git fetch --prune` 只清理了已不存在的 remote-tracking refs，没有删除本地分支。

当前非任务 worktree 中有 3 个 dirty 实验 worktree，内容与初始治理范围一致：Histogram 为 4 modified + 2 untracked，L3 为 5 untracked，vLLM industrial correction 为 2 modified + 1 untracked。完成审计 worktree 在 PR #39 修改前 clean，且只提交审计、文档事实同步和自动校验，不计入第二阶段冗余清理。其余 dirty 内容原样保留。

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
| `RaggedRoute` worktree | `codex/unpermute-sm86-v2`，clean，落后 `origin/main` 40 commits | 等当前 PR 合并后决定移除 worktree 或重新指向 main |
| `RaggedRoute-histogram-v2` worktree | 4 个 modified 文件、2 组未跟踪报告/制品 | 先查看 diff 与 evidence 完整性；可选择归档到独立分支或丢弃 |
| `RaggedRoute-l3-eval-8d4c281` worktree | 5 个未跟踪 L3/vLLM 报告目录 | 与 main 的 L3 报告去重后决定归档或删除 |
| `RaggedRoute-vllm-industrial-correction` worktree | 2 个 modified 脚本、1 个未跟踪同步脚本 | 需要代码 review，不得随 stacked branch 清理 |
| `RaggedRoute-six-ops-v2-eval` worktree | PR #39 完成审计已交付；本地任务 ref 保留，随后只做合并后事实同步 | 不得混入其他 worktree 的实验内容；是否移除仍进入第二阶段 review |
| `RaggedRoute-unpermute-v2` 独立 clone | clean，旧 `codex/unpermute-sm86-v2` | 可在确认无独立 object/evidence 后删除候选 |
| `RaggedRoute-histogram-candidate` 普通目录 | 非 RaggedRoute Git worktree，含历史源码/报告/out | 先做文件级 manifest，再判断是否与 Git 历史重复 |
| `stash@{0}` | `histogram-v2 accidental shared-worktree edits` | 必须单独 review；禁止自动 drop |

第二阶段如果批准清理，必须先记录绝对路径、Git SHA、dirty diff 摘要和未跟踪文件 SHA-256；可保留内容先进入独立归档分支、tag 或不可覆盖的压缩证据包。

## 3. 文档与证据冗余候选

最新 `main` 的 `docs/reports/` 包含：398 个 `artifacts` 文件（13,768,659 B）、70 个 compact 文件（1,274,767 B）；下列项目只进入 review，不在第一阶段删除：

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
