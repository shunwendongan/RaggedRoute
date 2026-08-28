# 分支管理约定

RaggedRoute 使用短生命周期分支和单目标 PR。默认分支为 `main`，开发分支统一使用 `codex/<area>-<task>`，例如 `codex/grouped-gemm-profile-docs`。

## 生命周期

1. 从最新 `main` 创建一个只承载单一目标的分支。
2. 在 PR 中同时提交实现、验证命令、性能证据边界和失败结论。
3. CI 与 review 通过后合入 `main`；GitHub 自动删除已合并远端分支。
4. evidence tag、未合并研究分支和含未归档证据的分支不得按名称自动删除。
5. 本地 worktree、stash 和未跟踪文件不属于远端自动清理范围，必须经过 `docs/cleanup-review.md` 的人工 review。

## 删除门禁

删除远端分支前必须重新确认：

- live tip SHA 与审计清单一致；
- 关联 PR 的 `merged_at` 非空；
- 分支没有 open PR；
- 分支不在 evidence tag、保留分支或当前工作分支允许列表；
- 删除不会触碰本地 dirty worktree、stash 或未跟踪 evidence。

私有仓库当前套餐不提供 branch protection/ruleset。替代门禁为 PR review、GitHub Actions、`scripts/repository_checks.py`、Python tests、CTest，以及与变更风险相称的 CUDA correctness/performance evidence。
