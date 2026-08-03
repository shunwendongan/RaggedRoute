# RTX 3080 Unpermute research candidate 证据包

结论：不晋升；保留 benchmark-only research implementation。

详见[正式报告](../../unpermute-sm86-candidate-eba8f02.md)。正式 benchmark 证据位于
`benchmark/formal-run1`、`benchmark/formal-rerun` 和 `benchmark/chain`；最终验证位于
`benchmark/validation`；标准化 profiler 与 SASS 证据位于 `profile`。

每个正式目录还包含 72 行 `throughput-comparison.csv`，字段包括 p50/p95、CV、logical bytes、
effective GB/s、tokens/s 和 elements/s。吞吐字段由同一个未插桩 p50 与固定工作量推导。

Profiler duration 仅用于诊断，Release comparison 才是晋升依据。证据包不包含 `.ncu-rep`、
`.nsys-rep`、SQLite、可执行文件、库文件或构建产物。
