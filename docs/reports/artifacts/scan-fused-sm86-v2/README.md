# Scan Fused SM86 v2 Evidence Bundle

这是本轮提交的 compact evidence index。大型 JSONL、`.nsys-rep`、`.ncu-rep`、
SQLite 和原始 sanitizer 输出保存在本地 `out/` raw archive；本目录只保存可审计
摘要、命令和 hash，避免超过仓库 5 MiB 发布限制。

`decision` 是“按用户要求提交 F2 的 draft candidate”，不是“严格性能门禁通过”。
正式数据受到并行 TopK campaign 和 Intel Graphics Overlay 干扰，CV/L3 gate 结果
必须在独占 GPU 上复测。
