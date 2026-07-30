# 实现状态与证据边界

更新时间：2026-07-29

## 已实现

- P0 公共运行时：无堆分配的 `Status`、公共 shape/dtype/layout/variant 类型、caller-stream `RuntimeContext` 与可单测的 compute-capability→SM86 分派；当前运行时只接受 `SM86 + FP32 + contiguous row-major + cuda_naive`，其他架构/精度/layout/variant 均显式返回不支持，避免把未实测路径包装成“兼容”；
- 两层 API：保留 `raggedroute::ops::launch_*_naive` 作为 L1 Kernel Entry；新增 `raggedroute::{dense_gemm, topk_gate, histogram, exclusive_scan, token_permute, grouped_gemm, unpermute}` 作为 L2/L3 Operator Wrapper。Wrapper 使用 caller stream，不在 hot path 分配/同步；Histogram 在 Wrapper 内清零 counts，Permute 使用 caller workspace（`E * sizeof(int32_t)`）并在 Wrapper 内清零 cursor；
- Benchmark 接入：L1 继续调用低层 launcher；L2 和两个 L3 chain 改为经过公开 Wrapper，架构查询在 setup 阶段缓存，不计入 event 计时；
- 公共 API correctness：新增纯 dispatch、参数/Workspace 拒绝、Dense GEMM、Histogram reset、Permute workspace reset 与 redzone 覆盖；
- 模块化 CMake 3.24+：`RAGGEDROUTE_ENABLE_CUDA=OFF` 时不启用 CUDA language，保留 host-side schema tests；开启后才发现 `CUDAToolkit`；
- CMake presets：本地 RTX 3080 `sm_86`、H100 portable `sm_90`、H100 accelerated `sm_90a` 各自独立 Debug/Release 输出目录；
- Debug CUDA 使用 `-G`；Release CUDA 使用 `-lineinfo` 且不带 `-G`；默认不开 fast-math，RDC 默认关闭；
- CUDA runtime/cuBLAS、CUDA 13 bundled CCCL、外部 CCCL/CUTLASS 的 `AUTO|SYSTEM|FETCH|OFF` 发现策略；CCCL/CUTLASS Fetch 固定 tag；
- CUDA 静态库的 install/export 与 `find_package(RaggedRoute)` package config；
- correctness framework 的 FP32/FP16/BF16/低精度 capability metadata、dtype roundtrip launcher 与 guarded-buffer 工具；低精度能力仍须按实卡等级区分；
- correctness framework 独立 CTest suite：dtype host/runtime roundtrip、reference invariants、failure/replay、redzone、zero-size、随机与 caller-stream 合同；
- correctness framework 的详细 dtype capability 边界、算子合同、failure JSON、sanitizer 和可选 `sm_90a`/`sm_100a` compile-only probe 见 [correctness-framework.md](correctness-framework.md)；其中跨架构编译不等同于 H100/Blackwell 实卡验证；
- 七个 FP32 naive CUDA launcher，均使用 caller stream，hot path 无分配和无条件同步；
- 一个公共 CUDA Event runner 和七个 typed adapter；
- `chain_from_tokens` 完整 7 算子 L3 与 `chain_from_logits` 6 算子 L3；
- L1/L2 reset 成本边界、状态型 repeat policy、raw JSONL 和聚合 JSON/CSV；
- correctness、benchmark smoke、release benchmark、Nsight profile 四个独立入口；
- RTX 3080（CUDA 13.3 / MSVC 19.44 / CMake 4.3.1）SM86 Debug/Release 均已构建；CTest、CUDA smoke 均已通过；Release binary 检查为 `sm_86`；
- commit `e37c132` 的 RTX 3080 正式 baseline suite 已完成：3 个独立进程、78 条 raw records、26 个聚合组、全部后置验证通过；结果与噪声限制见 [baseline report](reports/rtx3080-naive-baseline-e37c132.md)；
- H100 SM90 与 SM90a Debug/Release 均已完成本机交叉编译，分别检查为 `sm_90` 与 `sm_90a` cubin。

## 尚未实现，禁止据此宣称

- 本次 P0 runtime/API 改动尚未在本轮 Windows + RTX 3080（SM86）环境重新构建、运行 CTest 与 benchmark smoke；本机 macOS 仅完成 host-side 配置和脚本测试，不能代替 CUDA 验证；
- FP16/Tensor Core、`cp.async`、persistent grouped scheduler 等优化版本；
- cuBLAS/CUTLASS/CUB 强性能基线；
- shape-aware default dispatch 和 promotion policy 的实际候选数据；
- H100/Blackwell 实卡支持、正确性或性能；本机 SM90/SM90a 交叉编译不等同于 H100 验证；
- 完整 MoE FFN、训练、多 GPU 或 All-to-All。

因此当前提交是 benchmark 基础设施和 naive baseline milestone，不是“七个算子已经优化完成”。任何正式 speedup 必须等候选 variant 与同机同语义 performance baseline 接入后再生成。
