# 实现状态与证据边界

更新时间：2026-07-30

## 已实现

- v0.2 公共运行时：完成一次不保留旧 `float*` shim 的 source-breaking 收口；浮点 payload 使用带 dtype/layout/element-strides 的 `ConstTensorView`/`MutableTensorView`，路由 metadata 保持强类型 int32；runtime 与 correctness 共用一套 `ScalarType`，`KernelSelection` 分离 family 与 operator-local implementation id；
- 可单测的 compute-capability 分派显式区分 SM86、SM90 与其他架构；当前可执行路径只接受 `SM86 + 全 FP32 + zero-stride contiguous row-major + cuda_naive implementation 0`，FP16/BF16 签名虽可表达但显式返回不支持，SM90 也不据交叉编译结果宣称实卡支持；
- 两层 API：保留 `raggedroute::ops::launch_*_naive` 作为 L1 Kernel Entry；新增 `raggedroute::{dense_gemm, topk_gate, histogram, exclusive_scan, token_permute, grouped_gemm, unpermute}` 作为 L2/L3 Operator Wrapper。Wrapper 使用 caller stream，不在 hot path 分配/同步；Histogram 在 Wrapper 内清零 counts，Permute 使用 caller workspace（`E * sizeof(int32_t)`）并在 Wrapper 内清零 cursor；
- Benchmark 接入：L1 继续调用低层 launcher；L2 和两个 L3 chain 改为经过公开 Wrapper，架构查询在 setup 阶段缓存，不计入 event 计时；
- 公共 API correctness：覆盖七算子 role signature、SM86 FP32、尚未实现的 FP16/BF16、SM90、layout/stride、kernel family/id、参数/Workspace 拒绝、Dense GEMM、Histogram reset、Permute workspace reset 与 redzone；
- 模块化 CMake 3.24+：`RAGGEDROUTE_ENABLE_CUDA=OFF` 时不启用 CUDA language，保留 host-side schema tests；开启后才发现 `CUDAToolkit`；
- CMake presets：本地 RTX 3080 `sm_86`、H100 portable `sm_90`、H100 accelerated `sm_90a` 各自独立 Debug/Release 输出目录；
- Debug CUDA 使用 `-G`；Release CUDA 使用 `-lineinfo` 且不带 `-G`；默认不开 fast-math，RDC 默认关闭；
- CUDA runtime/cuBLAS、CUDA 13 bundled CCCL、外部 CCCL/CUTLASS 的 `AUTO|SYSTEM|FETCH|OFF` 发现策略；CCCL/CUTLASS Fetch 固定 tag；
- CUDA 静态库的 install/export 与 `find_package(RaggedRoute)` package config；
- correctness framework 的 FP32/FP16/BF16/低精度 capability metadata、dtype roundtrip launcher 与 guarded-buffer 工具；低精度能力仍须按实卡等级区分；
- correctness framework 独立 CTest suite：dtype host/runtime roundtrip、reference invariants、failure artifact 序列化 roundtrip、redzone、zero-size、随机与 caller-stream 合同；当前不提供可执行的 failure replay；
- 七个算子的 CPU oracle 实现分别归属 `src/<operator>/cpu_reference/reference.cpp`，统一 correctness facade 与既有 API 不变；浮点 reference 使用 `double`，路由 metadata 采用精确整数校验；
- correctness framework 的详细 dtype capability 边界、算子合同、failure JSON、sanitizer 和可选 `sm_90a`/`sm_100a` compile-only probe 见 [correctness-framework.md](correctness-framework.md)；其中跨架构编译不等同于 H100/Blackwell 实卡验证；
- Compute Sanitizer 默认执行七个非零尺寸 L2 operator 和两条 L3 chain，覆盖 memcheck、initcheck、racecheck 与 synccheck；
- 七个 FP32 naive CUDA launcher，均使用 caller stream，hot path 无分配和无条件同步；
- 一个公共 CUDA Event runner 和七个 typed adapter；
- `chain_from_tokens` 完整 7 算子 L3 与 `chain_from_logits` 6 算子 L3；
- L1/L2 reset 成本边界、状态型 repeat policy、raw JSONL 和聚合 JSON/CSV；
- `raggedroute.suite.v2` 多 variant logical case：每个 case 至少两个唯一 variant、恰好一个 `promotion_baseline`，并复用 case ID、seed、params、level、cache 与采样协议；suite v1 与 `raggedroute.benchmark.v1` raw evidence 保持兼容；
- registry 为 typed adapter 注入标准实现元数据；`raggedroute.aggregate.v2` 从 v2 manifest 保留完整配对字段，`raggedroute.comparison.v1` 对 GPU/build/语义/math/seed/level/cache/repeats/排除项严格 fail-closed，并输出 per-pair speedup、shape geometric mean 和具备权重时的 trace ratio-of-sums；
- correctness、benchmark smoke、release benchmark、Nsight profile 四个独立入口；
- RTX 3080（CUDA 13.3 / MSVC 19.44 / CMake 4.3.1）SM86 Debug/Release 均已构建；CTest、CUDA smoke 均已通过；Release binary 检查为 `sm_86`；
- commit `e37c132` 的 RTX 3080 正式 baseline suite 已完成：3 个独立进程、78 条 raw records、26 个聚合组、全部后置验证通过；结果与噪声限制见 [baseline report](reports/rtx3080-naive-baseline-e37c132.md)；
- H100 SM90 与 SM90a Debug/Release 均已完成本机交叉编译，分别检查为 `sm_90` 与 `sm_90a` cubin。

## 尚未实现，禁止据此宣称

- 通用的 failure artifact 自动重放、失败用例最小化与随机 GPU fuzz；当前 artifact 只保存和校验诊断信息；
- FP16/Tensor Core、`cp.async`、persistent grouped scheduler 等优化版本；
- cuBLAS/CUTLASS/CUB 强性能基线；
- shape-aware default dispatch、promotion evaluator 和实际 library/optimized 候选数据；现有 comparison 只计算严格配对结果，`configs/benchmark_promotion_policy.json` 仍不会自动产生晋升结论；
- H100/Blackwell 实卡支持、正确性或性能；本机 SM90/SM90a 交叉编译不等同于 H100 验证；
- 完整 MoE FFN、训练、多 GPU 或 All-to-All。

因此当前提交是 benchmark 基础设施和 naive baseline milestone，不是“七个算子已经优化完成”。任何正式 speedup 必须等候选 variant 与同机同语义 performance baseline 接入后再生成。

候选 variant 评估闭环、trace/working-set workload、profile metrics、图表与 release bundle 的后续实施顺序见 [development-roadmap.md](development-roadmap.md)。该文档全部是计划，不属于上方“已实现”事实。
