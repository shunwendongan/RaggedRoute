# 实现状态与证据边界

更新时间：2026-08-03

## 已实现

- v0.2 公共运行时：完成一次不保留旧 `float*` shim 的 source-breaking 收口；浮点 payload 使用带 dtype/layout/element-strides 的 `ConstTensorView`/`MutableTensorView`，路由 metadata 保持强类型 int32；runtime 与 correctness 共用一套 `ScalarType`，`KernelSelection` 分离 family 与 operator-local implementation id；
- 可单测的 compute-capability 分派显式区分 SM86、SM90 与其他架构；当前可执行路径接受 `SM86 + 全 FP32 + zero-stride contiguous row-major`，`Auto` 对 Histogram 选择已晋升的 shape-dispatched optimized candidate，其余六算子选择 `cuda_naive implementation 0`；FP16/BF16 签名虽可表达但显式返回不支持，SM90 也不据交叉编译结果宣称实卡支持；
- 显式 research dispatch：Dense GEMM、Top-K Gate、Histogram 与 Token Permute 有已验证的 operator-local `kCudaOptimized` implementation id；显式可调用不等于 `Auto` 晋升。Grouped GEMM 与 Unpermute candidate 只在 benchmark adapter 中保留，Scan 失败 candidate 源码已从最终树删除；
- 两层 API：保留 `raggedroute::ops::launch_*_naive` 作为 L1 Kernel Entry；新增 `raggedroute::{dense_gemm, topk_gate, histogram, exclusive_scan, token_permute, grouped_gemm, unpermute}` 作为 L2/L3 Operator Wrapper。Wrapper 使用 caller stream，不在 hot path 分配/同步；Histogram 在 Wrapper 内清零 counts，Permute 使用 caller workspace（`E * sizeof(int32_t)`）并在 Wrapper 内清零 cursor；
- Benchmark 接入：L1 继续调用低层 launcher；L2 和两个 L3 chain 改为经过公开 Wrapper，架构查询在 setup 阶段缓存，不计入 event 计时；
- 公共 API correctness：覆盖七算子 role signature、SM86 FP32、尚未实现的 FP16/BF16、SM90、layout/stride、kernel family/id、参数/Workspace 拒绝、Dense GEMM、Histogram reset、Permute workspace reset 与 redzone；
- 模块化 CMake 3.24+：`RAGGEDROUTE_ENABLE_CUDA=OFF` 时不启用 CUDA language，保留 host-side schema tests；开启后才发现 `CUDAToolkit`；
- CMake presets：本地 RTX 3080 `sm_86`、H100 portable `sm_90`、H100 accelerated `sm_90a` 各自独立 Debug/Release 输出目录；
- Debug CUDA 使用 `-G`；Release CUDA 使用 `-lineinfo` 且不带 `-G`；默认不开 fast-math，RDC 默认关闭；
- CUDA runtime/cuBLAS、CUDA 13 bundled CCCL、外部 CCCL/CUTLASS 的 `AUTO|SYSTEM|FETCH|OFF` 发现策略；CCCL/CUTLASS Fetch 固定 tag；
- Windows wrapper 通过 `RAGGEDROUTE_VSDEVCMD`、`RAGGEDROUTE_VS_INSTALL_ROOT`、Developer Shell、`VSINSTALLDIR`、`VS2022_HOME`、`vswhere` 和标准安装目录动态解析 MSVC；configure 强制 fresh cache，build 检测到缺失或不同的 cached MSVC/CUDA compiler 时自动重配，避免 Visual Studio/CUDA Toolkit 移动后复用失效绝对路径；
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
- benchmark-only library/production variants：Dense GEMM `cublaslt`/`cublas`；Histogram `cub_device_histogram`；Scan `cub_device_scan`/`cub_block_scan`/`cub_warp_scan`；Permute `cuda_naive_from_ids`、`vllm_moe_permute` 与 prepared-mapping `vllm_expand_rows`；Grouped GEMM `cublas_per_expert`、可选 `cutlass_grouped` 及 SM86 strict-FP32 C0-C4/final 实验候选；Unpermute `vllm_finalize_routing`。它们不进入 v0.2 runtime dispatch；Grouped final 因十 shape CUTLASS 门禁失败而明确不晋级；Top-K 因 tie/NaN/selected-softmax 合同尚无语义等价库实现，明确不注册伪基线；
- 每个 `src/<operator>/library_baseline/` 均有来源记录；vLLM 固定 commit `837eae64580c885101ee95b073aafb27a485e7ce` 的改写源码保留 Apache-2.0，CUTLASS 改写入口保留 BSD-3-Clause；不提交 CUDA/cuBLAS 二进制；
- 新增 `configs/benchmark_library_smoke.json`，覆盖六个可严格配对的 library/production 边界；本机 SM86 Debug 已通过 raw JSONL → aggregate.v2 → comparison.v1 全链路及所有后置 reference validation。该 tiny Debug smoke 只验证接口和公平 join，不产生性能结论；
- correctness、benchmark smoke、release benchmark、Nsight profile 四个独立入口；
- RTX 3080（CUDA 13.3 / MSVC 19.50 / CMake 4.3.1）SM86 Debug/Release 均已在当前工具链迁移后重新构建，CTest 与 CUDA smoke 均已通过；CUTLASS v4.6.1 也在独立 Fetch Debug build 中编译并通过完整 CTest 和 grouped GEMM correctness；
- commit `e37c132` 的 RTX 3080 正式 baseline suite 已完成：3 个独立进程、78 条 raw records、26 个聚合组、全部后置验证通过；结果与噪声限制见 [baseline report](reports/rtx3080-naive-baseline-e37c132.md)；
- commit `a9489ab` 的 clean-Git RTX 3080 Release evidence 已完成：naive 78 条 raw/26 组、library reference 45 条 raw/15 组/9 个严格 pair，全部 validation 通过；完整 7 算子 NSYS、7 条 basic NCU、3 条 detailed hotspot NCU、normalized JSON/CSV 与 SHA256 bundle 见 [当前报告](reports/rtx3080-naive-profile-a9489ab.md)；
- profile v2 固定一个完整 7 算子 system case 与每算子一个 compute case；metric alias 对当前 NCU 实际名字解析，缺失值保留 `not_collected`/`unsupported_or_unknown`，原始 Nsight 二进制不进入 Git；
- H100 SM90 与 SM90a Debug/Release 均已完成本机交叉编译，分别检查为 `sm_90` 与 `sm_90a` cubin。

## 尚未实现，禁止据此宣称

- 通用的 failure artifact 自动重放、失败用例最小化与随机 GPU fuzz；当前 artifact 只保存和校验诊断信息；
- 可发布的 Grouped GEMM optimized runtime；现有 `cp.async`/persistent SM86 版本仅为 benchmark-only 失败实验；
- 与 Top-K tie/NaN/selected-softmax 合同相同的外部库基线；
- 真实 route trace、working-set rotation 与 distribution-aware shape sweep；当前十 shape synthetic suite 不是完整部署分布；
- Histogram 之外的 shape-aware default dispatch、自动 promotion evaluator；当前 Histogram 晋级以手工审计的固定门禁为依据，Grouped GEMM 未晋级结论也由配对报告人工审计；
- H100/Blackwell 实卡支持、正确性或性能；本机 SM90/SM90a 交叉编译不等同于 H100 验证；
- 完整 MoE FFN、训练、多 GPU 或 All-to-All。

因此当前提交仍不是“七个算子已经优化完成”。Grouped GEMM 候选会保留 clean-Git、同机同语义的 Release 证据，但因相对 CUTLASS 的门禁失败而不发布 runtime optimized 路径；后续性能声明仍必须来自相同协议的未 profile A/B。

候选 variant 评估闭环、trace/working-set workload 与图表的后续实施顺序见 [development-roadmap.md](development-roadmap.md)。未勾选项不属于上方“已实现”事实。
