# 实现状态与证据边界

更新时间：2026-08-09
事实基线：`main@354e1ff0a3aaf602830c8d989eb358cfa857a4c9`。PR #31/#32 合并到实验分支而非 `main`，不计入下列当前能力。

## 已实现

- v0.2 公共运行时：完成一次不保留旧 `float*` shim 的 source-breaking 收口；浮点 payload 使用带 dtype/layout/element-strides 的 `ConstTensorView`/`MutableTensorView`，路由 metadata 保持强类型 int32；runtime 与 correctness 共用一套 `ScalarType`，`KernelSelection` 分离 family 与 operator-local implementation id；
- 可单测的 compute-capability 分派显式区分 SM86、SM90 与其他架构；当前可执行路径接受 `SM86 + 全 FP32 + zero-stride contiguous row-major`。七个语义算子中，`Auto` 仅对 Histogram 选择已通过门禁的 shape-dispatched optimized candidate，其余六个选择 `cuda_naive implementation 0`；另一个公开融合 primitive `HistogramExclusiveScan` 的 `Auto` 当前选择 F2，但其 release 证据仍须独占 GPU 环境复测。FP16/BF16 签名虽可表达但显式返回不支持，SM90 也不据交叉编译结果宣称实卡支持；
- 显式 research dispatch：Dense GEMM、Top-K Gate、Histogram、HistogramExclusiveScan 与 Token Permute 有 operator-local `kCudaOptimized` implementation id；显式可调用不等于默认晋升。Grouped GEMM 与 Unpermute candidate 只在 benchmark adapter 中保留，standalone Scan 失败 candidate 源码已从最终树删除；
- 两层 API：保留 `raggedroute::ops::launch_*_naive` 作为 L1 Kernel Entry；公开 L2/L3 wrapper 包含 `dense_gemm`、`topk_gate`、`histogram`、`exclusive_scan`、`histogram_exclusive_scan`、`token_permute`、`grouped_gemm` 与 `unpermute`。Wrapper 使用 caller stream，不在 hot path 分配/同步；Histogram 在 Wrapper 内清零 counts，Permute 使用 caller workspace（`E * sizeof(int32_t)`）并在 Wrapper 内清零 cursor；
- Benchmark 接入：L1 继续调用低层 launcher；L2 和两个 L3 chain 改为经过公开 Wrapper，架构查询在 setup 阶段缓存，不计入 event 计时；
- 公共 API correctness：覆盖七算子与融合 primitive 的 role signature、SM86 FP32、尚未实现的 FP16/BF16、SM90、layout/stride、kernel family/id、参数/Workspace 拒绝、Dense GEMM、Histogram/Fused reset、Permute workspace reset 与 redzone；
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
- 一个公共 CUDA Event runner 和八个 typed adapter（七个语义算子加 `histogram_exclusive_scan`）；
- `chain_from_tokens` 完整 7 算子 L3 与 `chain_from_logits` 6 算子 L3；
- L1/L2 reset 成本边界、状态型 repeat policy、raw JSONL 和聚合 JSON/CSV；
- `raggedroute.suite.v2` 多 variant logical case：每个 case 至少两个唯一 variant、恰好一个 `promotion_baseline`，并复用 case ID、seed、params、level、cache 与采样协议；suite v1 与 `raggedroute.benchmark.v1` raw evidence 保持兼容；
- registry 为 typed adapter 注入标准实现元数据；`raggedroute.aggregate.v2` 从 v2 manifest 保留完整配对字段，`raggedroute.comparison.v1` 对 GPU/build/语义/math/seed/level/cache/repeats/排除项严格 fail-closed，并输出 per-pair speedup、shape geometric mean 和具备权重时的 trace ratio-of-sums；
- `raggedroute.route_trace.v1`、normalized `.rrtrace`、frame working-set 展开和 trace SHA-256 provenance；Token Permute、Grouped GEMM 与 `chain_from_logits` 可消费固定 frame，`chain_from_tokens` 明确拒绝该研究输入；
- `raggedroute.promotion_decision.v1` 三态 evaluator：`promote`、`reject`、`insufficient_evidence`，检查 Release/clean SHA、correctness、GPU/case/process 配对、CV、coverage、ratio-of-sums、p50/p95 回退、workspace 与真实 trace provenance；
- Grouped GEMM v3、Permute v3 与六阶段 research-v3 显式路径，以及由 decision JSON 生成的 compact report、CSV、shape heatmap、manifest、`SHA256SUMS` 和 deterministic ZIP；这些研究路径不修改 `KernelFamily::kAuto`；
- benchmark-only library/production variants：Dense GEMM `cublaslt`/`cublas`；Histogram `cub_device_histogram`；Scan `cub_device_scan`/`cub_block_scan`/`cub_warp_scan`；Permute `cuda_naive_from_ids`、`vllm_moe_permute` 与 prepared-mapping `vllm_expand_rows`；Grouped GEMM `cublas_per_expert`、可选 `cutlass_grouped` 及 SM86 strict-FP32 C0-C4/final 实验候选；Unpermute `vllm_finalize_routing`。它们不进入 v0.2 runtime dispatch；Grouped final 因十 shape CUTLASS 门禁失败而明确不晋级；Top-K 因 tie/NaN/selected-softmax 合同尚无语义等价库实现，明确不注册伪基线；
- 每个 `src/<operator>/library_baseline/` 均有来源记录；vLLM 固定 commit `837eae64580c885101ee95b073aafb27a485e7ce` 的改写源码保留 Apache-2.0，CUTLASS 改写入口保留 BSD-3-Clause；不提交 CUDA/cuBLAS 二进制；
- 新增 `configs/project/benchmark/library_smoke.json`，覆盖六个可严格配对的 library/production 边界；本机 SM86 Debug 已通过 raw JSONL → aggregate.v2 → comparison.v1 全链路及所有后置 reference validation。该 tiny Debug smoke 只验证接口和公平 join，不产生性能结论；
- correctness、benchmark smoke、release benchmark、Nsight profile 四个独立入口；
- RTX 3080（CUDA 13.3 / MSVC 19.50 / CMake 4.3.1）SM86 Debug/Release 均已在当前工具链迁移后重新构建，CTest 与 CUDA smoke 均已通过；CUTLASS v4.6.1 也在独立 Fetch Debug build 中编译并通过完整 CTest 和 grouped GEMM correctness；
- commit `e37c132` 的 RTX 3080 正式 baseline suite 已完成：3 个独立进程、78 条 raw records、26 个聚合组、全部后置验证通过；结果与噪声限制见 [baseline report](reports/rtx3080-naive-baseline-e37c132.md)；
- commit `a9489ab` 的 clean-Git RTX 3080 Release evidence 已完成：naive 78 条 raw/26 组、library reference 45 条 raw/15 组/9 个严格 pair，全部 validation 通过；完整 7 算子 NSYS、7 条 basic NCU、3 条 detailed hotspot NCU、normalized JSON/CSV 与 SHA256 bundle 见 [当前报告](reports/rtx3080-naive-profile-a9489ab.md)；
- profile v2 固定一个完整 7 算子 system case 与每算子一个 compute case；metric alias 对当前 NCU 实际名字解析，缺失值保留 `not_collected`/`unsupported_or_unknown`。v2 evidence bundle 的既定策略是原始 Nsight 二进制进入不可覆盖 Release 资产；`docs/reports/l3_three_way_20260805/` 当前仍包含少量 `.ncu-rep`/`.nsys-rep`，属于待清理的策略漂移，不应作为新提交范例；
- H100 SM90 与 SM90a Debug/Release 均已完成交叉编译，分别检查为 `sm_90` 与 `sm_90a` cubin；这不是 H100 实卡支持；
- 2026-08-04：新增 exact-int32 `HistogramExclusiveScanArgs` 公共 API；F2 单 CTA
  SM86 fused Histogram→Scan 放入 `src/scan/cuda_candidate`，`R<=4096` 由 Auto
  dispatch 选择，较大 R 回退现有两阶段路径。该实现已经进入 `main`，但其正式运行
  受到并发 GPU workload/overlay 干扰，CV、fallback 与 L3 门禁均失败；源码存在不等于
  生产级晋级，后续必须在独占 CUDA 环境复测或撤回默认选择；
- 2026-08-04：Top-K v4（implementation id 4）完成 exact-E 与连续 bucket 门禁，零区间通过，`Auto` 保持 naive；
- 2026-08-05：Unpermute v2 screening 相对已有 V1 退化，未保留新 kernel；已有 warp/CTA 路径继续为 benchmark-only；
- 2026-08-05：Token Permute v2 作为显式研究路径保留，full-from-ids 的五 case 中心结果相对旧 candidate/vLLM 更快，但 pure-permute 28-case 存在全配对高 CV、覆盖不足和最差 shape 大幅退化，因此 `Auto` 保持 naive；
- 2026-08-05：合入单一固定 workload 的七阶段 L3 三线路诊断报告。Selected CUDA research chain p50 为 `69.734 us`，Triton reference 为 `245.760 us`，但跨工具链比值不具 promotion 资格；该链中 Grouped GEMM 占 NSYS kernel time `64.4%`。报告见 [L3 three-way analysis](reports/l3_three_way_20260805/RaggedRoute_L3_3way_comparison.md)；
- 2026-08-05：`main@354e1ff` 的 GitHub CI 通过 repository checks、Python tests 与 Windows/Linux CPU-only build/CTest。当前 macOS 工作区只执行 CPU-only/文档检查，不做 CUDA 能力或性能复测。

## 尚未实现，禁止据此宣称

- 通用的 failure artifact 自动重放、失败用例最小化与随机 GPU fuzz；当前 artifact 只保存和校验诊断信息；
- 可发布的 Grouped GEMM optimized runtime；现有 `cp.async`/persistent SM86 版本仅为 benchmark-only 失败实验；
- 与 Top-K tie/NaN/selected-softmax 合同相同的外部库基线；
- 匿名 captured/production route trace、通用 working-set rotation 与 distribution-aware cache sweep；当前 synthetic fixture 只验证工具链，不是部署分布证据；
- 除 Histogram 与独立 fused Histogram→Scan primitive 外的 shape-aware default dispatch；自动 evaluator 已实现，但当前 v3 因 WDDM CV 超限为 `insufficient_evidence`，且 aggregate 趋势不支持晋级；
- H100/Blackwell 实卡支持、正确性或性能；本机 SM90/SM90a 交叉编译不等同于 H100 验证；
- 完整 MoE FFN、训练、多 GPU 或 All-to-All。

因此当前提交仍不是“七个算子已经优化完成”。Grouped GEMM 候选会保留 clean-Git、同机同语义的 Release 证据，但因相对 CUTLASS 的门禁失败而不发布 runtime optimized 路径；后续性能声明仍必须来自相同协议的未 profile A/B。

候选 variant 评估闭环、trace/working-set workload 与图表的后续实施顺序见 [development-roadmap.md](development-roadmap.md)。未勾选项不属于上方“已实现”事实。
