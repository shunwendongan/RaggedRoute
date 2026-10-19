# RaggedRoute 本地工具链与运行环境核查

本次仅执行配置读取、依赖检查、版本命令及无 GPU 工作负载的 `--list`。没有安装软件、修改用户/系统环境变量或执行 GPU 正确性/性能测试。

## 已确认的版本与实际来源

| 项目 | 实际版本/来源 | 旧构建与新构建关系 |
|---|---|---|
| MSVC | `C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe`；编译器 `19.44.35226.0` | 实际检测版本一致；新构建最初依赖 PATH 选择 host compiler，修正后已明确固定绝对路径 |
| CMake | `D:/CMake/cmake-4.3.1-windows-x86_64/bin/cmake.exe`；4.3.1 | 一致 |
| Ninja | 原缓存为 MSYS2 1.13.2；修正后由 VS Developer Shell 选择其自带 Ninja，绝对路径见新 CMakeCache | 仅构建调度工具发生变化；编译器、flags、目标架构保持一致 |
| CUDA nvcc | `D:/Dev/toolchains/CUDA/v13.3/bin/nvcc.exe`；13.3.73 | 一致 |
| CUDA SDK / runtime | 安装目录 `version.json`：SDK 13.3.1，cudart 13.3.29 | 一致；组件版本和编译器版本是不同字段 |
| cuBLAS | 同一 Toolkit 的 `libcublas` 13.6.0.2；两个 `bin/x64` DLL 均校验 SHA256 | 一致 |
| CCCL / CUB / Thrust | 实际头文件来自 `D:/Dev/toolchains/CUDA/v13.3/include/cccl`，版本均为 **3.3.4** | 旧缓存写 FETCH、新缓存写 AUTO/SYSTEM，但最终实际包含同一套已安装头文件；不能将请求的 v3.4.0 当成有效版本 |
| CUTLASS | `v4.6.1`，提交 `e05f953a5b3d38adc240df2ff928e0421c2abba3`；复用原 `_deps/rr_cutlass-src/include` | 同一实际头文件；checkout 的 dirty 项仅为一个已删除的上游 example，未发现头文件修改 |
| NCU | `C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe`；2026.2.1.0，build 38283040 | 初始 PATH 中的 `ncu.bat` 为此 executable 的包装器 |
| NSYS | `C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe`；2026.1.3.425-261338342291v0 | 版本命令成功 |
| Compute Sanitizer | `D:/Dev/toolchains/CUDA/v13.3/compute-sanitizer/compute-sanitizer.exe`；2026.2.1.0，build 38334959 | 文件存在且版本命令成功，初始 PATH 未包含此目录 |
| cuobjdump / nvdisasm | 同一 Toolkit 的 `bin`；13.3.73 | 版本命令成功 |

两套构建均为 `Release`、`86-real`、C++17、MSVC 动态运行库。实际 GPU 编译命令使用 `-O3 -lineinfo`，未启用 `--use_fast_math`。依赖的关键路径和逐工具版本命令完整输出保存在 `toolchain-audit-agent.json`。

## 故障 1：默认 Visual Studio 发现混用了两个安装

可重复命令 `cmd /d /c call scripts\setup_msvc_env.bat` 在原 shell 中返回 **3**；显式指定 C 盘 BuildTools 的 `RAGGEDROUTE_VSDEVCMD` 后返回 **0**。

- 用户和机器环境均设置 `VS2022_HOME=D:\VisualStudio\Community`，该目录下的 VsDevCmd 不存在。
- 原脚本的父目录回退命中 `D:\VisualStudio\Common7\Tools\VsDevCmd.bat`，实际属于 **Visual Studio 2026 v18.0**。
- 用户环境又固定 `VCToolsVersion=14.44.35207` 和 C 盘 `VCToolsInstallDir`。D 盘安装只有 `14.50.35717`。
- 直接运行 D 盘 VsDevCmd 的首个明确错误为：`Version '14.44.35207' is not valid; directory does not exist`。

证据：`toolchain-env-scopes.json`、`setup-msvc-default-before.log`、`vsdevcmd-stale-root.log`、`setup-msvc-explicit-before.log`。当前配置已经固定使用经过验证的 C 盘 14.44，未调整全局环境。

## 故障 2：CUDA 与诊断工具发现遗漏已有环境变量

初始进程具有 `CUDA_TOOLKIT_ROOT_DIR`、`CUDA_BIN_PATH` 和 `CUDA_PATH_V13_3`，但缺少脚本原先查找的 `CUDA_PATH/CUDACXX` 及 nvcc PATH 项。因此实际存在的 CUDA 13.3 会在普通 shell 中被误判为不可用；Sanitizer 也位于独立子目录，不能只查 `bin`。

主代理已将配置/构建脚本及 profiler/sanitizer 发现逻辑调整到同一已选择 Toolkit，并对新的标准脚本入口执行重新配置与构建。审计快照已确认 `CMAKE_CUDA_HOST_COMPILER` 现在为 C 盘 MSVC 的绝对路径。最终标准脚本构建成功（exit 0）记录在 `out/tensorcore/environment/build-audited-final.log`；程序 `--list` 也已返回 0。

## 故障 3：Codex 的 MSIX 子进程 DLL 搜索不读取 PATH

旧、新 exe 均曾在进入 main 前静默返回 `-1073741515/0xc0000135`。cuBLAS DLL 文件完整存在，通过绝对路径 `LoadLibraryExW` 可加载，但将 CUDA `bin/x64` 加到 PATH 或改 CWD 均无效。

通过只修改诊断子进程内存启用 loader snaps，定位到真实搜索集合：Codex package 目录、exe 目录、`C:\WINDOWS\SYSTEM32`，其中没有 PATH/CWD。`loader-snaps-before.log:727` 给出搜索路径；843、869 行明确报告 `cublas64_13.dll/cublasLt64_13.dll` 未找到。

已将两个匹配 Toolkit 的 DLL 复制到旧、新任务构建目录并校验 SHA256。两套 benchmark `--list` 均恢复 **exit 0**。复制路径及哈希在 `task-local-dll-copies.json`；主代理已增加构建后的本地 DLL staging，以便后续 clean build 保持可执行。没有复制到系统目录。

## 本轮调整原则及剩余验收

- 继续使用原构建已验证的 MSVC 14.44/CUDA 13.3/相同 CCCL 与 CUTLASS 头文件，避免在 Tensor Core 实现同时切换编译器。
- 脚本针对实际可执行的 compiler/Toolkit 验证并选择；诊断工具使用绝对路径或所选 Toolkit 派生路径。
- 报告同时记录“请求 provider/tag”和“有效头文件版本”，保留现有有效依赖；无需为标签差异下载或更换 CCCL。
- 不修改已发现的用户环境变量，也不修复无关的 CUTLASS example 删除项。
- 本审计确认配置、路径、版本和 DLL loader 问题已定位；主代理完成标准脚本构建与 `--list` 后再恢复正确性、Sanitizer 和性能阶段。工具版本命令成功不能替代 GPU 执行验证。

## 核查后的执行确认

- 2026-10-19：修正后从 fresh cache 使用标准 configure/build 脚本成功；DLL staging 已自动执行。
- GPU 程序运行记录实际返回 `cuda_runtime=13040`、`cuda_driver=13040`。它与 `version.json` 中组件标签分别保留，不将目录名或组件清单等同于运行时查询值。
- 原仓库正确性复现通过；新分支 CTest 9/9 通过；Tensor Core 独立完整测试 518 case 通过。
- 初次 CTest 的唯一失败是 registry 数量旧断言（8）；新增两算子后更新为 10，重跑全套通过。
