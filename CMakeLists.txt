cmake_minimum_required(VERSION 3.24)

project(RaggedRoute VERSION 0.2.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RaggedRouteOptions)
include(RaggedRouteCuda)
include(RaggedRouteDependencies)
include(RaggedRouteCompilerOptions)

raggedroute_initialize_options()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  message(STATUS "RaggedRoute: compile_commands.json export enabled")
endif()

if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
      CACHE STRING "MSVC runtime library" FORCE)
endif()

if(RAGGEDROUTE_ENABLE_CUDA)
  raggedroute_enable_cuda()
  raggedroute_discover_cuda_dependencies()
else()
  message(STATUS "RaggedRoute: CUDA disabled; only host-side schema tests are available")
endif()

raggedroute_configure_compiler_options()

add_library(raggedroute_build_config INTERFACE)
add_library(RaggedRoute::build_config ALIAS raggedroute_build_config)
target_compile_features(raggedroute_build_config INTERFACE cxx_std_17)
target_compile_definitions(raggedroute_build_config INTERFACE
  RAGGEDROUTE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  RAGGEDROUTE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  RAGGEDROUTE_VERSION_PATCH=${PROJECT_VERSION_PATCH})

if(RAGGEDROUTE_ENABLE_CUDA)
  execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE RAGGEDROUTE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT RAGGEDROUTE_GIT_SHA)
    set(RAGGEDROUTE_GIT_SHA "unknown")
  endif()
  execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE RAGGEDROUTE_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(RAGGEDROUTE_GIT_STATUS)
    set(RAGGEDROUTE_GIT_DIRTY "true")
  else()
    set(RAGGEDROUTE_GIT_DIRTY "false")
  endif()

  add_library(raggedroute_baseline_ops STATIC
    src/benchmark/cache_scrub.cu
    src/dense_gemm/cuda_naive/baseline.cu
    src/dense_gemm/cuda_candidate/optimized.cu
    src/dense_gemm/cuda_candidate/optimized_v2.cu
    src/dense_gemm/cuda_candidate/optimized_v3.cu
    src/topk_gate/cuda_naive/baseline.cu
    src/topk_gate/cuda_candidate/optimized.cu
    src/histogram/cuda_naive/baseline.cu
    src/histogram/cuda_candidate/optimized.cu
    src/scan/cuda_naive/baseline.cu
    src/scan/cuda_candidate/optimized.cu
    src/permute/cuda_naive/baseline.cu
    src/permute/cuda_candidate/optimized.cu
    src/grouped_gemm/cuda_naive/baseline.cu
    src/grouped_gemm/cuda_candidate/optimized.cu
    src/grouped_gemm/cuda_candidate/optimized_register.cu
    src/grouped_gemm/cuda_candidate/optimized_descriptor.cu
    src/grouped_gemm/cuda_candidate/optimized_wide.cu
    src/unpermute/cuda_naive/baseline.cu
    src/unpermute/cuda_candidate/optimized.cu)
  add_library(RaggedRoute::baseline_ops ALIAS raggedroute_baseline_ops)
  target_include_directories(raggedroute_baseline_ops PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_compile_features(raggedroute_baseline_ops PUBLIC cxx_std_17)
  target_link_libraries(raggedroute_baseline_ops PUBLIC CUDA::cudart)
  set_target_properties(raggedroute_baseline_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${RAGGEDROUTE_ENABLE_RDC}
    EXPORT_NAME baseline_ops
    POSITION_INDEPENDENT_CODE ON)
  raggedroute_apply_target_settings(raggedroute_baseline_ops)

  # Benchmark-only production/library baselines. These never participate in
  # the public runtime dispatch or install/export surface.
  set(_raggedroute_library_baseline_sources
    src/unpermute/library_baseline/vllm_unpermute.cu
    src/topk_gate/library_baseline/vllm_row_packed_top2.cu)
  set(_raggedroute_library_baseline_libraries CUDA::cudart)
  set(_raggedroute_has_cublas 0)
  set(_raggedroute_has_cccl 0)
  set(_raggedroute_has_cutlass 0)
  if(RAGGEDROUTE_ENABLE_CUBLAS AND TARGET CUDA::cublas AND TARGET CUDA::cublasLt)
    list(APPEND _raggedroute_library_baseline_sources
      src/dense_gemm/library_baseline/cublas_gemm.cpp
      src/grouped_gemm/library_baseline/cublas_grouped.cpp)
    list(APPEND _raggedroute_library_baseline_libraries CUDA::cublas CUDA::cublasLt)
    set(_raggedroute_has_cublas 1)
  endif()
  if(TARGET RaggedRoute::cccl)
    list(APPEND _raggedroute_library_baseline_sources
      src/histogram/library_baseline/cub_histogram.cu
      src/scan/library_baseline/cub_scan.cu
      src/permute/library_baseline/vllm_permute.cu
      src/topk_gate/library_baseline/cub_block_radix_top2.cu)
    list(APPEND _raggedroute_library_baseline_libraries RaggedRoute::cccl)
    set(_raggedroute_has_cccl 1)
  endif()
  if(TARGET RaggedRoute::cutlass)
    list(APPEND _raggedroute_library_baseline_sources
      src/grouped_gemm/library_baseline/cutlass_grouped.cu)
    list(APPEND _raggedroute_library_baseline_libraries RaggedRoute::cutlass)
    set(_raggedroute_has_cutlass 1)
  endif()
  set(_raggedroute_library_baseline_definitions
    RAGGEDROUTE_HAS_VLLM_UNPERMUTE=1
    RAGGEDROUTE_HAS_CUBLAS=${_raggedroute_has_cublas}
    RAGGEDROUTE_HAS_CCCL=${_raggedroute_has_cccl}
    RAGGEDROUTE_HAS_CUTLASS=${_raggedroute_has_cutlass}
    RAGGEDROUTE_CUDATOOLKIT_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
    RAGGEDROUTE_CCCL_REQUESTED_FETCH_TAG="${RAGGEDROUTE_CCCL_GIT_TAG}"
    RAGGEDROUTE_CCCL_PROVIDER="${RAGGEDROUTE_CCCL_PROVIDER}"
    RAGGEDROUTE_CUTLASS_CONFIG_REVISION="${RAGGEDROUTE_CUTLASS_GIT_TAG}")
  add_library(raggedroute_library_baselines STATIC ${_raggedroute_library_baseline_sources})
  target_include_directories(raggedroute_library_baselines PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
  target_compile_features(raggedroute_library_baselines PUBLIC cxx_std_17)
  target_compile_definitions(raggedroute_library_baselines PUBLIC
    ${_raggedroute_library_baseline_definitions})
  target_link_libraries(raggedroute_library_baselines PUBLIC
    ${_raggedroute_library_baseline_libraries})
  set_target_properties(raggedroute_library_baselines PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${RAGGEDROUTE_ENABLE_RDC}
    POSITION_INDEPENDENT_CODE ON)
  raggedroute_apply_target_settings(raggedroute_library_baselines)

  # Public P0 operator layer. The low-level baseline launchers remain public
  # for L1 kernel-body studies; this target owns checked dispatch, workspace
  # contracts, and L2/L3 operator semantics.
  add_library(raggedroute_runtime STATIC
    src/runtime/dispatch.cpp
    src/dense_gemm/operator.cpp
    src/topk_gate/operator.cpp
    src/histogram/operator.cpp
    src/scan/operator.cpp
    src/permute/operator.cpp
    src/grouped_gemm/operator.cpp
    src/unpermute/operator.cpp)
  add_library(RaggedRoute::runtime ALIAS raggedroute_runtime)
  target_include_directories(raggedroute_runtime PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_compile_features(raggedroute_runtime PUBLIC cxx_std_17)
  target_link_libraries(raggedroute_runtime PUBLIC
    raggedroute_baseline_ops
    CUDA::cudart)
  set_target_properties(raggedroute_runtime PROPERTIES
    EXPORT_NAME runtime
    POSITION_INDEPENDENT_CODE ON)
  raggedroute_apply_target_settings(raggedroute_runtime)

  add_library(raggedroute_correctness_framework STATIC
    src/correctness/framework.cu
    src/correctness/reference_validation.cpp
    src/correctness/report.cpp
    src/dense_gemm/cpu_reference/reference.cpp
    src/topk_gate/cpu_reference/reference.cpp
    src/histogram/cpu_reference/reference.cpp
    src/scan/cpu_reference/reference.cpp
    src/permute/cpu_reference/reference.cpp
    src/grouped_gemm/cpu_reference/reference.cpp
    src/unpermute/cpu_reference/reference.cpp)
  add_library(RaggedRoute::correctness_framework ALIAS raggedroute_correctness_framework)
  target_include_directories(raggedroute_correctness_framework PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_compile_features(raggedroute_correctness_framework PUBLIC cxx_std_17)
  target_link_libraries(raggedroute_correctness_framework PUBLIC CUDA::cudart)
  set_target_properties(raggedroute_correctness_framework PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${RAGGEDROUTE_ENABLE_RDC}
    EXPORT_NAME correctness_framework
    POSITION_INDEPENDENT_CODE ON)
  raggedroute_apply_target_settings(raggedroute_correctness_framework)

  if(RAGGEDROUTE_BUILD_BENCHMARKS OR BUILD_TESTING)
    add_library(raggedroute_benchmark_core STATIC
      benchmarks/core/types.cpp
      benchmarks/core/registry.cpp
      benchmarks/core/runner.cpp
      benchmarks/adapters/dense_gemm_adapter.cpp
      benchmarks/adapters/topk_gate_adapter.cpp
      benchmarks/adapters/histogram_adapter.cpp
      benchmarks/adapters/exclusive_scan_adapter.cpp
      benchmarks/adapters/histogram_exclusive_scan_adapter.cpp
      benchmarks/adapters/token_permute_adapter.cpp
      benchmarks/adapters/grouped_gemm_adapter.cpp
      benchmarks/adapters/unpermute_adapter.cpp
      benchmarks/adapters/chain_adapter.cpp
      benchmarks/adapters/postroute_chain_adapter.cpp)
    target_include_directories(raggedroute_benchmark_core PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
    target_include_directories(raggedroute_benchmark_core PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dense_gemm"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/topk_gate"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/histogram"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/scan"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/permute"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/unpermute")
    target_link_libraries(raggedroute_benchmark_core PUBLIC
      raggedroute_runtime
      raggedroute_correctness_framework
      raggedroute_library_baselines
      CUDA::cudart)
    target_compile_definitions(raggedroute_benchmark_core PRIVATE
      RAGGEDROUTE_GIT_SHA="${RAGGEDROUTE_GIT_SHA}"
      RAGGEDROUTE_GIT_DIRTY="${RAGGEDROUTE_GIT_DIRTY}"
      RAGGEDROUTE_BUILD_TYPE="$<CONFIG>"
      RAGGEDROUTE_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
      RAGGEDROUTE_HOPPER_ACCELERATED_BUILD=$<BOOL:${RAGGEDROUTE_ENABLE_SM90A_EXPERIMENTS}>)
    raggedroute_apply_target_settings(raggedroute_benchmark_core)
  endif()

  if(RAGGEDROUTE_BUILD_BENCHMARKS)
    add_executable(raggedroute_benchmark benchmarks/benchmark_main.cpp)
    target_link_libraries(raggedroute_benchmark PRIVATE raggedroute_benchmark_core)
    raggedroute_apply_target_settings(raggedroute_benchmark)
  endif()

  if(BUILD_TESTING)
    add_executable(raggedroute_correctness_tests tests/correctness_tests.cpp)
    target_link_libraries(raggedroute_correctness_tests PRIVATE
      raggedroute_benchmark_core
      raggedroute_correctness_framework)
    raggedroute_apply_target_settings(raggedroute_correctness_tests)
    add_test(NAME raggedroute_correctness COMMAND raggedroute_correctness_tests)
    set_tests_properties(raggedroute_correctness PROPERTIES
      LABELS "correctness_smoke;correctness"
      SKIP_RETURN_CODE 77)

    add_executable(raggedroute_correctness_framework_tests
      tests/correctness_framework_tests.cpp)
    target_link_libraries(raggedroute_correctness_framework_tests PRIVATE
      raggedroute_baseline_ops
      raggedroute_correctness_framework)
    raggedroute_apply_target_settings(raggedroute_correctness_framework_tests)
    function(raggedroute_add_correctness_framework_test name suite labels)
      add_test(NAME ${name}
        COMMAND raggedroute_correctness_framework_tests --suite ${suite})
      set_tests_properties(${name} PROPERTIES
        LABELS "${labels};correctness"
        SKIP_RETURN_CODE 77)
    endfunction()
    raggedroute_add_correctness_framework_test(raggedroute_correctness_dtype dtype "dtype")
    raggedroute_add_correctness_framework_test(raggedroute_correctness_selftest selftest
      "correctness_selftest")
    raggedroute_add_correctness_framework_test(raggedroute_correctness_edge edge
      "correctness_edge")
    raggedroute_add_correctness_framework_test(raggedroute_correctness_randomized randomized
      "correctness_randomized")
    raggedroute_add_correctness_framework_test(raggedroute_correctness_stream stream
      "stream_contract")
    add_executable(raggedroute_operator_api_tests tests/operator_api_tests.cpp)
    target_link_libraries(raggedroute_operator_api_tests PRIVATE
      raggedroute_runtime
      raggedroute_correctness_framework)
    raggedroute_apply_target_settings(raggedroute_operator_api_tests)
    add_test(NAME raggedroute_operator_api COMMAND raggedroute_operator_api_tests)
    set_tests_properties(raggedroute_operator_api PROPERTIES
      LABELS "operator_api;correctness;sm86"
      SKIP_RETURN_CODE 77)
  endif()

  if(RAGGEDROUTE_ENABLE_ARCH_COMPILE_TESTS)
    if(CMAKE_VERSION VERSION_LESS "4.0")
      message(FATAL_ERROR
        "RAGGEDROUTE_ENABLE_ARCH_COMPILE_TESTS requires CMake 4.x for 90a/100a architecture names")
    endif()
    foreach(_raggedroute_probe_arch IN ITEMS 90a-real 100a-real)
      string(REPLACE "-" "_" _raggedroute_probe_suffix "${_raggedroute_probe_arch}")
      add_library(raggedroute_dtype_probe_${_raggedroute_probe_suffix} OBJECT EXCLUDE_FROM_ALL
        tests/correctness/dtype_compile_probe.cu)
      target_include_directories(raggedroute_dtype_probe_${_raggedroute_probe_suffix} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
      set_target_properties(raggedroute_dtype_probe_${_raggedroute_probe_suffix} PROPERTIES
        CUDA_ARCHITECTURES "${_raggedroute_probe_arch}")
      raggedroute_apply_target_settings(raggedroute_dtype_probe_${_raggedroute_probe_suffix})
    endforeach()
    add_custom_target(raggedroute_arch_compile_tests
      DEPENDS raggedroute_dtype_probe_90a_real raggedroute_dtype_probe_100a_real
      COMMENT "Compile dtype declarations for Hopper and Blackwell target architectures")
  endif()

  if(RAGGEDROUTE_BUILD_BENCHMARKS)
    add_custom_target(raggedroute_cuda_smoke
      COMMAND $<TARGET_FILE:raggedroute_benchmark> --list
      DEPENDS raggedroute_benchmark
      USES_TERMINAL
      COMMENT "Verifying that the CUDA benchmark binary launches")
  endif()

  if(RAGGEDROUTE_ENABLE_INSTALL)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RaggedRouteConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/RaggedRouteConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RaggedRoute")
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/RaggedRouteConfigVersion.cmake"
      VERSION "${PROJECT_VERSION}"
      COMPATIBILITY SameMinorVersion)

    install(TARGETS raggedroute_baseline_ops raggedroute_runtime raggedroute_correctness_framework
      EXPORT RaggedRouteTargets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/baseline_ops.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/status.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/types.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/runtime.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/dispatch.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/operators.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/raggedroute")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/raggedroute/correctness/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/raggedroute/correctness"
      FILES_MATCHING PATTERN "*.h")
    install(EXPORT RaggedRouteTargets
      FILE RaggedRouteTargets.cmake
      NAMESPACE RaggedRoute::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RaggedRoute")
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/RaggedRouteConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/RaggedRouteConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/RaggedRoute")
  endif()
elseif(RAGGEDROUTE_ENABLE_INSTALL)
  message(STATUS "RaggedRoute: install/export is disabled in CPU-only mode because the public API is CUDA-only")
endif()

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME raggedroute_script_tests
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scripts.py")
endif()

if(BUILD_TESTING)
  add_custom_target(raggedroute_check
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -C $<CONFIG>
    USES_TERMINAL
    COMMENT "Running RaggedRoute CTest suite")
endif()
