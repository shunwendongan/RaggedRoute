include_guard(GLOBAL)

function(raggedroute_initialize_options)
  option(RAGGEDROUTE_ENABLE_CUDA "Build CUDA kernels and CUDA-facing targets" ON)
  option(RAGGEDROUTE_BUILD_BENCHMARKS "Build the benchmark runner when CUDA is enabled" ${PROJECT_IS_TOP_LEVEL})
  option(RAGGEDROUTE_ENABLE_INSTALL "Generate install and find_package exports" ${PROJECT_IS_TOP_LEVEL})
  option(RAGGEDROUTE_ENABLE_LINEINFO "Embed CUDA line information in non-Debug builds" ON)
  option(RAGGEDROUTE_ENABLE_CUDA_DEVICE_DEBUG "Pass -G to NVCC for Debug builds" ON)
  option(RAGGEDROUTE_ENABLE_RDC "Enable CUDA relocatable device code" OFF)
  option(RAGGEDROUTE_ENABLE_CUBLAS "Discover CUDA::cublas for future baseline targets" ON)
  option(RAGGEDROUTE_ENABLE_SM90A_EXPERIMENTS "Mark this build as Hopper SM90a-specific" OFF)
  option(RAGGEDROUTE_WARNINGS_AS_ERRORS "Treat host compiler warnings as errors" OFF)

  set(RAGGEDROUTE_CUDA_ARCHITECTURES "" CACHE STRING
      "CUDA architectures when CMAKE_CUDA_ARCHITECTURES is not provided (default: 86-real)")

  set(RAGGEDROUTE_CCCL_PROVIDER "AUTO" CACHE STRING
      "CCCL provider: AUTO, SYSTEM, FETCH, or OFF")
  set_property(CACHE RAGGEDROUTE_CCCL_PROVIDER PROPERTY STRINGS AUTO SYSTEM FETCH OFF)
  set(RAGGEDROUTE_CCCL_ROOT "" CACHE PATH "Root of a CCCL checkout or installation")
  set(RAGGEDROUTE_CCCL_GIT_TAG "v3.4.0" CACHE STRING "Pinned CCCL tag used by FETCH")

  set(RAGGEDROUTE_CUTLASS_PROVIDER "AUTO" CACHE STRING
      "CUTLASS provider: AUTO, SYSTEM, FETCH, or OFF")
  set_property(CACHE RAGGEDROUTE_CUTLASS_PROVIDER PROPERTY STRINGS AUTO SYSTEM FETCH OFF)
  set(RAGGEDROUTE_CUTLASS_ROOT "" CACHE PATH "Root of a CUTLASS checkout or installation")
  set(RAGGEDROUTE_CUTLASS_GIT_TAG "v4.6.1" CACHE STRING "Pinned CUTLASS tag used by FETCH")
endfunction()
