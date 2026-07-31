include_guard(GLOBAL)

include(FetchContent)

function(raggedroute_validate_dependency_provider name value)
  set(_raggedroute_valid_providers AUTO SYSTEM FETCH OFF)
  if(NOT value IN_LIST _raggedroute_valid_providers)
    message(FATAL_ERROR
      "RAGGEDROUTE_${name}_PROVIDER must be one of AUTO, SYSTEM, FETCH, or OFF; got '${value}'.")
  endif()
endfunction()

function(raggedroute_add_header_dependency target)
  if(NOT TARGET ${target})
    add_library(${target} INTERFACE)
  endif()
  target_include_directories(${target} INTERFACE ${ARGN})
endfunction()

function(raggedroute_discover_cccl)
  raggedroute_validate_dependency_provider(CCCL "${RAGGEDROUTE_CCCL_PROVIDER}")
  if(RAGGEDROUTE_CCCL_PROVIDER STREQUAL "OFF")
    message(STATUS "RaggedRoute CCCL: disabled")
    return()
  endif()

  find_path(_raggedroute_cccl_cub_include NAMES cub/cub.cuh
    HINTS ${CUDAToolkit_INCLUDE_DIRS} "${RAGGEDROUTE_CCCL_ROOT}" "$ENV{CCCL_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cccl"
    PATH_SUFFIXES include cub)
  find_path(_raggedroute_cccl_libcudacxx_include NAMES cuda/std/version
    HINTS ${CUDAToolkit_INCLUDE_DIRS} "${RAGGEDROUTE_CCCL_ROOT}" "$ENV{CCCL_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cccl"
    PATH_SUFFIXES include libcudacxx/include)
  if(_raggedroute_cccl_cub_include)
    set(_raggedroute_cccl_includes "${_raggedroute_cccl_cub_include}")
    if(_raggedroute_cccl_libcudacxx_include)
      list(APPEND _raggedroute_cccl_includes "${_raggedroute_cccl_libcudacxx_include}")
    endif()
    raggedroute_add_header_dependency(raggedroute_cccl ${_raggedroute_cccl_includes})
    add_library(RaggedRoute::cccl ALIAS raggedroute_cccl)
    message(STATUS "RaggedRoute CCCL: system headers at ${_raggedroute_cccl_cub_include}")
    return()
  endif()

  if(RAGGEDROUTE_CCCL_PROVIDER STREQUAL "SYSTEM")
    message(FATAL_ERROR "CCCL headers were not found. Set RAGGEDROUTE_CCCL_ROOT, CCCL_ROOT, or use RAGGEDROUTE_CCCL_PROVIDER=FETCH.")
  elseif(RAGGEDROUTE_CCCL_PROVIDER STREQUAL "FETCH")
    # Keep FetchContent identifiers short: Ninja on Windows can otherwise exceed
    # MAX_PATH in the generated *-populate-stamp directory when the repository is
    # checked out below a normal user profile path.
    FetchContent_Declare(rr_cccl
      GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
      GIT_TAG "${RAGGEDROUTE_CCCL_GIT_TAG}" GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(rr_cccl)
    raggedroute_add_header_dependency(raggedroute_cccl
      "${rr_cccl_SOURCE_DIR}/cub"
      "${rr_cccl_SOURCE_DIR}/libcudacxx/include")
    add_library(RaggedRoute::cccl ALIAS raggedroute_cccl)
    message(STATUS "RaggedRoute CCCL: fetched ${RAGGEDROUTE_CCCL_GIT_TAG}")
  else()
    message(STATUS "RaggedRoute CCCL: not found (AUTO leaves it disabled; use FETCH to download the pinned dependency)")
  endif()
endfunction()

function(raggedroute_discover_cutlass)
  raggedroute_validate_dependency_provider(CUTLASS "${RAGGEDROUTE_CUTLASS_PROVIDER}")
  if(RAGGEDROUTE_CUTLASS_PROVIDER STREQUAL "OFF")
    message(STATUS "RaggedRoute CUTLASS: disabled")
    return()
  endif()

  find_path(_raggedroute_cutlass_include NAMES cutlass/cutlass.h
    HINTS "${RAGGEDROUTE_CUTLASS_ROOT}" "$ENV{CUTLASS_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass"
    PATH_SUFFIXES include)
  if(_raggedroute_cutlass_include)
    raggedroute_add_header_dependency(raggedroute_cutlass "${_raggedroute_cutlass_include}")
    add_library(RaggedRoute::cutlass ALIAS raggedroute_cutlass)
    message(STATUS "RaggedRoute CUTLASS: system headers at ${_raggedroute_cutlass_include}")
    return()
  endif()

  if(RAGGEDROUTE_CUTLASS_PROVIDER STREQUAL "SYSTEM")
    message(FATAL_ERROR "CUTLASS headers were not found. Set RAGGEDROUTE_CUTLASS_ROOT, CUTLASS_ROOT, or use RAGGEDROUTE_CUTLASS_PROVIDER=FETCH.")
  elseif(RAGGEDROUTE_CUTLASS_PROVIDER STREQUAL "FETCH")
    # See the CCCL note above: the short name is a Windows path-length safeguard.
    FetchContent_Declare(rr_cutlass
      GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG "${RAGGEDROUTE_CUTLASS_GIT_TAG}" GIT_SHALLOW TRUE)
    # RaggedRoute consumes CUTLASS as a header/template dependency only.  Do
    # not configure the upstream profiler/library generator: besides being
    # unrelated to our target, its generated filenames can exceed MAX_PATH on
    # Windows before our own code is even configured.
    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "RaggedRoute uses CUTLASS headers only" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable upstream CUTLASS examples" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "Disable upstream CUTLASS tools" FORCE)
    set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "Disable upstream CUTLASS library" FORCE)
    set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "Disable upstream CUTLASS profiler" FORCE)
    set(CUTLASS_ENABLE_PERFORMANCE OFF CACHE BOOL "Disable upstream CUTLASS performance targets" FORCE)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable upstream CUTLASS tests" FORCE)
    FetchContent_MakeAvailable(rr_cutlass)
    raggedroute_add_header_dependency(raggedroute_cutlass "${rr_cutlass_SOURCE_DIR}/include")
    add_library(RaggedRoute::cutlass ALIAS raggedroute_cutlass)
    message(STATUS "RaggedRoute CUTLASS: fetched ${RAGGEDROUTE_CUTLASS_GIT_TAG}")
  else()
    message(STATUS "RaggedRoute CUTLASS: not found (AUTO leaves it disabled; use FETCH to download the pinned dependency)")
  endif()
endfunction()

function(raggedroute_discover_cuda_dependencies)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "RaggedRoute CUDAToolkit: ${CUDAToolkit_VERSION}")
  if(RAGGEDROUTE_ENABLE_CUBLAS)
    if(TARGET CUDA::cublas)
      message(STATUS "RaggedRoute cuBLAS: available")
    else()
      message(WARNING "RaggedRoute cuBLAS: requested but CUDAToolkit did not provide CUDA::cublas")
    endif()
  else()
    message(STATUS "RaggedRoute cuBLAS: disabled")
  endif()
  raggedroute_discover_cccl()
  raggedroute_discover_cutlass()
endfunction()
