include_guard(GLOBAL)

include(CheckLanguage)

macro(raggedroute_enable_cuda)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(RAGGEDROUTE_CUDA_ARCHITECTURES)
      set(_raggedroute_architectures "${RAGGEDROUTE_CUDA_ARCHITECTURES}")
    else()
      set(_raggedroute_architectures "86-real")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_raggedroute_architectures}" CACHE STRING
        "CUDA architectures to generate device code for")
  endif()

  if(CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)90a(-real|-virtual)?(;|$)" AND
     CMAKE_VERSION VERSION_LESS "4.0")
    message(FATAL_ERROR
      "SM90a requires a CMake 4.x toolchain in RaggedRoute. Use the sm90 preset "
      "for the portable H100 path, or upgrade CMake for the SM90a preset.")
  endif()

  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "RAGGEDROUTE_ENABLE_CUDA=ON, but no CUDA compiler was found. Set CUDACXX or "
      "configure with -DRAGGEDROUTE_ENABLE_CUDA=OFF for the host-only build.")
  endif()
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  message(STATUS "RaggedRoute CUDA compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
  message(STATUS "RaggedRoute CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endmacro()
