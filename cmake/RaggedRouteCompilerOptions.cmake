include_guard(GLOBAL)

function(raggedroute_configure_compiler_options)
  if(MSVC)
    message(STATUS "RaggedRoute host compiler: MSVC ${MSVC_VERSION} at ${CMAKE_CXX_COMPILER}")
  else()
    message(STATUS "RaggedRoute host compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER}")
  endif()
endfunction()

function(raggedroute_apply_target_settings target)
  target_compile_features(${target} PRIVATE cxx_std_17)

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    if(RAGGEDROUTE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    if(RAGGEDROUTE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  endif()

  if(RAGGEDROUTE_ENABLE_CUDA)
    if(RAGGEDROUTE_ENABLE_CUDA_DEVICE_DEBUG)
      target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
    endif()
    if(RAGGEDROUTE_ENABLE_LINEINFO)
      target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>)
    endif()
  endif()
endfunction()
