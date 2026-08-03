@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"

call "%~dp0setup_msvc_env.bat"
if errorlevel 1 exit /b %errorlevel%

if /i not "%PRESET:~0,4%"=="cpu-" (
  call "%~dp0setup_cuda_env.bat"
  if errorlevel 1 exit /b %errorlevel%
)

where cmake.exe >nul 2>nul
if errorlevel 1 (
  echo CMake was not found on PATH. Install CMake 3.24 or newer and retry. 1>&2
  exit /b 5
)

where ninja.exe >nul 2>nul
if errorlevel 1 (
  echo Ninja was not found on PATH. Install Ninja or add it to PATH and retry. 1>&2
  exit /b 6
)

set "RAGGEDROUTE_NVCC=%CUDACXX%"
if /i not "%PRESET:~0,4%"=="cpu-" if not defined RAGGEDROUTE_NVCC (
  echo CUDA nvcc.exe was not resolved by setup_cuda_env.bat. 1>&2
  exit /b 7
)

set "RAGGEDROUTE_REFERENCE_ARGS="
if /i "%RAGGEDROUTE_FETCH_REFERENCES%"=="ON" set "RAGGEDROUTE_REFERENCE_ARGS=-DRAGGEDROUTE_CCCL_PROVIDER=FETCH -DRAGGEDROUTE_CUTLASS_PROVIDER=FETCH"

rem --fresh discards compiler paths cached before a Visual Studio/CUDA relocation.
if /i "%PRESET:~0,4%"=="cpu-" (
  cmake --fresh --preset "%PRESET%" "-DCMAKE_CXX_COMPILER=%RAGGEDROUTE_MSVC_CL%"
) else (
  rem Ninja otherwise prefers an earlier MSYS2 c++.exe on mixed developer machines.
  cmake --fresh --preset "%PRESET%" "-DCMAKE_CXX_COMPILER=%RAGGEDROUTE_MSVC_CL%" "-DCMAKE_CUDA_COMPILER=%RAGGEDROUTE_NVCC%" "-DCMAKE_CUDA_HOST_COMPILER=%RAGGEDROUTE_MSVC_CL%" %RAGGEDROUTE_REFERENCE_ARGS%
)
