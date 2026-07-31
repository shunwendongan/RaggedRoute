@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"

call "%~dp0setup_msvc_env.bat"
if errorlevel 1 exit /b %errorlevel%

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
if not defined RAGGEDROUTE_NVCC if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "RAGGEDROUTE_NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined RAGGEDROUTE_NVCC for /f "delims=" %%I in ('where nvcc.exe 2^>nul') do if not defined RAGGEDROUTE_NVCC set "RAGGEDROUTE_NVCC=%%I"
if /i not "%PRESET:~0,4%"=="cpu-" if not defined RAGGEDROUTE_NVCC (
  echo CUDA nvcc.exe was not found. Set CUDACXX or CUDA_PATH and retry. 1>&2
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
