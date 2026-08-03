@echo off
rem Intentionally do not use setlocal: callers need the resolved CUDA runtime PATH.
set "RAGGEDROUTE_RESOLVED_CUDA_ROOT="

where python.exe >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH; it is required for CUDA Toolkit discovery. 1>&2
  exit /b 7
)

for /f "usebackq delims=" %%I in (`python "%~dp0resolve_cuda_toolkit.py" --repo "%~dp0.."`) do if not defined RAGGEDROUTE_RESOLVED_CUDA_ROOT set "RAGGEDROUTE_RESOLVED_CUDA_ROOT=%%I"
if not defined RAGGEDROUTE_RESOLVED_CUDA_ROOT (
  echo A complete CUDA Toolkit with cuBLAS runtime DLLs was not found. 1>&2
  exit /b 7
)

set "CUDA_PATH=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%"
set "CUDA_HOME=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%"
set "CUDA_BIN_PATH=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%\bin"
set "CUDACXX=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%\bin\nvcc.exe"
if exist "%RAGGEDROUTE_RESOLVED_CUDA_ROOT%\bin\x64" set "PATH=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%\bin\x64;%PATH%"
set "PATH=%RAGGEDROUTE_RESOLVED_CUDA_ROOT%\bin;%PATH%"

if "%RAGGEDROUTE_PRINT_TOOLCHAIN%"=="1" (
  echo RaggedRoute CUDA Toolkit: %RAGGEDROUTE_RESOLVED_CUDA_ROOT%
  echo RaggedRoute CUDA compiler: %CUDACXX%
)
exit /b 0
