@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"
set "TARGET=%~2"

call "%~dp0setup_msvc_env.bat"
if errorlevel 1 exit /b %errorlevel%

set "CACHE=%~dp0..\out\build\%PRESET%\CMakeCache.txt"
set "RAGGEDROUTE_RECONFIGURE=0"
if not exist "%CACHE%" set "RAGGEDROUTE_RECONFIGURE=1"

set "RAGGEDROUTE_CACHED_CXX="
if exist "%CACHE%" for /f "tokens=1,* delims==" %%A in ('findstr /r /b "CMAKE_CXX_COMPILER:.*=" "%CACHE%"') do set "RAGGEDROUTE_CACHED_CXX=%%B"
if not defined RAGGEDROUTE_CACHED_CXX set "RAGGEDROUTE_RECONFIGURE=1"
if defined RAGGEDROUTE_CACHED_CXX if not exist "%RAGGEDROUTE_CACHED_CXX%" set "RAGGEDROUTE_RECONFIGURE=1"

if defined RAGGEDROUTE_CACHED_CXX if exist "%RAGGEDROUTE_CACHED_CXX%" (
  for %%I in ("%RAGGEDROUTE_CACHED_CXX%") do set "RAGGEDROUTE_CACHED_CXX=%%~fI"
  for %%I in ("%RAGGEDROUTE_MSVC_CL%") do set "RAGGEDROUTE_CURRENT_CXX=%%~fI"
)
if defined RAGGEDROUTE_CURRENT_CXX if /i not "%RAGGEDROUTE_CACHED_CXX%"=="%RAGGEDROUTE_CURRENT_CXX%" set "RAGGEDROUTE_RECONFIGURE=1"

rem CUDA caches can become stale independently when a Toolkit moves or upgrades.
if /i not "%PRESET:~0,4%"=="cpu-" call :check_cuda_cache

if "%RAGGEDROUTE_RECONFIGURE%"=="1" (
  echo RaggedRoute build tree is missing or references a different compiler/Toolkit; reconfiguring "%PRESET%" from a fresh cache.
  call "%~dp0configure_windows.bat" "%PRESET%"
  if errorlevel 1 exit /b %errorlevel%
)

if defined TARGET (
  cmake --build --preset "build-%PRESET%" --target "%TARGET%" --parallel
) else (
  cmake --build --preset "build-%PRESET%" --parallel
)
exit /b %errorlevel%

:check_cuda_cache
set "RAGGEDROUTE_CACHED_CUDA="
if exist "%CACHE%" for /f "tokens=1,* delims==" %%A in ('findstr /r /b "CMAKE_CUDA_COMPILER:.*=" "%CACHE%"') do set "RAGGEDROUTE_CACHED_CUDA=%%B"
if not defined RAGGEDROUTE_CACHED_CUDA set "RAGGEDROUTE_RECONFIGURE=1"
if defined RAGGEDROUTE_CACHED_CUDA if not exist "%RAGGEDROUTE_CACHED_CUDA%" set "RAGGEDROUTE_RECONFIGURE=1"

set "RAGGEDROUTE_CURRENT_CUDA="
if defined CUDACXX if exist "%CUDACXX%" set "RAGGEDROUTE_CURRENT_CUDA=%CUDACXX%"
if not defined RAGGEDROUTE_CURRENT_CUDA if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "RAGGEDROUTE_CURRENT_CUDA=%CUDA_PATH%\bin\nvcc.exe"
if not defined RAGGEDROUTE_CURRENT_CUDA for /f "delims=" %%I in ('where nvcc.exe 2^>nul') do if not defined RAGGEDROUTE_CURRENT_CUDA set "RAGGEDROUTE_CURRENT_CUDA=%%I"
if not defined RAGGEDROUTE_CURRENT_CUDA set "RAGGEDROUTE_RECONFIGURE=1"
if defined RAGGEDROUTE_CACHED_CUDA if defined RAGGEDROUTE_CURRENT_CUDA call :compare_cuda_cache
exit /b 0

:compare_cuda_cache
for %%I in ("%RAGGEDROUTE_CACHED_CUDA%") do set "RAGGEDROUTE_CACHED_CUDA=%%~fI"
for %%I in ("%RAGGEDROUTE_CURRENT_CUDA%") do set "RAGGEDROUTE_CURRENT_CUDA=%%~fI"
if /i not "%RAGGEDROUTE_CACHED_CUDA%"=="%RAGGEDROUTE_CURRENT_CUDA%" set "RAGGEDROUTE_RECONFIGURE=1"
exit /b 0
