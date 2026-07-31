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

rem --fresh discards compiler paths cached before a Visual Studio/CUDA relocation.
cmake --fresh --preset "%PRESET%"
