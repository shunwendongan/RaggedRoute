@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"
set "TARGET=%~2"

call "%~dp0setup_msvc_env.bat"
if errorlevel 1 exit /b %errorlevel%

if defined TARGET (
  cmake --build --preset "build-%PRESET%" --target "%TARGET%" --parallel
) else (
  cmake --build --preset "build-%PRESET%" --parallel
)
