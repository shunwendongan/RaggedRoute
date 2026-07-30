@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"

call "%~dp0setup_msvc_env.bat"
if errorlevel 1 exit /b %errorlevel%

cmake --preset "%PRESET%"
