@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=test-rtx3080-sm86-release"

call "%~dp0setup_cuda_env.bat"
if errorlevel 1 exit /b %errorlevel%

ctest --preset "%PRESET%" --output-on-failure
exit /b %errorlevel%
