@echo off
rem Intentionally do not use setlocal: callers need PATH/INCLUDE/LIB updates from VsDevCmd.

if defined VSCMD_VER goto verify_toolchain

set "RAGGEDROUTE_RESOLVED_VSDEVCMD="
if defined VSDEVCMD if exist "%VSDEVCMD%" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%VSDEVCMD%"

set "RAGGEDROUTE_RESOLVED_VSWHERE="
if defined VSWHERE if exist "%VSWHERE%" set "RAGGEDROUTE_RESOLVED_VSWHERE=%VSWHERE%"
if not defined RAGGEDROUTE_RESOLVED_VSWHERE if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" set "RAGGEDROUTE_RESOLVED_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined RAGGEDROUTE_RESOLVED_VSWHERE if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" set "RAGGEDROUTE_RESOLVED_VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if defined RAGGEDROUTE_RESOLVED_VSWHERE (
  for /f "usebackq delims=" %%I in (`"%RAGGEDROUTE_RESOLVED_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%%I\Common7\Tools\VsDevCmd.bat"
)

if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"

if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD (
  echo Visual Studio C++ Build Tools were not found. Set VSDEVCMD to VsDevCmd.bat or install the x64 C++ workload. 1>&2
  exit /b 2
)

call "%RAGGEDROUTE_RESOLVED_VSDEVCMD%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 exit /b %errorlevel%

:verify_toolchain
where cl.exe >nul 2>nul
if errorlevel 1 (
  echo MSVC environment initialization completed without a usable cl.exe. Check VSDEVCMD and the x64 C++ workload. 1>&2
  exit /b 3
)

for /f "delims=" %%I in ('where cl.exe') do if not defined RAGGEDROUTE_MSVC_CL set "RAGGEDROUTE_MSVC_CL=%%I"
if "%RAGGEDROUTE_PRINT_TOOLCHAIN%"=="1" echo RaggedRoute MSVC compiler: %RAGGEDROUTE_MSVC_CL%
exit /b 0
