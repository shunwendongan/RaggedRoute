@echo off
rem Intentionally do not use setlocal: callers need PATH/INCLUDE/LIB updates from VsDevCmd.

rem A stale Developer Shell marker is not sufficient: Visual Studio may have moved or
rem been upgraded since the shell was opened. Trust the compiler that is executable
rem now, otherwise run the complete discovery sequence again.
set "RAGGEDROUTE_MSVC_CL="
where cl.exe >nul 2>nul
if not errorlevel 1 goto record_toolchain

set "RAGGEDROUTE_RESOLVED_VSDEVCMD="
if defined RAGGEDROUTE_VSDEVCMD if exist "%RAGGEDROUTE_VSDEVCMD%" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%RAGGEDROUTE_VSDEVCMD%"
if defined VSDEVCMD if exist "%VSDEVCMD%" set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%VSDEVCMD%"

rem Accept install roots exported by CI, package managers, or a local developer.
rem The parent fallback handles an installation whose edition suffix was removed
rem during relocation (for example D:\VisualStudio\Community -> D:\VisualStudio).
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD call :try_vs_root "%RAGGEDROUTE_VS_INSTALL_ROOT%"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD call :try_vs_root "%VSINSTALLDIR%"
if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD call :try_vs_root "%VS2022_HOME%"

set "RAGGEDROUTE_RESOLVED_VSWHERE="
if defined VSWHERE if exist "%VSWHERE%" set "RAGGEDROUTE_RESOLVED_VSWHERE=%VSWHERE%"
if not defined RAGGEDROUTE_RESOLVED_VSWHERE for /f "delims=" %%I in ('where vswhere.exe 2^>nul') do if not defined RAGGEDROUTE_RESOLVED_VSWHERE set "RAGGEDROUTE_RESOLVED_VSWHERE=%%I"
if not defined RAGGEDROUTE_RESOLVED_VSWHERE if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" set "RAGGEDROUTE_RESOLVED_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined RAGGEDROUTE_RESOLVED_VSWHERE if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" set "RAGGEDROUTE_RESOLVED_VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD if defined RAGGEDROUTE_RESOLVED_VSWHERE (
  for /f "usebackq delims=" %%I in (`"%RAGGEDROUTE_RESOLVED_VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do if not defined RAGGEDROUTE_RESOLVED_VSDEVCMD call :try_vs_root "%%I"
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
  echo Visual Studio C++ Build Tools were not found. Set RAGGEDROUTE_VSDEVCMD to VsDevCmd.bat, set RAGGEDROUTE_VS_INSTALL_ROOT to the Visual Studio install root, or install the x64 C++ workload. 1>&2
  exit /b 2
)

call "%RAGGEDROUTE_RESOLVED_VSDEVCMD%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 (
  echo Failed to initialize the x64 MSVC environment through "%RAGGEDROUTE_RESOLVED_VSDEVCMD%". 1>&2
  exit /b 3
)

:verify_toolchain
where cl.exe >nul 2>nul
if errorlevel 1 (
  echo MSVC environment initialization completed without a usable cl.exe. Check VSDEVCMD and the x64 C++ workload. 1>&2
  exit /b 4
)

:record_toolchain
for /f "delims=" %%I in ('where cl.exe') do if not defined RAGGEDROUTE_MSVC_CL set "RAGGEDROUTE_MSVC_CL=%%I"
if "%RAGGEDROUTE_PRINT_TOOLCHAIN%"=="1" echo RaggedRoute MSVC compiler: %RAGGEDROUTE_MSVC_CL%
exit /b 0

:try_vs_root
if "%~1"=="" exit /b 0
if exist "%~1\Common7\Tools\VsDevCmd.bat" call :validate_vs_root "%~1"
if defined RAGGEDROUTE_RESOLVED_VSDEVCMD exit /b 0
if exist "%~1\..\Common7\Tools\VsDevCmd.bat" for %%I in ("%~1\..") do call :validate_vs_root "%%~fI"
exit /b 0

:validate_vs_root
rem A relocated VS shell can exist while its default VC toolset is absent.
rem Validate that toolset before selecting an automatically discovered root.
set "RAGGEDROUTE_DEFAULT_VCTOOLS="
if exist "%~1\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt" set /p RAGGEDROUTE_DEFAULT_VCTOOLS=<"%~1\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
rem VsDevCmd honors an inherited VCToolsVersion before the install default.
if defined VCToolsVersion set "RAGGEDROUTE_DEFAULT_VCTOOLS=%VCToolsVersion%"
if not defined RAGGEDROUTE_DEFAULT_VCTOOLS exit /b 0
if not exist "%~1\VC\Tools\MSVC\%RAGGEDROUTE_DEFAULT_VCTOOLS%\bin\Hostx64\x64\cl.exe" exit /b 0
set "RAGGEDROUTE_RESOLVED_VSDEVCMD=%~1\Common7\Tools\VsDevCmd.bat"
exit /b 0
