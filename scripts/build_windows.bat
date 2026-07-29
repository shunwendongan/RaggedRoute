@echo off
setlocal EnableExtensions

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=rtx3080-sm86-release"
set "TARGET=%~2"

set "RAGGEDROUTE_ENV_VSDEVCMD=%VSDEVCMD%"
set "VSDEVCMD="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDEVCMD=%%I\Common7\Tools\VsDevCmd.bat"
)
if not defined VSDEVCMD if defined RAGGEDROUTE_ENV_VSDEVCMD set "VSDEVCMD=%RAGGEDROUTE_ENV_VSDEVCMD%"
if not defined VSDEVCMD set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo Visual Studio C++ Build Tools were not found. Set VSDEVCMD or install Visual Studio 2022 Build Tools. 1>&2
  exit /b 2
)

call "%VSDEVCMD%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 exit /b %errorlevel%

if defined TARGET (
  cmake --build --preset "build-%PRESET%" --target "%TARGET%" --parallel
) else (
  cmake --build --preset "build-%PRESET%" --parallel
)
