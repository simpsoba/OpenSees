@echo off
setlocal
cd /d "%~dp0..\.."
call SCRIPTS\windows\opensees-cuda-env.bat
if errorlevel 1 exit /b 1

set "METHOD=%~1"
if "%METHOD%"=="" set "METHOD=CudaKRAlpha"
set "RHO=%~2"
if "%RHO%"=="" set "RHO=0.5"
set "SCALE=%~3"
if "%SCALE%"=="" set "SCALE=3.0"

cd EXAMPLES\KRAlphaExplicit\Two-Story_MRF
echo Running: OpenSees.exe two_story_MRF.tcl %METHOD% %RHO% %SCALE%
"%REPO_ROOT%\build\Release\OpenSees.exe" two_story_MRF.tcl %METHOD% %RHO% %SCALE%
if errorlevel 1 exit /b 1

for /f "delims=" %%f in ('dir /s /b results\results.txt 2^>nul') do (
  findstr /c:"COMPLETED successfully" "%%f" >nul 2>&1 && (
    echo MRF_TCL_OK: %%f
    exit /b 0
  )
)
echo ERROR: no successful results.txt found under results\
exit /b 1
