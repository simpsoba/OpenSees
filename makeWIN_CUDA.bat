@echo off
REM Build OpenSees + OpenSeesPy with CUDA and cuDSS on Windows (AmgX disabled for now).
REM Requires: VS 2022, Intel oneAPI (ifx/MKL), Anaconda env opensees-cuda, MUMPS build.
REM Override paths below via environment variables before running.

setlocal EnableDelayedExpansion

set "REPO_ROOT=%~dp0"
cd /d "%REPO_ROOT%"

REM --- Anaconda (Python 3.12 + conan) ---
set "ANACONDA_ROOT=%LOCALAPPDATA%\anaconda3"
if not exist "%ANACONDA_ROOT%\Scripts\activate.bat" set "ANACONDA_ROOT=%USERPROFILE%\AppData\Local\anaconda3"
if exist "%ANACONDA_ROOT%\Scripts\activate.bat" (
  call "%ANACONDA_ROOT%\Scripts\activate.bat" "%ANACONDA_ROOT%"
  call conda activate opensees-cuda
  if errorlevel 1 (
    echo WARNING: conda env opensees-cuda not found. Run SCRIPTS\windows\setup-opensees-cuda-env.bat
  )
)

REM --- Intel oneAPI (ifx, MKL) + GPU runtime PATH (cuDSS not covered by setvars) ---
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

REM --- User-tunable dependency paths (defaults for this machine) ---
if not defined CUDAToolkit_ROOT set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined MUMPS_DIR set "MUMPS_DIR=%REPO_ROOT%..\mumps\build"
REM Runtime PATH only (cuDSS DLLs). CMake discovers cuDSS via default install hints.
if not defined OPENSEES_CUDSS_DIR set "OPENSEES_CUDSS_DIR=C:\Program Files\NVIDIA cuDSS\v0.8"

set "PATH=%OPENSEES_CUDSS_DIR%\bin\12;%CUDAToolkit_ROOT%\bin;%PATH%"

if not exist "%MUMPS_DIR%\dmumps.lib" (
  echo ERROR: MUMPS not found at MUMPS_DIR=%MUMPS_DIR%
  echo Build mumps first, or set MUMPS_DIR to your mumps\build folder.
  exit /b 1
)

set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE_EXE%" (
  echo ERROR: cmake.exe not found at %CMAKE_EXE%
  exit /b 1
)

set "BUILD_DIR=build"
set "OUT_DIR=%BUILD_DIR%\Release"

echo CUDAToolkit_ROOT=%CUDAToolkit_ROOT%
echo MUMPS_DIR=%MUMPS_DIR%
echo OPENSEES_CUDSS_DIR=%OPENSEES_CUDSS_DIR%
echo Build tree: %BUILD_DIR%  (Release binaries: %OUT_DIR%)

REM VS generator: -B build + --config Release => build/Release/OpenSees.exe (not build/Release/Release/)
conan install . -of %BUILD_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator="Visual Studio 17 2022"
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" -S . -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=%BUILD_DIR%/generators/conan_toolchain.cmake ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_Fortran_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/ifx.exe" ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%" ^
  -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR ^
  -UAMGX_NO_MPI_DIR ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees-CUDA
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build %BUILD_DIR% --config Release --target OpenSees --parallel 10
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build %BUILD_DIR% --config Release --target OpenSeesPy
if errorlevel 1 exit /b 1

REM OpenSees.exe looks for Tcl under build/lib/tcl8.6; clock.tcl also needs build/lib/tcl8 (msgcat).
set "TCL_PKG_LIB="
for /d %%d in ("%USERPROFILE%\.conan2\p\b\tcl*") do (
  if exist "%%d\p\lib\tcl8.6\init.tcl" (
    set "TCL_PKG_LIB=%%d\p\lib"
    goto :tcl_found
  )
)
echo ERROR: Conan Tcl not found under %USERPROFILE%\.conan2\p\b
exit /b 1
:tcl_found
set "TCL_DEST=%BUILD_DIR%\lib"
if not exist "%TCL_DEST%" mkdir "%TCL_DEST%"
robocopy "%TCL_PKG_LIB%\tcl8.6" "%TCL_DEST%\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 (
  echo ERROR: failed to copy Tcl tcl8.6 to %TCL_DEST%\tcl8.6
  exit /b 1
)
robocopy "%TCL_PKG_LIB%\tcl8" "%TCL_DEST%\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 (
  echo ERROR: failed to copy Tcl tcl8 to %TCL_DEST%\tcl8
  exit /b 1
)
echo Staged Tcl runtime from %TCL_PKG_LIB% to %TCL_DEST%

cd %OUT_DIR%

if exist OpenSeesPy.dll (
  copy /Y OpenSeesPy.dll opensees.pyd
)

echo.
echo Build complete. Run from a shell with setvars + GPU PATH, e.g.:
echo   SCRIPTS\windows\run-cuda-smoke.bat

endlocal
