@echo off
REM Build OpenSees, OpenSeesMP, and OpenSeesPy with CUDA and cuDSS on Windows.
REM   build-cuda/  PARALLEL_PROCESSING=OFF  -> OpenSees.exe, OpenSeesMP.exe, OpenSeesPy
REM   (OpenSeesSP disabled for now — see commented build-sp-cuda sections below)
REM Non-CUDA devel builds use makeWIN_VS.bat (build/, build-sp/).
REM Requires: VS 2022, Intel oneAPI (ifx/MKL), Anaconda env opensees-cuda, MUMPS build.
REM Override BUILD_DIR, BUILD_SP_DIR, MUMPS_DIR, CUDAToolkit_ROOT before running.

setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "REPO_ROOT=%CD%"

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

REM --- User-tunable paths (defaults for this machine) ---
if not defined BUILD_DIR set "BUILD_DIR=build-cuda"
REM if not defined BUILD_SP_DIR set "BUILD_SP_DIR=build-sp-cuda"
if not defined MUMPS_DIR for %%I in ("%~dp0..\mumps\build") do set "MUMPS_DIR=%%~fI"
if not defined CUDAToolkit_ROOT set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
REM Runtime PATH only (cuDSS DLLs). CMake discovers cuDSS via default install hints.
if not defined CUDSS_DIR set "CUDSS_DIR=C:\Program Files\NVIDIA cuDSS\v0.8"

set "PATH=%CUDSS_DIR%\bin\12;%CUDAToolkit_ROOT%\bin;%PATH%"

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

echo CUDAToolkit_ROOT=%CUDAToolkit_ROOT%
echo MUMPS_DIR=%MUMPS_DIR%
echo CUDSS_DIR=%CUDSS_DIR%
echo Build tree: %BUILD_DIR% (OpenSees, OpenSeesMP, OpenSeesPy^)
REM echo Build tree: %BUILD_SP_DIR% (OpenSeesSP^)
echo devel/non-CUDA trees: build/ and build-sp/ via makeWIN_VS.bat

REM --- Configure %BUILD_DIR% (serial + MP): PARALLEL_PROCESSING=OFF ---
call :configure_cuda_tree "%BUILD_DIR%" OFF
if errorlevel 1 exit /b 1

REM --- Configure %BUILD_SP_DIR% (SP): PARALLEL_PROCESSING=ON ---
REM call :configure_cuda_tree "%BUILD_SP_DIR%" ON
REM if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build %BUILD_DIR% --config Release --target OpenSees OpenSeesMP --parallel 10
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build %BUILD_DIR% --config Release --target OpenSeesPy
if errorlevel 1 exit /b 1

REM "%CMAKE_EXE%" --build %BUILD_SP_DIR% --config Release --target OpenSeesSP --parallel 10
REM if errorlevel 1 exit /b 1

REM Tcl runtimes: OpenSees*.exe look for lib\tcl8.6 next to the build tree root.
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

REM Original: for %%B in ("%BUILD_DIR%" "%BUILD_SP_DIR%") do (
for %%B in ("%BUILD_DIR%") do (
  set "TCL_DEST=%%~B\lib"
  if not exist "!TCL_DEST!" mkdir "!TCL_DEST!"
  robocopy "!TCL_PKG_LIB!\tcl8.6" "!TCL_DEST!\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
  if errorlevel 8 (
    echo ERROR: failed to copy Tcl tcl8.6 to !TCL_DEST!\tcl8.6
    exit /b 1
  )
  robocopy "!TCL_PKG_LIB!\tcl8" "!TCL_DEST!\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
  if errorlevel 8 (
    echo ERROR: failed to copy Tcl tcl8 to !TCL_DEST!\tcl8
    exit /b 1
  )
  echo Staged Tcl runtime to !TCL_DEST!
)
REM OpenSeesSP Tcl staging (re-enable with build-sp-cuda sections above):
REM for %%B in ("%BUILD_SP_DIR%") do (
REM   set "TCL_DEST=%%~B\lib"
REM   if not exist "!TCL_DEST!" mkdir "!TCL_DEST!"
REM   robocopy "!TCL_PKG_LIB!\tcl8.6" "!TCL_DEST!\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
REM   if errorlevel 8 exit /b 1
REM   robocopy "!TCL_PKG_LIB!\tcl8" "!TCL_DEST!\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
REM   if errorlevel 8 exit /b 1
REM   echo Staged Tcl runtime to !TCL_DEST!
REM )

cd %BUILD_DIR%\Release
if exist OpenSeesPy.dll (
  copy /Y OpenSeesPy.dll opensees.pyd
)

echo.
echo Build complete.
echo   %BUILD_DIR%\Release\OpenSees.exe
echo   %BUILD_DIR%\Release\OpenSeesMP.exe
REM echo   %BUILD_SP_DIR%\Release\OpenSeesSP.exe
echo Run: call SCRIPTS\windows\opensees-cuda-env.bat
REM echo SP runs: set TCL_LIBRARY=%%REPO_ROOT%%\%BUILD_SP_DIR%\lib\tcl8.6

endlocal
exit /b 0

:configure_cuda_tree
set "CFG_BUILD_DIR=%~1"
set "CFG_PARALLEL_PROCESSING=%~2"

echo.
echo === Configuring %CFG_BUILD_DIR% (PARALLEL_PROCESSING=%CFG_PARALLEL_PROCESSING%^) ===

conan install . -of %CFG_BUILD_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator="Visual Studio 17 2022"
if errorlevel 1 exit /b 1

if /I "%CFG_PARALLEL_PROCESSING%"=="ON" (
  set "CFG_PARALLEL_FLAG=-DPARALLEL_PROCESSING=ON"
) else (
  set "CFG_PARALLEL_FLAG=-DPARALLEL_PROCESSING=OFF"
)

set "CFG_TOOLCHAIN=%CFG_BUILD_DIR%\build\generators\conan_toolchain.cmake"
if not exist "%CFG_TOOLCHAIN%" set "CFG_TOOLCHAIN=%CFG_BUILD_DIR%\generators\conan_toolchain.cmake"
if not exist "%CFG_TOOLCHAIN%" (
  echo ERROR: Conan toolchain not found under %CFG_BUILD_DIR%\build\generators or %CFG_BUILD_DIR%\generators
  exit /b 1
)

"%CMAKE_EXE%" -S . -B %CFG_BUILD_DIR% -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=%CFG_TOOLCHAIN% ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_Fortran_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/ifx.exe" ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%" ^
  %CFG_PARALLEL_FLAG% ^
  -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR ^
  -UAMGX_NO_MPI_DIR ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees-CUDA
if errorlevel 1 exit /b 1

exit /b 0
