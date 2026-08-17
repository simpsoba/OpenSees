@echo off
setlocal EnableDelayedExpansion

REM ##############################################################################
REM
REM  makeOpsFresco_WIN.bat
REM  ---------------------
REM  Double-click this file, or run it from a cmd.exe window in this folder.
REM
REM  What it does (for the RTHS-CUDA branch):
REM    1. Switches git to branch RTHS-CUDA (if needed)
REM    2. Builds OpenSees.exe and OpenSeesMP.exe
REM       (linked to dynamic tcl86t.dll so OpenFresco loadPackage works)
REM    3. Enables CUDA when nvcc is found; otherwise CPU-only (no FATAL_ERROR)
REM
REM  You usually do NOT need to edit anything if your machine matches the
REM  "expected layout" below. If a check fails, the script tells you what
REM  to install or which line to change.
REM
REM  Expected layout: MUMPS (and optionally METIS) sit NEXT TO this OpenSees
REM  folder. The parent directory can be named anything (repos, src, code, ...).
REM
REM    <any-parent>\
REM      OpenSees\              <-- this repo (run the .bat from here)
REM      mumps\build\           <-- must contain dmumps.lib
REM      metis-5.1.0\install\   <-- optional but needed for partition()
REM
REM  METIS 5.1.0 Windows install steps:
REM    see README-Windows-METIS5.md  (clone eric2003 fork, VS2022 /MT build)
REM
REM  If your MUMPS/METIS live somewhere else, set MUMPS_DIR / METIS5_DIR below.
REM
REM  Also required on the machine:
REM    - Git
REM    - CMake, Ninja, Conan  (on PATH)
REM    - Visual Studio 2022 with C++ tools
REM    - Intel oneAPI (setvars.bat + ifx / MKL / MPI as used by OpenSees)
REM    - Tcl 8.6 with tcl86t.dll  (default: C:\Program Files\Tcl)
REM
REM ##############################################################################

REM =============================================================================
REM  SETTINGS -- only change these if a check fails or you want different outputs
REM =============================================================================

REM Which programs to build: 1 = yes, 0 = no
set "BUILD_OPENSEES=1"
set "BUILD_OPENSEESMP=1"
set "BUILD_OPENSEESSP=0"

REM How many CPU cores to use while compiling
set "JOBS=10"

REM Switch to this git branch before building? 1 = yes, 0 = leave current branch
set "BRANCH=RTHS-CUDA"
set "CHECKOUT_BRANCH=1"

REM Copy finished .exe files into the ShearBuilding40 example folder? 1 = yes
set "STAGE_SHEAR40=0"

REM --- Paths (leave blank "" to auto-detect; fill in only if auto-detect fails) ---
REM Example overrides:
REM   set "TCL_ROOT=D:\Tcl"
REM   set "MUMPS_DIR=D:\libs\mumps\build"
REM   set "METIS5_DIR=D:\libs\metis-5.1.0\install"
REM   set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "TCL_ROOT="
set "MUMPS_DIR="
set "METIS5_DIR="
set "ONEAPI_SETVARS="
set "CUDAToolkit_ROOT="

REM Build folder names (rarely need changing)
set "BUILD_DIR=build"
set "BUILD_MP_DIR=build-mp"
set "BUILD_SP_DIR=build-sp"
set "SHEAR40_BIN=%~dp0EXAMPLES\ShearBuilding40SP\bin"
set "CMAKE=cmake"

REM Optional overrides for scripting/CI (leave unset for normal use):
REM   set OPS_FRESCO_BUILD_OPENSEES=0
REM   set OPS_FRESCO_BUILD_OPENSEESMP=0
REM   set OPS_FRESCO_NOPAUSE=1
if defined OPS_FRESCO_BUILD_OPENSEES set "BUILD_OPENSEES=%OPS_FRESCO_BUILD_OPENSEES%"
if defined OPS_FRESCO_BUILD_OPENSEESMP set "BUILD_OPENSEESMP=%OPS_FRESCO_BUILD_OPENSEESMP%"
if defined OPS_FRESCO_BUILD_OPENSEESSP set "BUILD_OPENSEESSP=%OPS_FRESCO_BUILD_OPENSEESSP%"
if defined OPS_FRESCO_NOPAUSE set "NOPAUSE=1"

REM =============================================================================
REM  End of settings -- you should not need to edit below this line
REM =============================================================================

cd /d "%~dp0"
set "REPO=%CD%"
set "FAILED=0"

echo.
echo ============================================================
echo  OpenSees RTHS-CUDA Windows build
echo ============================================================
echo  Folder: !REPO!
echo.

REM ---- Auto-detect paths if not set ------------------------------------------
if "!TCL_ROOT!"=="" (
  if exist "C:\Program Files\Tcl\lib\tcl86t.lib" set "TCL_ROOT=C:\Program Files\Tcl"
)
if "!TCL_ROOT!"=="" (
  if exist "C:\Program Files (x86)\Tcl\lib\tcl86t.lib" set "TCL_ROOT=C:\Program Files (x86)\Tcl"
)

if "!MUMPS_DIR!"=="" (
  if exist "%~dp0..\mumps\build\dmumps.lib" (
    for %%I in ("%~dp0..\mumps\build") do set "MUMPS_DIR=%%~fI"
  )
)

if "!METIS5_DIR!"=="" (
  if exist "%~dp0..\metis-5.1.0\install\include\metis.h" (
    for %%I in ("%~dp0..\metis-5.1.0\install") do set "METIS5_DIR=%%~fI"
  )
)

if "!ONEAPI_SETVARS!"=="" (
  if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
  )
)
if "!ONEAPI_SETVARS!"=="" (
  if exist "C:\Program Files\Intel\oneAPI\setvars.bat" (
    set "ONEAPI_SETVARS=C:\Program Files\Intel\oneAPI\setvars.bat"
  )
)

if "!CUDAToolkit_ROOT!"=="" (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
    set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
  )
)
if not defined OPS_SKIP_CUDA set "OPS_SKIP_CUDA=0"

REM ---- Preflight: install locations -----------------------------------------
echo Checking prerequisites...
echo.

if "!ONEAPI_SETVARS!"=="" (
  echo [MISSING] Intel oneAPI setvars.bat
  echo          Install Intel oneAPI Base + HPC toolkits, then either:
  echo            - accept the default install path, or
  echo            - set ONEAPI_SETVARS in the SETTINGS section
  set "FAILED=1"
) else if not exist "!ONEAPI_SETVARS!" (
  echo [MISSING] oneAPI setvars.bat not found:
  echo          !ONEAPI_SETVARS!
  echo          Fix: edit ONEAPI_SETVARS in the SETTINGS section
  set "FAILED=1"
) else (
  echo [OK]      Intel oneAPI: !ONEAPI_SETVARS!
)

if "!TCL_ROOT!"=="" (
  echo [MISSING] Tcl 8.6 with tcl86t ^(needed for OpenFresco loadPackage^)
  echo          Install Tcl/ActiveTcl so these exist:
  echo            ...\Tcl\lib\tcl86t.lib
  echo            ...\Tcl\bin\tcl86t.dll
  echo          Then set TCL_ROOT in the SETTINGS section ^(e.g. C:\Program Files\Tcl^)
  set "FAILED=1"
) else (
  if not exist "!TCL_ROOT!\lib\tcl86t.lib" (
    echo [MISSING] !TCL_ROOT!\lib\tcl86t.lib
    echo          Fix: install tcl86t or edit TCL_ROOT
    set "FAILED=1"
  ) else if not exist "!TCL_ROOT!\bin\tcl86t.dll" (
    echo [MISSING] !TCL_ROOT!\bin\tcl86t.dll
    echo          Fix: install tcl86t or edit TCL_ROOT
    set "FAILED=1"
  ) else (
    echo [OK]      Tcl ^(tcl86t^): !TCL_ROOT!
  )
)

if "!MUMPS_DIR!"=="" (
  echo [MISSING] MUMPS libraries
  echo          Expected next to this OpenSees folder:
  echo            !REPO!\..\mumps\build\dmumps.lib
  echo          ^(Parent folder name does not matter.^)
  echo          Fix: put MUMPS there, or set MUMPS_DIR in SETTINGS to the
  echo          folder that contains dmumps.lib
  set "FAILED=1"
) else if not exist "!MUMPS_DIR!\dmumps.lib" (
  echo [MISSING] dmumps.lib not found in:
  echo          !MUMPS_DIR!
  echo          Fix: set MUMPS_DIR to the folder that contains dmumps.lib
  set "FAILED=1"
) else (
  echo [OK]      MUMPS: !MUMPS_DIR!
)

if "!METIS5_DIR!"=="" (
  echo [WARN]    METIS 5 not found ^(OpenSeesMP partition^(^) will be a stub^)
  echo          Optional: install next to OpenSees as ..\metis-5.1.0\install
  echo          Full Windows steps: README-Windows-METIS5.md
  echo          Or set METIS5_DIR in SETTINGS if it lives elsewhere
) else if not exist "!METIS5_DIR!\include\metis.h" (
  echo [WARN]    METIS 5 headers missing under !METIS5_DIR!
  echo          See README-Windows-METIS5.md
) else (
  echo [OK]      METIS 5: !METIS5_DIR!
)

set "WITH_CUDA=0"
if /I not "!OPS_SKIP_CUDA!"=="1" (
  if not "!CUDAToolkit_ROOT!"=="" if exist "!CUDAToolkit_ROOT!\bin\nvcc.exe" (
    set "WITH_CUDA=1"
  )
)
if "!WITH_CUDA!"=="1" (
  echo [OK]      CUDA: !CUDAToolkit_ROOT!
) else (
  echo [INFO]    CUDA skipped ^(CPU-only^). Set CUDAToolkit_ROOT if nvcc exists, or OPS_SKIP_CUDA=1.
  set "CUDAToolkit_ROOT="
)

echo.
if "!FAILED!"=="1" (
  echo ============================================================
  echo  Setup incomplete. Fix the [MISSING] items above, then
  echo  re-run this script. Only edit SETTINGS if a path is wrong.
  echo ============================================================
  echo.
  if not defined NOPAUSE pause
  exit /b 1
)

REM ---- oneAPI env + tool PATH ------------------------------------------------
echo Loading Intel oneAPI environment...
call "!ONEAPI_SETVARS!" intel64 mod
if errorlevel 1 (
  echo ERROR: oneAPI setvars.bat failed.
  if not defined NOPAUSE pause
  exit /b 1
)

REM VS installs CMake/Ninja here; make sure a plain cmd.exe can find them.
set "VS_CMAKE_BIN=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "VS_NINJA_BIN=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "!VS_CMAKE_BIN!\cmake.exe" set "PATH=!VS_CMAKE_BIN!;!PATH!"
if exist "!VS_NINJA_BIN!\ninja.exe" set "PATH=!VS_NINJA_BIN!;!PATH!"
if "!WITH_CUDA!"=="1" set "PATH=!CUDAToolkit_ROOT!\bin;!PATH!"

echo.
echo Checking build tools on PATH...
call :check_cmd git "Git - https://git-scm.com/"
call :check_cmd cmake "CMake - install VS 2022 C++ workload, or https://cmake.org/"
call :check_cmd ninja "Ninja - comes with VS CMake tools, or https://ninja-build.org/"
call :check_cmd conan "Conan - run: pip install conan - and ensure Scripts is on PATH"
echo.
if "!FAILED!"=="1" (
  echo ============================================================
  echo  Build tools missing. Fix the [MISSING] items above, then
  echo  re-run this script.
  echo ============================================================
  echo.
  if not defined NOPAUSE pause
  exit /b 1
)

REM ---- Git branch ------------------------------------------------------------
if "!CHECKOUT_BRANCH!"=="1" (
  git rev-parse --is-inside-work-tree >nul 2>&1
  if errorlevel 1 (
    echo ERROR: This folder is not a git clone of OpenSees.
    if not defined NOPAUSE pause
    exit /b 1
  )
  for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "CUR_BRANCH=%%b"
  if /I not "!CUR_BRANCH!"=="!BRANCH!" (
    echo Switching branch: !CUR_BRANCH! -^> !BRANCH!
    git checkout "!BRANCH!"
    if errorlevel 1 (
      echo ERROR: could not check out !BRANCH!
      if not defined NOPAUSE pause
      exit /b 1
    )
  ) else (
    echo Already on branch !BRANCH!.
  )
  echo.
)

REM ---- METIS cmake args ------------------------------------------------------
set "METIS5_CMAKE_ARGS=-UOPENSEES_METIS5_LIBRARY"
if not "!METIS5_DIR!"=="" if exist "!METIS5_DIR!\include\metis.h" (
  set "METIS5_CMAKE_ARGS=-DMETIS5_DIR=!METIS5_DIR! -UOPENSEES_METIS5_LIBRARY"
)

set "CUDA_CMAKE_ARGS=-UCUDToolkit_ROOT"
if "!WITH_CUDA!"=="1" (
  set "CUDA_CMAKE_ARGS=-DCUDAToolkit_ROOT=!CUDAToolkit_ROOT! -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR -UAMGX_NO_MPI_DIR"
)

REM ---- Builds ----------------------------------------------------------------
if "!BUILD_OPENSEES!"=="1" (
  call :configure_and_build "!BUILD_DIR!" "!BUILD_DIR!\Release" OFF OpenSees
  if errorlevel 1 goto :fail
)

if "!BUILD_OPENSEESMP!"=="1" (
  call :configure_and_build "!BUILD_MP_DIR!" "!BUILD_MP_DIR!\Release" OFF OpenSeesMP
  if errorlevel 1 goto :fail
)

if "!BUILD_OPENSEESSP!"=="1" (
  call :configure_and_build "!BUILD_SP_DIR!" "!BUILD_SP_DIR!\Release" ON OpenSeesSP
  if errorlevel 1 goto :fail
)

if "!STAGE_SHEAR40!"=="1" (
  if not exist "!SHEAR40_BIN!" mkdir "!SHEAR40_BIN!"
  if "!BUILD_OPENSEES!"=="1" if exist "!BUILD_DIR!\Release\OpenSees.exe" (
    copy /Y "!BUILD_DIR!\Release\OpenSees.exe" "!SHEAR40_BIN!\" >nul
  )
  if "!BUILD_OPENSEESMP!"=="1" if exist "!BUILD_MP_DIR!\Release\OpenSeesMP.exe" (
    copy /Y "!BUILD_MP_DIR!\Release\OpenSeesMP.exe" "!SHEAR40_BIN!\" >nul
  )
  if "!BUILD_OPENSEESSP!"=="1" if exist "!BUILD_SP_DIR!\Release\OpenSeesSP.exe" (
    copy /Y "!BUILD_SP_DIR!\Release\OpenSeesSP.exe" "!SHEAR40_BIN!\" >nul
  )
  copy /Y "!TCL_ROOT!\bin\tcl86t.dll" "!SHEAR40_BIN!\" >nul
  echo Copied executables to !SHEAR40_BIN!
)

echo.
echo ============================================================
echo  SUCCESS
echo ============================================================
if "!BUILD_OPENSEES!"=="1"   echo   OpenSees:   !REPO!\!BUILD_DIR!\Release\OpenSees.exe
if "!BUILD_OPENSEESMP!"=="1" echo   OpenSeesMP: !REPO!\!BUILD_MP_DIR!\Release\OpenSeesMP.exe
if "!BUILD_OPENSEESSP!"=="1" echo   OpenSeesSP: !REPO!\!BUILD_SP_DIR!\Release\OpenSeesSP.exe
echo   Tcl DLL:    !TCL_ROOT!\bin\tcl86t.dll
if "!WITH_CUDA!"=="1" (echo   CUDA:        enabled at !CUDAToolkit_ROOT!) else (echo   CUDA:        skipped / CPU-only)
echo.
if not defined NOPAUSE pause
endlocal
exit /b 0

:fail
echo.
echo ============================================================
echo  BUILD FAILED -- scroll up for the first error message.
echo ============================================================
echo.
if not defined NOPAUSE pause
exit /b 1

REM ---- helpers ---------------------------------------------------------------
:check_cmd
where %~1 >nul 2>&1
if not errorlevel 1 (
  echo [OK]      %~1
  exit /b 0
)
echo [MISSING] %~1
echo          Install: %~2
set "FAILED=1"
exit /b 0

:configure_and_build
set "_OF=%~1"
set "_BIN=%~2"
set "_PAR=%~3"
shift
shift
shift

echo.
echo ------------------------------------------------------------
echo  Conan packages -^> !_OF!
echo ------------------------------------------------------------
conan install . -of "!_OF!" ^
  -s build_type=Release ^
  -s arch=x86_64 ^
  -s compiler.runtime=static ^
  --build=missing ^
  -c tools.cmake.cmaketoolchain:generator=Ninja
if errorlevel 1 exit /b 1

set "TOOLCHAIN=!_OF!\build\Release\generators\conan_toolchain.cmake"
if not exist "!TOOLCHAIN!" set "TOOLCHAIN=!_OF!\Release\generators\conan_toolchain.cmake"
if not exist "!TOOLCHAIN!" set "TOOLCHAIN=!_BIN!\generators\conan_toolchain.cmake"
if not exist "!TOOLCHAIN!" (
  echo ERROR: Conan did not create a CMake toolchain file under !_OF!
  exit /b 1
)

echo.
echo ------------------------------------------------------------
echo  Configure !_BIN!  ^(PARALLEL_PROCESSING=!_PAR!^)
echo ------------------------------------------------------------
"!CMAKE!" -S . -B "!_BIN!" -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="!TOOLCHAIN!" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="!MUMPS_DIR!" ^
  !METIS5_CMAKE_ARGS! ^
  -UOPENMPI ^
  -DPARALLEL_PROCESSING=!_PAR! ^
  !CUDA_CMAKE_ARGS! ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON
if errorlevel 1 exit /b 1

:build_next_target
if "%~1"=="" exit /b 0
echo.
echo ------------------------------------------------------------
echo  Compiling %~1 ...
echo ------------------------------------------------------------
"!CMAKE!" --build "!_BIN!" --target %~1 --parallel !JOBS!
if errorlevel 1 exit /b 1
shift
goto build_next_target
