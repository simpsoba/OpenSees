REM rd /s /q build
REM rd /s /q build-sp

cd /d "%~dp0"

set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set BUILD_DIR=build
set BUILD_SP_DIR=build-sp
set TOOLCHAIN=%BUILD_DIR%/build/generators/conan_toolchain.cmake
set TOOLCHAIN_SP=%BUILD_SP_DIR%/build/generators/conan_toolchain.cmake
for %%I in ("%~dp0..\mumps\build") do set MUMPS_DIR=%%~fI

REM METIS 5 for OpenSeesMP partition() (not bundled OTHER/METIS 4).
REM Prefer env METIS5_DIR; else use sibling ..\metis-5.1.0\install if present
REM (build with IDXTYPEWIDTH 32 / static /MT). Without it, partition stays a stub.
set METIS5_OPT=-UOPENSEES_METIS5_LIBRARY
if not defined METIS5_DIR (
  for %%I in ("%~dp0..\metis-5.1.0\install") do (
    if exist "%%~fI\include\metis.h" set METIS5_DIR=%%~fI
  )
)
if defined METIS5_DIR (
  set METIS5_OPT=-DMETIS5_DIR="%METIS5_DIR%" -UOPENSEES_METIS5_LIBRARY
  echo METIS5_DIR=%METIS5_DIR%
) else (
  echo WARNING: METIS 5 not found; OpenSeesMP partition^(^) will remain a stub.
)

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

REM build/ -> OpenSees.exe, OpenSeesMP.exe  (PARALLEL_PROCESSING=OFF)
conan install . -of %BUILD_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator="Visual Studio 17 2022"

"%CMAKE%" -S . -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN% ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  %METIS5_OPT% ^
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

"%CMAKE%" --build %BUILD_DIR% --config Release --target OpenSees OpenSeesMP --parallel 10

REM build-sp/ -> OpenSeesSP.exe  (PARALLEL_PROCESSING=ON)
conan install . -of %BUILD_SP_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator="Visual Studio 17 2022"

"%CMAKE%" -S . -B %BUILD_SP_DIR% -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_SP% ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DPARALLEL_PROCESSING=ON ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

"%CMAKE%" --build %BUILD_SP_DIR% --config Release --target OpenSeesSP --parallel 10

REM Tcl runtimes: OpenSees*.exe looks for lib\tcl8.6 next to the build tree root.
REM Must match conanfile.py tcl/8.6.11 (not an older cached tcl/8.6.10 package).
set TCL_VERSION=8.6.11
set TCL_PKG_LIB=
for /d %%d in ("%USERPROFILE%\.conan2\p\b\tcl*") do (
  if exist "%%d\p\lib\tcl8.6\init.tcl" (
    findstr /C:"package require -exact Tcl %TCL_VERSION%" "%%d\p\lib\tcl8.6\init.tcl" >nul 2>&1 && (
      set "TCL_PKG_LIB=%%d\p\lib"
      goto tcl_found
    )
  )
)
echo ERROR: Conan Tcl %TCL_VERSION% not found under %USERPROFILE%\.conan2\p\b
exit /b 1

:tcl_found
if not exist "%BUILD_DIR%\lib" mkdir "%BUILD_DIR%\lib"
robocopy "%TCL_PKG_LIB%\tcl8.6" "%BUILD_DIR%\lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1
robocopy "%TCL_PKG_LIB%\tcl8" "%BUILD_DIR%\lib\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1

if not exist "%BUILD_SP_DIR%\lib" mkdir "%BUILD_SP_DIR%\lib"
robocopy "%TCL_PKG_LIB%\tcl8.6" "%BUILD_SP_DIR%\lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1
robocopy "%TCL_PKG_LIB%\tcl8" "%BUILD_SP_DIR%\lib\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1
