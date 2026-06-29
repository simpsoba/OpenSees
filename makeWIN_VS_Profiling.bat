REM rd /s /q build-prof
REM rd /s /q build-sp-prof

REM Same as makeWIN_VS.bat, but RelWithDebInfo (Release-like + PDBs) in separate trees.

cd /d "%~dp0"

set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set BUILD_DIR=build-prof
set BUILD_SP_DIR=build-sp-prof
set BUILD_CFG=RelWithDebInfo
set TOOLCHAIN=%BUILD_DIR%/build/generators/conan_toolchain.cmake
set TOOLCHAIN_SP=%BUILD_SP_DIR%/build/generators/conan_toolchain.cmake
for %%I in ("%~dp0..\mumps\build") do set MUMPS_DIR=%%~fI

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

REM build-prof/ -> OpenSees.exe, OpenSeesMP.exe  (PARALLEL_PROCESSING=OFF)
conan install . -of %BUILD_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator="Visual Studio 17 2022"

"%CMAKE%" -S . -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN% ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

"%CMAKE%" --build %BUILD_DIR% --config %BUILD_CFG% --target OpenSees OpenSeesMP --parallel 10

REM build-sp-prof/ -> OpenSeesSP.exe  (PARALLEL_PROCESSING=ON)
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

"%CMAKE%" --build %BUILD_SP_DIR% --config %BUILD_CFG% --target OpenSeesSP --parallel 10

REM Tcl runtimes: OpenSees*.exe looks for lib\tcl8.6 next to the build tree root.
set TCL_PKG_LIB=
for /d %%d in ("%USERPROFILE%\.conan2\p\b\tcl*") do (
  if exist "%%d\p\lib\tcl8.6\init.tcl" (
    set "TCL_PKG_LIB=%%d\p\lib"
    goto tcl_found
  )
)
echo ERROR: Conan Tcl not found under %USERPROFILE%\.conan2\p\b
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
