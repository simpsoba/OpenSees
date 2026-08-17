REM rd /s /q build
REM rd /s /q build-sp

cd /d "%~dp0"

set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set BUILD_DIR=build
set BUILD_RELEASE=build\Release
set BUILD_SP_DIR=build-sp
set BUILD_SP_RELEASE=build-sp\Release
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

REM build/Release -> OpenSees.exe, OpenSeesMP.exe  (PARALLEL_PROCESSING=OFF)
conan install . -of %BUILD_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja

set TOOLCHAIN=%BUILD_RELEASE%/generators/conan_toolchain.cmake
if not exist "%TOOLCHAIN%" set TOOLCHAIN=%BUILD_DIR%/build/Release/generators/conan_toolchain.cmake

"%CMAKE%" -S . -B %BUILD_RELEASE% -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/ifx.exe" ^
  -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  %METIS5_OPT% ^
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

REM build-sp/Release -> OpenSeesSP.exe  (PARALLEL_PROCESSING=ON)
conan install . -of %BUILD_SP_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja
set TOOLCHAIN_SP=%BUILD_SP_RELEASE%/generators/conan_toolchain.cmake
if not exist "%TOOLCHAIN_SP%" set TOOLCHAIN_SP=%BUILD_SP_DIR%/build/Release/generators/conan_toolchain.cmake
"%CMAKE%" -S . -B %BUILD_SP_RELEASE% -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_SP% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/ifx.exe" ^
  -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DPARALLEL_PROCESSING=ON ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

cd %BUILD_RELEASE%
cmake --build . --target OpenSees --parallel 10
if errorlevel 1 exit /b 1
echo OpenSees built successfully: %BUILD_RELEASE%\OpenSees.exe

cmake --build . --target OpenSeesMP --parallel 10
if errorlevel 1 exit /b 1
echo OpenSeesMP built successfully: %BUILD_RELEASE%\OpenSeesMP.exe

REM cmake --build . --target OpenSeesPy
REM cmake --install .
cd ..\..

cd %BUILD_SP_RELEASE%
cmake --build . --target OpenSeesSP --parallel 10
if errorlevel 1 exit /b 1
echo OpenSeesSP built successfully: %BUILD_SP_RELEASE%\OpenSeesSP.exe
cd ..\..

REM Tcl runtimes: OpenSees*.exe in build\Release looks for ..\lib\tcl8.6
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
echo Tcl copied successfully to %BUILD_DIR%\lib\

if not exist "%BUILD_SP_DIR%\lib" mkdir "%BUILD_SP_DIR%\lib"
robocopy "%TCL_PKG_LIB%\tcl8.6" "%BUILD_SP_DIR%\lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1
robocopy "%TCL_PKG_LIB%\tcl8" "%BUILD_SP_DIR%\lib\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 exit /b 1
echo Tcl copied successfully to %BUILD_SP_DIR%\lib\

echo Build complete: %BUILD_RELEASE%\OpenSees.exe, %BUILD_RELEASE%\OpenSeesMP.exe, %BUILD_SP_RELEASE%\OpenSeesSP.exe
