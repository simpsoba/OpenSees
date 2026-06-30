REM rd /s /q build
REM rd /s /q build-sp

cd /d "%~dp0"

set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set BUILD_DIR=build
set BUILD_RELEASE=build\Release
set BUILD_SP_DIR=build-sp
set BUILD_SP_RELEASE=build-sp\Release
for %%I in ("%~dp0..\mumps\build") do set MUMPS_DIR=%%~fI

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
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

REM build-sp/Release -> OpenSeesSP.exe  (PARALLEL_PROCESSING=ON)
REM conan install . -of %BUILD_SP_DIR% -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja
REM set TOOLCHAIN_SP=%BUILD_SP_RELEASE%/generators/conan_toolchain.cmake
REM if not exist "%TOOLCHAIN_SP%" set TOOLCHAIN_SP=%BUILD_SP_DIR%/build/Release/generators/conan_toolchain.cmake
REM "%CMAKE%" -S . -B %BUILD_SP_RELEASE% -G Ninja ^
REM   -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_SP% ^
REM   -DCMAKE_BUILD_TYPE=Release ^
REM   -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
REM   -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
REM   -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
REM   -DCMAKE_Fortran_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/ifx.exe" ^
REM   -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 ^
REM   -DMUMPS_DIR="%MUMPS_DIR%" ^
REM   -DPARALLEL_PROCESSING=ON ^
REM   -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
REM   -DCMAKE_INSTALL_PREFIX=%USERPROFILE%\bin\OpenSees3.8.0

cd %BUILD_RELEASE%
cmake --build . --target OpenSees --parallel 10
if errorlevel 1 exit /b 1
echo OpenSees built successfully: %BUILD_RELEASE%\OpenSees.exe

REM cmake --build . --target OpenSeesMP --parallel 10
REM if errorlevel 1 exit /b 1
REM echo OpenSeesMP built successfully: %BUILD_RELEASE%\OpenSeesMP.exe

REM cmake --build . --target OpenSeesPy
REM cmake --install .
cd ..\..

REM cd %BUILD_SP_RELEASE%
REM cmake --build . --target OpenSeesSP --parallel 10
REM if errorlevel 1 exit /b 1
REM echo OpenSeesSP built successfully: %BUILD_SP_RELEASE%\OpenSeesSP.exe
REM cd ..\..

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

REM if not exist "%BUILD_SP_DIR%\lib" mkdir "%BUILD_SP_DIR%\lib"
REM robocopy "%TCL_PKG_LIB%\tcl8.6" "%BUILD_SP_DIR%\lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS >nul
REM robocopy "%TCL_PKG_LIB%\tcl8" "%BUILD_SP_DIR%\lib\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS >nul

echo Build complete: %BUILD_RELEASE%\OpenSees.exe
