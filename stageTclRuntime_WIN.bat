@echo off
setlocal EnableDelayedExpansion
REM Sync ActiveTcl tcl86t.dll + lib\tcl8.6 to OpenSeesMP runtime paths.
REM Run from OpenSees repo root after building OpenSeesMP.
REM Fixes: "package require -exact Tcl 8.6.11" vs tcl86t.dll 8.6.16 mismatch.

if not defined BUILD_DIR set "BUILD_DIR=build-mp"
if not defined TCL_ROOT set "TCL_ROOT=C:\Program Files\Tcl"

cd /d "%~dp0"

if not exist "!TCL_ROOT!\bin\tcl86t.dll" (
  echo ERROR: !TCL_ROOT!\bin\tcl86t.dll not found
  exit /b 1
)
if not exist "!TCL_ROOT!\lib\tcl8.6\init.tcl" (
  echo ERROR: !TCL_ROOT!\lib\tcl8.6\init.tcl not found
  exit /b 1
)
if not exist "!BUILD_DIR!\Release\OpenSeesMP.exe" (
  echo ERROR: !BUILD_DIR!\Release\OpenSeesMP.exe not found — build OpenSeesMP first
  exit /b 1
)

if not exist "!BUILD_DIR!\lib" mkdir "!BUILD_DIR!\lib"
robocopy "!TCL_ROOT!\lib\tcl8.6" "!BUILD_DIR!\lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS
if errorlevel 8 exit /b 1
if exist "!TCL_ROOT!\lib\tcl8" (
  robocopy "!TCL_ROOT!\lib\tcl8" "!BUILD_DIR!\lib\tcl8" /E /NFL /NDL /NJH /NJS /NC /NS
  if errorlevel 8 exit /b 1
)
if not exist "lib" mkdir "lib"
robocopy "!TCL_ROOT!\lib\tcl8.6" "lib\tcl8.6" /E /NFL /NDL /NJH /NJS /NC /NS
if errorlevel 8 exit /b 1
copy /Y "!TCL_ROOT!\bin\tcl86t.dll" "!BUILD_DIR!\Release\" >nul

echo OK: Tcl runtime staged for OpenSeesMP
echo   DLL:  !BUILD_DIR!\Release\tcl86t.dll
echo   lib:  !BUILD_DIR!\lib\tcl8.6
echo   lib:  %CD%\lib\tcl8.6
echo.
echo Set before mpiexec:
echo   set TCL_LIBRARY=%CD%\!BUILD_DIR!\lib\tcl8.6
endlocal
