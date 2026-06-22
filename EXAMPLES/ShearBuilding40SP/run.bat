@echo off
setlocal

set "SCRIPT=%~1"
if "%SCRIPT%"=="" set "SCRIPT=ShearBuilding40SP.tcl"

set "ROOT=%~dp0"
set "EXE=%ROOT%bin\OpenSeesSPFresco.exe"
set "MPIEXEC=C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin\mpiexec.exe"

if not exist "%EXE%" (
    echo ERROR: OpenSeesSPFresco.exe not found in bin\
    exit /b 1
)

call "C:\Program Files (x86)\Intel\oneAPI\setVars.bat" intel64 mod >nul 2>&1
cd /d "%ROOT%"

REM OpenSeesSPFresco uses dynamic tcl86t.dll; set AFTER setVars so it is not cleared
if exist "%ROOT%lib\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=%ROOT%lib\tcl8.6"
) else if exist "C:\Program Files\Tcl\lib\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=C:\Program Files\Tcl\lib\tcl8.6"
) else (
    echo ERROR: Tcl 8.6 script library not found.
    echo Expected %ROOT%lib\tcl8.6\init.tcl or a Tcl install under "C:\Program Files\Tcl"
    exit /b 1
)

echo Running %SCRIPT% with 4 MPI ranks...
echo TCL_LIBRARY=%TCL_LIBRARY%
echo.

set "LOG=%TEMP%\ShearBuilding40SP_run_%RANDOM%.log"
"%MPIEXEC%" -genv TCL_LIBRARY "%TCL_LIBRARY%" -n 4 "%EXE%" "%SCRIPT%" > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

type "%LOG%"
del /f /q "%LOG%" 2>nul
echo.
echo Exit code: %RC%
exit /b %RC%
