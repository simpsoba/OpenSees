@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "BUILD=C:\Users\garaujor\source\repos\OpenSees\build\bin-fresco"
set "MPIEXEC=C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin\mpiexec.exe"

if exist "%ROOT%bin\OpenSeesFresco.exe" (
    set "EXE_SEQ=%ROOT%bin\OpenSeesFresco.exe"
    set "EXE_SP=%ROOT%bin\OpenSeesSPFresco.exe"
    if exist "%ROOT%bin\OpenSeesMPFresco.exe" (
        set "EXE_MP=%ROOT%bin\OpenSeesMPFresco.exe"
    ) else (
        set "EXE_MP=%BUILD%\OpenSeesMPFresco.exe"
    )
) else (
    set "EXE_SEQ=%BUILD%\OpenSeesFresco.exe"
    set "EXE_SP=%BUILD%\OpenSeesSPFresco.exe"
    set "EXE_MP=%BUILD%\OpenSeesMPFresco.exe"
)

call "C:\Program Files (x86)\Intel\oneAPI\setVars.bat" intel64 mod >nul 2>&1
cd /d "%ROOT%"

if exist "%ROOT%lib\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=%ROOT%lib\tcl8.6"
) else if exist "C:\Program Files\Tcl\lib\tcl8.6\init.tcl" (
    set "TCL_LIBRARY=C:\Program Files\Tcl\lib\tcl8.6"
)

set RC=0

echo ============================================================
echo 1/3 Sequential OpenSeesFresco - ShearBuilding40.tcl
echo ============================================================
"%EXE_SEQ%" ShearBuilding40.tcl
if errorlevel 1 set RC=1
echo.

echo ============================================================
echo 2/3 OpenSeesSPFresco (4 ranks) - ShearBuilding40SP.tcl
echo ============================================================
"%MPIEXEC%" -genv TCL_LIBRARY "%TCL_LIBRARY%" -n 4 "%EXE_SP%" ShearBuilding40SP.tcl
if errorlevel 1 set RC=1
echo.

echo ============================================================
echo 3/3 OpenSeesMPFresco (4 ranks) - ShearBuilding40MP.tcl
echo ============================================================
"%MPIEXEC%" -genv TCL_LIBRARY "%TCL_LIBRARY%" -n 4 "%EXE_MP%" ShearBuilding40MP.tcl
if errorlevel 1 set RC=1
echo.

echo ============================================================
echo Plot displacement comparison (Python)
echo ============================================================
set "PYTHON="
where python >nul 2>&1 && python -c "import matplotlib" >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\anaconda3\python.exe"
)
if not defined PYTHON (
    echo ERROR: python with matplotlib not found
    set RC=1
    goto :done
)
"%PYTHON%" "%ROOT%plot_compare.py"
if errorlevel 1 set RC=1

:done

echo.
echo Done. exit code: %RC%
exit /b %RC%
