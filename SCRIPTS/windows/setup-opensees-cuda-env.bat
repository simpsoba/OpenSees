@echo off
REM Create/update the opensees-cuda conda environment for OpenSees-CUDA builds.
setlocal

set "ANACONDA_ROOT=%LOCALAPPDATA%\anaconda3"
if not exist "%ANACONDA_ROOT%\Scripts\conda.exe" set "ANACONDA_ROOT=%USERPROFILE%\AppData\Local\anaconda3"
if not exist "%ANACONDA_ROOT%\Scripts\conda.exe" (
  echo ERROR: Anaconda not found at %ANACONDA_ROOT%
  exit /b 1
)

call "%ANACONDA_ROOT%\Scripts\activate.bat" "%ANACONDA_ROOT%"

set "REPO_ROOT=%~dp0..\..\"
pushd "%REPO_ROOT%"

conda env update -f SCRIPTS\windows\environment-opensees-cuda.yml --prune
if errorlevel 1 (
  conda env create -f SCRIPTS\windows\environment-opensees-cuda.yml
)

echo.
echo Done. Activate with:
echo   call "%ANACONDA_ROOT%\Scripts\activate.bat" "%ANACONDA_ROOT%"
echo   conda activate opensees-cuda
echo Then build from repo root: makeWIN_CUDA.bat

popd
endlocal
