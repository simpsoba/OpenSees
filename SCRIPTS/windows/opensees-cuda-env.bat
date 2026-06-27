@echo off
REM Shared runtime env for OpenSees-CUDA on Windows (conda python, oneAPI, GPU PATH, Tcl).
set "REPO_ROOT=%~dp0..\.."
cd /d "%REPO_ROOT%"

set "ANACONDA_ROOT=%LOCALAPPDATA%\anaconda3"
if not exist "%ANACONDA_ROOT%\Scripts\conda.exe" set "ANACONDA_ROOT=%USERPROFILE%\AppData\Local\anaconda3"
set "OPENSEES_PYTHON=%ANACONDA_ROOT%\envs\opensees-cuda\python.exe"
if not exist "%OPENSEES_PYTHON%" (
  echo ERROR: conda env opensees-cuda not found at %OPENSEES_PYTHON%
  echo Run SCRIPTS\windows\setup-opensees-cuda-env.bat
  exit /b 1
)

call "%ANACONDA_ROOT%\Scripts\activate.bat" "%ANACONDA_ROOT%"
call conda activate opensees-cuda >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod >nul

if not defined CUDAToolkit_ROOT set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if not defined OPENSEES_CUDSS_DIR set "OPENSEES_CUDSS_DIR=C:\Program Files\NVIDIA cuDSS\v0.8"
set "PATH=%OPENSEES_CUDSS_DIR%\bin\12;%CUDAToolkit_ROOT%\bin;%PATH%"
if not defined OPENSEES_BUILD_DIR set "OPENSEES_BUILD_DIR=build-cuda"
if not defined OPENSEES_BUILD_SP_DIR set "OPENSEES_BUILD_SP_DIR=build-sp-cuda"
set "PYTHONPATH=%REPO_ROOT%\%OPENSEES_BUILD_DIR%\Release"
set "TCL_LIBRARY=%REPO_ROOT%\%OPENSEES_BUILD_DIR%\lib\tcl8.6"
