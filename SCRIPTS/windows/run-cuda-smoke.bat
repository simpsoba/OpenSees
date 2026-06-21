@echo off
setlocal
cd /d "%~dp0..\.."
call SCRIPTS\windows\opensees-cuda-env.bat
if errorlevel 1 exit /b 1
"%OPENSEES_PYTHON%" SCRIPTS\windows\run_cuda_smoke_test.py
endlocal
