@echo off
setlocal
cd /d "%~dp0..\.."
call SCRIPTS\windows\opensees-cuda-env.bat
if errorlevel 1 exit /b 1
cd EXAMPLES\KRAlphaExplicit\SDOF-OpenSees
"%OPENSEES_PYTHON%" run_one_integrator.py --method %1 --params "[0.5]" --ic init_disp --dt_tag dt_0.2
exit /b %ERRORLEVEL%
