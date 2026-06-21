@echo off
setlocal
cd /d "%~dp0..\.."
call SCRIPTS\windows\opensees-cuda-env.bat
if errorlevel 1 exit /b 1
cd EXAMPLES\KRAlphaExplicit\SDOF-OpenSees
"%OPENSEES_PYTHON%" plotResults.py %*
exit /b %ERRORLEVEL%
