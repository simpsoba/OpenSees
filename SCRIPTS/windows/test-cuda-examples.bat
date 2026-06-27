@echo off
setlocal
cd /d "%~dp0..\.."
call SCRIPTS\windows\opensees-cuda-env.bat
if errorlevel 1 exit /b 1

echo === CudaExplicitAlpha smoke (Python) ===
"%OPENSEES_PYTHON%" SCRIPTS\windows\run_cuda_smoke_test.py
if errorlevel 1 exit /b 1

echo === CudaExplicitAlpha_TP smoke (Python) ===
"%OPENSEES_PYTHON%" SCRIPTS\windows\run_cuda_tp_smoke_test.py
if errorlevel 1 exit /b 1

echo === CudaExplicitAlpha_TP smoke (Tcl / OpenSees.exe) ===
"%REPO_ROOT%\%OPENSEES_BUILD_DIR%\Release\OpenSees.exe" tests\cuda_explicit_alpha_tp_smoke.tcl
if errorlevel 1 exit /b 1

echo === KRAlphaExplicit SDOF: CudaKRAlpha ===
cd EXAMPLES\KRAlphaExplicit\SDOF-OpenSees
"%OPENSEES_PYTHON%" run_one_integrator.py --method CudaKRAlpha --params "[0.5]" --ic init_disp --dt_tag dt_0.2
if errorlevel 1 exit /b 1

echo === KRAlphaExplicit SDOF: CudaKRAlpha_TP ===
"%OPENSEES_PYTHON%" run_one_integrator.py --method CudaKRAlpha_TP --params "[0.5]" --ic init_disp --dt_tag dt_0.2
if errorlevel 1 exit /b 1

echo ALL_EXAMPLES_OK
endlocal
