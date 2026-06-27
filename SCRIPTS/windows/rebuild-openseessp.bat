@echo off
setlocal
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "REPO=C:\Users\garaujor\source\repos\simpsoba\OpenSees"
if not defined OPENSEES_BUILD_SP_DIR set "OPENSEES_BUILD_SP_DIR=build-sp-cuda"
for %%I in ("%REPO%\..\mumps\build") do set "MUMPS_DIR=%%~fI"
"%CMAKE%" -S "%REPO%" -B "%REPO%\%OPENSEES_BUILD_SP_DIR%" -DPARALLEL_PROCESSING=ON -DMUMPS_DIR=%MUMPS_DIR%
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%REPO%\%OPENSEES_BUILD_SP_DIR%" --config Release --target OpenSeesSP --parallel 4
exit /b %ERRORLEVEL%
