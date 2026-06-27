@echo off
setlocal
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "REPO=C:\Users\garaujor\source\repos\simpsoba\OpenSees"
if not defined OPENSEES_BUILD_DIR set "OPENSEES_BUILD_DIR=build-cuda"
"%CMAKE%" -S "%REPO%" -B "%REPO%\%OPENSEES_BUILD_DIR%"
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%REPO%\%OPENSEES_BUILD_DIR%" --config Release --target OpenSeesMP --parallel 4
exit /b %ERRORLEVEL%
