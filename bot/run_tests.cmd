@echo off
setlocal
set "PROJECT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%PROJECT_DIR%run_tests.ps1" %*
exit /b %ERRORLEVEL%
