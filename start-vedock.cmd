@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Vedock environment is missing. Running setup first...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-portable.ps1"
  if errorlevel 1 exit /b 1
)
echo Starting Vedock at http://127.0.0.1:5464
echo Press Ctrl+C to stop the server.
"%~dp0.venv\Scripts\python.exe" "%~dp0serve.py"
endlocal
