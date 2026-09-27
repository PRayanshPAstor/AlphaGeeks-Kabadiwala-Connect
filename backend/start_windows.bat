@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo   Kabadiwala Connect - SIH Backend
echo ========================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed or not in PATH.
  echo Install Python 3.10+ and try again.
  pause
  exit /b 1
)
echo Installing/checking required packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install required packages.
  pause
  exit /b 1
)
echo.
echo Starting backend at http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
