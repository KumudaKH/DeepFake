@echo off
echo ==============================================================================
echo Starting Deepfake Detection ^& Digital Forensics FastAPI Backend...
echo ==============================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment (.venv) not found.
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.11 is installed and in PATH.
        pause
        exit /b 1
    )
    echo [INFO] Installing dependencies from requirements.txt...
    ".venv\Scripts\pip.exe" install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Backend starting on http://127.0.0.1:8000
echo Health check available at: http://127.0.0.1:8000/api/health
echo Press Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m uvicorn src.backend.app.main:app --host 127.0.0.1 --port 8000 --reload
