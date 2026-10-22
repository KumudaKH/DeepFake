@echo off
echo ==============================================================================
echo Starting Deepfake Detection ^& Digital Forensics React Dashboard...
echo ==============================================================================
setlocal
cd /d "%~dp0frontend"

rem Check for Node.js / npm
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "..\..\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\npm.cmd" (
        set "PATH=%~dp0..\..\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;%PATH%"
    ) else (
        echo [ERROR] Node.js and npm were not found in PATH.
        echo Please install Node.js 18+ from https://nodejs.org/
        pause
        exit /b 1
    )
)

echo Frontend starting on http://localhost:5173
echo Ensure backend is running on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

if not exist "node_modules" (
    echo [INFO] Installing dependencies (first-time setup)...
    call npm install
)

call npm run dev
