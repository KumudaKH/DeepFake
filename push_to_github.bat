@echo off
echo ==============================================================================
echo Pushing Deepfake Forensics Project to GitHub (KumudaKH/DeepFake)...
echo ==============================================================================
setlocal
cd /d "%~dp0"

echo Current Git Remote:
git remote -v
echo.
echo Pushing branch 'main' to GitHub origin...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] Code successfully pushed to https://github.com/KumudaKH/DeepFake
    echo ==============================================================================
) else (
    echo.
    echo [ERROR] Push failed. If prompted, please sign in to GitHub in your browser.
)
pause
