@echo off
echo ========================================
echo Face Kinship Verification - Frontend
echo ========================================
echo.

REM Check if node_modules exists
if not exist node_modules (
    echo Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
)

echo Starting React development server...
echo App will open at http://localhost:3000
echo Press Ctrl+C to stop
echo.

npm start
