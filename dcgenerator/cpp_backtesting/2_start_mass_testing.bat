@echo off
echo =======================================================
echo STEP 2: Mass Testing of All Symbols
echo =======================================================
echo.
echo This will test ALL discovered symbols one by one:
echo - Automatic checkpoint/resume for each symbol
echo - Detailed report saved for each symbol
echo - Progress tracking across all symbols
echo - Can be stopped and resumed anytime
echo.
echo WARNING: This is a VERY long process!
echo - 5000 symbols x average 30 minutes each = ~2500 hours
echo - Recommended to run in batches or on dedicated machine
echo.
echo You can stop anytime with Ctrl+C and resume later
echo by running this script again.
echo.
set /p confirm="Are you sure you want to start? (y/N): "
if /i "%confirm%" NEQ "y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo Starting mass testing...
echo Note: Progress is saved automatically
echo.

mass_testing.exe --test-all

echo.
echo =======================================================
echo Mass testing session ended!
echo =======================================================
echo.
echo To resume: Run this script again
echo To check progress: Run 3_check_progress.bat
echo.
pause
