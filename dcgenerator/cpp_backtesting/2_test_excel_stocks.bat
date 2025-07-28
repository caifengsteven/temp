@echo off
echo =======================================================
echo STEP 2: Test Excel Stock List with DC Generator
echo =======================================================
echo.
echo This will test all stocks from your Excel files:
echo - Automatic checkpoint/resume for each stock
echo - Detailed report saved for each stock
echo - Progress tracking across all stocks
echo - Can be stopped and resumed anytime
echo.
echo Make sure you have run 1_read_excel_files.bat first!
echo.
set /p confirm="Start testing Excel stocks? (y/N): "
if /i "%confirm%" NEQ "y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo Starting Excel stock testing...
echo Found 1254 stocks from Excel files (948 Shanghai + 306 Shenzhen)
echo Note: Progress is saved automatically
echo You can stop anytime with Ctrl+C and resume later
echo.

mass_testing.exe --test-excel

echo.
echo =======================================================
echo Excel stock testing session ended!
echo =======================================================
echo.
echo To resume: Run this script again
echo To check progress: Run 3_check_progress.bat
echo.
