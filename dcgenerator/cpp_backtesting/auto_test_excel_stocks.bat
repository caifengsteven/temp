@echo off
echo =======================================================
echo AUTOMATED Excel Stock Testing (No Prompts)
echo =======================================================
echo.
echo This will automatically test all 1254 stocks from Excel files
echo without any keyboard prompts or pauses.
echo.
echo Features:
echo - Fully automated (no user input required)
echo - Automatic checkpoint/resume for each stock
echo - Detailed report saved for each stock
echo - Progress tracking across all stocks
echo - Can be stopped with Ctrl+C and resumed later
echo.
echo Starting automated testing in 3 seconds...
timeout /t 3 /nobreak >nul
echo.
echo Starting Excel stock testing...
echo Found 1254 stocks from Excel files (948 Shanghai + 306 Shenzhen)
echo Note: Progress is saved automatically
echo You can stop anytime with Ctrl+C and resume later
echo.

mass_testing.exe --test-excel

echo.
echo =======================================================
echo Automated Excel stock testing completed!
echo =======================================================
echo.
echo To resume: Run this script again
echo To check progress: Run 3_check_progress.bat
echo.
