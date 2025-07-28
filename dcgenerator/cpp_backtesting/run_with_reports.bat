@echo off
echo =======================================================
echo DC Generator with Checkpoint and Report Generation
echo =======================================================
echo.
echo Usage: run_with_reports.bat [SYMBOL]
echo Example: run_with_reports.bat AAPL
echo.

if "%1"=="" (
    echo Error: Please provide a stock symbol as parameter
    echo.
    echo Example usage:
    echo   run_with_reports.bat AAPL
    echo   run_with_reports.bat MSFT
    echo   run_with_reports.bat TSLA
    echo.
    echo Running without symbol to see available symbols...
    echo.
    dc_database_test_with_reports.exe
) else (
    echo Testing symbol: %1
    echo.
    echo Features:
    echo - Automatic checkpoint/resume (can stop and restart anytime)
    echo - Detailed report saved to report_%1.txt
    echo - Tests ALL data from 2018-2025 databases
    echo - Tests 3 DC strategies with thresholds 0.5%% and above
    echo.
    echo Starting test for %1...
    echo Note: You can press Ctrl+C to stop and resume later
    echo.
    dc_database_test_with_reports.exe %1
    echo.
    echo Report saved to: report_%1.txt
    echo Checkpoint saved to: checkpoint_%1.txt
)

echo.
echo =======================================================
echo Test completed!
echo =======================================================
pause
