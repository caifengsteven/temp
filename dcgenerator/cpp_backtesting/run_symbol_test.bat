@echo off
echo =======================================================
echo DC Generator Symbol-Specific Test
echo =======================================================
echo.
echo Usage: run_symbol_test.bat [SYMBOL]
echo Example: run_symbol_test.bat AAPL
echo.

if "%1"=="" (
    echo Error: Please provide a stock symbol as parameter
    echo.
    echo Example usage:
    echo   run_symbol_test.bat AAPL
    echo   run_symbol_test.bat MSFT
    echo   run_symbol_test.bat TSLA
    echo.
    echo Running without symbol to see available symbols...
    echo.
    dc_database_test_fixed.exe
) else (
    echo Testing symbol: %1
    echo.
    echo This will:
    echo - Load ALL data for %1 from 2018-2025 databases
    echo - Test thresholds 0.5%% and above
    echo - Test 3 different DC trading strategies
    echo - Calculate P^&L for each strategy and threshold
    echo.
    echo Starting test for %1...
    echo.
    dc_database_test_fixed.exe %1
)

echo.
echo =======================================================
echo Test completed!
echo =======================================================
pause
