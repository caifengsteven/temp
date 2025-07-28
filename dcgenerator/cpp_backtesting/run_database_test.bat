@echo off
echo =======================================================
echo DC Generator Multi-Year Database Test with Multiple Strategies
echo =======================================================
echo.
echo This will connect to ALL your trading databases and:
echo - Test DC algorithm with ALL data from 2018-2025
echo - Test thresholds 0.5%% and above only
echo - Test 3 different DC trading strategies
echo - Calculate P^&L for each strategy and threshold
echo.
echo Databases: I:/zhubi/cpp_implementation/sqlite_databases/2018-2025/
echo Strategies: Simple DC, Contrarian DC, Long Only DC
echo Thresholds: 0.5%%, 1.0%%, 1.5%%, 2.0%%, 3.0%%, 5.0%%
echo.
echo Starting comprehensive test...
echo.

dc_database_test.exe

echo.
echo =======================================================
echo Multi-year test completed!
echo =======================================================
pause
