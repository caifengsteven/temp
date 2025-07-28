@echo off
echo =======================================================
echo US FUTURES 1-MINUTE DC TESTING
echo =======================================================
echo.
echo This will test US futures 1-minute data with the
echo CORRECTED DCGenerator using realistic thresholds.
echo.
echo Features:
echo - Explores database structure first
echo - Tests common futures symbols (ES, NQ, CL, GC, etc.)
echo - Uses corrected thresholds (0.5%% to 5%%)
echo - Shows both Simple and Contrarian DC strategies
echo - Up to 100,000 data points per symbol
echo.
echo Database: F:\database\us futures 1mins\us_fut_1min.db
echo.
echo Expected results:
echo - Realistic trade counts for futures volatility
echo - Proper DC event detection
echo - Performance comparison between strategies
echo.
pause

echo.
echo Starting US futures DC testing...
echo.

us_futures_dc_test.exe

echo.
echo =======================================================
echo US Futures DC testing completed!
echo =======================================================
echo.
echo The results show how the corrected DCGenerator performs
echo on high-frequency futures data with proper thresholds.
echo.
pause
