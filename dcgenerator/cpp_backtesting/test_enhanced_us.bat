@echo off
echo =======================================================
echo Enhanced US Market Database Test
echo =======================================================
echo.
echo This will test the enhanced Windows-specific database discovery
echo and exploration features.
echo.

echo Step 1: Compiling enhanced version...
g++ -std=c++17 -O3 -o us_market_dc_test.exe us_market_dc_test.cpp

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    pause
    exit /b
)

echo Compilation successful!
echo.

echo Step 2: Testing database discovery and exploration...
echo.
us_market_dc_test.exe --explore

echo.
echo =======================================================
echo Test completed!
echo =======================================================
echo.
echo If databases were found, you can also try:
echo   us_market_dc_test.exe --symbols
echo.
pause
