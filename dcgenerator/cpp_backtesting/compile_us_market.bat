@echo off
echo =======================================================
echo Compiling US Market DC Test
echo =======================================================
echo.

g++ -std=c++17 -O3 -o us_market_dc_test.exe us_market_dc_test.cpp

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Compilation successful!
    echo.
    echo Testing database exploration...
    echo.
    us_market_dc_test.exe --explore
) else (
    echo.
    echo Compilation failed!
)

echo.
pause
