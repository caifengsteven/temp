@echo off
echo =======================================================
echo STEP 1: Scan All Databases for Stock Symbols
echo =======================================================
echo.
echo This will scan ALL databases from 2018-2025 to find
echo every unique stock symbol in your trading data.
echo.
echo Expected: Around 5000 unique symbols
echo Time: 10-30 minutes depending on database size
echo.
echo Starting scan...
echo.

mass_testing.exe --scan

echo.
echo =======================================================
echo Scan completed!
echo =======================================================
echo.
echo Next step: Run 2_start_mass_testing.bat to test all symbols
echo.
pause
