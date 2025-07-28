@echo off
echo =======================================================
echo Testing US Market with FIXED Thresholds (50 symbols)
echo =======================================================
echo.
echo This will test 50 US symbols with corrected thresholds:
echo - 0.5%% to 5.0%% (instead of 0.05%% to 2%%)
echo - Should show realistic results like China market
echo.

echo Clearing old US progress...
if exist us_corrected_dc_progress.txt del us_corrected_dc_progress.txt

echo.
echo Starting test with 50 symbols...
echo.

us_mass_dc_corrected.exe

echo.
echo Test completed! Check a few reports to verify realistic results.
echo.
pause
