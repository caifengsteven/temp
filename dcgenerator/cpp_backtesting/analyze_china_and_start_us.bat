@echo off
echo =======================================================
echo Analyze China Results + Start US Testing
echo =======================================================
echo.
echo This will:
echo 1. Analyze the completed China results
echo 2. Start US market testing in parallel
echo.

echo Step 1: Analyzing China results...
python analyze_corrected_results.py

echo.
echo Step 2: Starting US market testing in new window...
start "US Market DC Testing" cmd /c "us_mass_dc_corrected.exe & pause"

echo.
echo =======================================================
echo China analysis completed, US testing started!
echo =======================================================
echo.
echo - China results analyzed and saved to CSV
echo - US testing running in separate window
echo - You can monitor US progress in the new window
echo.
pause
