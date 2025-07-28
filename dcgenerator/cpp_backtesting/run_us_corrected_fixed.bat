@echo off
echo =======================================================
echo US Market DC Testing - FIXED THRESHOLDS
echo =======================================================
echo.
echo This will run US market testing with CORRECTED thresholds:
echo - Changed from 0.05%% (too aggressive) to 0.5%% (realistic)
echo - Thresholds: 0.5%%, 1.0%%, 1.5%%, 2.0%%, 3.0%%, 5.0%%
echo - Should eliminate false DC events
echo - Expected realistic trade counts and returns
echo.
echo IMPORTANT: This will clear old US progress to start fresh
echo with corrected thresholds.
echo.
pause

echo.
echo Clearing old US progress file...
if exist us_corrected_dc_progress.txt (
    del us_corrected_dc_progress.txt
    echo Old progress cleared.
) else (
    echo No old progress file found.
)

echo.
echo Starting US market testing with corrected thresholds...
echo.

us_mass_dc_corrected.exe

echo.
echo =======================================================
echo US Market testing with corrected thresholds completed!
echo =======================================================
echo.
echo Results should now show:
echo - Realistic trade counts (similar to China results)
echo - Modest returns (not extreme percentages)
echo - Proper DC event detection
echo.
echo Run analyze_us_corrected_results.py to see the corrected analysis.
echo.
pause
