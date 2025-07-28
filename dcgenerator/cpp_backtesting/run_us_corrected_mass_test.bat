@echo off
echo =======================================================
echo US Market Mass DC Testing (CORRECTED VERSION)
echo =======================================================
echo.
echo This will test ALL available US stocks and ETFs with
echo the CORRECTED DCGenerator that fixes the bugs found in
echo the original implementation.
echo.
echo Features:
echo - Individual report file for each US symbol
echo - Checkpoint/resume functionality
echo - Progress tracking
echo - Optimized for 1-minute high-frequency data
echo - Realistic DC thresholds (0.05%% to 2%%)
echo.
echo Output files:
echo - us_corrected_report_[symbol].txt (individual reports)
echo - us_corrected_dc_progress.txt (checkpoint file)
echo.
echo You can stop the process anytime with Ctrl+C and resume
echo later by running this script again.
echo.
pause

echo.
echo Starting US market corrected mass testing...
echo.

us_mass_dc_corrected.exe

echo.
echo =======================================================
echo US Market mass testing completed!
echo =======================================================
echo.
echo Check the generated files:
echo - us_corrected_report_*.txt for individual stock results
echo - us_corrected_dc_progress.txt for progress tracking
echo.
pause
