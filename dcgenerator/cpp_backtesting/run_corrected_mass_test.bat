@echo off
echo =======================================================
echo China A-Share Mass DC Testing (CORRECTED VERSION)
echo =======================================================
echo.
echo This will test ALL available China A-share stocks with
echo the CORRECTED DCGenerator that fixes the bugs found in
echo the original implementation.
echo.
echo Features:
echo - Individual report file for each stock
echo - Checkpoint/resume functionality
echo - Progress tracking
echo - Comparison with original buggy results
echo.
echo Output files:
echo - corrected_report_[symbol].txt (individual reports)
echo - corrected_dc_progress.txt (checkpoint file)
echo.
echo You can stop the process anytime with Ctrl+C and resume
echo later by running this script again.
echo.
pause

echo.
echo Starting corrected mass testing...
echo.

china_mass_dc_corrected.exe

echo.
echo =======================================================
echo Mass testing completed!
echo =======================================================
echo.
echo Check the generated files:
echo - corrected_report_*.txt for individual stock results
echo - corrected_dc_progress.txt for progress tracking
echo.
pause
