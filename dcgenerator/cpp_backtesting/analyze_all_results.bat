@echo off
echo =======================================================
echo COMPREHENSIVE RESULTS ANALYSIS
echo =======================================================
echo.
echo This will analyze all corrected DC testing results from
echo both China and US markets.
echo.
echo Analysis includes:
echo - Individual market summaries
echo - Cross-market comparisons
echo - Original vs Corrected DCGenerator comparison
echo - Top performers identification
echo - Statistical insights
echo.

echo Analyzing China market results...
python analyze_corrected_results.py

echo.
echo Analyzing US market results...
python analyze_us_corrected_results.py

echo.
echo Analyzing combined market results...
python analyze_both_markets.py

echo.
echo =======================================================
echo Analysis completed!
echo =======================================================
echo.
echo Generated files:
echo - corrected_dc_summary.csv (China results)
echo - us_corrected_dc_summary.csv (US results)
echo - combined_corrected_dc_results.csv (Both markets)
echo.
pause
