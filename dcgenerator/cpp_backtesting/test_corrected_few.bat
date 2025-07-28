@echo off
echo =======================================================
echo Testing Corrected DC Generator (First 5 Symbols)
echo =======================================================
echo.
echo This will test just the first 5 symbols to verify
echo the corrected DCGenerator is working properly.
echo.

echo Starting test...
echo.

china_mass_dc_corrected.exe

echo.
echo Test completed! Check the generated files:
echo.
dir corrected_report_*.txt
echo.
pause
