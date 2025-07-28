@echo off
echo =======================================================
echo DUAL MARKET DC TESTING (CHINA + US)
echo =======================================================
echo.
echo This will run BOTH China and US market testing in parallel
echo using the CORRECTED DCGenerator.
echo.
echo Two separate processes will be started:
echo 1. China A-share market testing
echo 2. US stock and ETF market testing
echo.
echo Each process has its own:
echo - Individual report files
echo - Checkpoint/resume capability
echo - Progress tracking
echo.
echo You can monitor both processes and stop/resume them
echo independently.
echo.
pause

echo.
echo Starting China market testing in new window...
start "China Market DC Testing" cmd /c "china_mass_dc_corrected.exe & pause"

echo.
echo Waiting 5 seconds before starting US market testing...
timeout /t 5 /nobreak

echo.
echo Starting US market testing in new window...
start "US Market DC Testing" cmd /c "us_mass_dc_corrected.exe & pause"

echo.
echo =======================================================
echo Both market testing processes started!
echo =======================================================
echo.
echo Two separate windows are now running:
echo - China Market DC Testing
echo - US Market DC Testing
echo.
echo You can close this window. The testing will continue
echo in the separate windows.
echo.
echo Output files:
echo China: corrected_report_*.txt, corrected_dc_progress.txt
echo US: us_corrected_report_*.txt, us_corrected_dc_progress.txt
echo.
pause
