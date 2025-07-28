@echo off
echo =======================================================
echo Running PowerShell Script to Create Links
echo =======================================================
echo.
echo This will run a PowerShell script to create symbolic links
echo for the US market databases.
echo.

powershell -ExecutionPolicy Bypass -File create_links.ps1

echo.
echo =======================================================
echo PowerShell script completed!
echo =======================================================
pause
