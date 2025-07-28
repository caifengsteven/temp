@echo off
echo =======================================================
echo Test Single Excel Stock (Quick Verification)
echo =======================================================
echo.
echo This will test one stock from your Excel list to verify
echo everything is working before starting the full mass testing.
echo.
echo Testing: sh600000 (first stock from Excel list)
echo.

dc_database_test_with_reports.exe sh600000

echo.
echo =======================================================
echo Single stock test completed!
echo =======================================================
echo.
echo If this worked successfully, you can now run:
echo   2_test_excel_stocks.bat
echo.
echo to start testing all 1254 stocks from your Excel files.
echo.
pause
