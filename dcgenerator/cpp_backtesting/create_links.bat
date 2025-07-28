@echo off
echo =======================================================
echo Creating Hard Links for US Market Databases (Windows)
echo =======================================================
echo.
echo This will create hard links in the current directory
echo pointing to the actual database files.
echo.
echo Note: This script must be run as Administrator for symbolic links,
echo or we'll use hard links which don't require admin privileges.
echo.

echo Creating hard link for US_ETF_1min.db...
mklink /H US_ETF_1min.db "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_ETF_1min.db"

if %ERRORLEVEL% NEQ 0 (
    echo Hard link failed, trying copy instead...
    copy "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_ETF_1min.db" US_ETF_1min.db
)

echo.
echo Creating hard link for US_stock_1min.db...
mklink /H US_stock_1min.db "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_stock_1min.db"

if %ERRORLEVEL% NEQ 0 (
    echo Hard link failed, trying copy instead...
    copy "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_stock_1min.db" US_stock_1min.db
)

echo.
echo =======================================================
echo Links/copies created! You can now use:
echo   us_market_dc_test.exe --explore
echo =======================================================
pause
