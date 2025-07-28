@echo off
echo =======================================================
echo Copy US Market Databases (Windows Solution)
echo =======================================================
echo.
echo This will copy the database files to the current directory
echo to avoid encoding issues with Chinese characters in path.
echo.
echo WARNING: This requires 428 GB of disk space!
echo - US_ETF_1min.db: 51 GB
echo - US_stock_1min.db: 377 GB
echo.
set /p confirm="Continue with copy? (y/N): "
if /i "%confirm%" NEQ "y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo Copying US_ETF_1min.db (51 GB)...
echo This may take several minutes...
copy "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_ETF_1min.db" US_ETF_1min.db

if %ERRORLEVEL% EQU 0 (
    echo US_ETF_1min.db copied successfully!
) else (
    echo Failed to copy US_ETF_1min.db
    echo Please check the source path exists
    pause
    exit /b
)

echo.
echo Copying US_stock_1min.db (377 GB)...
echo This may take 30+ minutes...
copy "F:\BaiduNetdiskDownload\US stock ane etf 1mins\US_stock_1min.db" US_stock_1min.db

if %ERRORLEVEL% EQU 0 (
    echo US_stock_1min.db copied successfully!
) else (
    echo Failed to copy US_stock_1min.db
    echo Please check the source path exists
    pause
    exit /b
)

echo.
echo =======================================================
echo Databases copied successfully!
echo You can now use:
echo   us_market_dc_test.exe --explore
echo =======================================================
pause
