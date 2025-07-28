@echo off
echo =======================================================
echo STEP 1: Read Stock Codes from Excel Files
echo =======================================================
echo.
echo This will read stock codes from .xls files in current directory
echo and apply the mapping rules:
echo - Codes starting with 6 (e.g., 600001) -^> sh600001
echo - Other codes (e.g., 000001) -^> sz000001
echo.
echo Make sure your .xls files are in this directory!
echo.
echo Starting Excel file processing...
echo.

python read_excel_stocks.py

echo.
echo =======================================================
echo Excel processing completed!
echo =======================================================
echo.
echo Generated files:
echo - excel_stocks.txt: Stock codes for testing
echo - stock_mapping_details.txt: Detailed mapping info
echo.
echo Next step: Run 2_test_excel_stocks.bat
echo.
pause
