# How to Use DC Database Test with Stock Symbol Parameter

## ✅ **Updated Executable: `dc_database_test_fixed.exe`**

The executable now accepts a stock symbol as a command line parameter to test specific symbols.

## **Usage:**

### **Method 1: Direct Command Line**
```bash
cd cpp_backtesting
./dc_database_test_fixed.exe SYMBOL
```

**Examples:**
```bash
./dc_database_test_fixed.exe AAPL
./dc_database_test_fixed.exe MSFT
./dc_database_test_fixed.exe TSLA
./dc_database_test_fixed.exe GOOGL
```

### **Method 2: Using Batch File**
```bash
cd cpp_backtesting
./run_symbol_test.bat SYMBOL
```

**Examples:**
```bash
./run_symbol_test.bat AAPL
./run_symbol_test.bat MSFT
```

### **Method 3: See Available Symbols**
Run without parameters to see available symbols:
```bash
./dc_database_test_fixed.exe
```

This will show you a list of available symbols from your database.

## **What the Program Does:**

1. **Loads ALL data** for the specified symbol from ALL databases (2018-2025)
2. **Tests 3 DC strategies**:
   - Simple DC: Buy on downturn end, sell on upturn end
   - Contrarian DC: Buy on upturn end, sell on downturn end  
   - Long Only DC: Only buy on downturn end, hold position
3. **Tests 6 thresholds**: 0.5%, 1.0%, 1.5%, 2.0%, 3.0%, 5.0%
4. **Calculates P&L** and returns for each strategy/threshold combination

## **Expected Output:**
```
=== DC Generator Multi-Year Database Test ===
Target symbol: AAPL
Testing period: 2018-2025
Testing thresholds: 0.5% and above
Testing multiple DC strategies

SQLite library loaded successfully!

=== Testing with symbol: AAPL ===
Loading ALL available data from 2018-2025 databases...
Found 84 database files to process
  2018_01.db: 123,456 records
  2018_02.db: 134,567 records
  ...
Total loaded: 12,345,678 price points across all years
Price range: $150.25 to $180.75

=== Simple DC Strategy Results ===
Initial Capital: $100,000

   Threshold      Trades      Final PnL    Return %
--------------------------------------------------
        0.5%        2,456        $15,670      15.67%
        1.0%        1,234         $8,450       8.45%
        1.5%          678         $5,230       5.23%
        2.0%          345         $3,120       3.12%
        3.0%          156         $1,890       1.89%
        5.0%           67           $890       0.89%

=== Contrarian DC Strategy Results ===
...

=== Long Only DC Strategy Results ===
...
```

## **Error Handling:**

- **No symbol provided**: Shows usage and available symbols
- **Symbol not found**: Shows error message and exits
- **No data found**: Shows error message for the specific symbol
- **Invalid prices**: Automatically filtered out with safety checks

## **Performance Notes:**

- Processing time depends on data volume for the symbol
- Large symbols (high trading volume) will take longer to process
- Progress indicators show database loading status
- All calculations include safety checks to prevent NaN/infinity values

## **Tips:**

1. **Start with common symbols** like AAPL, MSFT, GOOGL
2. **Check available symbols first** by running without parameters
3. **Use batch file** for easier repeated testing
4. **Monitor output** for data loading progress
5. **Compare strategies** to find optimal DC parameters for your symbol
