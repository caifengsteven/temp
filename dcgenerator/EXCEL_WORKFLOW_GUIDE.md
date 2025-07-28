# Excel-Based Stock Testing Workflow

## 🎯 **Objective**
Read stock codes from your three .xls files (column E), apply mapping rules, and test each stock with DC Generator strategies.

## 📋 **Mapping Rules**
- **Codes starting with 6** (e.g., 600001) → **sh600001** (Shanghai Exchange)
- **Other codes** (e.g., 000001) → **sz000001** (Shenzhen Exchange)

## 🔧 **Prerequisites**
1. **Place .xls files** in the `cpp_backtesting` directory
2. **Install Python dependencies**:
   ```bash
   pip install pandas xlrd
   ```

## 📋 **Step-by-Step Workflow**

### **Step 1: Read Excel Files**
```bash
./1_read_excel_files.bat
```
**What it does:**
- Reads all .xls files in current directory
- Extracts stock codes from column E
- Applies mapping rules (6xxxxx → sh6xxxxx, others → szxxxxxx)
- Saves mapped codes to `excel_stocks.txt`
- Creates detailed mapping report

**Output Files:**
- `excel_stocks.txt` - Mapped stock codes for testing
- `stock_mapping_details.txt` - Detailed mapping information

### **Step 2: Test Excel Stocks**
```bash
./2_test_excel_stocks.bat
```
**What it does:**
- Tests each mapped stock code automatically
- Saves detailed report for each stock: `report_SYMBOL.txt`
- Tracks progress in `testing_progress.txt`
- Can be stopped and resumed anytime (Ctrl+C)

**Output Files:**
- `report_sh600001.txt`, `report_sz000001.txt`, etc. - Individual reports
- `testing_progress.txt` - Overall progress tracking
- `checkpoint_SYMBOL.txt` - Individual stock checkpoints

### **Step 3: Monitor Progress**
```bash
./3_check_progress.bat
```
**Shows:**
- Total stocks from Excel files
- Number completed
- Number failed
- Percentage complete

### **Step 4: List Excel Stocks**
```bash
./3_list_excel_stocks.bat
```
**Shows:**
- All mapped stock codes
- Count by exchange (Shanghai vs Shenzhen)

## 📊 **Sample Output**

### **Excel File Reading:**
```
=== Excel Stock Code Reader (.xls files) ===
Found 3 .xls files:
  - file1.xls
  - file2.xls  
  - file3.xls

Reading: file1.xls
  Stock codes extracted: 150
  Sample codes: ['sh600001', 'sh600002', 'sz000001', 'sz000002', 'sh600003']

=== SUMMARY ===
Total unique stock codes found: 450
Shanghai Exchange (sh): 200
Shenzhen Exchange (sz): 250
```

### **Stock Testing Progress:**
```
=== TESTING PROGRESS ===
Total symbols: 450
Completed: 125
Failed: 5
Remaining: 320
Progress: 28.9%
```

### **Generated Reports:**
Each stock gets a detailed report like:
```
=== DATA SUMMARY ===
Symbol: sh600001
Total Price Points: 1,234,567
Price Range: $10.25 to $15.75
Testing Period: 2018-2025

=== Simple DC STRATEGY RESULTS ===
   Threshold    Trades    Final PnL    Return %
   0.5%         1,456     $12,450      12.45%
   1.0%           734     $8,230       8.23%
   ...

=== BEST PERFORMING CONFIGURATION ===
Best Strategy: Simple DC
Best Threshold: 0.5%
Best Return: 12.45%
```

## 🔄 **Resume Capability**

### **If Interrupted:**
- Simply run `./2_test_excel_stocks.bat` again
- Automatically resumes from where it stopped
- No progress lost

### **Individual Stock Resume:**
- Each stock has its own checkpoint file
- If a stock test is interrupted, it resumes from checkpoint
- Complete fault tolerance

## 📁 **File Organization**

### **Input Files:**
```
cpp_backtesting/
├── file1.xls                    # Your Excel files
├── file2.xls
├── file3.xls
└── read_excel_stocks.py         # Excel reader script
```

### **Generated Files:**
```
cpp_backtesting/
├── excel_stocks.txt             # Mapped stock codes
├── stock_mapping_details.txt    # Mapping details
├── testing_progress.txt         # Overall progress
├── report_sh600001.txt          # Individual reports
├── report_sz000001.txt
├── ...                          # One report per stock
├── checkpoint_sh600001.txt      # Individual checkpoints
├── checkpoint_sz000001.txt
└── ...                          # One checkpoint per stock
```

## 🎯 **Expected Results**

### **Typical Excel File Results:**
- **~300-500 unique stocks** from three Excel files
- **Mix of Shanghai and Shenzhen** exchange stocks
- **Complete DC analysis** for each stock
- **Best strategy identification** per stock

### **Analysis Benefits:**
- **Focused testing** on your specific stock universe
- **Exchange comparison** (Shanghai vs Shenzhen performance)
- **Sector analysis** if Excel files are organized by sector
- **Portfolio optimization** based on DC strategy performance

## 🛠️ **Troubleshooting**

### **Excel Reading Issues:**
```bash
# Install required library
pip install xlrd

# Check file format
# Make sure files are .xls (not .xlsx)

# Verify column E contains stock codes
# Open Excel file and check column E has 6-digit codes
```

### **Missing Stock Data:**
- Some mapped codes might not exist in your database
- Check `testing_progress.txt` for failed symbols
- Review individual error messages during testing

### **Resume Testing:**
```bash
# To resume interrupted testing
./2_test_excel_stocks.bat

# To check what's completed
./3_check_progress.bat

# To see which stocks are being tested
./3_list_excel_stocks.bat
```

## 📈 **Analysis Workflow**

### **During Testing:**
1. **Monitor progress** with `3_check_progress.bat`
2. **Review completed reports** as they're generated
3. **Identify patterns** in Shanghai vs Shenzhen performance

### **After Completion:**
1. **Compare exchange performance** (sh vs sz stocks)
2. **Identify best performing stocks** from your Excel lists
3. **Analyze sector patterns** if Excel files represent different sectors
4. **Create portfolio** based on best DC configurations

This Excel-based workflow provides targeted testing of your specific stock universe with complete automation and fault tolerance.
