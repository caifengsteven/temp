# 📊 Daily QDII Fund Data Retriever - Usage Guide

## 🎯 **What This Script Does**

The `daily_qdii_data_retriever.py` script automatically retrieves:

1. **Latest NAV** (Net Asset Value) for all 153 QDII funds
2. **Latest Close Prices** with volume and trading amount
3. **Underlying Index Information** (index code and name)
4. **Price vs NAV Analysis** (differences and percentages)

## 🚀 **How to Run Daily**

### Simple Command
```bash
python daily_qdii_data_retriever.py
```

### What Happens
1. ✅ Connects to Wind terminal automatically
2. 📋 Loads all 155 QDII funds from `qdii_funds_parsed.csv`
3. 📊 Retrieves latest NAV, close prices, and index data
4. 💾 Saves everything to `qdii_daily_data_YYYYMMDD.csv`
5. 📈 Shows summary statistics
6. 🔌 Disconnects from Wind

## 📁 **Output File Structure**

Each day creates: `qdii_daily_data_YYYYMMDD.csv`

### Columns Included:
- **wind_code**: Fund identifier (e.g., 513090.SH)
- **fund_name**: Fund name in Chinese
- **latest_nav**: Current Net Asset Value
- **latest_close**: Current closing price
- **price_nav_diff**: Difference between close and NAV
- **price_nav_diff_pct**: Percentage difference
- **latest_volume**: Trading volume
- **latest_amount**: Trading amount
- **index_code**: Underlying index code
- **index_name**: Underlying index name
- **data_date**: Date of data (YYYYMMDD)
- **retrieval_timestamp**: When data was retrieved

## 📊 **Sample Output**

```csv
wind_code,fund_name,latest_nav,latest_close,price_nav_diff,price_nav_diff_pct,index_code,index_name
513090.SH,香港证券ETF,1.8841,1.882,-0.0021,-0.1115,930709.CSI,香港证券
513120.SH,港股创新药ETF,1.1739,1.179,0.0051,0.4344,931787CNY00.CSI,港股创新药(CNY)
513130.SH,恒生科技ETF,0.693,0.693,0.0,0.0,HSTECH.HI,恒生科技
```

## ⏰ **Recommended Daily Schedule**

### Best Times to Run:
- **Morning**: 9:00 AM (after market open)
- **Evening**: 6:00 PM (after market close)
- **Late Evening**: 10:00 PM (for final data)

### Automation Options:

#### Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 6:00 PM
4. Action: Start Program
5. Program: `python`
6. Arguments: `daily_qdii_data_retriever.py`
7. Start in: Your project directory

#### Linux/Mac Cron Job:
```bash
# Edit crontab
crontab -e

# Add line for daily 6 PM execution
0 18 * * * cd /path/to/project && python daily_qdii_data_retriever.py
```

## 📈 **What You Get Each Day**

### Data Coverage (Typical):
- ✅ **153/155 funds** with NAV data (98.7%)
- ✅ **153/155 funds** with close prices (98.7%)
- ✅ **153/155 funds** with index info (98.7%)

### Analysis Included:
- 💰 **Price vs NAV differences** (average ~0.2%)
- 📊 **Trading volume and amounts**
- 🎯 **Underlying index tracking**

## 🔧 **Requirements**

### Prerequisites:
1. **WindPy installed** (Wind API)
2. **Wind terminal access**
3. **Python packages**: pandas, datetime
4. **Input file**: `qdii_funds_parsed.csv` (fund list)

### File Dependencies:
- `qdii_funds_parsed.csv` - Must be in same directory
- Wind terminal - Must be accessible

## 📋 **Daily Workflow Example**

```bash
# 1. Navigate to project directory
cd "C:\path\to\wind qdii fund nav"

# 2. Run daily script
python daily_qdii_data_retriever.py

# 3. Check output
# File created: qdii_daily_data_20250708.csv
```

## 🎯 **Key Benefits**

### ✅ **Complete Data**:
- All 153 QDII funds in one file
- NAV + Close prices + Index info
- Automatic price-NAV difference calculation

### ✅ **Ready for Analysis**:
- Clean CSV format
- Consistent column structure
- Date-stamped files

### ✅ **Reliable**:
- Direct Wind API access
- Error handling included
- Automatic connection management

## 📊 **Data Quality**

### Typical Results:
- **98.7% data coverage** (153/155 funds)
- **Real-time data** from Wind
- **Accurate NAV values** (not closing prices)
- **Complete index mapping**

## 🔍 **Troubleshooting**

### Common Issues:

#### "WindPy not available"
- Install WindPy from Wind terminal
- Ensure Wind terminal is running

#### "Error loading fund list"
- Check `qdii_funds_parsed.csv` exists
- Verify file encoding (UTF-8)

#### "Connection failed"
- Restart Wind terminal
- Check Wind login credentials

## 📁 **File Management**

### Daily Files:
- `qdii_daily_data_20250708.csv` (today)
- `qdii_daily_data_20250707.csv` (yesterday)
- `qdii_daily_data_20250706.csv` (day before)

### Recommended:
- Keep last 30 days of files
- Archive older files monthly
- Backup important data

---

## 🎉 **Ready to Use!**

Your daily QDII fund data retrieval is now automated and ready to run every day. Simply execute the script and get comprehensive, up-to-date fund information in minutes!
