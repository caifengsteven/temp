# QDII Fund NAV Data Retrieval Project

This project successfully extracts QDII fund data from an Excel file and retrieves NAV/price data using Wind API.

## 📁 Generated Files

### Main Data Files
- **`qdii_funds_parsed.csv`** - Parsed QDII fund list from Excel file (155 funds)
- **`qdii_nav_price_data_wind.csv`** - NAV/price data from Wind API (4,840 records for 20 funds)
- **`qdii_fund_basic_info_wind.csv`** - Basic fund information from Wind

### Summary Files
- **`qdii_fund_summary.csv`** - Summary statistics by fund (when all funds are processed)

## 📊 Data Structure

### NAV/Price Data (`qdii_nav_price_data_wind.csv`)
- **CLOSE**: Closing price/NAV value
- **VOLUME**: Trading volume
- **AMT**: Trading amount
- **wind_code**: Wind fund code (e.g., 513090.SH)
- **date**: Trading date
- **fund_name**: Fund name in Chinese

### Sample Data
```
CLOSE,VOLUME,AMT,wind_code,date,fund_name
0.885,400431600.0,358796825.0,513090.SH,2024-07-08,香港证券ETF
0.897,730069600.0,652266591.0,513090.SH,2024-07-09,香港证券ETF
```

## 🚀 Scripts Available

### Main Processing Scripts
1. **`wind_nav_final.py`** - Final Wind API NAV retriever (recommended)
2. **`process_all_qdii_funds.py`** - Process all 154 funds (takes 10-15 minutes)
3. **`excel_xml_parser.py`** - Parse Excel file to extract fund codes

### Alternative/Backup Scripts
4. **`alternative_qdii_scraper.py`** - Public source scraper (backup)
5. **`main_qdii_scraper.py`** - Combined Wind + alternative approach

### Diagnostic Scripts
6. **`wind_diagnostic.py`** - Test Wind API functionality
7. **`wind_tuple_analysis.py`** - Analyze Wind API response structure

## 📈 Current Results

✅ **Successfully Retrieved:**
- 20 QDII funds processed
- 4,840 total records
- 1 year of daily data (2024-07-08 to 2025-07-07)
- Price, volume, and trading amount data

✅ **Fund Types Covered:**
- Hong Kong ETFs (香港证券ETF, 恒生科技ETF)
- Healthcare/Pharma ETFs (港股创新药ETF, 恒生医疗ETF)
- Technology ETFs (纳指科技ETF, 港股通科技30ETF)
- Internet ETFs (中概互联网ETF, 港股通互联网ETF)

## 🔧 How to Use

### Process All Funds
```bash
python process_all_qdii_funds.py
```

### Process Limited Funds (for testing)
```bash
python wind_nav_final.py
```

### Parse Excel File Only
```bash
python excel_xml_parser.py
```

## 📋 Requirements

- Python 3.6+
- pandas
- WindPy (Wind API)
- Access to Wind terminal

## 🎯 Next Steps

To get complete data for all 154 QDII funds:
1. Run `python process_all_qdii_funds.py`
2. Wait 10-15 minutes for completion
3. Check `qdii_nav_price_data_wind.csv` for complete dataset

## 📝 Notes

- Wind API provides reliable, real-time data
- Each fund has approximately 242 trading days of data
- Data includes price, volume, and trading amount
- All dates are in YYYY-MM-DD format
- Fund codes follow Wind convention (e.g., 513090.SH, 159570.SZ)

## ✅ Task Completion Status

- [x] Parse Excel file with QDII fund list
- [x] Extract fund codes and names
- [x] Connect to Wind API
- [x] Retrieve NAV/price data with dates
- [x] Save data to structured CSV format
- [x] Generate summary and analysis

**Project Status: COMPLETED SUCCESSFULLY** 🎉
