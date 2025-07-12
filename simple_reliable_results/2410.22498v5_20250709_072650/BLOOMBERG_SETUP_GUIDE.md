# Bloomberg Data Setup Guide for VIX Stochastic Volatility Model

This guide explains how to properly set up Bloomberg data access for the enhanced VIX Stochastic Volatility Model implementation.

## Prerequisites

### 1. Bloomberg Terminal or API Access
- **Bloomberg Terminal**: Active Bloomberg Terminal subscription
- **Bloomberg API**: Bloomberg Server API (SAPI) or Desktop API (DAPI) access
- **Network Access**: Proper network connectivity to Bloomberg servers

### 2. Bloomberg Python API
The Bloomberg Python API must be installed before xbbg:

```bash
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

**Note**: This requires Bloomberg credentials and may not work without proper Bloomberg access.

### 3. Bloomberg C++ SDK
- Download from [Bloomberg API Library](https://www.bloomberg.com/professional/support/api-library/)
- Extract and copy `blpapi3_32.dll` and `blpapi3_64.dll` to Bloomberg `BLPAPI_ROOT` folder
- Usually located at `blp/DAPI`

## Installation Steps

### Step 1: Install Dependencies
```bash
# Install required Python packages
pip install numpy pandas matplotlib scipy statsmodels

# Install Bloomberg-specific packages
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
pip install xbbg
```

### Step 2: Verify Bloomberg Connection
```python
# Test Bloomberg connectivity
from xbbg import blp

# Simple test
try:
    data = blp.bdp(tickers='SPX Index', flds=['PX_LAST'])
    print("Bloomberg connection successful!")
    print(data)
except Exception as e:
    print(f"Bloomberg connection failed: {e}")
```

### Step 3: Run Enhanced Implementation
```bash
python 2410.22498v5_test_strategy.py
```

## Bloomberg Data Sources Used

### VIX Data
- **Ticker**: `VIX Index`
- **Fields**: `PX_LAST`, `PX_HIGH`, `PX_LOW`
- **Description**: CBOE Volatility Index

### Corporate Bond Data
- **HYG US Equity**: iShares High Yield Corporate Bond ETF
- **LQD US Equity**: iShares Investment Grade Corporate Bond ETF
- **VCIT US Equity**: Vanguard Intermediate-Term Corporate Bond ETF
- **USIG US Equity**: iShares Broad USD Investment Grade Corporate Bond ETF

### Corporate Bond Indices
- **LF98TRUU Index**: Bloomberg US Corporate Bond Index
- **LUACTRUU Index**: Bloomberg US Credit Index

### Spread Data
- **LUACOAS Index**: US Credit Option Adjusted Spread
- **LUCROAS Index**: US Corporate OAS
- **C0A0 Index**: US Investment Grade Corporate OAS
- **H0A0 Index**: US High Yield Corporate OAS

### Treasury Data
- **USGG2YR Index**: 2-Year Treasury Yield
- **USGG5YR Index**: 5-Year Treasury Yield
- **USGG10YR Index**: 10-Year Treasury Yield

## API Usage Examples

### Historical Data (BDH)
```python
from xbbg import blp

# Load VIX data
vix_data = blp.bdh(
    tickers='VIX Index',
    flds=['PX_LAST'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Load corporate bond data
bond_data = blp.bdh(
    tickers=['HYG US Equity', 'LQD US Equity'],
    flds=['PX_LAST'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### Current Data (BDP)
```python
# Get current market snapshot
current_data = blp.bdp(
    tickers=['VIX Index', 'SPX Index', 'HYG US Equity'],
    flds=['PX_LAST', 'CHG_PCT_1D', 'NAME']
)
```

### Reference Data (BDS)
```python
# Get dividend history
dividends = blp.bds(
    'HYG US Equity', 
    'DVD_Hist_All',
    DVD_Start_Dt='20230101',
    DVD_End_Dt='20231231'
)
```

## Troubleshooting

### Common Issues

#### 1. Import Error: "No module named 'ruamel'"
```bash
pip install ruamel.yaml
```

#### 2. Import Error: "No module named 'blpapi'"
```bash
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

#### 3. Connection Error: "Failed to connect to Bloomberg"
- Ensure Bloomberg Terminal is running
- Check Bloomberg API permissions
- Verify network connectivity
- Contact Bloomberg support for API access issues

#### 4. Data Error: "No data returned"
- Check ticker symbols are correct
- Verify date ranges are valid
- Ensure you have permissions for requested data
- Some data may have delays or restrictions

### Fallback Mode
If Bloomberg data is not available, the implementation automatically falls back to simulated data:

```
⚠ Bloomberg data not available - used simulated data only
✓ Demonstrated VIX stochastic volatility model
✓ Verified residual normalization effect
```

## Data Compliance

### Bloomberg Data Usage Policy
- Data accessed via Bloomberg API must comply with Bloomberg Datafeed Addendum
- Data cannot leave the local PC used to access Bloomberg Professional service
- Redistribution of Bloomberg data is prohibited
- See `DAPI<GO>` on Bloomberg Terminal for full terms

### Local Data Storage
If `BBG_ROOT` environment variable is set, xbbg can cache data locally:

```bash
export BBG_ROOT=/path/to/bloomberg/data
```

## Performance Optimization

### Data Range Selection
- Use shorter date ranges for faster processing
- Default implementation uses 3 years of data
- Adjust `start_date` parameter as needed

### Ticker Selection
- Start with liquid, reliable tickers (VIX, SPX, HYG, LQD)
- Add additional tickers as needed
- Some corporate bond indices may have limited history

### Error Handling
The implementation includes robust error handling:
- Graceful fallback to simulated data
- Detailed error messages
- Connection testing before data requests

## Support

### Bloomberg Support
- Bloomberg Help Desk: `HELP HELP` on Terminal
- API Support: `WAPI<GO>` on Terminal
- Documentation: `DAPI<GO>` on Terminal

### Implementation Support
- Check error messages in console output
- Review log files if available
- Test with simple examples first
- Verify all dependencies are installed

## Example Output

When Bloomberg data is successfully loaded:

```
✓ Bloomberg connection successful
  S&P 500 Index: 4,567.89

Current Market Snapshot:
========================================
VIX Index      :    18.45 ( +2.34%)
SPX Index      : 4,567.89 ( +0.87%)
HYG US Equity  :    82.15 ( -0.23%)
LQD US Equity  :   108.76 ( +0.12%)
LUACOAS Index  :   125.50 ( +3.45%)

✓ Loaded 1,095 VIX observations
✓ Loaded 1,095 bond price observations
✓ Ready for VIX stochastic volatility model analysis
```

This setup enables full integration of real Bloomberg market data with the VIX stochastic volatility model, providing authentic market validation of the academic research.
