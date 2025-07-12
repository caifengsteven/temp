# Bloomberg Data Setup Guide for GSPHAR Model

This guide helps you set up Bloomberg data access using xbbg for the enhanced GSPHAR model implementation.

## Prerequisites

1. **Bloomberg Terminal Access**: You need access to a Bloomberg Terminal or Bloomberg API license
2. **Bloomberg API Installation**: Bloomberg C++ SDK and Python API must be installed
3. **Active Bloomberg Session**: Bloomberg Terminal must be running during data requests

## Installation Steps

### 1. Install Bloomberg C++ SDK

1. Visit [Bloomberg API Library](https://www.bloomberg.com/professional/support/api-library/)
2. Download the C++ Supported Release (version 3.12.1 or higher)
3. Extract the downloaded zip file
4. Copy `blpapi3_32.dll` and `blpapi3_64.dll` from the `bin` folder to your Bloomberg `BLPAPI_ROOT` folder (usually `blp/DAPI`)

### 2. Install Bloomberg Python API

```bash
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

### 3. Install xbbg and dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Or install xbbg directly with dependencies
pip install xbbg
pip install ruamel.yaml pyarrow
```

### 4. Verify Installation

```bash
python test_bloomberg_connection.py
```

## Configuration

### Environment Variables (Optional)

You can set the following environment variables for better data management:

```bash
# For local data storage (optional)
export BBG_ROOT=/path/to/your/bloomberg/data/folder
```

### Testing Bloomberg Connection

The implementation includes a comprehensive test script:

```bash
python test_bloomberg_connection.py
```

This will test:
- xbbg library import
- Basic Bloomberg connection
- Historical data retrieval
- Multiple ticker access
- Intraday data availability

You can also test manually:

```python
from xbbg import blp

# Test basic connection with proper xbbg syntax
try:
    data = blp.bdp(tickers='SPX Index', flds='PX_LAST', timeout=10)
    print("Bloomberg connection successful!")
    print(f"SPX last price: {data.iloc[0, 0]}")
except Exception as e:
    print(f"Bloomberg connection failed: {e}")
```

### Key xbbg Features Used

The enhanced implementation leverages these xbbg capabilities:

1. **BDP (Bloomberg Data Point)**: Current market data
   ```python
   blp.bdp(tickers=['SPX Index'], flds=['PX_LAST', 'VOLATILITY_30D'])
   ```

2. **BDH (Bloomberg Data History)**: Historical time series
   ```python
   blp.bdh(tickers=['SPX Index'], flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'],
           start_date='2020-01-01', end_date='2023-12-31')
   ```

3. **BDIB (Bloomberg Data Intraday Bars)**: High-frequency data
   ```python
   blp.bdib(ticker='SPX Index', dt='2023-12-01', session='allday')
   ```

## Usage

### Running with Bloomberg Data

```bash
python 2410.22706v2_test_strategy.py
```

The script will automatically:
1. Try to connect to Bloomberg
2. Fetch real market data for major indices
3. Calculate realized volatility
4. Train the GSPHAR model
5. Generate visualizations

### Fallback to Synthetic Data

If Bloomberg is not available, the script automatically falls back to synthetic data generation.

## Customization

### Modifying Tickers

Edit the `bloomberg_tickers` list in the `run_bloomberg_experiment()` function:

```python
bloomberg_tickers = [
    'SPX Index',      # S&P 500
    'SX5E Index',     # Euro Stoxx 50
    'NKY Index',      # Nikkei 225
    # Add your preferred indices here
]
```

### Adjusting Date Range

Modify the date range in `run_bloomberg_experiment()`:

```python
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
```

### Realized Volatility Calculation

The script supports different RV calculation methods:
- `'daily'`: Uses OHLC data with Garman-Klass estimator
- `'5min'`: Uses 5-minute intraday data (requires more setup)
- `'1min'`: Uses 1-minute intraday data (requires more setup)

## Troubleshooting

### Common Issues

1. **"xbbg not available"**: Install xbbg using pip
2. **"Bloomberg connection failed"**: Check Bloomberg Terminal is running and API is properly configured
3. **"No data returned"**: Verify ticker symbols and date ranges are valid
4. **Permission errors**: Ensure you have proper Bloomberg data permissions

### Data Compliance

Remember that Bloomberg data usage must comply with the Bloomberg Datafeed Addendum:
- Data cannot leave the local PC used to access Bloomberg
- Proper licensing is required for data redistribution
- Follow your organization's Bloomberg data usage policies

## Output Files

The script generates several output files:
- `training_curve.png`: Model training progress
- `prediction_sample.png`: Sample volatility predictions
- `weights_comparison.png`: Learned vs traditional HAR weights
- `spillover_network.png`: Volatility spillover network visualization
- `rv_timeseries.png`: Realized volatility time series plots

## Support

For Bloomberg API issues, consult:
- Bloomberg API documentation
- Bloomberg Help Desk (HELP HELP on terminal)
- xbbg GitHub repository: https://github.com/alpha-xone/xbbg
