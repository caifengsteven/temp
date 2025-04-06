# ETF Premium/Discount Monitor

This script monitors ETF premiums and discounts relative to their Indicative Net Asset Values (INAVs) in real-time using Bloomberg data. It displays the results in the terminal with color coding and flashing effects based on the premium/discount level.

## Features

- Reads ETF and INAV ticker pairs from a CSV file
- Connects to Bloomberg to get real-time prices
- Calculates premium/discount in basis points
- Displays results in the terminal with color coding:
  - GREEN (flashing): ETF at discount > 20 bps
  - RED (flashing): ETF at premium > 20 bps
  - BLUE: ETF within 20 bps of INAV
- Updates every minute (configurable)

## Prerequisites

- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `colorama`, `blpapi`

## Installation

1. Install the required packages:

```bash
pip install pandas colorama
pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi
```

2. Make sure your Bloomberg Terminal is running and the API service is enabled.

## Usage

1. Prepare a CSV file named `etf_pairs.csv` with ETF and INAV ticker pairs:

```
ETF Ticker,INAV Ticker
2823 HK Equity,2823IV Index
3067 HK Equity,3067IV Index
9834 HK Equity,9834IV Index
```

2. Run the script:

```bash
python etf_premium_monitor.py
```

3. The script will:
   - Connect to Bloomberg
   - Read ETF pairs from `etf_pairs.csv`
   - Check real-time prices every minute
   - Display the premium/discount in the terminal with color coding

4. Press Ctrl+C to stop the monitoring.

## Mock Version

For testing without a Bloomberg connection, you can use the mock version:

```bash
python mock_etf_premium_monitor.py
```

The mock version simulates price movements and premium/discount events for demonstration purposes.

## Customization

You can customize the script by modifying the following parameters:

- `threshold_bps`: Premium/discount threshold in basis points (default: 20)
- `interval_seconds`: Interval between checks in seconds (default: 60)

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the tickers in your CSV file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration
