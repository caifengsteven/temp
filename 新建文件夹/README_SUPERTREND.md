# Supertrend Indicator Monitor

This script monitors a list of instruments from a file and checks for supertrend indicator breakups and breakdowns every 30 minutes. When a signal is detected, it flashes the instrument name and signal type in the terminal.

## What is the Supertrend Indicator?

The Supertrend indicator is a trend-following indicator that uses the ATR (Average True Range) to identify the current market trend. It plots a line above or below the price depending on the trend direction:

- When price is above the Supertrend line, the market is in an uptrend (bullish)
- When price is below the Supertrend line, the market is in a downtrend (bearish)

A "breakup" occurs when the price crosses above the Supertrend line, indicating a potential bullish trend. A "breakdown" occurs when the price crosses below the Supertrend line, indicating a potential bearish trend.

## Features

- Reads instruments from a text file
- Connects to Bloomberg to get real-time price data
- Calculates the Supertrend indicator for each instrument
- Checks for breakup and breakdown signals every 30 minutes
- Displays signals in the terminal with color coding:
  - GREEN (flashing): Breakup signal
  - RED (flashing): Breakdown signal

## Prerequisites

- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `numpy`, `colorama`, `blpapi`

## Installation

1. Install the required packages:

```bash
pip install pandas numpy colorama
pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi
```

2. Make sure your Bloomberg Terminal is running and the API service is enabled.

## Usage

1. Create a file named `instruments.txt` with one Bloomberg security identifier per line, for example:

```
AAPL US Equity
MSFT US Equity
AMZN US Equity
```

2. Run the script:

```bash
python supertrend_monitor.py
```

3. The script will:
   - Connect to Bloomberg
   - Read instruments from `instruments.txt`
   - Check for supertrend signals every 30 minutes
   - Display signals in the terminal when detected

4. Press Ctrl+C to stop the monitoring.

## Mock Version

For testing without a Bloomberg connection, you can use the mock version:

```bash
python mock_supertrend_monitor.py
```

The mock version simulates price movements and supertrend signals for demonstration purposes. It updates more frequently (every 5 seconds) to show signals more quickly.

## Customization

You can customize the script by modifying the following parameters:

- `atr_period`: Period for ATR calculation (default: 10)
- `atr_multiplier`: Multiplier for ATR to calculate Supertrend bands (default: 3.0)
- `interval_minutes`: Interval between checks in minutes (default: 30)

## Supertrend Calculation

The Supertrend indicator is calculated as follows:

1. Calculate the ATR (Average True Range) over a specified period
2. Calculate the basic upper and lower bands:
   - Basic Upper Band = (High + Low) / 2 + (ATR * Multiplier)
   - Basic Lower Band = (High + Low) / 2 - (ATR * Multiplier)
3. Calculate the final Supertrend bands based on the previous values and price action
4. Determine the trend direction:
   - If Close > Supertrend: Uptrend (1)
   - If Close < Supertrend: Downtrend (-1)
5. Detect crossovers (signals):
   - Breakup: Direction changes from -1 to 1
   - Breakdown: Direction changes from 1 to -1

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the instruments in your file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration
