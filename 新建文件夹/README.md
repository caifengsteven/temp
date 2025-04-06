# Bloomberg Data Fetcher

This script connects to Bloomberg, reads instruments from a file, fetches 30-minute bar data for each instrument, and saves the data to files.

## Prerequisites

- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `blpapi`

## Installation

1. Install the required packages:

```bash
pip install pandas
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
python bloomberg_data_fetcher.py
```

3. The script will:
   - Connect to Bloomberg
   - Read instruments from `instruments.txt`
   - Fetch 30-minute bar data for each instrument (for the maximum time period allowed by Bloomberg, typically 140 days)
   - Save the data to CSV files in the `bloomberg_data` directory

## Output

The script creates a directory called `bloomberg_data` and saves one CSV file per instrument with the naming format:
`{instrument_identifier}_30min_bars.csv`

Each CSV file contains the following columns:
- time: Timestamp for the bar
- open: Opening price
- high: Highest price during the interval
- low: Lowest price during the interval
- close: Closing price
- volume: Trading volume
- numEvents: Number of events in the bar

## Customization

You can modify the script to change:
- Bar interval (currently set to 30 minutes)
- Date range (currently set to maximum allowed by Bloomberg)
- Output directory
- Event type (currently set to TRADE)

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the instruments in your file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration
