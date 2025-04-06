# Bloomberg News Retrieval Tools (Standard Version)

This repository contains Python scripts for retrieving news from Bloomberg Terminal using the standard Bloomberg API services that are commonly available to all Bloomberg Terminal users.

## Scripts

1. **bloomberg_news_standard.py**: A continuous monitoring tool that checks for new headlines for a list of instruments at regular intervals.

2. **get_latest_news.py**: A simple script that retrieves the latest news for a list of instruments in a single run.

## Why These Scripts?

The original scripts (`bloomberg_news_retriever.py` and `instrument_news_monitor.py`) used the Bloomberg News API service (`//blp/newsapi`), which may not be available in all Bloomberg Terminal subscriptions. These new scripts use only the standard Reference Data Service (`//blp/refdata`), which is available to all Bloomberg Terminal users.

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

### Bloomberg News Standard Monitor

This script continuously monitors news for a list of instruments:

```bash
python bloomberg_news_standard.py
```

The script:
1. Reads instruments from `instruments.txt`
2. Checks for new headlines every 30 minutes
3. Displays and saves new headlines
4. Highlights important keywords in the headlines

### Get Latest News

This script retrieves the latest news for a list of instruments in a single run:

```bash
python get_latest_news.py
```

The script:
1. Reads instruments from `instruments.txt`
2. Retrieves the latest headlines for each instrument
3. Displays and saves the headlines
4. Highlights important keywords in the headlines

## Integrating with Supertrend Monitor

These news retrieval tools can be used alongside the `supertrend_monitor.py` script for comprehensive market analysis:

1. Run the supertrend monitor to detect technical signals:

```bash
python supertrend_monitor.py
```

2. In a separate terminal, run the news monitor to track fundamental developments:

```bash
python bloomberg_news_standard.py
```

This combination provides both technical and fundamental insights for your trading decisions.

## Customization

### Bloomberg News Standard Monitor

- `interval_minutes`: Interval between checks in minutes (default: 30)
- `max_headlines`: Maximum number of headlines to retrieve per instrument (default: 5)
- `highlight_keywords`: List of keywords to highlight in headlines

### Get Latest News

- `max_headlines`: Maximum number of headlines to retrieve per instrument (default: 10)
- `keywords`: List of keywords to highlight in headlines

## Output

Both scripts save retrieved news to the `bloomberg_news` directory as CSV files.

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the instruments in your file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration

### Common Errors

- **"The request schema or service doesn't exist"**: This error occurs when trying to use a Bloomberg service that is not available in your subscription. Use the standard version scripts instead.
- **"Invalid security"**: Check that the security identifiers in your instruments.txt file are valid Bloomberg identifiers (e.g., "AAPL US Equity").
- **"Invalid field"**: The field you're trying to access is not available in your subscription. Try using a different field.
