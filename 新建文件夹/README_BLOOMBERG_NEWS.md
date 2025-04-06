# Bloomberg News Retrieval Tools

This repository contains Python scripts for retrieving and monitoring news from Bloomberg Terminal using the Bloomberg API (BLPAPI).

## Scripts

1. **bloomberg_news_retriever.py**: A versatile command-line tool for retrieving news from Bloomberg by topic, security, or search query.

2. **instrument_news_monitor.py**: A monitoring tool that continuously checks for new headlines for a list of instruments.

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

### Bloomberg News Retriever

This script provides several ways to retrieve news from Bloomberg:

1. **Get news by topic**:

```bash
python bloomberg_news_retriever.py --topic TOP --max 10
```

Common topics include:
- `TOP`: Top news
- `FIRST`: First word news
- `MARKET_STORIES`: Market stories
- `WORLDWIDE`: Worldwide news

2. **Get news by security**:

```bash
python bloomberg_news_retriever.py --security "AAPL US Equity" --max 10
```

3. **Search for news**:

```bash
python bloomberg_news_retriever.py --query "Apple iPhone" --max 10 --days 7
```

4. **Retrieve a specific news story**:

```bash
python bloomberg_news_retriever.py --story "NEWS_STORY_ID"
```

5. **Save results to a file**:

```bash
python bloomberg_news_retriever.py --topic TOP --output "top_news.csv"
```

### Instrument News Monitor

This script continuously monitors news for a list of instruments:

```bash
python instrument_news_monitor.py
```

The script:
1. Reads instruments from `instruments.txt`
2. Checks for new headlines every 30 minutes
3. Displays and saves new headlines
4. Highlights important keywords in the headlines

## Integrating with Supertrend Monitor

These news retrieval tools can be used alongside the `supertrend_monitor.py` script for comprehensive market analysis:

1. Run the supertrend monitor to detect technical signals:

```bash
python supertrend_monitor.py
```

2. In a separate terminal, run the news monitor to track fundamental developments:

```bash
python instrument_news_monitor.py
```

This combination provides both technical and fundamental insights for your trading decisions.

## Customization

### Bloomberg News Retriever

- `--max`: Maximum number of results to retrieve (default: 10)
- `--days`: Number of days to look back for search (default: 7)
- `--output`: Custom output file path

### Instrument News Monitor

- `interval_minutes`: Interval between checks in minutes (default: 30)
- `max_headlines`: Maximum number of headlines to retrieve per instrument (default: 5)
- `highlight_keywords`: List of keywords to highlight in headlines

## Output

Both scripts save retrieved news to the `bloomberg_news` directory:

- News headlines are saved as CSV files
- Full news stories are saved as text files

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the instruments in your file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration
