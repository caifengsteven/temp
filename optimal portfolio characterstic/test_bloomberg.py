"""
Test script for Bloomberg connection and data retrieval
"""

import pandas as pd
import datetime as dt
from bloomberg_data import BloombergDataRetriever

def test_bloomberg_connection():
    """Test Bloomberg connection and data retrieval"""
    print("Testing Bloomberg connection...")

    # Initialize Bloomberg data retriever
    bbg = BloombergDataRetriever()

    # Define test tickers (major stocks that should be available)
    test_tickers = [
        "AAPL US Equity",
        "MSFT US Equity",
        "AMZN US Equity",
        "GOOGL US Equity",
        "META US Equity"  # FB is now META
    ]

    # Test market cap retrieval
    print("\nTesting market cap retrieval...")
    market_caps = bbg.get_market_caps(test_tickers)
    print(market_caps)

    # Test book-to-market retrieval
    print("\nTesting book-to-market retrieval...")
    bm_ratios = bbg.get_book_to_market(test_tickers)
    print(bm_ratios)

    # Test momentum calculation
    print("\nTesting momentum calculation...")
    momentum = bbg.get_momentum(test_tickers, lookback_months=12, skip_months=1)
    print(momentum)

    # Test returns retrieval
    print("\nTesting returns retrieval...")
    start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime('%Y%m%d')
    end_date = dt.datetime.now().strftime('%Y%m%d')

    print(f"Retrieving returns from {start_date} to {end_date}")
    returns = bbg.get_returns(test_tickers, start_date, end_date, frequency='monthly')

    if returns.empty:
        print("No returns data retrieved")
    else:
        print(f"Returns shape: {returns.shape}")
        print(returns.head())

    # Test risk-free rate retrieval
    print("\nTesting risk-free rate retrieval...")
    rf_rate = bbg.get_risk_free_rate(start_date, end_date, frequency='monthly')

    if isinstance(rf_rate, pd.Series) and not rf_rate.empty:
        print(f"Risk-free rate shape: {rf_rate.shape}")
        print(rf_rate.head())
    else:
        print("No risk-free rate data retrieved")

    # Close connection
    bbg.close_connection()

    print("\nBloomberg connection test completed.")

def test_simulated_data():
    """Test the fallback to simulated data"""
    print("Testing simulated data generation...")

    # Initialize Bloomberg data retriever with invalid connection parameters
    # to force fallback to simulated data
    bbg = BloombergDataRetriever(host='invalid_host', port=9999)

    # Define test tickers
    test_tickers = [f"STOCK{i} US Equity" for i in range(1, 6)]

    # Test returns retrieval with simulated data
    print("\nTesting simulated returns...")
    start_date = '20200101'
    end_date = '20201231'
    returns = bbg.get_returns(test_tickers, start_date, end_date, frequency='monthly')

    if returns.empty:
        print("No simulated returns data generated")
    else:
        print(f"Simulated returns shape: {returns.shape}")
        print(returns.head())

    # Test risk-free rate retrieval with simulated data
    print("\nTesting simulated risk-free rate...")
    rf_rate = bbg.get_risk_free_rate(start_date, end_date, frequency='monthly')

    if isinstance(rf_rate, pd.Series) and not rf_rate.empty:
        print(f"Simulated risk-free rate shape: {rf_rate.shape}")
        print(rf_rate.head())
    else:
        print("No simulated risk-free rate data generated")

    print("\nSimulated data test completed.")

if __name__ == "__main__":
    # Test real Bloomberg connection
    test_bloomberg_connection()

    # Test simulated data fallback
    print("\n" + "="*50 + "\n")
    test_simulated_data()
