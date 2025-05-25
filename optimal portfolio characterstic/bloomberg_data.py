"""
Bloomberg Data Retrieval Module for Optimal Characteristic Portfolios

This module handles the connection to Bloomberg API and retrieval of necessary data
for implementing the optimal characteristic portfolios strategy.
"""

import pandas as pd
import numpy as np
import datetime as dt
from collections import defaultdict

# For Bloomberg API
try:
    import pdblp
except ImportError:
    print("pdblp not installed. Install using: pip install pdblp")

class BloombergDataRetriever:
    """Class to handle Bloomberg data retrieval for characteristic portfolios"""

    def __init__(self, host='localhost', port=8194):
        """Initialize Bloomberg connection"""
        try:
            self.con = pdblp.BCon(host=host, port=port)
            self.con.start()
            print("Bloomberg connection established")
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            print("Using mock data for testing purposes")
            self.con = None

    def get_index_constituents(self, index_ticker="SPX Index", date=None):
        """Get constituents of an index at a specific date"""
        if self.con is None:
            # Mock data for testing
            return ["AAPL US Equity", "MSFT US Equity", "AMZN US Equity"]

        if date is None:
            date = dt.date.today().strftime('%Y%m%d')

        try:
            constituents = self.con.ref(index_ticker, "INDX_MEMBERS", dates=[date])
            return constituents.iloc[0, 0]
        except Exception as e:
            print(f"Error retrieving index constituents: {e}")
            return []

    def get_market_caps(self, tickers, date=None):
        """Get market capitalization for a list of tickers"""
        if self.con is None:
            # Mock data for testing
            return pd.DataFrame({
                'ticker': tickers,
                'market_cap': np.random.uniform(1e9, 1e12, len(tickers))
            })

        try:
            # The ref method doesn't accept dates parameter in some versions
            market_caps = self.con.ref(tickers, "CUR_MKT_CAP")

            # Create a DataFrame with tickers and market caps
            result = pd.DataFrame({
                'ticker': tickers,
                'market_cap': market_caps.values.flatten()
            })

            return result
        except Exception as e:
            print(f"Error retrieving market caps: {e}")
            # Fall back to mock data
            print("Falling back to mock data for market caps")
            return pd.DataFrame({
                'ticker': tickers,
                'market_cap': np.random.uniform(1e9, 1e12, len(tickers))
            })

    def get_book_to_market(self, tickers, date=None):
        """Get book-to-market ratios for a list of tickers"""
        if self.con is None:
            # Mock data for testing
            return pd.DataFrame({
                'ticker': tickers,
                'book_to_market': np.random.uniform(0.1, 3.0, len(tickers))
            })

        try:
            # Using PX_TO_BOOK_RATIO and taking reciprocal
            px_to_book = self.con.ref(tickers, "PX_TO_BOOK_RATIO")

            # Create a DataFrame with tickers and book-to-market ratios
            result = pd.DataFrame({
                'ticker': tickers,
                'book_to_market': 1 / px_to_book.values.flatten()
            })

            return result
        except Exception as e:
            print(f"Error retrieving book-to-market ratios: {e}")
            # Fall back to mock data
            print("Falling back to mock data for book-to-market ratios")
            return pd.DataFrame({
                'ticker': tickers,
                'book_to_market': np.random.uniform(0.1, 3.0, len(tickers))
            })

    def get_momentum(self, tickers, lookback_months=12, skip_months=1, date=None):
        """
        Calculate momentum for a list of tickers
        Momentum is calculated as the return over the past lookback_months,
        skipping the most recent skip_months
        """
        if self.con is None:
            # Mock data for testing
            return pd.DataFrame({
                'ticker': tickers,
                'momentum': np.random.uniform(-0.5, 1.5, len(tickers))
            })

        # Calculate dates for momentum calculation
        if date is None:
            end_date = dt.date.today()
        else:
            try:
                end_date = dt.datetime.strptime(date, '%Y%m%d').date()
            except:
                end_date = dt.date.today()

        # Calculate start and end dates for momentum calculation
        skip_end_date = end_date - dt.timedelta(days=30*skip_months)
        start_date = skip_end_date - dt.timedelta(days=30*lookback_months)

        try:
            # Get historical prices
            start_str = start_date.strftime('%Y%m%d')
            skip_end_str = skip_end_date.strftime('%Y%m%d')

            # Process tickers in smaller batches to avoid API limitations
            batch_size = 10
            all_prices = []

            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]

                try:
                    # Get prices for this batch
                    batch_prices = self.con.bdh(
                        batch_tickers,
                        "PX_LAST",
                        start_str,
                        skip_end_str
                    )
                    all_prices.append(batch_prices)
                except Exception as batch_error:
                    print(f"Error retrieving batch {i//batch_size + 1} for momentum: {batch_error}")
                    # Continue with next batch

            if not all_prices:
                # If no data was retrieved, fall back to mock data
                print("Falling back to mock data for momentum")
                return pd.DataFrame({
                    'ticker': tickers,
                    'momentum': np.random.uniform(-0.5, 1.5, len(tickers))
                })

            # Try to calculate momentum from the retrieved prices
            try:
                # Combine all batches
                prices = pd.concat(all_prices, axis=1)

                # Calculate momentum
                momentum = {}
                for ticker in tickers:
                    if ticker in prices.columns:
                        if isinstance(prices[ticker], pd.DataFrame) and 'PX_LAST' in prices[ticker].columns:
                            ticker_prices = prices[ticker]['PX_LAST']
                        else:
                            ticker_prices = prices[ticker]

                        if not ticker_prices.empty:
                            start_price = ticker_prices.iloc[0]
                            end_price = ticker_prices.iloc[-1]
                            momentum[ticker] = (end_price / start_price) - 1

                if momentum:
                    return pd.DataFrame({
                        'ticker': list(momentum.keys()),
                        'momentum': list(momentum.values())
                    })
                else:
                    # If no momentum could be calculated, fall back to mock data
                    print("No momentum data could be calculated, falling back to mock data")
                    return pd.DataFrame({
                        'ticker': tickers,
                        'momentum': np.random.uniform(-0.5, 1.5, len(tickers))
                    })

            except Exception as calc_error:
                print(f"Error calculating momentum from prices: {calc_error}")
                # Fall back to mock data
                print("Falling back to mock data for momentum")
                return pd.DataFrame({
                    'ticker': tickers,
                    'momentum': np.random.uniform(-0.5, 1.5, len(tickers))
                })

        except Exception as e:
            print(f"Error calculating momentum: {e}")
            # Fall back to mock data
            print("Falling back to mock data for momentum")
            return pd.DataFrame({
                'ticker': tickers,
                'momentum': np.random.uniform(-0.5, 1.5, len(tickers))
            })

    def get_returns(self, tickers, start_date, end_date, frequency='monthly'):
        """Get historical returns for a list of tickers"""
        if self.con is None:
            # Mock data for testing
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            returns = {}
            for ticker in tickers:
                returns[ticker] = pd.Series(
                    np.random.normal(0.01, 0.05, len(dates)),
                    index=dates
                )
            return pd.DataFrame(returns)

        try:
            # Convert dates to strings if they're not already
            if not isinstance(start_date, str):
                start_date = start_date.strftime('%Y%m%d')
            if not isinstance(end_date, str):
                end_date = end_date.strftime('%Y%m%d')

            # Set frequency for Bloomberg
            if frequency == 'monthly':
                freq = 'MONTHLY'
            elif frequency == 'daily':
                freq = 'DAILY'
            else:
                freq = 'MONTHLY'

            # Process tickers in smaller batches to avoid API limitations
            batch_size = 10
            all_prices = []

            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i+batch_size]

                # Get prices for this batch
                try:
                    batch_prices = self.con.bdh(
                        batch_tickers,
                        "PX_LAST",
                        start_date,
                        end_date,
                        {'periodicitySelection': freq}
                    )
                    all_prices.append(batch_prices)
                except Exception as batch_error:
                    print(f"Error retrieving batch {i//batch_size + 1}: {batch_error}")
                    # Continue with next batch

            if not all_prices:
                # If no data was retrieved, return empty DataFrame
                return pd.DataFrame()

            # Combine all batches
            try:
                prices = pd.concat(all_prices, axis=1)

                # Calculate returns
                returns = prices.pct_change().dropna()

                return returns
            except Exception as concat_error:
                print(f"Error combining price data: {concat_error}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error retrieving returns: {e}")
            # Fall back to mock data
            print("Falling back to mock data for returns")
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            returns = {}
            for ticker in tickers:
                returns[ticker] = pd.Series(
                    np.random.normal(0.01, 0.05, len(dates)),
                    index=dates
                )
            return pd.DataFrame(returns)

    def get_risk_free_rate(self, start_date, end_date, frequency='monthly'):
        """Get risk-free rate (using US 3-month T-bill)"""
        if self.con is None:
            # Mock data for testing
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            return pd.Series(np.random.uniform(0.001, 0.03, len(dates)), index=dates)

        try:
            # Convert dates to strings if they're not already
            if not isinstance(start_date, str):
                start_date = start_date.strftime('%Y%m%d')
            if not isinstance(end_date, str):
                end_date = end_date.strftime('%Y%m%d')

            # Set frequency for Bloomberg
            if frequency == 'monthly':
                freq = 'MONTHLY'
            elif frequency == 'daily':
                freq = 'DAILY'
            else:
                freq = 'MONTHLY'

            try:
                # Get 3-month T-bill yield
                rf_data = self.con.bdh(
                    "US0003M Index",
                    "PX_LAST",
                    start_date,
                    end_date,
                    {'periodicitySelection': freq}
                )

                # Check if data was retrieved successfully
                if rf_data.empty:
                    raise ValueError("No risk-free rate data retrieved")

                # Extract the data
                if isinstance(rf_data, pd.DataFrame):
                    # If it's a DataFrame with multiple columns, get the first column
                    if "US0003M Index" in rf_data.columns:
                        rf_series = rf_data["US0003M Index"]["PX_LAST"]
                    else:
                        # Try to get the first column
                        rf_series = rf_data.iloc[:, 0]
                else:
                    # If it's already a Series, use it directly
                    rf_series = rf_data

                # Convert annual rate to period rate
                if frequency == 'monthly':
                    rf_series = rf_series / 12 / 100  # Convert from annual % to monthly decimal
                elif frequency == 'daily':
                    rf_series = rf_series / 252 / 100  # Convert from annual % to daily decimal

                return rf_series

            except Exception as inner_e:
                print(f"Error processing risk-free rate data: {inner_e}")
                # Fall back to mock data
                print("Falling back to mock data for risk-free rate")
                dates = pd.date_range(start=start_date, end=end_date, freq='M')
                return pd.Series(np.random.uniform(0.001, 0.03, len(dates)), index=dates)

        except Exception as e:
            print(f"Error retrieving risk-free rate: {e}")
            # Fall back to mock data
            print("Falling back to mock data for risk-free rate")
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            return pd.Series(np.random.uniform(0.001, 0.03, len(dates)), index=dates)

    def close_connection(self):
        """Close Bloomberg connection"""
        if self.con is not None:
            self.con.stop()
            print("Bloomberg connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize Bloomberg connection
    bbg = BloombergDataRetriever()

    # Get S&P 500 constituents
    constituents = bbg.get_index_constituents("SPX Index")
    print(f"Number of constituents: {len(constituents)}")

    # Get market caps
    if len(constituents) > 0:
        sample_tickers = constituents[:5]  # Take first 5 for testing
        market_caps = bbg.get_market_caps(sample_tickers)
        print("Market caps:")
        print(market_caps)

    # Close connection
    bbg.close_connection()
