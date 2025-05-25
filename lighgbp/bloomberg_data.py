"""
Bloomberg Data Fetching Module

This module provides functions to fetch historical market data using Bloomberg API.
"""

import pandas as pd
import numpy as np
import datetime as dt
import pdblp  # Python wrapper for Bloomberg API

class BloombergDataFetcher:
    """Class for fetching data from Bloomberg."""
    
    def __init__(self):
        """Initialize Bloomberg connection."""
        try:
            self.con = pdblp.BCon(timeout=5000)
            self.con.start()
            print("Bloomberg connection established.")
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            print("Using fallback data source.")
            self.con = None
    
    def fetch_historical_data(self, ticker, start_date, end_date=None, fields=None):
        """
        Fetch historical OHLCV data for a given ticker.
        
        Parameters:
        -----------
        ticker : str
            Bloomberg ticker symbol (e.g., 'AAPL US Equity')
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str, optional
            End date in format 'YYYYMMDD', defaults to today
        fields : list, optional
            List of fields to fetch, defaults to OHLCV
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the requested data
        """
        if end_date is None:
            end_date = dt.datetime.now().strftime('%Y%m%d')
            
        if fields is None:
            fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
        
        try:
            if self.con is not None:
                data = self.con.bdh(
                    tickers=ticker,
                    flds=fields,
                    start_date=start_date,
                    end_date=end_date,
                    elms=[("periodicitySelection", "DAILY")]
                )
                
                # Rename columns to standard OHLCV names
                rename_map = {
                    'PX_OPEN': 'open',
                    'PX_HIGH': 'high',
                    'PX_LOW': 'low',
                    'PX_LAST': 'close',
                    'PX_VOLUME': 'volume'
                }
                
                # Only rename columns that exist in the data
                rename_cols = {col: rename_map.get(col, col) for col in data.columns.levels[1] if col in rename_map}
                if rename_cols:
                    data = data.rename(columns=rename_cols)
                
                return data
            else:
                raise Exception("Bloomberg connection not available")
        except Exception as e:
            print(f"Error fetching data from Bloomberg: {e}")
            print("Using fallback data source or sample data.")
            return self._get_sample_data(ticker, start_date, end_date)
    
    def fetch_technical_indicators(self, ticker, start_date, end_date=None):
        """
        Fetch pre-calculated technical indicators from Bloomberg.
        
        Parameters:
        -----------
        ticker : str
            Bloomberg ticker symbol
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str, optional
            End date in format 'YYYYMMDD', defaults to today
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing technical indicators
        """
        if end_date is None:
            end_date = dt.datetime.now().strftime('%Y%m%d')
        
        # List of Bloomberg fields for technical indicators
        indicator_fields = [
            'RSI_14D',           # 14-day Relative Strength Index
            'MACD_SIGNAL_LINE',  # MACD Signal Line
            'STOCH_K',           # Stochastic %K
            'STOCH_D',           # Stochastic %D
            'CCI_20D',           # 20-day Commodity Channel Index
            'ATR_14D',           # 14-day Average True Range
            'ROC_10D'            # 10-day Rate of Change
        ]
        
        try:
            if self.con is not None:
                data = self.con.bdh(
                    tickers=ticker,
                    flds=indicator_fields,
                    start_date=start_date,
                    end_date=end_date,
                    elms=[("periodicitySelection", "DAILY")]
                )
                return data
            else:
                raise Exception("Bloomberg connection not available")
        except Exception as e:
            print(f"Error fetching technical indicators from Bloomberg: {e}")
            return None
    
    def _get_sample_data(self, ticker, start_date, end_date):
        """
        Generate sample data when Bloomberg connection is not available.
        This is for testing purposes only.
        """
        start = dt.datetime.strptime(start_date, '%Y%m%d')
        end = dt.datetime.strptime(end_date, '%Y%m%d')
        
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Create sample data
        np.random.seed(42)  # For reproducibility
        n = len(date_range)
        
        # Start with a base price and add random walks
        base_price = 100
        daily_returns = np.random.normal(0.0005, 0.015, n)
        prices = base_price * (1 + np.cumsum(daily_returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.lognormal(15, 1, n).astype(int)
        }, index=date_range)
        
        # Ensure high >= open, close, low and low <= open, close
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        
        return data

# Example usage
if __name__ == "__main__":
    fetcher = BloombergDataFetcher()
    data = fetcher.fetch_historical_data('AAPL US Equity', '20200101', '20210101')
    print(data.head())
