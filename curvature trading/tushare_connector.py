"""
Tushare Data Connector for Chinese Stock Market
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class TushareConnector:
    """
    Connector for fetching Chinese stock data from Tushare
    """
    
    def __init__(self, token: str):
        """
        Initialize Tushare connector
        
        Parameters:
        -----------
        token : str
            Tushare API token
        """
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_top_stocks(self, n: int = 100) -> list:
        """
        Get top N Chinese stocks by market cap
        
        Parameters:
        -----------
        n : int
            Number of top stocks to return
            
        Returns:
        --------
        list : List of stock codes
        """
        try:
            # Get all A-share stocks
            stock_basic = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            # Filter for main board stocks (exclude ST stocks)
            stock_basic = stock_basic[~stock_basic['name'].str.contains('ST', na=False)]
            
            # Get latest market cap data
            trade_date = self.get_latest_trade_date()
            
            print(f"Fetching market cap data for {trade_date}...")
            daily_basic = self.pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,total_mv'
            )
            
            # Merge and sort by market cap
            merged = stock_basic.merge(daily_basic, on='ts_code', how='inner')
            merged = merged.sort_values('total_mv', ascending=False)
            
            # Get top N stocks
            top_stocks = merged.head(n)['ts_code'].tolist()
            
            print(f"✓ Found top {len(top_stocks)} stocks by market cap")
            return top_stocks
            
        except Exception as e:
            print(f"Error getting top stocks: {str(e)}")
            # Fallback: return some major stocks
            return [
                '600519.SH',  # Moutai
                '000858.SZ',  # Wuliangye
                '600036.SH',  # China Merchants Bank
                '601318.SH',  # Ping An Insurance
                '000333.SZ',  # Midea Group
            ]
    
    def get_latest_trade_date(self) -> str:
        """
        Get the latest trading date
        
        Returns:
        --------
        str : Latest trade date in YYYYMMDD format
        """
        try:
            # Get trade calendar
            today = datetime.now()
            start_date = (today - timedelta(days=10)).strftime('%Y%m%d')
            end_date = today.strftime('%Y%m%d')
            
            cal = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )
            
            if len(cal) > 0:
                return cal.iloc[-1]['cal_date']
            else:
                return (today - timedelta(days=1)).strftime('%Y%m%d')
                
        except Exception as e:
            print(f"Error getting latest trade date: {str(e)}")
            return datetime.now().strftime('%Y%m%d')
    
    def fetch_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a stock
        
        Parameters:
        -----------
        ts_code : str
            Tushare stock code (e.g., '600519.SH')
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame : OHLCV data with columns [date, open, high, low, close, volume]
        """
        try:
            # Convert date format
            start_date_ts = start_date.replace('-', '')
            end_date_ts = end_date.replace('-', '')
            
            # Fetch daily data
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date_ts,
                end_date=end_date_ts
            )
            
            if df is None or len(df) == 0:
                return None
            
            # Sort by date ascending
            df = df.sort_values('trade_date')
            
            # Rename columns to match our format
            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume'
            })
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # Select required columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ts_code}: {str(e)}")
            return None
    
    def get_stock_name(self, ts_code: str) -> str:
        """
        Get stock name from code
        
        Parameters:
        -----------
        ts_code : str
            Tushare stock code
            
        Returns:
        --------
        str : Stock name
        """
        try:
            stock_basic = self.pro.stock_basic(
                ts_code=ts_code,
                fields='ts_code,name'
            )
            
            if len(stock_basic) > 0:
                return stock_basic.iloc[0]['name']
            else:
                return ts_code
                
        except Exception as e:
            return ts_code


def test_connection():
    """
    Test Tushare connection
    """
    token = 'bfd5b1c0e45f3c7288e35d6ac2a0f0cc55279b233b37c6980cb61ab3'
    
    print("=" * 80)
    print("Testing Tushare Connection")
    print("=" * 80)
    print()
    
    connector = TushareConnector(token)
    
    # Test 1: Get top stocks
    print("Test 1: Getting top 10 stocks...")
    top_stocks = connector.get_top_stocks(n=10)
    print(f"Top 10 stocks: {top_stocks}")
    print()
    
    # Test 2: Fetch data for one stock
    if len(top_stocks) > 0:
        test_stock = top_stocks[0]
        print(f"Test 2: Fetching data for {test_stock}...")
        
        data = connector.fetch_stock_data(
            ts_code=test_stock,
            start_date='2020-01-01',
            end_date='2025-01-01'
        )
        
        if data is not None:
            print(f"✓ Fetched {len(data)} rows")
            print(f"\nFirst 5 rows:")
            print(data.head())
            print(f"\nLast 5 rows:")
            print(data.tail())
        else:
            print("✗ Failed to fetch data")
    
    print()
    print("=" * 80)
    print("✅ Connection test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_connection()

