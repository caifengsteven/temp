"""
Database Connector for US Stock SIP Day Aggregates

This module provides functions to fetch OHLC data from the NAS database.
"""

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class StockDataConnector:
    """
    Connector to fetch stock data from the database
    """
    
    def __init__(self):
        """Initialize database connection parameters"""
        self.db_config = {
            'host': '192.168.50.230',
            'port': 3306,
            'user': 'root',
            'password': '352471Cf!1',
            'database': 'us_stock_sip_day_aggs'
        }
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        if self.conn is None or not self.conn.open:
            self.conn = pymysql.connect(**self.db_config)
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn and self.conn.open:
            self.conn.close()
    
    def get_table_names(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of table names between start and end dates
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        List[str] : List of table names (YYYYMM format)
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        tables = []
        current = start
        while current <= end:
            table_name = current.strftime('%Y%m')
            tables.append(table_name)
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return tables
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        min_volume: int = 100000
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        min_volume : int
            Minimum volume filter (default: 100,000)
            
        Returns:
        --------
        pd.DataFrame : DataFrame with columns [date, open, high, low, close, volume]
        """
        self.connect()
        
        # Get table names for the date range
        tables = self.get_table_names(start_date, end_date)
        
        all_data = []
        
        for table in tables:
            try:
                query = f"""
                    SELECT 
                        ticker,
                        window_start,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        transactions
                    FROM `{table}`
                    WHERE ticker = %s
                    AND volume >= %s
                    ORDER BY window_start
                """
                
                df = pd.read_sql(query, self.conn, params=(ticker, min_volume))
                
                if not df.empty:
                    all_data.append(df)
                    
            except Exception as e:
                # Table might not exist or other error
                print(f"Warning: Could not fetch data from table {table}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Convert window_start (nanoseconds) to datetime
        result['date'] = pd.to_datetime(result['window_start'], unit='ns')
        
        # Filter by exact date range
        result = result[
            (result['date'] >= start_date) & 
            (result['date'] <= end_date)
        ]
        
        # Convert decimal to float
        for col in ['open', 'high', 'low', 'close']:
            result[col] = result[col].astype(float)
        
        # Sort by date
        result = result.sort_values('date').reset_index(drop=True)
        
        # Select and reorder columns
        result = result[['date', 'open', 'high', 'low', 'close', 'volume', 'transactions']]
        
        return result
    
    def get_available_tickers(
        self,
        date: str,
        min_volume: int = 1000000,
        limit: int = 100
    ) -> List[str]:
        """
        Get list of available tickers for a specific date
        
        Parameters:
        -----------
        date : str
            Date in format 'YYYY-MM-DD'
        min_volume : int
            Minimum volume filter (default: 1,000,000)
        limit : int
            Maximum number of tickers to return
            
        Returns:
        --------
        List[str] : List of ticker symbols
        """
        self.connect()
        
        # Get table name for the date
        dt = datetime.strptime(date, '%Y-%m-%d')
        table = dt.strftime('%Y%m')
        
        try:
            query = f"""
                SELECT DISTINCT ticker
                FROM `{table}`
                WHERE volume >= %s
                ORDER BY ticker
                LIMIT %s
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, (min_volume, limit))
            tickers = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            return tickers
            
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return []
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        min_volume: int = 100000
    ) -> dict:
        """
        Fetch data for multiple tickers
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        min_volume : int
            Minimum volume filter
            
        Returns:
        --------
        dict : Dictionary with ticker as key and DataFrame as value
        """
        results = {}
        
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            df = self.fetch_stock_data(ticker, start_date, end_date, min_volume)
            
            if not df.empty:
                results[ticker] = df
            else:
                print(f"  No data found for {ticker}")
        
        return results
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def example_usage():
    """Example usage of the StockDataConnector"""
    
    print("Stock Data Connector - Example Usage")
    print("=" * 60)
    
    # Create connector
    connector = StockDataConnector()
    
    try:
        # Example 1: Fetch data for a single stock
        print("\n1. Fetching AAPL data for 2023...")
        aapl_data = connector.fetch_stock_data(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_volume=1000000
        )
        
        print(f"   Retrieved {len(aapl_data)} trading days")
        print(f"   Date range: {aapl_data['date'].min()} to {aapl_data['date'].max()}")
        print(f"\n   Sample data:")
        print(aapl_data.head())
        
        # Example 2: Get available tickers
        print("\n2. Getting available tickers for 2023-01-01...")
        tickers = connector.get_available_tickers(
            date='2023-01-01',
            min_volume=5000000,
            limit=20
        )
        print(f"   Found {len(tickers)} tickers: {tickers[:10]}...")
        
        # Example 3: Fetch multiple stocks
        print("\n3. Fetching data for multiple stocks...")
        stocks = connector.get_multiple_stocks(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-03-31',
            min_volume=1000000
        )
        
        for ticker, df in stocks.items():
            print(f"   {ticker}: {len(df)} days")
        
    finally:
        connector.close()
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    example_usage()

