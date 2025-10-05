"""
Data Collection Module - Fetch data from Bloomberg via xbbg
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import pytz

try:
    from xbbg import blp
except ImportError:
    print("Warning: xbbg not installed. Bloomberg data collection will not work.")
    blp = None

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collect market data from Bloomberg"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize DataCollector
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        self.hk_tz = pytz.timezone(config.HK_TIMEZONE)
        self.us_tz = pytz.timezone(config.US_TIMEZONE)
        
    def fetch_price_data(self, ticker: str, fields: List[str] = None) -> pd.DataFrame:
        """
        Fetch price data from Bloomberg

        Args:
            ticker: Bloomberg ticker
            fields: List of fields to fetch

        Returns:
            DataFrame with price data
        """
        if blp is None:
            raise ImportError("xbbg package is required for Bloomberg data collection")

        # Use xbbg field names (they are case-insensitive but use specific names)
        if fields is None:
            fields = ['Open', 'High', 'Low', 'Last_Price', 'Volume']

        try:
            logger.info(f"Fetching data for {ticker} from {self.start_date} to {self.end_date}")
            df = blp.bdh(
                tickers=ticker,
                flds=fields,
                start_date=self.start_date,
                end_date=self.end_date
            )

            # xbbg returns DataFrame with date as index and multi-level columns (ticker, field)
            # Flatten the columns
            if isinstance(df.columns, pd.MultiIndex):
                # Extract just the field names (second level)
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

            # Reset index to make date a column
            df = df.reset_index()
            if 'index' in df.columns:
                df.rename(columns={'index': 'date'}, inplace=True)

            # Standardize column names to match expected format
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Last_Price': 'close',
                'Volume': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)

            logger.info(f"Fetched {len(df)} records for {ticker}")
            logger.info(f"Columns: {list(df.columns)}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise
    
    def fetch_baba_us_data(self) -> pd.DataFrame:
        """Fetch BABA US stock data"""
        df = self.fetch_price_data(config.BABA_US_TICKER)
        df['ticker'] = 'BABA'
        return df
    
    def fetch_baba_hk_data(self) -> pd.DataFrame:
        """Fetch 9988.HK stock data"""
        df = self.fetch_price_data(config.BABA_HK_TICKER)
        df['ticker'] = '9988.HK'
        return df
    
    def fetch_usdhkd_data(self) -> pd.DataFrame:
        """Fetch USDHKD exchange rate data"""
        df = self.fetch_price_data(config.USDHKD_TICKER, fields=['Last_Price'])
        # Rename close column to usdhkd_rate
        if 'close' in df.columns:
            df.rename(columns={'close': 'usdhkd_rate'}, inplace=True)
        return df

    def fetch_vix_data(self) -> pd.DataFrame:
        """Fetch VIX data"""
        df = self.fetch_price_data(config.VIX_TICKER, fields=['Last_Price'])
        # Rename close column to vix_level
        if 'close' in df.columns:
            df.rename(columns={'close': 'vix_level'}, inplace=True)
        return df

    def fetch_treasury_data(self) -> pd.DataFrame:
        """Fetch US Treasury yield data"""
        # Fetch 10-year yield
        df_10y = self.fetch_price_data(config.US_10Y_TICKER, fields=['Last_Price'])
        if 'close' in df_10y.columns:
            df_10y.rename(columns={'close': 'us_10y_yield'}, inplace=True)

        # Fetch 3-month yield
        df_3m = self.fetch_price_data(config.US_3M_TICKER, fields=['Last_Price'])
        if 'close' in df_3m.columns:
            df_3m.rename(columns={'close': 'us_3m_yield'}, inplace=True)

        # Merge
        df = pd.merge(df_10y, df_3m, on='date', how='outer')
        return df
    
    def fetch_pdd_data(self) -> pd.DataFrame:
        """Fetch PDD (Pinduoduo) stock data"""
        df = self.fetch_price_data(config.PDD_TICKER)
        df['ticker'] = 'PDD'
        return df
    
    def fetch_implied_volatility(self) -> pd.DataFrame:
        """Fetch BABA implied volatility from options"""
        try:
            # Fetch ATM implied volatility
            df = blp.bdh(
                tickers=config.BABA_US_TICKER,
                flds=['IVOL_MID'],
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            df = df.reset_index()
            df.rename(columns={'index': 'date', f'{config.BABA_US_TICKER}_IVOL_MID': 'implied_vol'}, inplace=True)
            
            return df
        except Exception as e:
            logger.warning(f"Could not fetch implied volatility: {str(e)}")
            return pd.DataFrame(columns=['date', 'implied_vol'])
    
    def identify_common_trading_days(self, 
                                     baba_df: pd.DataFrame, 
                                     hk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify days where both US and HK markets are open on day T and T+1
        
        Args:
            baba_df: BABA US data
            hk_df: 9988.HK data
            
        Returns:
            DataFrame with common trading days
        """
        # Get trading days for each market
        us_trading_days = set(pd.to_datetime(baba_df['date']).dt.date)
        hk_trading_days = set(pd.to_datetime(hk_df['date']).dt.date)
        
        # Find days where both markets are open
        common_days = sorted(us_trading_days & hk_trading_days)
        
        # Filter to ensure T+1 is also a common trading day
        valid_days = []
        for i in range(len(common_days) - 1):
            current_day = common_days[i]
            next_day = common_days[i + 1]
            
            # Check if next_day is actually T+1 (consecutive trading day)
            if (next_day - current_day).days <= 5:  # Allow for weekends
                valid_days.append(current_day)
        
        logger.info(f"Found {len(valid_days)} valid trading days where both markets are open on T and T+1")
        
        return pd.DataFrame({'date': valid_days})
    
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data from Bloomberg
        
        Returns:
            Dictionary of DataFrames with all market data
        """
        logger.info("Starting data collection from Bloomberg...")
        
        data = {}
        
        # Fetch all data
        data['baba_us'] = self.fetch_baba_us_data()
        data['baba_hk'] = self.fetch_baba_hk_data()
        data['usdhkd'] = self.fetch_usdhkd_data()
        data['vix'] = self.fetch_vix_data()
        data['treasury'] = self.fetch_treasury_data()
        data['pdd'] = self.fetch_pdd_data()
        data['implied_vol'] = self.fetch_implied_volatility()
        
        # Identify common trading days
        data['common_days'] = self.identify_common_trading_days(
            data['baba_us'], 
            data['baba_hk']
        )
        
        logger.info("Data collection completed")
        
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = None):
        """
        Save collected data to parquet files
        
        Args:
            data: Dictionary of DataFrames
            output_dir: Output directory
        """
        output_dir = output_dir or config.DATA_DIR
        
        for name, df in data.items():
            output_path = f"{output_dir}/{name}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {name} data to {output_path}")


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    data = collector.collect_all_data()
    collector.save_data(data)

