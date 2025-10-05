"""
CSV Data Processing Module - Process minute-level and option data using DuckDB
"""
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class CSVProcessor:
    """Process CSV files using DuckDB for efficient querying"""
    
    def __init__(self, minute_data_file: str = None, option_data_file: str = None):
        """
        Initialize CSVProcessor
        
        Args:
            minute_data_file: Path to minute-level data CSV
            option_data_file: Path to option trades CSV
        """
        self.minute_data_file = minute_data_file or config.MINUTE_DATA_FILE
        self.option_data_file = option_data_file or config.OPTION_DATA_FILE
        self.conn = duckdb.connect(':memory:')
        self.us_tz = pytz.timezone(config.US_TIMEZONE)
        self.hk_tz = pytz.timezone(config.HK_TIMEZONE)
        
    def _nanoseconds_to_datetime(self, ns_timestamp: int, timezone: str = 'US') -> datetime:
        """
        Convert nanosecond timestamp to datetime
        
        Args:
            ns_timestamp: Timestamp in nanoseconds
            timezone: 'US' or 'HK'
            
        Returns:
            datetime object
        """
        dt = pd.to_datetime(ns_timestamp, unit='ns')
        
        # Localize to US timezone (data is in US time)
        dt = dt.tz_localize('UTC').tz_convert(self.us_tz)
        
        # Convert to HK time if requested
        if timezone == 'HK':
            dt = dt.tz_convert(self.hk_tz)
        
        return dt
    
    def load_minute_data(self):
        """Load minute-level data into DuckDB"""
        logger.info(f"Loading minute data from {self.minute_data_file}")
        
        # Create table from CSV
        self.conn.execute(f"""
            CREATE TABLE minute_data AS 
            SELECT * FROM read_csv_auto('{self.minute_data_file}')
        """)
        
        # Add date column (convert nanosecond timestamp to date)
        self.conn.execute("""
            ALTER TABLE minute_data 
            ADD COLUMN trade_date DATE
        """)
        
        self.conn.execute("""
            UPDATE minute_data 
            SET trade_date = CAST(to_timestamp(window_start / 1000000000) AS DATE)
        """)
        
        logger.info("Minute data loaded successfully")
    
    def load_option_data(self):
        """Load option trades data into DuckDB"""
        logger.info(f"Loading option data from {self.option_data_file}")
        
        # Create table from CSV
        self.conn.execute(f"""
            CREATE TABLE option_data AS 
            SELECT * FROM read_csv_auto('{self.option_data_file}')
        """)
        
        # Add date column
        self.conn.execute("""
            ALTER TABLE option_data 
            ADD COLUMN trade_date DATE
        """)
        
        self.conn.execute("""
            UPDATE option_data 
            SET trade_date = CAST(to_timestamp(sip_timestamp / 1000000000) AS DATE)
        """)
        
        logger.info("Option data loaded successfully")
    
    def get_last_30min_price_change(self, date: str = None) -> pd.DataFrame:
        """
        Calculate BABA last 30 minutes price change for each trading day
        
        Args:
            date: Specific date to query (optional)
            
        Returns:
            DataFrame with date and last_30min_pct_change
        """
        logger.info("Calculating last 30 minutes price change")
        
        date_filter = f"WHERE trade_date = '{date}'" if date else ""
        
        query = f"""
        WITH last_30min AS (
            SELECT 
                trade_date,
                window_start,
                close,
                ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY window_start DESC) as rn
            FROM minute_data
            WHERE EXTRACT(HOUR FROM to_timestamp(window_start / 1000000000)) * 60 + 
                  EXTRACT(MINUTE FROM to_timestamp(window_start / 1000000000)) >= 15 * 60 + 30
            {date_filter}
        ),
        price_at_1530 AS (
            SELECT 
                trade_date,
                close as price_1530
            FROM last_30min
            WHERE rn = (SELECT MAX(rn) FROM last_30min l WHERE l.trade_date = last_30min.trade_date)
        ),
        price_at_close AS (
            SELECT 
                trade_date,
                close as price_close
            FROM last_30min
            WHERE rn = 1
        )
        SELECT 
            p1.trade_date as date,
            ((p2.price_close - p1.price_1530) / p1.price_1530 * 100) as last_30min_pct_change
        FROM price_at_1530 p1
        JOIN price_at_close p2 ON p1.trade_date = p2.trade_date
        ORDER BY p1.trade_date
        """
        
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Calculated last 30min price change for {len(df)} days")
        
        return df
    
    def get_first_30min_volume(self, date: str = None) -> pd.DataFrame:
        """
        Calculate BABA first 30 minutes trading volume for each trading day
        
        Args:
            date: Specific date to query (optional)
            
        Returns:
            DataFrame with date and first_30min_volume
        """
        logger.info("Calculating first 30 minutes volume")
        
        date_filter = f"AND trade_date = '{date}'" if date else ""
        
        query = f"""
        SELECT 
            trade_date as date,
            SUM(volume) as first_30min_volume
        FROM minute_data
        WHERE EXTRACT(HOUR FROM to_timestamp(window_start / 1000000000)) * 60 + 
              EXTRACT(MINUTE FROM to_timestamp(window_start / 1000000000)) 
              BETWEEN 9 * 60 + 30 AND 10 * 60
        {date_filter}
        GROUP BY trade_date
        ORDER BY trade_date
        """
        
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Calculated first 30min volume for {len(df)} days")
        
        return df
    
    def get_daily_option_volume(self, date: str = None) -> pd.DataFrame:
        """
        Calculate total BABA option trading volume for each trading day
        
        Args:
            date: Specific date to query (optional)
            
        Returns:
            DataFrame with date and option_volume
        """
        logger.info("Calculating daily option volume")
        
        date_filter = f"WHERE trade_date = '{date}'" if date else ""
        
        query = f"""
        SELECT 
            trade_date as date,
            SUM(size) as option_volume
        FROM option_data
        {date_filter}
        GROUP BY trade_date
        ORDER BY trade_date
        """
        
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Calculated option volume for {len(df)} days")
        
        return df
    

    
    def process_all_intraday_features(self) -> pd.DataFrame:
        """
        Process all intraday features from CSV files

        Returns:
            DataFrame with all intraday features
        """
        logger.info("Processing all intraday features")

        # Load data into DuckDB
        self.load_minute_data()
        self.load_option_data()

        # Calculate all features
        last_30min = self.get_last_30min_price_change()
        first_30min = self.get_first_30min_volume()
        option_vol = self.get_daily_option_volume()

        # Merge all features
        df = last_30min
        df = df.merge(first_30min, on='date', how='outer')
        df = df.merge(option_vol, on='date', how='outer')

        # Rename columns
        df.rename(columns={
            'last_30min_pct_change': 'baba_last_30min_pct_change',
            'first_30min_volume': 'baba_first_30min_volume',
            'option_volume': 'baba_option_volume'
        }, inplace=True)

        logger.info(f"Processed intraday features for {len(df)} days")

        return df
    
    def close(self):
        """Close DuckDB connection"""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


if __name__ == "__main__":
    # Example usage
    with CSVProcessor() as processor:
        intraday_features = processor.process_all_intraday_features()
        print(intraday_features.head())
        
        # Save to file
        output_path = f"{config.DATA_DIR}/intraday_features.parquet"
        intraday_features.to_parquet(output_path, index=False)
        logger.info(f"Saved intraday features to {output_path}")

