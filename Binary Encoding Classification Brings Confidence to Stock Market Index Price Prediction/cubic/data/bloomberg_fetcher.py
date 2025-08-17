"""
Bloomberg Data Fetcher using xbbg package for CUBIC framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple
import time
import warnings

try:
    from xbbg import blp
except ImportError:
    warnings.warn("xbbg package not found. Please install it to use Bloomberg data.")
    blp = None

from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class BloombergDataFetcher:
    """
    Fetches financial data from Bloomberg Terminal using xbbg package
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Bloomberg Data Fetcher
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.bloomberg_config = self.config.get('data.bloomberg', {})
        self.indices_config = self.config.get('data.indices', {})
        self.timeout = self.bloomberg_config.get('timeout', 30000)
        self.max_retries = self.bloomberg_config.get('max_retries', 3)
        
        # Check if Bloomberg connection is available
        self._check_bloomberg_connection()
    
    def _check_bloomberg_connection(self) -> bool:
        """
        Check if Bloomberg Terminal connection is available
        
        Returns:
            bool: True if connection is available, False otherwise
        """
        if blp is None:
            logger.error("xbbg package not installed. Cannot connect to Bloomberg.")
            return False
        
        try:
            # Test connection with a simple query
            test_data = blp.bdh('SPY US Equity', 'PX_LAST', '2024-01-01', '2024-01-02')
            if test_data is not None and not test_data.empty:
                logger.info("Bloomberg connection established successfully.")
                return True
            else:
                logger.warning("Bloomberg connection test returned empty data.")
                return False
        except Exception as e:
            logger.error(f"Bloomberg connection failed: {str(e)}")
            return False
    
    def get_index_constituents(self, index_ticker: str) -> List[str]:
        """
        Get constituent stocks for a given index
        
        Args:
            index_ticker: Bloomberg ticker for the index (e.g., 'SPX Index')
            
        Returns:
            List of constituent stock tickers
        """
        try:
            # Get index members
            constituents = blp.bds(index_ticker, 'INDX_MEMBERS')
            
            if constituents is not None and not constituents.empty:
                # Extract tickers from the result
                tickers = constituents['Member Ticker and Exchange Code'].tolist()
                logger.info(f"Retrieved {len(tickers)} constituents for {index_ticker}")
                return tickers
            else:
                logger.warning(f"No constituents found for {index_ticker}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching constituents for {index_ticker}: {str(e)}")
            return []
    
    def get_historical_data(self, 
                          tickers: List[str], 
                          fields: List[str],
                          start_date: str, 
                          end_date: str,
                          retry_count: int = 0) -> pd.DataFrame:
        """
        Fetch historical data for given tickers and fields
        
        Args:
            tickers: List of Bloomberg tickers
            fields: List of Bloomberg fields (e.g., ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            retry_count: Current retry attempt
            
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Fetching historical data for {len(tickers)} tickers from {start_date} to {end_date}")
            
            # Fetch data using xbbg
            data = blp.bdh(tickers, fields, start_date, end_date)
            
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched data: {data.shape}")
                return data
            else:
                logger.warning("Received empty data from Bloomberg")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying... Attempt {retry_count + 1}/{self.max_retries}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.get_historical_data(tickers, fields, start_date, end_date, retry_count + 1)
            else:
                logger.error("Max retries exceeded. Returning empty DataFrame.")
                return pd.DataFrame()
    
    def get_index_data(self, 
                      index_name: str, 
                      start_date: str, 
                      end_date: str,
                      include_constituents: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive data for a market index including constituents
        
        Args:
            index_name: Name of the index (e.g., 'SPX', 'HSI')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            include_constituents: Whether to fetch constituent stock data
            
        Returns:
            Dictionary containing index data and constituent data
        """
        if index_name not in self.indices_config:
            raise ValueError(f"Index {index_name} not found in configuration")
        
        index_config = self.indices_config[index_name]
        index_ticker = index_config['bloomberg_ticker']
        
        result = {}
        
        # Define fields to fetch
        price_fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
        
        # Fetch index data
        logger.info(f"Fetching index data for {index_name}")
        index_data = self.get_historical_data([index_ticker], price_fields, start_date, end_date)
        result['index'] = index_data
        
        if include_constituents:
            # Get constituent tickers
            constituents = self.get_index_constituents(index_ticker)
            
            if constituents:
                # Limit constituents based on configuration
                max_constituents = index_config.get('constituent_count', len(constituents))
                constituents = constituents[:max_constituents]
                
                # Fetch constituent data in batches to avoid Bloomberg limits
                batch_size = 50  # Bloomberg typically limits to 50-100 securities per request
                constituent_data_list = []
                
                for i in range(0, len(constituents), batch_size):
                    batch = constituents[i:i + batch_size]
                    logger.info(f"Fetching batch {i//batch_size + 1}: {len(batch)} constituents")
                    
                    batch_data = self.get_historical_data(batch, price_fields, start_date, end_date)
                    if not batch_data.empty:
                        constituent_data_list.append(batch_data)
                
                # Combine all constituent data
                if constituent_data_list:
                    constituent_data = pd.concat(constituent_data_list, axis=1)
                    result['constituents'] = constituent_data
                    logger.info(f"Successfully fetched data for {len(constituents)} constituents")
                else:
                    logger.warning("No constituent data retrieved")
                    result['constituents'] = pd.DataFrame()
            else:
                logger.warning("No constituents found")
                result['constituents'] = pd.DataFrame()
        
        return result
    
    def get_real_time_data(self, tickers: List[str], fields: List[str]) -> pd.DataFrame:
        """
        Get real-time data for given tickers
        
        Args:
            tickers: List of Bloomberg tickers
            fields: List of Bloomberg fields
            
        Returns:
            DataFrame with real-time data
        """
        try:
            logger.info(f"Fetching real-time data for {len(tickers)} tickers")
            
            # Fetch real-time data using xbbg
            data = blp.bdp(tickers, fields)
            
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched real-time data: {data.shape}")
                return data
            else:
                logger.warning("Received empty real-time data from Bloomberg")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
            return pd.DataFrame()
    
    def save_data(self, data: Dict[str, pd.DataFrame], filepath: str) -> None:
        """
        Save fetched data to file
        
        Args:
            data: Dictionary containing DataFrames to save
            filepath: Path to save the data
        """
        try:
            # Save as pickle for preserving data types and structure
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
    
    def load_data(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """
        Load previously saved data
        
        Args:
            filepath: Path to the saved data file
            
        Returns:
            Dictionary containing loaded DataFrames
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return {}
