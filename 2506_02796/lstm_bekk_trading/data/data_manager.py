"""
Data Manager Module

Central data management class that coordinates data fetching, processing, and storage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import yaml
from pathlib import Path

from .data_fetcher import DataFetcherFactory
from .data_processor import DataProcessor


class DataManager:
    """
    Central data manager for the LSTM-BEKK trading system.
    
    Coordinates data fetching, processing, and provides a unified interface
    for accessing financial data.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize data manager.
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides config_path)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Initialize components
        self.data_fetcher = DataFetcherFactory.create_fetcher(
            self.config.get('data', {}).get('source', 'auto')
        )
        self.data_processor = DataProcessor(self.config)
        
        # Data storage
        self.raw_prices = None
        self.returns = None
        self.demeaned_returns = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self, universe: str = "sp500_sample", 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load and process data for a given universe.
        
        Args:
            universe: Name of the stock universe to load
            start_date: Start date (overrides config)
            end_date: End date (overrides config)
            
        Returns:
            Processed returns data
        """
        self.logger.info(f"Loading data for universe: {universe}")
        
        # Get symbols from config
        symbols = self.config['data']['universes'][universe]
        
        # Get date range
        start_date = start_date or self.config['data']['bloomberg']['start_date']
        end_date = end_date or self.config['data']['bloomberg']['end_date']
        
        # Fetch raw price data
        self.raw_prices = self.data_fetcher.fetch_prices(symbols, start_date, end_date)
        
        # Clean price data
        self.raw_prices = self.data_processor.clean_data(self.raw_prices)
        
        # Calculate returns
        self.returns = self.data_processor.calculate_returns(self.raw_prices)
        
        # De-mean returns for LSTM-BEKK
        self.demeaned_returns = self.data_processor.demean_returns(self.returns)
        
        # Create train/test splits
        self.train_data, self.val_data, self.test_data = self.data_processor.create_train_test_split(
            self.demeaned_returns,
            train_ratio=self.config.get('backtesting', {}).get('train_ratio', 0.7),
            validation_ratio=self.config.get('backtesting', {}).get('validation_ratio', 0.15)
        )
        
        self.logger.info(f"Data loading complete. Shape: {self.demeaned_returns.shape}")
        return self.demeaned_returns
    
    def get_returns(self, demeaned: bool = True) -> pd.DataFrame:
        """
        Get return data.
        
        Args:
            demeaned: Whether to return de-meaned returns
            
        Returns:
            Returns DataFrame
        """
        if demeaned:
            if self.demeaned_returns is None:
                raise ValueError("No de-meaned returns available. Call load_data() first.")
            return self.demeaned_returns
        else:
            if self.returns is None:
                raise ValueError("No returns available. Call load_data() first.")
            return self.returns
    
    def get_prices(self) -> pd.DataFrame:
        """Get raw price data."""
        if self.raw_prices is None:
            raise ValueError("No price data available. Call load_data() first.")
        return self.raw_prices
    
    def get_train_data(self) -> pd.DataFrame:
        """Get training data."""
        if self.train_data is None:
            raise ValueError("No training data available. Call load_data() first.")
        return self.train_data
    
    def get_validation_data(self) -> pd.DataFrame:
        """Get validation data."""
        if self.val_data is None:
            raise ValueError("No validation data available. Call load_data() first.")
        return self.val_data
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test data."""
        if self.test_data is None:
            raise ValueError("No test data available. Call load_data() first.")
        return self.test_data
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive data statistics."""
        if self.demeaned_returns is None:
            raise ValueError("No data available. Call load_data() first.")
        
        return self.data_processor.get_data_statistics(self.demeaned_returns)
    
    def validate_data_quality(self) -> Dict[str, bool]:
        """Validate data quality for modeling."""
        if self.demeaned_returns is None:
            raise ValueError("No data available. Call load_data() first.")
        
        return self.data_processor.validate_data_quality(self.demeaned_returns)
    
    def save_data(self, filepath: str, data_type: str = "returns") -> None:
        """
        Save data to file.
        
        Args:
            filepath: Output file path
            data_type: Type of data to save ('prices', 'returns', 'demeaned_returns')
        """
        data_map = {
            'prices': self.raw_prices,
            'returns': self.returns,
            'demeaned_returns': self.demeaned_returns
        }
        
        if data_type not in data_map:
            raise ValueError(f"Unknown data type: {data_type}")
        
        data = data_map[data_type]
        if data is None:
            raise ValueError(f"No {data_type} data available.")
        
        # Save based on file extension
        if filepath.endswith('.csv'):
            data.to_csv(filepath)
        elif filepath.endswith('.parquet'):
            data.to_parquet(filepath)
        elif filepath.endswith('.pickle'):
            data.to_pickle(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .pickle")
        
        self.logger.info(f"Saved {data_type} data to {filepath}")
    
    def load_saved_data(self, filepath: str, data_type: str = "returns") -> pd.DataFrame:
        """
        Load previously saved data.
        
        Args:
            filepath: Input file path
            data_type: Type of data to load
            
        Returns:
            Loaded DataFrame
        """
        # Load based on file extension
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        elif filepath.endswith('.pickle'):
            data = pd.read_pickle(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .pickle")
        
        # Store in appropriate attribute
        if data_type == "prices":
            self.raw_prices = data
        elif data_type == "returns":
            self.returns = data
        elif data_type == "demeaned_returns":
            self.demeaned_returns = data
        
        self.logger.info(f"Loaded {data_type} data from {filepath}")
        return data
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'source': 'auto',
                'bloomberg': {
                    'start_date': '2020-01-01',
                    'end_date': '2024-01-01'
                },
                'universes': {
                    'default': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                }
            },
            'backtesting': {
                'train_ratio': 0.7,
                'validation_ratio': 0.15
            }
        }
