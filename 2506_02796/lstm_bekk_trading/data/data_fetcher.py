"""
Data Fetcher Module

Implements data fetching from various sources including Bloomberg and Yahoo Finance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

try:
    import xbbg
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    logging.warning("xbbg not available. Bloomberg data fetching disabled.")

try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    logging.warning("yfinance not available. Yahoo Finance data fetching disabled.")


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    @abstractmethod
    def fetch_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data for given symbols and date range."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass


class BloombergFetcher(DataFetcher):
    """Bloomberg data fetcher using xbbg package."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Bloomberg data is available."""
        return BLOOMBERG_AVAILABLE
    
    def fetch_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price data from Bloomberg.
        
        Args:
            symbols: List of Bloomberg tickers (e.g., ['AAPL US Equity'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with dates as index and symbols as columns
        """
        if not self.is_available():
            raise RuntimeError("Bloomberg data not available. Install xbbg package.")
        
        try:
            self.logger.info(f"Fetching Bloomberg data for {len(symbols)} symbols")
            
            # Fetch data using xbbg
            data = xbbg.blp.bdh(
                tickers=symbols,
                flds=['PX_LAST'],  # Last price
                start_date=start_date,
                end_date=end_date
            )
            
            # Reshape data if needed
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-index columns
                data.columns = [col[0] for col in data.columns]
            
            # Clean symbol names (remove ' US Equity' etc.)
            data.columns = [self._clean_symbol_name(col) for col in data.columns]
            
            self.logger.info(f"Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Bloomberg data: {e}")
            raise
    
    def _clean_symbol_name(self, symbol: str) -> str:
        """Clean Bloomberg symbol names."""
        # Remove common Bloomberg suffixes
        suffixes = [' US Equity', ' LN Equity', ' JP Equity', ' Equity']
        for suffix in suffixes:
            if symbol.endswith(suffix):
                return symbol.replace(suffix, '')
        return symbol


class YahooFetcher(DataFetcher):
    """Yahoo Finance data fetcher using yfinance package."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Yahoo Finance data is available."""
        return YAHOO_AVAILABLE
    
    def fetch_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance.
        
        Args:
            symbols: List of Yahoo Finance tickers (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with dates as index and symbols as columns
        """
        if not self.is_available():
            raise RuntimeError("Yahoo Finance data not available. Install yfinance package.")
        
        try:
            self.logger.info(f"Fetching Yahoo Finance data for {len(symbols)} symbols")
            
            # Convert Bloomberg symbols to Yahoo symbols if needed
            yahoo_symbols = [self._convert_to_yahoo_symbol(symbol) for symbol in symbols]
            
            # Fetch data
            data = yf.download(
                tickers=yahoo_symbols,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Extract adjusted close prices
            if len(yahoo_symbols) == 1:
                prices = data['Adj Close'].to_frame()
                prices.columns = [symbols[0]]
            else:
                prices = data['Adj Close']
                # Map back to original symbol names
                symbol_mapping = dict(zip(yahoo_symbols, symbols))
                prices.columns = [symbol_mapping.get(col, col) for col in prices.columns]
            
            self.logger.info(f"Successfully fetched {len(prices)} rows of data")
            return prices
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data: {e}")
            raise
    
    def _convert_to_yahoo_symbol(self, bloomberg_symbol: str) -> str:
        """Convert Bloomberg symbol to Yahoo Finance symbol."""
        # Basic conversion mapping
        if ' US Equity' in bloomberg_symbol:
            return bloomberg_symbol.replace(' US Equity', '')
        elif ' LN Equity' in bloomberg_symbol:
            # London stocks often have .L suffix
            base = bloomberg_symbol.replace(' LN Equity', '')
            return f"{base}.L"
        elif ' JP Equity' in bloomberg_symbol:
            # Japanese stocks often have .T suffix
            base = bloomberg_symbol.replace(' JP Equity', '')
            return f"{base}.T"
        else:
            return bloomberg_symbol


class DataFetcherFactory:
    """Factory class for creating appropriate data fetchers."""
    
    @staticmethod
    def create_fetcher(source: str = "auto") -> DataFetcher:
        """
        Create a data fetcher based on available sources.
        
        Args:
            source: Data source ('bloomberg', 'yahoo', or 'auto')
            
        Returns:
            DataFetcher instance
        """
        if source == "bloomberg":
            if not BLOOMBERG_AVAILABLE:
                raise RuntimeError("Bloomberg data not available. Install xbbg package.")
            return BloombergFetcher()
        
        elif source == "yahoo":
            if not YAHOO_AVAILABLE:
                raise RuntimeError("Yahoo Finance data not available. Install yfinance package.")
            return YahooFetcher()
        
        elif source == "auto":
            # Try Bloomberg first, then Yahoo
            if BLOOMBERG_AVAILABLE:
                return BloombergFetcher()
            elif YAHOO_AVAILABLE:
                return YahooFetcher()
            else:
                raise RuntimeError("No data sources available. Install xbbg or yfinance.")
        
        else:
            raise ValueError(f"Unknown data source: {source}")
