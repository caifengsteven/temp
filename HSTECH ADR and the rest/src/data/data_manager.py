"""
Data Manager for HSTECH Estimation System

This module coordinates all data operations including fetching, caching,
and providing data to the estimation system.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import pandas as pd

from ..models import PriceData, CurrencyRate, IndexData, Config
from .data_fetcher import DataFetcher
from .adr_mapper import ADRMapper
from ...data.hstech_components import HSTECH_COMPONENTS, MARKET_INDICATORS, get_adr_mapped_components

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages all data operations for the HSTECH estimation system.
    
    Responsibilities:
    - Coordinate data fetching from multiple sources
    - Manage data caching and refresh schedules
    - Provide clean, validated data to estimation algorithms
    - Handle data quality checks and fallback mechanisms
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.adr_mapper = ADRMapper()
        
        # Data storage
        self.current_adr_prices = {}
        self.current_hk_prices = {}
        self.current_indicator_prices = {}
        self.previous_indicator_prices = {}
        self.current_exchange_rate = None
        self.last_hstech_close = None
        
        # Update tracking
        self.last_update_time = None
        self.update_frequency = config.data_sources.price_update_frequency
        
        logger.info("Initialized data manager")
    
    async def fetch_all_current_data(self) -> bool:
        """
        Fetch all current market data needed for estimation.
        
        Returns:
            True if all critical data was fetched successfully
        """
        logger.info("Fetching all current market data")
        
        success = True
        
        try:
            # Fetch ADR prices
            adr_symbols = self.adr_mapper.get_all_adr_symbols()
            self.current_adr_prices = await self.data_fetcher.fetch_adr_prices(adr_symbols)
            
            if not self.current_adr_prices:
                logger.error("Failed to fetch any ADR prices")
                success = False
            
            # Fetch HK prices (last known)
            hk_symbols = [stock.symbol for stock in HSTECH_COMPONENTS]
            self.current_hk_prices = await self.data_fetcher.fetch_last_hk_prices(hk_symbols)
            
            if not self.current_hk_prices:
                logger.error("Failed to fetch any HK prices")
                success = False
            
            # Fetch market indicator prices
            self.previous_indicator_prices = self.current_indicator_prices.copy()  # Save previous
            self.current_indicator_prices = await self.data_fetcher.fetch_market_indicator_prices(MARKET_INDICATORS)
            
            if not self.current_indicator_prices:
                logger.warning("Failed to fetch market indicator prices")
            
            # Fetch currency rate
            self.current_exchange_rate = await self.data_fetcher.fetch_currency_rate("USD", "HKD")
            
            if not self.current_exchange_rate:
                logger.error("Failed to fetch currency rate")
                success = False
            
            # Fetch HSTECH index
            self.last_hstech_close = await self.data_fetcher.fetch_hstech_index()
            
            if not self.last_hstech_close:
                logger.error("Failed to fetch HSTECH index")
                success = False
            
            self.last_update_time = datetime.now(timezone.utc)
            
            logger.info(f"Data fetch complete. Success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return False
    
    async def get_estimation_data(self) -> Tuple[Dict, Dict, CurrencyRate, IndexData, Dict, Dict]:
        """
        Get all data needed for HSTECH estimation.
        
        Returns:
            Tuple of (current_adr_prices, last_hk_prices, exchange_rate, 
                     last_hstech_close, current_indicators, previous_indicators)
        """
        # Check if data needs refresh
        if self._needs_data_refresh():
            await self.fetch_all_current_data()
        
        return (
            self.current_adr_prices,
            self.current_hk_prices,
            self.current_exchange_rate,
            self.last_hstech_close,
            self.current_indicator_prices,
            self.previous_indicator_prices
        )
    
    async def fetch_historical_data(
        self, 
        symbols: List[str], 
        days: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for covariance modeling.
        
        Args:
            symbols: List of stock symbols
            days: Number of days of historical data
            
        Returns:
            Dict of {symbol: DataFrame with date, close, returns columns}
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols, {days} days")
        
        historical_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra buffer for weekends/holidays
        
        # Try Bloomberg first for historical data
        try:
            # Determine market for symbols
            hk_symbols = [s for s in symbols if s.endswith('.HK')]
            us_symbols = [s for s in symbols if not s.endswith('.HK')]

            # Fetch HK historical data
            if hk_symbols:
                hk_historical = await self.data_fetcher.bloomberg_fetcher.fetch_historical_data(
                    hk_symbols, days, "HK"
                )
                historical_data.update(hk_historical)

            # Fetch US historical data
            if us_symbols:
                us_historical = await self.data_fetcher.bloomberg_fetcher.fetch_historical_data(
                    us_symbols, days, "US"
                )
                historical_data.update(us_historical)

        except Exception as e:
            logger.warning(f"Bloomberg historical data fetch failed: {e}")

        # Fill missing data with fallback sources
        missing_symbols = [s for s in symbols if s not in historical_data]
        if missing_symbols:
            logger.info(f"Fetching fallback historical data for {len(missing_symbols)} symbols")

            for symbol in missing_symbols:
                try:
                    # Use yfinance for fallback historical data
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)

                    if not hist.empty:
                        # Prepare DataFrame with required columns
                        df = pd.DataFrame({
                            'date': hist.index,
                            'close': hist['Close'],
                        })

                        # Calculate returns
                        df['returns'] = df['close'].pct_change()
                        df = df.dropna()

                        # Keep only the requested number of days
                        df = df.tail(days)

                        if len(df) >= days // 2:  # At least half the requested data
                            historical_data[symbol] = df
                            logger.debug(f"Fetched {len(df)} days of fallback data for {symbol}")
                        else:
                            logger.warning(f"Insufficient historical data for {symbol}: {len(df)} days")

                except Exception as e:
                    logger.warning(f"Failed to fetch fallback historical data for {symbol}: {e}")
        
        logger.info(f"Successfully fetched historical data for {len(historical_data)} symbols")
        return historical_data
    
    def get_data_quality_report(self) -> Dict[str, any]:
        """
        Generate a report on current data quality.
        
        Returns:
            Dict with data quality metrics
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "data_age_minutes": None,
            "adr_coverage": {},
            "hk_coverage": {},
            "indicator_coverage": {},
            "currency_data": {},
            "hstech_data": {},
            "overall_quality": "unknown"
        }
        
        if self.last_update_time:
            age = (datetime.now(timezone.utc) - self.last_update_time).total_seconds() / 60
            report["data_age_minutes"] = age
        
        # ADR coverage
        expected_adr_symbols = self.adr_mapper.get_all_adr_symbols()
        report["adr_coverage"] = {
            "expected": len(expected_adr_symbols),
            "available": len(self.current_adr_prices),
            "missing": [s for s in expected_adr_symbols if s not in self.current_adr_prices],
            "coverage_percent": len(self.current_adr_prices) / len(expected_adr_symbols) * 100 if expected_adr_symbols else 0
        }
        
        # HK coverage
        expected_hk_symbols = [stock.symbol for stock in HSTECH_COMPONENTS]
        report["hk_coverage"] = {
            "expected": len(expected_hk_symbols),
            "available": len(self.current_hk_prices),
            "missing": [s for s in expected_hk_symbols if s not in self.current_hk_prices],
            "coverage_percent": len(self.current_hk_prices) / len(expected_hk_symbols) * 100 if expected_hk_symbols else 0
        }
        
        # Indicator coverage
        report["indicator_coverage"] = {
            "expected": len(MARKET_INDICATORS),
            "available": len(self.current_indicator_prices),
            "missing": [s for s in MARKET_INDICATORS if s not in self.current_indicator_prices],
            "coverage_percent": len(self.current_indicator_prices) / len(MARKET_INDICATORS) * 100 if MARKET_INDICATORS else 0
        }
        
        # Currency data
        report["currency_data"] = {
            "available": self.current_exchange_rate is not None,
            "rate": float(self.current_exchange_rate.rate) if self.current_exchange_rate else None,
            "source": self.current_exchange_rate.source if self.current_exchange_rate else None
        }
        
        # HSTECH data
        report["hstech_data"] = {
            "available": self.last_hstech_close is not None,
            "value": float(self.last_hstech_close.value) if self.last_hstech_close else None
        }
        
        # Overall quality assessment
        critical_coverage = min(
            report["adr_coverage"]["coverage_percent"],
            report["hk_coverage"]["coverage_percent"]
        )
        
        if critical_coverage >= 90 and report["currency_data"]["available"] and report["hstech_data"]["available"]:
            report["overall_quality"] = "excellent"
        elif critical_coverage >= 75:
            report["overall_quality"] = "good"
        elif critical_coverage >= 50:
            report["overall_quality"] = "fair"
        else:
            report["overall_quality"] = "poor"
        
        return report
    
    def _needs_data_refresh(self) -> bool:
        """Check if data needs to be refreshed based on age and update frequency."""
        if not self.last_update_time:
            return True
        
        age_seconds = (datetime.now(timezone.utc) - self.last_update_time).total_seconds()
        return age_seconds >= self.update_frequency
    
    async def start_auto_refresh(self):
        """Start automatic data refresh in the background."""
        logger.info(f"Starting auto-refresh every {self.update_frequency} seconds")
        
        while True:
            try:
                await asyncio.sleep(self.update_frequency)
                await self.fetch_all_current_data()
            except Exception as e:
                logger.error(f"Error in auto-refresh: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def clear_all_data(self):
        """Clear all cached data."""
        self.current_adr_prices.clear()
        self.current_hk_prices.clear()
        self.current_indicator_prices.clear()
        self.previous_indicator_prices.clear()
        self.current_exchange_rate = None
        self.last_hstech_close = None
        self.last_update_time = None
        
        self.data_fetcher.clear_cache()
        logger.info("Cleared all data")


def create_data_manager(config: Config) -> DataManager:
    """Create and return a DataManager instance."""
    return DataManager(config)
