"""
Bloomberg Data Fetcher for HSTECH Estimation System

This module provides functionality to fetch real-time market data from Bloomberg
using both Bloomberg Terminal API (BLPAPI) and Bloomberg Cloud API.
"""

from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import json
import httpx
import pandas as pd

try:
    import blpapi
    BLPAPI_AVAILABLE = True
except ImportError:
    BLPAPI_AVAILABLE = False
    logging.warning("Bloomberg BLPAPI not available. Install with: pip install blpapi")

from ..models import PriceData, CurrencyRate, IndexData, Config

logger = logging.getLogger(__name__)


class BloombergFetcher:
    """
    Fetches real-time market data from Bloomberg sources.
    
    Supports:
    - Bloomberg Terminal API (BLPAPI) - for terminal users
    - Bloomberg Cloud API - for cloud/web access
    - Automatic fallback between sources
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Bloomberg API configuration
        self.bloomberg_api_key = getattr(config.api_keys, 'bloomberg_api_key', None)
        self.bloomberg_cloud_url = getattr(config.api_keys, 'bloomberg_cloud_url', None)
        
        # Bloomberg Terminal session
        self.session = None
        self.ref_data_service = None
        self.terminal_available = False
        
        # Cache for recent data
        self.price_cache = {}
        self.currency_cache = {}
        self.cache_expiry = {}
        
        # Initialize Bloomberg Terminal if available
        if BLPAPI_AVAILABLE:
            self._init_bloomberg_terminal()
        
        logger.info(f"Initialized Bloomberg fetcher - Terminal: {self.terminal_available}, "
                   f"Cloud API: {bool(self.bloomberg_api_key)}")
    
    def _init_bloomberg_terminal(self):
        """Initialize Bloomberg Terminal connection."""
        try:
            # Create session options
            session_options = blpapi.SessionOptions()
            session_options.setServerHost("localhost")
            session_options.setServerPort(8194)
            
            # Create session
            self.session = blpapi.Session(session_options)
            
            # Start session
            if self.session.start():
                # Open reference data service
                if self.session.openService("//blp/refdata"):
                    self.ref_data_service = self.session.getService("//blp/refdata")
                    self.terminal_available = True
                    logger.info("Bloomberg Terminal connection established")
                else:
                    logger.warning("Failed to open Bloomberg reference data service")
            else:
                logger.warning("Failed to start Bloomberg Terminal session")
                
        except Exception as e:
            logger.warning(f"Bloomberg Terminal initialization failed: {e}")
            self.terminal_available = False
    
    async def fetch_adr_prices(self, adr_symbols: List[str]) -> Dict[str, PriceData]:
        """
        Fetch current prices for ADR symbols from Bloomberg.
        
        Args:
            adr_symbols: List of ADR ticker symbols
            
        Returns:
            Dict of {symbol: PriceData}
        """
        logger.info(f"Fetching ADR prices for {len(adr_symbols)} symbols from Bloomberg")
        
        prices = {}
        
        # Convert symbols to Bloomberg format
        bloomberg_symbols = [self._convert_to_bloomberg_symbol(symbol, "US") for symbol in adr_symbols]
        
        # Try Bloomberg Terminal first
        if self.terminal_available:
            try:
                terminal_prices = await self._fetch_terminal_prices(bloomberg_symbols, adr_symbols)
                prices.update(terminal_prices)
            except Exception as e:
                logger.warning(f"Bloomberg Terminal fetch failed: {e}")
        
        # Fill missing data with Bloomberg Cloud API
        missing_symbols = [s for s in adr_symbols if s not in prices]
        if missing_symbols and self.bloomberg_api_key:
            try:
                cloud_prices = await self._fetch_cloud_prices(missing_symbols, "US")
                prices.update(cloud_prices)
            except Exception as e:
                logger.warning(f"Bloomberg Cloud API fetch failed: {e}")
        
        logger.info(f"Successfully fetched prices for {len(prices)}/{len(adr_symbols)} ADR symbols")
        return prices
    
    async def fetch_hk_prices(self, hk_symbols: List[str]) -> Dict[str, PriceData]:
        """
        Fetch Hong Kong stock prices from Bloomberg.
        
        Args:
            hk_symbols: List of Hong Kong stock symbols (e.g., '0700.HK')
            
        Returns:
            Dict of {symbol: PriceData}
        """
        logger.info(f"Fetching HK prices for {len(hk_symbols)} symbols from Bloomberg")
        
        prices = {}
        
        # Convert symbols to Bloomberg format
        bloomberg_symbols = [self._convert_to_bloomberg_symbol(symbol, "HK") for symbol in hk_symbols]
        
        # Try Bloomberg Terminal first
        if self.terminal_available:
            try:
                terminal_prices = await self._fetch_terminal_prices(bloomberg_symbols, hk_symbols)
                prices.update(terminal_prices)
            except Exception as e:
                logger.warning(f"Bloomberg Terminal HK fetch failed: {e}")
        
        # Fill missing data with Bloomberg Cloud API
        missing_symbols = [s for s in hk_symbols if s not in prices]
        if missing_symbols and self.bloomberg_api_key:
            try:
                cloud_prices = await self._fetch_cloud_prices(missing_symbols, "HK")
                prices.update(cloud_prices)
            except Exception as e:
                logger.warning(f"Bloomberg Cloud API HK fetch failed: {e}")
        
        logger.info(f"Successfully fetched prices for {len(prices)}/{len(hk_symbols)} HK symbols")
        return prices
    
    async def fetch_currency_rate(self, base: str = "USD", quote: str = "HKD") -> CurrencyRate:
        """
        Fetch currency exchange rate from Bloomberg.
        
        Args:
            base: Base currency (default: USD)
            quote: Quote currency (default: HKD)
            
        Returns:
            CurrencyRate object
        """
        cache_key = f"{base}/{quote}"
        
        # Check cache first
        if self._is_cache_valid(cache_key, self.config.currency.conversion_cache_minutes * 60):
            logger.debug(f"Using cached currency rate for {cache_key}")
            return self.currency_cache[cache_key]
        
        logger.info(f"Fetching currency rate: {base}/{quote} from Bloomberg")
        
        # Bloomberg currency symbol format
        currency_symbol = f"{base}{quote} Curncy"
        
        rate = None
        source = None
        
        # Try Bloomberg Terminal first
        if self.terminal_available:
            try:
                rate, source = await self._fetch_terminal_currency(currency_symbol)
            except Exception as e:
                logger.warning(f"Bloomberg Terminal currency fetch failed: {e}")
        
        # Try Bloomberg Cloud API as backup
        if rate is None and self.bloomberg_api_key:
            try:
                rate, source = await self._fetch_cloud_currency(currency_symbol)
            except Exception as e:
                logger.warning(f"Bloomberg Cloud API currency fetch failed: {e}")
        
        if rate is None:
            raise ValueError(f"Failed to fetch currency rate for {base}/{quote} from Bloomberg")
        
        # Create CurrencyRate object
        currency_rate = CurrencyRate(
            base_currency=base,
            quote_currency=quote,
            rate=rate,
            timestamp=datetime.now(timezone.utc),
            source=source
        )
        
        # Cache the result
        self.currency_cache[cache_key] = currency_rate
        self.cache_expiry[cache_key] = datetime.now(timezone.utc)
        
        logger.info(f"Fetched currency rate: {base}/{quote} = {rate:.4f} from {source}")
        return currency_rate
    
    async def fetch_hstech_index(self) -> Optional[IndexData]:
        """
        Fetch HSTECH index data from Bloomberg.
        
        Returns:
            IndexData object or None if fetch fails
        """
        logger.info("Fetching HSTECH index data from Bloomberg")
        
        # Bloomberg HSTECH index symbol
        hstech_symbol = "HSTECH Index"
        
        try:
            if self.terminal_available:
                # Fetch from Bloomberg Terminal
                index_data = await self._fetch_terminal_index(hstech_symbol)
                if index_data:
                    return index_data
            
            if self.bloomberg_api_key:
                # Fetch from Bloomberg Cloud API
                index_data = await self._fetch_cloud_index(hstech_symbol)
                if index_data:
                    return index_data
            
            logger.warning("Failed to fetch HSTECH index from all Bloomberg sources")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch HSTECH index: {e}")
            return None
    
    def _convert_to_bloomberg_symbol(self, symbol: str, market: str) -> str:
        """Convert symbol to Bloomberg format."""
        if market == "US":
            # US symbols are typically just the ticker
            return f"{symbol} US Equity"
        elif market == "HK":
            # Hong Kong symbols: remove .HK and add HK Equity
            if symbol.endswith('.HK'):
                ticker = symbol[:-3]
                return f"{ticker} HK Equity"
            else:
                return f"{symbol} HK Equity"
        else:
            return symbol
    
    async def _fetch_terminal_prices(self, bloomberg_symbols: List[str], original_symbols: List[str]) -> Dict[str, PriceData]:
        """Fetch prices using Bloomberg Terminal API."""
        if not self.terminal_available:
            return {}
        
        prices = {}
        
        try:
            # Create request
            request = self.ref_data_service.createRequest("ReferenceDataRequest")
            
            # Add securities
            for symbol in bloomberg_symbols:
                request.getElement("securities").appendValue(symbol)
            
            # Add fields
            request.getElement("fields").appendValue("PX_LAST")
            request.getElement("fields").appendValue("VOLUME")
            request.getElement("fields").appendValue("CURRENCY")
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            while True:
                event = self.session.nextEvent(500)  # 500ms timeout
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        
                        for i in range(security_data.numValues()):
                            security = security_data.getValue(i)
                            symbol = str(security.getElement("security"))
                            field_data = security.getElement("fieldData")
                            
                            # Map back to original symbol
                            original_symbol = original_symbols[bloomberg_symbols.index(symbol)] if symbol in bloomberg_symbols else symbol
                            
                            if field_data.hasElement("PX_LAST"):
                                price = Decimal(str(field_data.getElement("PX_LAST").getValue()))
                                volume = None
                                currency = "USD"
                                
                                if field_data.hasElement("VOLUME"):
                                    volume = int(field_data.getElement("VOLUME").getValue())
                                
                                if field_data.hasElement("CURRENCY"):
                                    currency = str(field_data.getElement("CURRENCY").getValue())
                                
                                price_data = PriceData(
                                    symbol=original_symbol,
                                    price=price,
                                    currency=currency,
                                    timestamp=datetime.now(timezone.utc),
                                    volume=volume,
                                    source="bloomberg_terminal"
                                )
                                
                                prices[original_symbol] = price_data
                    break
                
                elif event.eventType() == blpapi.Event.TIMEOUT:
                    logger.warning("Bloomberg Terminal request timeout")
                    break
        
        except Exception as e:
            logger.error(f"Bloomberg Terminal price fetch error: {e}")
        
        return prices

    async def _fetch_cloud_prices(self, symbols: List[str], market: str) -> Dict[str, PriceData]:
        """Fetch prices using Bloomberg Cloud API."""
        if not self.bloomberg_api_key:
            return {}

        prices = {}

        try:
            async with httpx.AsyncClient() as client:
                for symbol in symbols:
                    bloomberg_symbol = self._convert_to_bloomberg_symbol(symbol, market)

                    # Bloomberg Cloud API request
                    headers = {
                        "Authorization": f"Bearer {self.bloomberg_api_key}",
                        "Content-Type": "application/json"
                    }

                    payload = {
                        "securities": [bloomberg_symbol],
                        "fields": ["PX_LAST", "VOLUME", "CURRENCY"]
                    }

                    response = await client.post(
                        f"{self.bloomberg_cloud_url}/refdata",
                        headers=headers,
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()

                        if "securityData" in data:
                            for security_data in data["securityData"]:
                                if "fieldData" in security_data:
                                    field_data = security_data["fieldData"]

                                    if "PX_LAST" in field_data:
                                        price = Decimal(str(field_data["PX_LAST"]))
                                        volume = field_data.get("VOLUME")
                                        currency = field_data.get("CURRENCY", "USD")

                                        price_data = PriceData(
                                            symbol=symbol,
                                            price=price,
                                            currency=currency,
                                            timestamp=datetime.now(timezone.utc),
                                            volume=int(volume) if volume else None,
                                            source="bloomberg_cloud"
                                        )

                                        prices[symbol] = price_data
                    else:
                        logger.warning(f"Bloomberg Cloud API error for {symbol}: {response.status_code}")

        except Exception as e:
            logger.error(f"Bloomberg Cloud API price fetch error: {e}")

        return prices

    async def _fetch_terminal_currency(self, currency_symbol: str) -> Tuple[Decimal, str]:
        """Fetch currency rate using Bloomberg Terminal."""
        if not self.terminal_available:
            raise ValueError("Bloomberg Terminal not available")

        try:
            # Create request
            request = self.ref_data_service.createRequest("ReferenceDataRequest")
            request.getElement("securities").appendValue(currency_symbol)
            request.getElement("fields").appendValue("PX_LAST")

            # Send request
            self.session.sendRequest(request)

            # Process response
            while True:
                event = self.session.nextEvent(500)

                if event.eventType() == blpapi.Event.RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")

                        for i in range(security_data.numValues()):
                            security = security_data.getValue(i)
                            field_data = security.getElement("fieldData")

                            if field_data.hasElement("PX_LAST"):
                                rate = Decimal(str(field_data.getElement("PX_LAST").getValue()))
                                return rate, "bloomberg_terminal"
                    break

                elif event.eventType() == blpapi.Event.TIMEOUT:
                    break

        except Exception as e:
            logger.error(f"Bloomberg Terminal currency fetch error: {e}")

        raise ValueError(f"Failed to fetch currency rate for {currency_symbol}")

    async def _fetch_cloud_currency(self, currency_symbol: str) -> Tuple[Decimal, str]:
        """Fetch currency rate using Bloomberg Cloud API."""
        if not self.bloomberg_api_key:
            raise ValueError("Bloomberg Cloud API key not available")

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.bloomberg_api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "securities": [currency_symbol],
                    "fields": ["PX_LAST"]
                }

                response = await client.post(
                    f"{self.bloomberg_cloud_url}/refdata",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()

                    if "securityData" in data and data["securityData"]:
                        field_data = data["securityData"][0].get("fieldData", {})

                        if "PX_LAST" in field_data:
                            rate = Decimal(str(field_data["PX_LAST"]))
                            return rate, "bloomberg_cloud"

                logger.warning(f"Bloomberg Cloud API currency error: {response.status_code}")

        except Exception as e:
            logger.error(f"Bloomberg Cloud API currency fetch error: {e}")

        raise ValueError(f"Failed to fetch currency rate for {currency_symbol}")

    def _is_cache_valid(self, cache_key: str, max_age_seconds: int) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_expiry:
            return False

        age = (datetime.now(timezone.utc) - self.cache_expiry[cache_key]).total_seconds()
        return age < max_age_seconds

    def clear_cache(self):
        """Clear all cached data."""
        self.price_cache.clear()
        self.currency_cache.clear()
        self.cache_expiry.clear()
        logger.info("Cleared Bloomberg data cache")


def create_bloomberg_fetcher(config: Config) -> BloombergFetcher:
    """Create and return a BloombergFetcher instance."""
    return BloombergFetcher(config)
