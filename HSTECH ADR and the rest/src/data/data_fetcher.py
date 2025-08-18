"""
Real-time Data Fetcher for HSTECH Estimation System

This module provides functionality to fetch real-time market data from Bloomberg
with fallback to other sources for reliability.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import asyncio

from ..models import PriceData, CurrencyRate, IndexData, Config
from .bloomberg_fetcher import BloombergFetcher

# Fallback imports
try:
    import yfinance as yf
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.foreignexchange import ForeignExchange
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    logger.warning("Fallback data sources not available")

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches real-time market data with Bloomberg as primary source.

    Supports:
    - Bloomberg Terminal API (primary)
    - Bloomberg Cloud API (primary)
    - Yahoo Finance (fallback)
    - Alpha Vantage (fallback)
    """

    def __init__(self, config: Config):
        self.config = config

        # Initialize Bloomberg fetcher as primary
        self.bloomberg_fetcher = BloombergFetcher(config)

        # Initialize fallback sources if available
        self.alpha_vantage_key = config.api_keys.alpha_vantage
        self.ts = None
        self.fx = None

        if FALLBACK_AVAILABLE and self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.fx = ForeignExchange(key=self.alpha_vantage_key, output_format='pandas')

        # Cache for recent data to avoid excessive API calls
        self.price_cache = {}
        self.currency_cache = {}
        self.cache_expiry = {}

        logger.info(f"Initialized data fetcher - Bloomberg primary, fallbacks: {FALLBACK_AVAILABLE}")
    
    async def fetch_adr_prices(self, adr_symbols: List[str]) -> Dict[str, PriceData]:
        """
        Fetch current prices for ADR symbols.

        Args:
            adr_symbols: List of ADR ticker symbols

        Returns:
            Dict of {symbol: PriceData}
        """
        logger.info(f"Fetching ADR prices for {len(adr_symbols)} symbols")

        prices = {}

        # Try Bloomberg first (primary source)
        try:
            bloomberg_prices = await self.bloomberg_fetcher.fetch_adr_prices(adr_symbols)
            prices.update(bloomberg_prices)
        except Exception as e:
            logger.warning(f"Bloomberg fetch failed: {e}")

        # Fill missing data with fallback sources
        missing_symbols = [s for s in adr_symbols if s not in prices]
        if missing_symbols and FALLBACK_AVAILABLE:
            try:
                fallback_prices = await self._fetch_fallback_prices(missing_symbols)
                prices.update(fallback_prices)
            except Exception as e:
                logger.warning(f"Fallback fetch failed: {e}")

        logger.info(f"Successfully fetched prices for {len(prices)}/{len(adr_symbols)} ADR symbols")
        return prices
    
    async def fetch_market_indicator_prices(self, indicator_symbols: List[str]) -> Dict[str, PriceData]:
        """
        Fetch current prices for market indicators (PDD, KWEB, etc.).

        Args:
            indicator_symbols: List of market indicator symbols

        Returns:
            Dict of {symbol: PriceData}
        """
        logger.info(f"Fetching market indicator prices for {len(indicator_symbols)} symbols")

        # Use same logic as ADR prices
        return await self.fetch_adr_prices(indicator_symbols)
    
    async def fetch_currency_rate(self, base: str = "USD", quote: str = "HKD") -> CurrencyRate:
        """
        Fetch current currency exchange rate.

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

        logger.info(f"Fetching currency rate: {base}/{quote}")

        # Try Bloomberg first
        try:
            currency_rate = await self.bloomberg_fetcher.fetch_currency_rate(base, quote)

            # Cache the result
            self.currency_cache[cache_key] = currency_rate
            self.cache_expiry[cache_key] = datetime.now(timezone.utc)

            return currency_rate

        except Exception as e:
            logger.warning(f"Bloomberg currency fetch failed: {e}")

        # Try fallback sources
        if FALLBACK_AVAILABLE:
            try:
                rate, source = await self._fetch_fallback_currency(base, quote)

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

            except Exception as e:
                logger.warning(f"Fallback currency fetch failed: {e}")

        raise ValueError(f"Failed to fetch currency rate for {base}/{quote} from all sources")
    
    async def fetch_last_hk_prices(self, hk_symbols: List[str]) -> Dict[str, PriceData]:
        """
        Fetch last known Hong Kong stock prices.

        Args:
            hk_symbols: List of Hong Kong stock symbols (e.g., '0700.HK')

        Returns:
            Dict of {symbol: PriceData}
        """
        logger.info(f"Fetching last HK prices for {len(hk_symbols)} symbols")

        prices = {}

        # Try Bloomberg first
        try:
            bloomberg_prices = await self.bloomberg_fetcher.fetch_hk_prices(hk_symbols)
            prices.update(bloomberg_prices)
        except Exception as e:
            logger.warning(f"Bloomberg HK fetch failed: {e}")

        # Fill missing data with fallback sources
        missing_symbols = [s for s in hk_symbols if s not in prices]
        if missing_symbols and FALLBACK_AVAILABLE:
            try:
                fallback_prices = await self._fetch_fallback_prices(missing_symbols)
                prices.update(fallback_prices)
            except Exception as e:
                logger.warning(f"Fallback HK fetch failed: {e}")

        logger.info(f"Successfully fetched HK prices for {len(prices)}/{len(hk_symbols)} symbols")
        return prices
    
    async def fetch_hstech_index(self) -> Optional[IndexData]:
        """
        Fetch last known HSTECH index value.

        Returns:
            IndexData object or None if fetch fails
        """
        logger.info("Fetching HSTECH index data")

        # Try Bloomberg first
        try:
            index_data = await self.bloomberg_fetcher.fetch_hstech_index()
            if index_data:
                logger.info(f"Fetched HSTECH index from Bloomberg: {index_data.value:.2f}")
                return index_data
        except Exception as e:
            logger.warning(f"Bloomberg HSTECH fetch failed: {e}")

        # Try fallback sources
        if FALLBACK_AVAILABLE:
            try:
                index_data = await self._fetch_fallback_hstech_index()
                if index_data:
                    logger.info(f"Fetched HSTECH index from fallback: {index_data.value:.2f}")
                    return index_data
            except Exception as e:
                logger.warning(f"Fallback HSTECH fetch failed: {e}")

        logger.error("Failed to fetch HSTECH index from all sources")
        return None
    
    async def _fetch_fallback_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Fetch prices using fallback sources (Yahoo Finance, Alpha Vantage)."""
        if not FALLBACK_AVAILABLE:
            return {}

        prices = {}

        # Try Yahoo Finance first
        try:
            yf_prices = await self._fetch_yahoo_prices(symbols)
            prices.update(yf_prices)
        except Exception as e:
            logger.warning(f"Yahoo Finance fallback failed: {e}")

        # Fill missing data with Alpha Vantage
        missing_symbols = [s for s in symbols if s not in prices]
        if missing_symbols and self.ts:
            try:
                av_prices = await self._fetch_alpha_vantage_prices(missing_symbols)
                prices.update(av_prices)
            except Exception as e:
                logger.warning(f"Alpha Vantage fallback failed: {e}")

        return prices

    async def _fetch_yahoo_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Fetch prices using Yahoo Finance."""
        prices = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")

                if not hist.empty:
                    latest = hist.iloc[-1]

                    price_data = PriceData(
                        symbol=symbol,
                        price=Decimal(str(latest['Close'])),
                        currency="USD" if not symbol.endswith('.HK') else "HKD",
                        timestamp=datetime.now(timezone.utc),
                        volume=int(latest['Volume']) if 'Volume' in latest else None,
                        source="yahoo_finance"
                    )

                    prices[symbol] = price_data

            except Exception as e:
                logger.warning(f"Failed to fetch Yahoo price for {symbol}: {e}")

        return prices
    
    async def _fetch_alpha_vantage_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Fetch prices using Alpha Vantage."""
        prices = {}

        if not self.ts:
            return prices

        for symbol in symbols:
            try:
                data, meta_data = self.ts.get_quote_endpoint(symbol=symbol)

                if not data.empty:
                    latest = data.iloc[0]

                    price_data = PriceData(
                        symbol=symbol,
                        price=Decimal(str(latest['05. price'])),
                        currency="USD",  # Alpha Vantage typically returns USD
                        timestamp=datetime.now(timezone.utc),
                        volume=int(latest['06. volume']) if '06. volume' in latest else None,
                        source="alpha_vantage"
                    )

                    prices[symbol] = price_data

            except Exception as e:
                logger.warning(f"Failed to fetch Alpha Vantage price for {symbol}: {e}")

        return prices
    
    async def _fetch_fallback_currency(self, base: str, quote: str) -> Tuple[Decimal, str]:
        """Fetch currency rate using fallback sources."""
        # Try Yahoo Finance first
        try:
            return await self._fetch_yahoo_currency(base, quote)
        except Exception as e:
            logger.warning(f"Yahoo Finance currency fallback failed: {e}")

        # Try Alpha Vantage as backup
        if self.fx:
            try:
                return await self._fetch_alpha_vantage_currency(base, quote)
            except Exception as e:
                logger.warning(f"Alpha Vantage currency fallback failed: {e}")

        raise ValueError(f"Failed to fetch currency rate for {base}/{quote} from fallback sources")

    async def _fetch_yahoo_currency(self, base: str, quote: str) -> Tuple[Decimal, str]:
        """Fetch currency rate using Yahoo Finance."""
        symbol = f"{base}{quote}=X"

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")

        if hist.empty:
            raise ValueError(f"No currency data for {symbol}")

        latest = hist.iloc[-1]
        rate = Decimal(str(latest['Close']))

        return rate, "yahoo_finance"

    async def _fetch_alpha_vantage_currency(self, base: str, quote: str) -> Tuple[Decimal, str]:
        """Fetch currency rate using Alpha Vantage."""
        if not self.fx:
            raise ValueError("Alpha Vantage not configured")

        data, meta_data = self.fx.get_currency_exchange_rate(from_currency=base, to_currency=quote)

        if data.empty:
            raise ValueError(f"No currency data for {base}/{quote}")

        rate = Decimal(str(data.iloc[0]['5. Exchange Rate']))

        return rate, "alpha_vantage"

    async def _fetch_fallback_hstech_index(self) -> Optional[IndexData]:
        """Fetch HSTECH index using fallback sources."""
        # HSTECH index symbol on Yahoo Finance
        hstech_symbol = "^HSTECH"

        try:
            ticker = yf.Ticker(hstech_symbol)
            hist = ticker.history(period="1d")

            if hist.empty:
                logger.warning("No HSTECH index data available from Yahoo Finance")
                return None

            latest = hist.iloc[-1]

            index_data = IndexData(
                value=Decimal(str(latest['Close'])),
                timestamp=datetime.now(timezone.utc),
                change=Decimal(str(latest['Close'] - latest['Open'])),
                change_percent=Decimal(str((latest['Close'] - latest['Open']) / latest['Open'] * 100)),
                volume=int(latest['Volume']) if 'Volume' in latest else None
            )

            return index_data

        except Exception as e:
            logger.error(f"Failed to fetch HSTECH index from Yahoo Finance: {e}")
            return None
    
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

        # Clear Bloomberg cache too
        self.bloomberg_fetcher.clear_cache()

        logger.info("Cleared all data caches")


def create_data_fetcher(config: Config) -> DataFetcher:
    """Create and return a DataFetcher instance."""
    return DataFetcher(config)
