"""
Unit tests for Bloomberg data fetcher.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from src.data.bloomberg_fetcher import BloombergFetcher
from src.models import Config, PriceData, CurrencyRate, IndexData


class TestBloombergFetcher:
    """Test Bloomberg data fetcher functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.config.api_keys.bloomberg_api_key = "test_api_key"
        self.config.api_keys.bloomberg_cloud_url = "https://test.bloomberg.com"
        
    @patch('src.data.bloomberg_fetcher.BLPAPI_AVAILABLE', False)
    def test_init_without_blpapi(self):
        """Test initialization without Bloomberg Terminal API."""
        fetcher = BloombergFetcher(self.config)
        
        assert fetcher.terminal_available == False
        assert fetcher.session is None
        assert fetcher.bloomberg_api_key == "test_api_key"
    
    def test_convert_to_bloomberg_symbol(self):
        """Test symbol conversion to Bloomberg format."""
        fetcher = BloombergFetcher(self.config)
        
        # Test US symbols
        us_symbol = fetcher._convert_to_bloomberg_symbol("TCEHY", "US")
        assert us_symbol == "TCEHY US Equity"
        
        # Test HK symbols
        hk_symbol = fetcher._convert_to_bloomberg_symbol("0700.HK", "HK")
        assert hk_symbol == "0700 HK Equity"
        
        hk_symbol_no_suffix = fetcher._convert_to_bloomberg_symbol("0700", "HK")
        assert hk_symbol_no_suffix == "0700 HK Equity"
    
    @pytest.mark.asyncio
    async def test_fetch_adr_prices_no_sources(self):
        """Test ADR price fetching with no available sources."""
        # Mock no Bloomberg sources available
        with patch('src.data.bloomberg_fetcher.BLPAPI_AVAILABLE', False):
            fetcher = BloombergFetcher(self.config)
            fetcher.bloomberg_api_key = None
            
            prices = await fetcher.fetch_adr_prices(["TCEHY", "BABA"])
            assert prices == {}
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_cloud_prices_success(self, mock_client):
        """Test successful cloud API price fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "securityData": [{
                "fieldData": {
                    "PX_LAST": 45.50,
                    "VOLUME": 1000000,
                    "CURRENCY": "USD"
                }
            }]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        fetcher = BloombergFetcher(self.config)
        prices = await fetcher._fetch_cloud_prices(["TCEHY"], "US")
        
        assert "TCEHY" in prices
        assert prices["TCEHY"].price == Decimal("45.50")
        assert prices["TCEHY"].currency == "USD"
        assert prices["TCEHY"].source == "bloomberg_cloud"
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_cloud_currency_success(self, mock_client):
        """Test successful cloud API currency fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "securityData": [{
                "fieldData": {
                    "PX_LAST": 7.8
                }
            }]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        fetcher = BloombergFetcher(self.config)
        rate, source = await fetcher._fetch_cloud_currency("USDHKD Curncy")
        
        assert rate == Decimal("7.8")
        assert source == "bloomberg_cloud"
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_cloud_index_success(self, mock_client):
        """Test successful cloud API index fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "securityData": [{
                "fieldData": {
                    "PX_LAST": 10250.75,
                    "CHG_NET_1D": 25.50,
                    "CHG_PCT_1D": 0.25
                }
            }]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        fetcher = BloombergFetcher(self.config)
        index_data = await fetcher._fetch_cloud_index("HSTECH Index")
        
        assert index_data is not None
        assert index_data.value == Decimal("10250.75")
        assert index_data.change == Decimal("25.50")
        assert index_data.change_percent == Decimal("0.25")
    
    @pytest.mark.asyncio
    async def test_fetch_currency_rate_with_cache(self):
        """Test currency rate fetching with caching."""
        fetcher = BloombergFetcher(self.config)
        
        # Mock a cached rate
        cache_key = "USD/HKD"
        cached_rate = CurrencyRate(
            base_currency="USD",
            quote_currency="HKD",
            rate=Decimal("7.8"),
            timestamp=datetime.now(timezone.utc),
            source="test_cache"
        )
        
        fetcher.currency_cache[cache_key] = cached_rate
        fetcher.cache_expiry[cache_key] = datetime.now(timezone.utc)
        
        # Should return cached rate
        rate = await fetcher.fetch_currency_rate("USD", "HKD")
        assert rate.rate == Decimal("7.8")
        assert rate.source == "test_cache"
    
    def test_cache_validation(self):
        """Test cache validation logic."""
        fetcher = BloombergFetcher(self.config)
        
        # Test invalid cache (not in cache)
        assert not fetcher._is_cache_valid("test_key", 300)
        
        # Test valid cache
        fetcher.cache_expiry["test_key"] = datetime.now(timezone.utc)
        assert fetcher._is_cache_valid("test_key", 300)
        
        # Test expired cache
        from datetime import timedelta
        fetcher.cache_expiry["test_key"] = datetime.now(timezone.utc) - timedelta(seconds=400)
        assert not fetcher._is_cache_valid("test_key", 300)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        fetcher = BloombergFetcher(self.config)
        
        # Add some cache data
        fetcher.price_cache["test"] = "data"
        fetcher.currency_cache["test"] = "data"
        fetcher.cache_expiry["test"] = datetime.now(timezone.utc)
        
        # Clear cache
        fetcher.clear_cache()
        
        assert len(fetcher.price_cache) == 0
        assert len(fetcher.currency_cache) == 0
        assert len(fetcher.cache_expiry) == 0
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_cloud_prices_error(self, mock_client):
        """Test cloud API error handling."""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 401
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        fetcher = BloombergFetcher(self.config)
        prices = await fetcher._fetch_cloud_prices(["TCEHY"], "US")
        
        # Should return empty dict on error
        assert prices == {}
    
    @pytest.mark.asyncio
    async def test_fetch_currency_rate_no_sources(self):
        """Test currency rate fetching with no available sources."""
        fetcher = BloombergFetcher(self.config)
        fetcher.terminal_available = False
        fetcher.bloomberg_api_key = None
        
        with pytest.raises(ValueError, match="Failed to fetch currency rate"):
            await fetcher.fetch_currency_rate("USD", "HKD")
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_historical_data_cloud(self, mock_client):
        """Test historical data fetching via cloud API."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "securityData": [{
                "fieldData": [
                    {"date": "20240801", "PX_LAST": 45.0},
                    {"date": "20240802", "PX_LAST": 46.0},
                    {"date": "20240803", "PX_LAST": 45.5}
                ]
            }]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        fetcher = BloombergFetcher(self.config)
        fetcher.terminal_available = False  # Force cloud API usage
        
        historical_data = await fetcher.fetch_historical_data(["TCEHY"], days=30, market="US")
        
        assert "TCEHY" in historical_data
        df = historical_data["TCEHY"]
        assert len(df) == 3
        assert "date" in df.columns
        assert "close" in df.columns
        assert "returns" in df.columns


if __name__ == "__main__":
    pytest.main([__file__])
