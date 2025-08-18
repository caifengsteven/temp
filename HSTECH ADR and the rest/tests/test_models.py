"""
Unit tests for HSTECH estimation system models.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.models import (
    Stock, ADRMapping, PriceData, CurrencyRate, IndexData, 
    EstimationResult, Config, EstimationWeights
)


class TestStock:
    """Test Stock model."""
    
    def test_valid_stock_creation(self):
        """Test creating a valid stock."""
        stock = Stock(
            symbol="0700.HK",
            name="Tencent Holdings Ltd",
            weight=0.085,
            sector="Internet & Direct Marketing Retail",
            market_cap=3500000000000
        )
        
        assert stock.symbol == "0700.HK"
        assert stock.name == "Tencent Holdings Ltd"
        assert stock.weight == 0.085
        assert stock.sector == "Internet & Direct Marketing Retail"
        assert stock.market_cap == 3500000000000
    
    def test_invalid_symbol(self):
        """Test that invalid symbols raise validation error."""
        with pytest.raises(ValueError, match="Hong Kong symbols must end with .HK"):
            Stock(
                symbol="TCEHY",  # US symbol, not HK
                name="Tencent Holdings Ltd",
                weight=0.085,
                sector="Internet & Direct Marketing Retail"
            )
    
    def test_invalid_weight(self):
        """Test that invalid weights raise validation error."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            Stock(
                symbol="0700.HK",
                name="Tencent Holdings Ltd",
                weight=1.5,  # Invalid weight > 1
                sector="Internet & Direct Marketing Retail"
            )


class TestADRMapping:
    """Test ADRMapping model."""
    
    def test_valid_adr_mapping(self):
        """Test creating a valid ADR mapping."""
        mapping = ADRMapping(
            hk_symbol="0700.HK",
            us_symbol="TCEHY",
            conversion_ratio=5.0,
            currency_base="HKD",
            currency_quote="USD"
        )
        
        assert mapping.hk_symbol == "0700.HK"
        assert mapping.us_symbol == "TCEHY"
        assert mapping.conversion_ratio == 5.0
        assert mapping.currency_base == "HKD"
        assert mapping.currency_quote == "USD"
    
    def test_invalid_hk_symbol(self):
        """Test that invalid HK symbols raise validation error."""
        with pytest.raises(ValueError, match="Hong Kong symbols must end with .HK"):
            ADRMapping(
                hk_symbol="TCEHY",  # US symbol, not HK
                us_symbol="TCEHY",
                conversion_ratio=5.0
            )


class TestPriceData:
    """Test PriceData model."""
    
    def test_valid_price_data(self):
        """Test creating valid price data."""
        price_data = PriceData(
            symbol="TCEHY",
            price=Decimal("45.50"),
            currency="USD",
            timestamp=datetime.now(timezone.utc),
            volume=1000000,
            source="yahoo_finance"
        )
        
        assert price_data.symbol == "TCEHY"
        assert price_data.price == Decimal("45.50")
        assert price_data.currency == "USD"
        assert price_data.volume == 1000000
        assert price_data.source == "yahoo_finance"
    
    def test_invalid_price(self):
        """Test that invalid prices raise validation error."""
        with pytest.raises(ValueError):
            PriceData(
                symbol="TCEHY",
                price=Decimal("-10.00"),  # Negative price
                currency="USD",
                timestamp=datetime.now(timezone.utc),
                source="yahoo_finance"
            )


class TestCurrencyRate:
    """Test CurrencyRate model."""
    
    def test_valid_currency_rate(self):
        """Test creating valid currency rate."""
        rate = CurrencyRate(
            base_currency="USD",
            quote_currency="HKD",
            rate=Decimal("7.8"),
            timestamp=datetime.now(timezone.utc),
            source="yahoo_finance"
        )
        
        assert rate.base_currency == "USD"
        assert rate.quote_currency == "HKD"
        assert rate.rate == Decimal("7.8")
        assert rate.source == "yahoo_finance"


class TestIndexData:
    """Test IndexData model."""
    
    def test_valid_index_data(self):
        """Test creating valid index data."""
        index_data = IndexData(
            value=Decimal("10000.50"),
            timestamp=datetime.now(timezone.utc),
            change=Decimal("50.25"),
            change_percent=Decimal("0.5")
        )
        
        assert index_data.value == Decimal("10000.50")
        assert index_data.change == Decimal("50.25")
        assert index_data.change_percent == Decimal("0.5")


class TestEstimationResult:
    """Test EstimationResult model."""
    
    def test_valid_estimation_result(self):
        """Test creating valid estimation result."""
        result = EstimationResult(
            estimated_value=Decimal("10050.75"),
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
            method_weights={"adr_based": 0.6, "covariance_based": 0.25, "market_indicators": 0.15},
            component_contributions={"0700.HK": 0.002, "9988.HK": 0.001}
        )
        
        assert result.estimated_value == Decimal("10050.75")
        assert result.confidence == 0.85
        assert len(result.method_weights) == 3
        assert len(result.component_contributions) == 2


class TestConfig:
    """Test Config model."""
    
    def test_default_config(self):
        """Test creating default configuration."""
        config = Config()
        
        assert config.estimation.lookback_days == 252
        assert config.estimation.min_correlation_threshold == 0.3
        assert config.currency.base == "HKD"
        assert config.currency.quote == "USD"
        assert config.logging.level == "INFO"
    
    def test_estimation_weights_validation(self):
        """Test that estimation weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Estimation weights must sum to 1.0"):
            EstimationWeights(
                adr_based=0.5,
                covariance_based=0.3,
                market_indicators=0.3  # Sum = 1.1, should fail
            )
    
    def test_valid_estimation_weights(self):
        """Test valid estimation weights."""
        weights = EstimationWeights(
            adr_based=0.6,
            covariance_based=0.25,
            market_indicators=0.15
        )
        
        assert weights.adr_based == 0.6
        assert weights.covariance_based == 0.25
        assert weights.market_indicators == 0.15


if __name__ == "__main__":
    pytest.main([__file__])
