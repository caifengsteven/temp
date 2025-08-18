"""
Unit tests for ADR mapping functionality.
"""

import pytest
from decimal import Decimal

from src.data.adr_mapper import ADRMapper


class TestADRMapper:
    """Test ADRMapper functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mapper = ADRMapper()
    
    def test_has_adr_mapping(self):
        """Test checking for ADR mappings."""
        # Test known mapping
        assert self.mapper.has_adr_mapping("0700.HK") == True
        
        # Test unknown mapping
        assert self.mapper.has_adr_mapping("9999.XX") == False
    
    def test_get_adr_symbol(self):
        """Test getting ADR symbol for HK stock."""
        # Test known mapping
        adr_symbol = self.mapper.get_adr_symbol("0700.HK")
        assert adr_symbol == "TCEHY"
        
        # Test unknown mapping
        adr_symbol = self.mapper.get_adr_symbol("9999.XX")
        assert adr_symbol is None
    
    def test_get_conversion_ratio(self):
        """Test getting conversion ratio."""
        # Test known mapping
        ratio = self.mapper.get_conversion_ratio("0700.HK")
        assert ratio == 5.0
        
        # Test unknown mapping
        ratio = self.mapper.get_conversion_ratio("9999.XX")
        assert ratio is None
    
    def test_convert_adr_to_hk_price(self):
        """Test converting ADR price to HK equivalent."""
        # Test conversion for Tencent (5:1 ratio)
        adr_price = Decimal("45.00")  # USD
        exchange_rate = Decimal("7.8")  # USD/HKD
        
        hk_price = self.mapper.convert_adr_to_hk_price(
            adr_price, "0700.HK", exchange_rate
        )
        
        # Expected: 45 * 7.8 / 5 = 70.2 HKD
        expected = Decimal("70.2")
        assert hk_price == expected
    
    def test_convert_hk_to_adr_price(self):
        """Test converting HK price to ADR equivalent."""
        # Test conversion for Tencent (5:1 ratio)
        hk_price = Decimal("351.00")  # HKD
        exchange_rate = Decimal("7.8")  # USD/HKD
        
        adr_price = self.mapper.convert_hk_to_adr_price(
            hk_price, "0700.HK", exchange_rate
        )
        
        # Expected: 351 * 5 / 7.8 = 225 USD
        expected = Decimal("225")
        assert adr_price == expected
    
    def test_calculate_adr_impact(self):
        """Test calculating ADR impact on HK stocks."""
        adr_changes = {
            "TCEHY": Decimal("0.02"),  # 2% increase
            "BABA": Decimal("-0.01"),  # 1% decrease
        }
        exchange_rate = Decimal("7.8")
        stock_weights = {
            "0700.HK": 0.085,
            "9988.HK": 0.075
        }
        
        impacts = self.mapper.calculate_adr_impact(
            adr_changes, exchange_rate, stock_weights
        )
        
        # Should have impacts for both stocks
        assert "0700.HK" in impacts
        assert "9988.HK" in impacts
        
        # Check that percentage changes are preserved
        assert impacts["0700.HK"] == Decimal("0.02")
        assert impacts["9988.HK"] == Decimal("-0.01")
    
    def test_get_all_adr_symbols(self):
        """Test getting all ADR symbols."""
        adr_symbols = self.mapper.get_all_adr_symbols()
        
        assert isinstance(adr_symbols, list)
        assert len(adr_symbols) > 0
        assert "TCEHY" in adr_symbols
        assert "BABA" in adr_symbols
    
    def test_get_all_hk_symbols_with_adrs(self):
        """Test getting all HK symbols with ADR mappings."""
        hk_symbols = self.mapper.get_all_hk_symbols_with_adrs()
        
        assert isinstance(hk_symbols, list)
        assert len(hk_symbols) > 0
        assert "0700.HK" in hk_symbols
        assert "9988.HK" in hk_symbols
    
    def test_validate_mapping(self):
        """Test validating ADR mappings."""
        # Test valid mapping
        assert self.mapper.validate_mapping("0700.HK", "TCEHY") == True
        
        # Test invalid mapping
        assert self.mapper.validate_mapping("0700.HK", "BABA") == False
        
        # Test non-existent HK symbol
        assert self.mapper.validate_mapping("9999.XX", "TCEHY") == False
    
    def test_get_mapping_info(self):
        """Test getting detailed mapping information."""
        # Test valid mapping
        info = self.mapper.get_mapping_info("0700.HK")
        
        assert info is not None
        assert info["hk_symbol"] == "0700.HK"
        assert info["us_symbol"] == "TCEHY"
        assert info["conversion_ratio"] == 5.0
        assert info["currency_base"] == "HKD"
        assert info["currency_quote"] == "USD"
        
        # Test invalid mapping
        info = self.mapper.get_mapping_info("9999.XX")
        assert info is None
    
    def test_estimate_index_impact_from_adrs(self):
        """Test estimating overall index impact from ADR movements."""
        adr_changes = {
            "TCEHY": Decimal("0.02"),  # 2% increase
            "BABA": Decimal("-0.01"),  # 1% decrease
        }
        stock_weights = {
            "0700.HK": 0.085,  # 8.5% weight
            "9988.HK": 0.075   # 7.5% weight
        }
        exchange_rate = Decimal("7.8")
        
        impact = self.mapper.estimate_index_impact_from_adrs(
            adr_changes, stock_weights, exchange_rate
        )
        
        # Expected: (0.02 * 0.085) + (-0.01 * 0.075) = 0.0017 - 0.00075 = 0.00095
        expected = Decimal("0.00095")
        assert abs(impact - expected) < Decimal("0.0001")  # Allow small floating point errors


if __name__ == "__main__":
    pytest.main([__file__])
