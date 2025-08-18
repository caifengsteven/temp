"""
ADR Mapping Service for HSTECH Index Components

This module provides functionality to map Hong Kong stocks to their US ADR equivalents
and handle currency conversions and ratio adjustments.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import logging

from ..models import Stock, ADRMapping, PriceData, CurrencyRate
from ...data.hstech_components import ADR_MAPPINGS, get_adr_mapping

logger = logging.getLogger(__name__)


class ADRMapper:
    """Handles mapping between Hong Kong stocks and their US ADR equivalents."""
    
    def __init__(self):
        self.adr_mappings = {mapping.hk_symbol: mapping for mapping in ADR_MAPPINGS}
        
    def has_adr_mapping(self, hk_symbol: str) -> bool:
        """Check if a Hong Kong stock has a US ADR mapping."""
        return hk_symbol in self.adr_mappings
    
    def get_adr_symbol(self, hk_symbol: str) -> Optional[str]:
        """Get the US ADR symbol for a Hong Kong stock."""
        mapping = self.adr_mappings.get(hk_symbol)
        return mapping.us_symbol if mapping else None
    
    def get_conversion_ratio(self, hk_symbol: str) -> Optional[float]:
        """Get the conversion ratio (HK shares per ADR) for a stock."""
        mapping = self.adr_mappings.get(hk_symbol)
        return mapping.conversion_ratio if mapping else None
    
    def convert_adr_to_hk_price(
        self, 
        adr_price: Decimal, 
        hk_symbol: str, 
        exchange_rate: Decimal
    ) -> Optional[Decimal]:
        """
        Convert ADR price to equivalent Hong Kong stock price.
        
        Args:
            adr_price: Price of the ADR in USD
            hk_symbol: Hong Kong stock symbol
            exchange_rate: USD/HKD exchange rate
            
        Returns:
            Equivalent Hong Kong stock price in HKD
        """
        if not self.has_adr_mapping(hk_symbol):
            return None
            
        mapping = self.adr_mappings[hk_symbol]
        
        # Convert USD to HKD
        adr_price_hkd = adr_price * exchange_rate
        
        # Adjust for conversion ratio (ADR represents multiple HK shares)
        hk_price = adr_price_hkd / Decimal(str(mapping.conversion_ratio))
        
        return hk_price
    
    def convert_hk_to_adr_price(
        self,
        hk_price: Decimal,
        hk_symbol: str, 
        exchange_rate: Decimal
    ) -> Optional[Decimal]:
        """
        Convert Hong Kong stock price to equivalent ADR price.
        
        Args:
            hk_price: Price of HK stock in HKD
            hk_symbol: Hong Kong stock symbol
            exchange_rate: USD/HKD exchange rate
            
        Returns:
            Equivalent ADR price in USD
        """
        if not self.has_adr_mapping(hk_symbol):
            return None
            
        mapping = self.adr_mappings[hk_symbol]
        
        # Adjust for conversion ratio
        adjusted_hk_price = hk_price * Decimal(str(mapping.conversion_ratio))
        
        # Convert HKD to USD
        adr_price = adjusted_hk_price / exchange_rate
        
        return adr_price
    
    def calculate_adr_impact(
        self,
        adr_price_changes: Dict[str, Decimal],
        exchange_rate: Decimal,
        stock_weights: Dict[str, float]
    ) -> Dict[str, Decimal]:
        """
        Calculate the impact of ADR price changes on Hong Kong stock prices.
        
        Args:
            adr_price_changes: Dict of {adr_symbol: price_change_percent}
            exchange_rate: Current USD/HKD exchange rate
            stock_weights: Dict of {hk_symbol: weight_in_index}
            
        Returns:
            Dict of {hk_symbol: estimated_price_change_percent}
        """
        impacts = {}
        
        # Create reverse mapping from ADR to HK symbols
        adr_to_hk = {mapping.us_symbol: mapping.hk_symbol 
                     for mapping in self.adr_mappings.values()}
        
        for adr_symbol, price_change in adr_price_changes.items():
            if adr_symbol in adr_to_hk:
                hk_symbol = adr_to_hk[adr_symbol]
                
                # For price percentage changes, conversion ratio doesn't affect the percentage
                # Only currency effects matter, which we assume are minimal for percentage changes
                impacts[hk_symbol] = price_change
                
                logger.debug(f"ADR {adr_symbol} change {price_change:.2%} -> "
                           f"HK {hk_symbol} estimated change {price_change:.2%}")
        
        return impacts
    
    def get_all_adr_symbols(self) -> List[str]:
        """Get list of all US ADR symbols that map to HSTECH components."""
        return [mapping.us_symbol for mapping in self.adr_mappings.values()]
    
    def get_all_hk_symbols_with_adrs(self) -> List[str]:
        """Get list of all Hong Kong symbols that have ADR mappings."""
        return list(self.adr_mappings.keys())
    
    def validate_mapping(self, hk_symbol: str, adr_symbol: str) -> bool:
        """Validate that a Hong Kong symbol correctly maps to an ADR symbol."""
        mapping = self.adr_mappings.get(hk_symbol)
        return mapping is not None and mapping.us_symbol == adr_symbol
    
    def get_mapping_info(self, hk_symbol: str) -> Optional[Dict]:
        """Get detailed mapping information for a Hong Kong stock."""
        if hk_symbol not in self.adr_mappings:
            return None
            
        mapping = self.adr_mappings[hk_symbol]
        return {
            "hk_symbol": mapping.hk_symbol,
            "us_symbol": mapping.us_symbol,
            "conversion_ratio": mapping.conversion_ratio,
            "currency_base": mapping.currency_base,
            "currency_quote": mapping.currency_quote
        }
    
    def estimate_index_impact_from_adrs(
        self,
        adr_price_changes: Dict[str, Decimal],
        stock_weights: Dict[str, float],
        exchange_rate: Decimal
    ) -> Decimal:
        """
        Estimate the overall HSTECH index impact from ADR price movements.
        
        Args:
            adr_price_changes: Dict of {adr_symbol: price_change_percent}
            stock_weights: Dict of {hk_symbol: weight_in_index}
            exchange_rate: Current USD/HKD exchange rate
            
        Returns:
            Estimated index change percentage
        """
        total_impact = Decimal('0')
        
        stock_impacts = self.calculate_adr_impact(adr_price_changes, exchange_rate, stock_weights)
        
        for hk_symbol, price_change in stock_impacts.items():
            if hk_symbol in stock_weights:
                weight = Decimal(str(stock_weights[hk_symbol]))
                weighted_impact = price_change * weight
                total_impact += weighted_impact
                
                logger.debug(f"Stock {hk_symbol}: change {price_change:.2%}, "
                           f"weight {weight:.2%}, impact {weighted_impact:.4%}")
        
        logger.info(f"Total ADR-based index impact: {total_impact:.2%}")
        return total_impact


# Utility functions for easy access
def create_adr_mapper() -> ADRMapper:
    """Create and return an ADRMapper instance."""
    return ADRMapper()


def get_adr_symbols_for_hstech() -> List[str]:
    """Get all ADR symbols that correspond to HSTECH components."""
    mapper = create_adr_mapper()
    return mapper.get_all_adr_symbols()


def check_adr_availability(hk_symbol: str) -> bool:
    """Quick check if a Hong Kong stock has an ADR mapping."""
    mapper = create_adr_mapper()
    return mapper.has_adr_mapping(hk_symbol)
