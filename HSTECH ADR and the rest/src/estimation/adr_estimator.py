"""
Step 1: Static ADR-based HSTECH Index Estimator

This module implements the first step of the HSTECH estimation process:
calculating index updates based on ADR price movements of dual-listed stocks.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import logging
import asyncio

from ..models import PriceData, CurrencyRate, IndexData, EstimationResult
from ..data.adr_mapper import ADRMapper
from ...data.hstech_components import HSTECH_COMPONENTS, get_adr_mapped_components

logger = logging.getLogger(__name__)


class ADRBasedEstimator:
    """
    Estimates HSTECH index movements based on ADR price changes.
    
    This estimator:
    1. Fetches current ADR prices for dual-listed HSTECH components
    2. Converts ADR prices to HK equivalent using currency rates and conversion ratios
    3. Calculates weighted impact on HSTECH index
    4. Accounts for time zone differences and last known HK closing prices
    """
    
    def __init__(self, adr_mapper: ADRMapper):
        self.adr_mapper = adr_mapper
        self.adr_components = get_adr_mapped_components()
        self.component_weights = {stock.symbol: stock.weight for stock in HSTECH_COMPONENTS}
        
    def calculate_adr_based_update(
        self,
        current_adr_prices: Dict[str, PriceData],
        last_hk_prices: Dict[str, PriceData],
        current_exchange_rate: CurrencyRate,
        last_hstech_close: IndexData
    ) -> EstimationResult:
        """
        Calculate HSTECH index update based on ADR price movements.
        
        Args:
            current_adr_prices: Current ADR prices {adr_symbol: PriceData}
            last_hk_prices: Last known HK closing prices {hk_symbol: PriceData}
            current_exchange_rate: Current USD/HKD exchange rate
            last_hstech_close: Last HSTECH index closing value
            
        Returns:
            EstimationResult with updated index value and confidence
        """
        logger.info("Starting ADR-based HSTECH estimation")
        
        # Calculate price changes for each ADR-mapped component
        component_changes = {}
        total_weight_covered = Decimal('0')
        
        for stock in self.adr_components:
            hk_symbol = stock.symbol
            adr_symbol = self.adr_mapper.get_adr_symbol(hk_symbol)
            
            if not adr_symbol or adr_symbol not in current_adr_prices:
                logger.warning(f"Missing ADR price data for {adr_symbol} ({hk_symbol})")
                continue
                
            if hk_symbol not in last_hk_prices:
                logger.warning(f"Missing last HK price for {hk_symbol}")
                continue
            
            # Get current ADR price and last HK price
            adr_price = current_adr_prices[adr_symbol]
            last_hk_price = last_hk_prices[hk_symbol]
            
            # Convert ADR price to HK equivalent
            equivalent_hk_price = self.adr_mapper.convert_adr_to_hk_price(
                adr_price.price,
                hk_symbol,
                current_exchange_rate.rate
            )
            
            if equivalent_hk_price is None:
                logger.warning(f"Failed to convert ADR price for {hk_symbol}")
                continue
            
            # Calculate percentage change
            price_change = (equivalent_hk_price - last_hk_price.price) / last_hk_price.price
            component_changes[hk_symbol] = price_change
            total_weight_covered += Decimal(str(stock.weight))
            
            logger.debug(f"{hk_symbol}: ADR {adr_symbol} -> "
                        f"HK equivalent {equivalent_hk_price:.2f}, "
                        f"last HK {last_hk_price.price:.2f}, "
                        f"change {price_change:.2%}")
        
        # Calculate weighted index impact
        total_impact = self._calculate_weighted_impact(component_changes)
        
        # Estimate new index value
        estimated_value = last_hstech_close.value * (Decimal('1') + total_impact)
        
        # Calculate confidence based on coverage
        confidence = self._calculate_confidence(total_weight_covered, len(component_changes))
        
        # Create estimation result
        result = EstimationResult(
            estimated_value=estimated_value,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            method_weights={"adr_based": 1.0},
            component_contributions={
                hk_symbol: float(change * Decimal(str(self.component_weights[hk_symbol])))
                for hk_symbol, change in component_changes.items()
            }
        )
        
        logger.info(f"ADR-based estimation complete: "
                   f"Index {estimated_value:.2f} (change {total_impact:.2%}), "
                   f"confidence {confidence:.1%}, "
                   f"coverage {total_weight_covered:.1%}")
        
        return result
    
    def _calculate_weighted_impact(self, component_changes: Dict[str, Decimal]) -> Decimal:
        """Calculate the weighted impact on the index from component changes."""
        total_impact = Decimal('0')
        
        for hk_symbol, price_change in component_changes.items():
            if hk_symbol in self.component_weights:
                weight = Decimal(str(self.component_weights[hk_symbol]))
                weighted_impact = price_change * weight
                total_impact += weighted_impact
                
                logger.debug(f"Component {hk_symbol}: change {price_change:.2%}, "
                           f"weight {weight:.2%}, impact {weighted_impact:.4%}")
        
        return total_impact
    
    def _calculate_confidence(self, weight_covered: Decimal, num_components: int) -> float:
        """
        Calculate confidence score based on coverage and data quality.
        
        Confidence factors:
        - Weight coverage: How much of the index is covered by ADR data
        - Component count: Number of components with valid data
        - Data freshness: How recent the data is (could be enhanced)
        """
        # Base confidence from weight coverage (0-0.8)
        weight_confidence = min(float(weight_covered) * 2.0, 0.8)
        
        # Additional confidence from component count (0-0.2)
        max_adr_components = len(self.adr_components)
        component_confidence = (num_components / max_adr_components) * 0.2
        
        total_confidence = weight_confidence + component_confidence
        
        # Cap at 0.9 since this is only one estimation method
        return min(total_confidence, 0.9)
    
    def get_adr_coverage_stats(self) -> Dict[str, float]:
        """Get statistics about ADR coverage of the HSTECH index."""
        total_weight = sum(stock.weight for stock in self.adr_components)
        total_components = len(self.adr_components)
        total_hstech_components = len(HSTECH_COMPONENTS)
        
        return {
            "adr_weight_coverage": total_weight,
            "adr_component_count": total_components,
            "total_hstech_components": total_hstech_components,
            "component_coverage_ratio": total_components / total_hstech_components
        }
    
    def validate_input_data(
        self,
        adr_prices: Dict[str, PriceData],
        hk_prices: Dict[str, PriceData],
        exchange_rate: CurrencyRate
    ) -> List[str]:
        """
        Validate input data quality and return list of issues found.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check exchange rate
        if exchange_rate.rate <= 0:
            issues.append("Invalid exchange rate")
        
        # Check data freshness (within last 24 hours)
        now = datetime.now(timezone.utc)
        max_age_hours = 24
        
        for symbol, price_data in adr_prices.items():
            age_hours = (now - price_data.timestamp).total_seconds() / 3600
            if age_hours > max_age_hours:
                issues.append(f"Stale ADR data for {symbol}: {age_hours:.1f} hours old")
        
        for symbol, price_data in hk_prices.items():
            age_hours = (now - price_data.timestamp).total_seconds() / 3600
            if age_hours > max_age_hours:
                issues.append(f"Stale HK data for {symbol}: {age_hours:.1f} hours old")
        
        # Check for missing critical components (top 5 by weight)
        top_components = sorted(self.adr_components, key=lambda x: x.weight, reverse=True)[:5]
        for stock in top_components:
            adr_symbol = self.adr_mapper.get_adr_symbol(stock.symbol)
            if adr_symbol not in adr_prices:
                issues.append(f"Missing ADR data for major component {stock.symbol} ({adr_symbol})")
            if stock.symbol not in hk_prices:
                issues.append(f"Missing HK data for major component {stock.symbol}")
        
        return issues


def create_adr_estimator() -> ADRBasedEstimator:
    """Create and return an ADRBasedEstimator instance."""
    adr_mapper = ADRMapper()
    return ADRBasedEstimator(adr_mapper)
