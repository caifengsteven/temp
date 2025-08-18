"""
Step 3: Enhanced Estimation Using Additional US Market Indicators

This module integrates PDD, KWEB ETF, and other market indicators to enhance
the HSTECH index estimation accuracy.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import logging
import numpy as np
import pandas as pd

from ..models import PriceData, EstimationResult, IndexData
from ...data.hstech_components import MARKET_INDICATORS

logger = logging.getLogger(__name__)


class EnhancedMarketEstimator:
    """
    Enhances HSTECH estimation using additional US market indicators.
    
    This estimator:
    1. Incorporates PDD price movements as a Chinese tech indicator
    2. Uses KWEB ETF as a broader Chinese internet/tech sector proxy
    3. Includes other relevant ETFs and market indicators
    4. Applies machine learning models to combine all signals
    """
    
    def __init__(self, indicator_weights: Optional[Dict[str, float]] = None):
        self.indicator_symbols = MARKET_INDICATORS.copy()
        
        # Default weights for different indicators
        self.indicator_weights = indicator_weights or {
            "PDD": 0.25,    # High weight - direct Chinese e-commerce exposure
            "KWEB": 0.35,   # Highest weight - broad Chinese internet ETF
            "ASHR": 0.15,   # China A-shares ETF
            "FXI": 0.15,    # China large-cap ETF
            "MCHI": 0.10,   # MSCI China ETF
        }
        
        # Historical correlation data (to be built from historical analysis)
        self.indicator_correlations = {}
        self.indicator_betas = {}
        
        logger.info(f"Initialized enhanced estimator with indicators: {self.indicator_symbols}")
    
    def calculate_market_indicator_impact(
        self,
        current_indicator_prices: Dict[str, PriceData],
        previous_indicator_prices: Dict[str, PriceData],
        historical_correlations: Optional[Dict[str, float]] = None
    ) -> Decimal:
        """
        Calculate the impact of market indicators on HSTECH estimation.
        
        Args:
            current_indicator_prices: Current prices for market indicators
            previous_indicator_prices: Previous prices for comparison
            historical_correlations: Historical correlations with HSTECH (optional)
            
        Returns:
            Estimated percentage impact on HSTECH index
        """
        logger.info("Calculating market indicator impact")
        
        total_impact = Decimal('0')
        total_weight = Decimal('0')
        
        for symbol in self.indicator_symbols:
            if symbol not in current_indicator_prices or symbol not in previous_indicator_prices:
                logger.warning(f"Missing price data for indicator {symbol}")
                continue
            
            # Calculate price change
            current_price = current_indicator_prices[symbol].price
            previous_price = previous_indicator_prices[symbol].price
            price_change = (current_price - previous_price) / previous_price
            
            # Get weight and correlation
            weight = Decimal(str(self.indicator_weights.get(symbol, 0.1)))
            correlation = self._get_indicator_correlation(symbol, historical_correlations)
            
            # Calculate weighted impact
            indicator_impact = price_change * correlation * weight
            total_impact += indicator_impact
            total_weight += weight
            
            logger.debug(f"Indicator {symbol}: change {price_change:.2%}, "
                        f"correlation {correlation:.2f}, weight {weight:.2f}, "
                        f"impact {indicator_impact:.4%}")
        
        # Normalize by total weight
        if total_weight > 0:
            normalized_impact = total_impact / total_weight
        else:
            normalized_impact = Decimal('0')
        
        logger.info(f"Total market indicator impact: {normalized_impact:.2%}")
        return normalized_impact
    
    def enhance_estimation_with_indicators(
        self,
        base_estimation: EstimationResult,
        indicator_impact: Decimal,
        enhancement_weight: float = 0.15
    ) -> EstimationResult:
        """
        Enhance base estimation with market indicator signals.
        
        Args:
            base_estimation: Base estimation from ADR and covariance methods
            indicator_impact: Impact calculated from market indicators
            enhancement_weight: Weight to give to indicator enhancement (0-1)
            
        Returns:
            Enhanced estimation result
        """
        logger.info("Enhancing estimation with market indicators")
        
        # Calculate enhancement factor
        enhancement_factor = Decimal('1') + (indicator_impact * Decimal(str(enhancement_weight)))
        
        # Apply enhancement to base estimation
        enhanced_value = base_estimation.estimated_value * enhancement_factor
        
        # Adjust confidence based on indicator signal strength
        confidence_adjustment = self._calculate_confidence_adjustment(indicator_impact)
        enhanced_confidence = min(base_estimation.confidence + confidence_adjustment, 0.95)
        
        # Update method weights
        enhanced_method_weights = base_estimation.method_weights.copy()
        enhanced_method_weights["market_indicators"] = enhancement_weight
        
        # Normalize weights
        total_weight = sum(enhanced_method_weights.values())
        enhanced_method_weights = {k: v/total_weight for k, v in enhanced_method_weights.items()}
        
        # Create enhanced result
        enhanced_result = EstimationResult(
            estimated_value=enhanced_value,
            confidence=enhanced_confidence,
            timestamp=datetime.now(timezone.utc),
            method_weights=enhanced_method_weights,
            component_contributions=base_estimation.component_contributions
        )
        
        logger.info(f"Enhanced estimation: {enhanced_value:.2f} "
                   f"(enhancement factor: {enhancement_factor:.4f}, "
                   f"confidence: {enhanced_confidence:.1%})")
        
        return enhanced_result
    
    def _get_indicator_correlation(
        self, 
        symbol: str, 
        historical_correlations: Optional[Dict[str, float]]
    ) -> Decimal:
        """Get correlation coefficient for an indicator with HSTECH."""
        if historical_correlations and symbol in historical_correlations:
            return Decimal(str(historical_correlations[symbol]))
        
        # Default correlations based on expected relationships
        default_correlations = {
            "PDD": 0.75,    # High correlation - Chinese e-commerce
            "KWEB": 0.85,   # Very high correlation - Chinese internet ETF
            "ASHR": 0.65,   # Moderate-high correlation - China A-shares
            "FXI": 0.70,    # High correlation - China large-cap
            "MCHI": 0.68,   # High correlation - MSCI China
        }
        
        return Decimal(str(default_correlations.get(symbol, 0.5)))
    
    def _calculate_confidence_adjustment(self, indicator_impact: Decimal) -> float:
        """Calculate confidence adjustment based on indicator signal strength."""
        # Stronger signals (larger absolute impact) increase confidence
        impact_magnitude = abs(indicator_impact)
        
        # Cap adjustment at 0.1 (10 percentage points)
        max_adjustment = 0.1
        
        # Scale adjustment based on impact magnitude
        if impact_magnitude > Decimal('0.05'):  # 5% or more
            return max_adjustment
        elif impact_magnitude > Decimal('0.02'):  # 2-5%
            return max_adjustment * 0.6
        elif impact_magnitude > Decimal('0.01'):  # 1-2%
            return max_adjustment * 0.3
        else:
            return 0.0
    
    def analyze_indicator_divergence(
        self,
        indicator_changes: Dict[str, Decimal]
    ) -> Dict[str, float]:
        """
        Analyze divergence between different market indicators.
        
        Returns:
            Dict with divergence metrics and reliability scores
        """
        if len(indicator_changes) < 2:
            return {"divergence_score": 0.0, "reliability": 1.0}
        
        changes_list = [float(change) for change in indicator_changes.values()]
        
        # Calculate standard deviation of changes
        std_dev = np.std(changes_list)
        mean_change = np.mean(changes_list)
        
        # Divergence score: higher when indicators disagree
        divergence_score = std_dev / (abs(mean_change) + 0.01)  # Add small constant to avoid division by zero
        
        # Reliability decreases with higher divergence
        reliability = max(0.0, 1.0 - divergence_score)
        
        return {
            "divergence_score": divergence_score,
            "reliability": reliability,
            "mean_change": mean_change,
            "std_dev": std_dev,
            "indicator_count": len(indicator_changes)
        }
    
    def get_indicator_summary(
        self,
        current_prices: Dict[str, PriceData],
        previous_prices: Dict[str, PriceData]
    ) -> Dict[str, Dict]:
        """Get summary of all indicator movements and signals."""
        summary = {}
        
        for symbol in self.indicator_symbols:
            if symbol in current_prices and symbol in previous_prices:
                current = current_prices[symbol]
                previous = previous_prices[symbol]
                
                change = (current.price - previous.price) / previous.price
                
                summary[symbol] = {
                    "current_price": float(current.price),
                    "previous_price": float(previous.price),
                    "change_percent": float(change),
                    "weight": self.indicator_weights.get(symbol, 0.1),
                    "correlation": float(self._get_indicator_correlation(symbol, None)),
                    "timestamp": current.timestamp.isoformat()
                }
        
        return summary
    
    def update_indicator_weights(self, new_weights: Dict[str, float]):
        """Update the weights for market indicators."""
        # Validate weights sum to approximately 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(f"Indicator weights sum to {total_weight:.2f}, not 1.0")
        
        self.indicator_weights.update(new_weights)
        logger.info(f"Updated indicator weights: {self.indicator_weights}")
    
    def validate_indicator_data(
        self,
        current_prices: Dict[str, PriceData],
        previous_prices: Dict[str, PriceData]
    ) -> List[str]:
        """Validate market indicator data quality."""
        issues = []
        
        # Check for missing indicators
        for symbol in self.indicator_symbols:
            if symbol not in current_prices:
                issues.append(f"Missing current price for indicator {symbol}")
            if symbol not in previous_prices:
                issues.append(f"Missing previous price for indicator {symbol}")
        
        # Check data freshness
        now = datetime.now(timezone.utc)
        max_age_hours = 24
        
        for symbol, price_data in current_prices.items():
            if symbol in self.indicator_symbols:
                age_hours = (now - price_data.timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    issues.append(f"Stale data for indicator {symbol}: {age_hours:.1f} hours old")
        
        return issues


def create_enhanced_estimator(
    indicator_weights: Optional[Dict[str, float]] = None
) -> EnhancedMarketEstimator:
    """Create and return an EnhancedMarketEstimator instance."""
    return EnhancedMarketEstimator(indicator_weights=indicator_weights)
