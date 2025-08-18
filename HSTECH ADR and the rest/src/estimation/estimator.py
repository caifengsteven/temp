"""
Main HSTECH Index Estimator

This module combines all three estimation methods to provide the final
HSTECH index price estimation when Hong Kong markets are closed.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import logging

import pandas as pd
from ..models import PriceData, CurrencyRate, IndexData, EstimationResult, Config
from .adr_estimator import ADRBasedEstimator
from .covariance_estimator import CovarianceBasedEstimator
from .enhanced_estimator import EnhancedMarketEstimator
from ...data.hstech_components import HSTECH_COMPONENTS

logger = logging.getLogger(__name__)


class HSTECHEstimator:
    """
    Main HSTECH Index Estimator that combines all estimation methods.
    
    This estimator:
    1. Uses ADR-based estimation for dual-listed components
    2. Applies covariance-based estimation for non-dual-listed components  
    3. Enhances with market indicators (PDD, KWEB, etc.)
    4. Combines all methods with configurable weights
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize sub-estimators
        self.adr_estimator = ADRBasedEstimator()
        self.covariance_estimator = CovarianceBasedEstimator(
            lookback_days=self.config.estimation.lookback_days,
            min_correlation=self.config.estimation.min_correlation_threshold
        )
        self.enhanced_estimator = EnhancedMarketEstimator()
        
        # Estimation weights from config
        self.method_weights = {
            "adr_based": self.config.estimation.weights.adr_based,
            "covariance_based": self.config.estimation.weights.covariance_based,
            "market_indicators": self.config.estimation.weights.market_indicators
        }
        
        logger.info(f"Initialized HSTECH estimator with weights: {self.method_weights}")
    
    def estimate_current_price(
        self,
        current_adr_prices: Dict[str, PriceData],
        last_hk_prices: Dict[str, PriceData],
        current_exchange_rate: CurrencyRate,
        last_hstech_close: IndexData,
        current_indicator_prices: Dict[str, PriceData],
        previous_indicator_prices: Dict[str, PriceData],
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> EstimationResult:
        """
        Estimate current HSTECH index price using all available methods.
        
        Args:
            current_adr_prices: Current ADR prices
            last_hk_prices: Last known HK closing prices
            current_exchange_rate: Current USD/HKD exchange rate
            last_hstech_close: Last HSTECH index closing value
            current_indicator_prices: Current market indicator prices
            previous_indicator_prices: Previous market indicator prices
            historical_data: Historical price data for covariance modeling
            
        Returns:
            Combined estimation result
        """
        logger.info("Starting comprehensive HSTECH estimation")
        
        # Validate input data
        validation_issues = self._validate_input_data(
            current_adr_prices, last_hk_prices, current_exchange_rate,
            current_indicator_prices, previous_indicator_prices
        )
        
        if validation_issues:
            logger.warning(f"Data validation issues: {validation_issues}")
        
        # Step 1: ADR-based estimation
        adr_result = self.adr_estimator.calculate_adr_based_update(
            current_adr_prices,
            last_hk_prices,
            current_exchange_rate,
            last_hstech_close
        )
        
        logger.info(f"ADR estimation: {adr_result.estimated_value:.2f} "
                   f"(confidence: {adr_result.confidence:.1%})")
        
        # Step 2: Covariance-based estimation for non-ADR components
        covariance_adjustments = {}
        if historical_data:
            # Build/update covariance model
            model_built = self.covariance_estimator.build_covariance_model(historical_data)
            
            if model_built:
                # Extract ADR price changes from Step 1
                adr_changes = self._extract_adr_changes(adr_result)
                
                # Estimate non-ADR component movements
                covariance_adjustments = self.covariance_estimator.estimate_non_adr_movements(
                    adr_changes
                )
                
                logger.info(f"Covariance estimation: {len(covariance_adjustments)} "
                           f"non-ADR components estimated")
        
        # Step 3: Market indicator enhancement
        indicator_impact = self.enhanced_estimator.calculate_market_indicator_impact(
            current_indicator_prices,
            previous_indicator_prices
        )
        
        logger.info(f"Market indicator impact: {indicator_impact:.2%}")
        
        # Combine all estimations
        final_result = self._combine_estimations(
            adr_result,
            covariance_adjustments,
            indicator_impact,
            last_hstech_close
        )
        
        logger.info(f"Final HSTECH estimation: {final_result.estimated_value:.2f} "
                   f"(confidence: {final_result.confidence:.1%})")
        
        return final_result
    
    def _extract_adr_changes(self, adr_result: EstimationResult) -> Dict[str, Decimal]:
        """Extract individual ADR component changes from the ADR estimation result."""
        adr_changes = {}
        
        # Convert component contributions back to individual changes
        for hk_symbol, contribution in adr_result.component_contributions.items():
            # Get component weight
            component_weight = None
            for stock in self.adr_estimator.adr_components:
                if stock.symbol == hk_symbol:
                    component_weight = stock.weight
                    break
            
            if component_weight and component_weight > 0:
                # Back-calculate individual change from weighted contribution
                individual_change = Decimal(str(contribution)) / Decimal(str(component_weight))
                adr_changes[hk_symbol] = individual_change
        
        return adr_changes
    
    def _combine_estimations(
        self,
        adr_result: EstimationResult,
        covariance_adjustments: Dict[str, Decimal],
        indicator_impact: Decimal,
        last_hstech_close: IndexData
    ) -> EstimationResult:
        """Combine all estimation methods into final result."""
        
        # Start with ADR-based estimation
        base_value = adr_result.estimated_value
        
        # Apply covariance-based adjustments for non-ADR components
        covariance_impact = self._calculate_covariance_impact(covariance_adjustments)
        covariance_adjusted_value = base_value * (Decimal('1') + covariance_impact)
        
        # Apply market indicator enhancement
        indicator_factor = Decimal('1') + (indicator_impact * Decimal(str(self.method_weights["market_indicators"])))
        final_value = covariance_adjusted_value * indicator_factor
        
        # Calculate combined confidence
        final_confidence = self._calculate_combined_confidence(
            adr_result.confidence,
            len(covariance_adjustments),
            indicator_impact
        )
        
        # Combine component contributions
        combined_contributions = adr_result.component_contributions.copy()
        
        # Add covariance-based contributions
        for symbol, change in covariance_adjustments.items():
            # Get component weight
            component_weight = 0.0
            for stock in self.covariance_estimator.non_adr_components:
                if stock.symbol == symbol:
                    component_weight = stock.weight
                    break
            
            contribution = float(change * Decimal(str(component_weight)))
            combined_contributions[symbol] = contribution
        
        # Create final result
        final_result = EstimationResult(
            estimated_value=final_value,
            confidence=final_confidence,
            timestamp=datetime.now(timezone.utc),
            method_weights=self.method_weights,
            component_contributions=combined_contributions
        )
        
        return final_result
    
    def _calculate_covariance_impact(self, covariance_adjustments: Dict[str, Decimal]) -> Decimal:
        """Calculate the overall index impact from covariance-based adjustments."""
        total_impact = Decimal('0')
        
        for symbol, change in covariance_adjustments.items():
            # Get component weight
            component_weight = 0.0
            for stock in self.covariance_estimator.non_adr_components:
                if stock.symbol == symbol:
                    component_weight = stock.weight
                    break
            
            if component_weight > 0:
                weighted_impact = change * Decimal(str(component_weight))
                total_impact += weighted_impact
        
        # Apply covariance method weight
        return total_impact * Decimal(str(self.method_weights["covariance_based"]))
    
    def _calculate_combined_confidence(
        self,
        adr_confidence: float,
        num_covariance_estimates: int,
        indicator_impact: Decimal
    ) -> float:
        """Calculate combined confidence score from all methods."""
        
        # Base confidence from ADR estimation (weighted)
        weighted_adr_confidence = adr_confidence * self.method_weights["adr_based"]
        
        # Covariance confidence based on number of estimates
        max_non_adr_components = len(self.covariance_estimator.non_adr_components)
        covariance_coverage = num_covariance_estimates / max_non_adr_components if max_non_adr_components > 0 else 0
        covariance_confidence = covariance_coverage * 0.8  # Max 80% confidence for covariance
        weighted_covariance_confidence = covariance_confidence * self.method_weights["covariance_based"]
        
        # Indicator confidence based on signal strength
        indicator_strength = min(abs(float(indicator_impact)) * 10, 1.0)  # Scale to 0-1
        indicator_confidence = indicator_strength * 0.7  # Max 70% confidence for indicators
        weighted_indicator_confidence = indicator_confidence * self.method_weights["market_indicators"]
        
        # Combined confidence
        total_confidence = weighted_adr_confidence + weighted_covariance_confidence + weighted_indicator_confidence
        
        # Cap at 95% to account for model uncertainty
        return min(total_confidence, 0.95)
    
    def _validate_input_data(
        self,
        adr_prices: Dict[str, PriceData],
        hk_prices: Dict[str, PriceData],
        exchange_rate: CurrencyRate,
        current_indicators: Dict[str, PriceData],
        previous_indicators: Dict[str, PriceData]
    ) -> List[str]:
        """Validate all input data."""
        issues = []
        
        # Validate ADR data
        issues.extend(self.adr_estimator.validate_input_data(adr_prices, hk_prices, exchange_rate))
        
        # Validate indicator data
        issues.extend(self.enhanced_estimator.validate_indicator_data(current_indicators, previous_indicators))
        
        return issues
    
    def get_estimation_summary(self) -> Dict[str, any]:
        """Get summary of estimator configuration and capabilities."""
        adr_stats = self.adr_estimator.get_adr_coverage_stats()
        
        return {
            "method_weights": self.method_weights,
            "adr_coverage": adr_stats,
            "covariance_lookback_days": self.covariance_estimator.lookback_days,
            "market_indicators": self.enhanced_estimator.indicator_symbols,
            "total_hstech_components": len(HSTECH_COMPONENTS)
        }
