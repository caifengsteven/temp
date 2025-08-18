"""
Step 2: Covariance-based Estimation for Non-dual-listed HSTECH Components

This module implements statistical methods to estimate price movements of 
non-dual-listed HSTECH components based on the movements of dual-listed stocks.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from ..models import PriceData, EstimationResult
from ..data.adr_mapper import ADRMapper
from ...data.hstech_components import HSTECH_COMPONENTS, get_adr_mapped_components, get_non_adr_components

logger = logging.getLogger(__name__)


class CovarianceBasedEstimator:
    """
    Estimates non-dual-listed HSTECH component movements using statistical models.
    
    This estimator:
    1. Builds historical covariance matrices between all HSTECH components
    2. Uses factor models to predict non-ADR stock movements from ADR movements
    3. Applies regression models with regularization for robust predictions
    4. Validates predictions using historical correlation patterns
    """
    
    def __init__(self, lookback_days: int = 252, min_correlation: float = 0.3):
        self.lookback_days = lookback_days
        self.min_correlation = min_correlation
        self.adr_mapper = ADRMapper()
        
        self.adr_components = get_adr_mapped_components()
        self.non_adr_components = get_non_adr_components()
        
        # Model storage
        self.covariance_matrix = None
        self.correlation_matrix = None
        self.regression_models = {}
        self.feature_scalers = {}
        self.model_scores = {}
        
        logger.info(f"Initialized covariance estimator: "
                   f"{len(self.adr_components)} ADR components, "
                   f"{len(self.non_adr_components)} non-ADR components")
    
    def build_covariance_model(self, historical_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Build covariance matrix and regression models from historical data.
        
        Args:
            historical_data: Dict of {symbol: DataFrame with 'date', 'close', 'returns' columns}
            
        Returns:
            True if model building successful, False otherwise
        """
        logger.info("Building covariance model from historical data")
        
        try:
            # Prepare return data matrix
            returns_data = self._prepare_returns_matrix(historical_data)
            
            if returns_data.empty:
                logger.error("No valid returns data available")
                return False
            
            # Calculate covariance and correlation matrices
            self.covariance_matrix = returns_data.cov()
            self.correlation_matrix = returns_data.corr()
            
            # Build regression models for each non-ADR component
            self._build_regression_models(returns_data)
            
            logger.info(f"Covariance model built successfully: "
                       f"{len(returns_data.columns)} components, "
                       f"{len(returns_data)} observations")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build covariance model: {e}")
            return False
    
    def estimate_non_adr_movements(
        self,
        adr_price_changes: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Estimate price movements for non-ADR components based on ADR movements.
        
        Args:
            adr_price_changes: Dict of {hk_symbol: price_change_percent} for ADR components
            
        Returns:
            Dict of {hk_symbol: estimated_price_change_percent} for non-ADR components
        """
        if self.correlation_matrix is None:
            logger.error("Covariance model not built yet")
            return {}
        
        logger.info("Estimating non-ADR component movements")
        
        estimated_changes = {}
        
        for non_adr_stock in self.non_adr_components:
            target_symbol = non_adr_stock.symbol
            
            # Method 1: Regression-based prediction
            regression_estimate = self._predict_using_regression(target_symbol, adr_price_changes)
            
            # Method 2: Correlation-weighted average
            correlation_estimate = self._predict_using_correlation(target_symbol, adr_price_changes)
            
            # Method 3: Factor model approach
            factor_estimate = self._predict_using_factor_model(target_symbol, adr_price_changes)
            
            # Combine estimates with weights based on model quality
            final_estimate = self._combine_estimates(
                target_symbol,
                regression_estimate,
                correlation_estimate, 
                factor_estimate
            )
            
            estimated_changes[target_symbol] = final_estimate
            
            logger.debug(f"Estimated change for {target_symbol}: {final_estimate:.2%} "
                        f"(regression: {regression_estimate:.2%}, "
                        f"correlation: {correlation_estimate:.2%}, "
                        f"factor: {factor_estimate:.2%})")
        
        return estimated_changes
    
    def _prepare_returns_matrix(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare a matrix of returns for all components."""
        returns_dict = {}
        
        for symbol, data in historical_data.items():
            if 'returns' in data.columns and len(data) >= self.lookback_days // 2:
                # Use most recent data up to lookback_days
                recent_data = data.tail(self.lookback_days)
                returns_dict[symbol] = recent_data['returns']
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align all series by date and drop NaN values
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _build_regression_models(self, returns_data: pd.DataFrame):
        """Build regression models for each non-ADR component."""
        adr_symbols = [stock.symbol for stock in self.adr_components]
        non_adr_symbols = [stock.symbol for stock in self.non_adr_components]
        
        # Filter to available data
        available_adr = [s for s in adr_symbols if s in returns_data.columns]
        available_non_adr = [s for s in non_adr_symbols if s in returns_data.columns]
        
        if len(available_adr) < 2:
            logger.warning("Insufficient ADR components for regression modeling")
            return
        
        for target_symbol in available_non_adr:
            try:
                # Prepare features (ADR component returns) and target
                X = returns_data[available_adr].values
                y = returns_data[target_symbol].values
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit Ridge regression (with regularization)
                model = Ridge(alpha=0.1)
                model.fit(X_scaled, y)
                
                # Calculate R² score
                y_pred = model.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                
                # Store model and scaler
                self.regression_models[target_symbol] = model
                self.feature_scalers[target_symbol] = scaler
                self.model_scores[target_symbol] = r2
                
                logger.debug(f"Built regression model for {target_symbol}: R² = {r2:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to build regression model for {target_symbol}: {e}")
    
    def _predict_using_regression(
        self, 
        target_symbol: str, 
        adr_changes: Dict[str, Decimal]
    ) -> Decimal:
        """Predict using trained regression model."""
        if target_symbol not in self.regression_models:
            return Decimal('0')
        
        model = self.regression_models[target_symbol]
        scaler = self.feature_scalers[target_symbol]
        
        # Prepare feature vector
        adr_symbols = [stock.symbol for stock in self.adr_components]
        features = []
        
        for symbol in adr_symbols:
            if symbol in adr_changes:
                features.append(float(adr_changes[symbol]))
            else:
                features.append(0.0)  # Missing data treated as no change
        
        if not features:
            return Decimal('0')
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        return Decimal(str(prediction))
    
    def _predict_using_correlation(
        self, 
        target_symbol: str, 
        adr_changes: Dict[str, Decimal]
    ) -> Decimal:
        """Predict using correlation-weighted average."""
        if target_symbol not in self.correlation_matrix.index:
            return Decimal('0')
        
        weighted_sum = Decimal('0')
        weight_sum = Decimal('0')
        
        for adr_symbol, change in adr_changes.items():
            if adr_symbol in self.correlation_matrix.columns:
                correlation = self.correlation_matrix.loc[target_symbol, adr_symbol]
                
                # Only use correlations above minimum threshold
                if abs(correlation) >= self.min_correlation:
                    weight = Decimal(str(abs(correlation)))
                    weighted_sum += change * weight
                    weight_sum += weight
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return Decimal('0')
    
    def _predict_using_factor_model(
        self, 
        target_symbol: str, 
        adr_changes: Dict[str, Decimal]
    ) -> Decimal:
        """Predict using a simple factor model approach."""
        if target_symbol not in self.correlation_matrix.index:
            return Decimal('0')
        
        # Use the average change of ADR components, weighted by market cap
        adr_symbols = [stock.symbol for stock in self.adr_components]
        total_weight = Decimal('0')
        weighted_change = Decimal('0')
        
        for stock in self.adr_components:
            if stock.symbol in adr_changes:
                weight = Decimal(str(stock.weight))
                change = adr_changes[stock.symbol]
                weighted_change += change * weight
                total_weight += weight
        
        if total_weight > 0:
            market_factor = weighted_change / total_weight
            
            # Apply beta-like adjustment based on historical correlation with market
            avg_correlation = Decimal('0')
            correlation_count = 0
            
            for adr_symbol in adr_symbols:
                if adr_symbol in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[target_symbol, adr_symbol]
                    if not pd.isna(corr):
                        avg_correlation += Decimal(str(corr))
                        correlation_count += 1
            
            if correlation_count > 0:
                beta = avg_correlation / correlation_count
                return market_factor * beta
        
        return Decimal('0')
    
    def _combine_estimates(
        self,
        target_symbol: str,
        regression_est: Decimal,
        correlation_est: Decimal,
        factor_est: Decimal
    ) -> Decimal:
        """Combine multiple estimates using quality-based weights."""
        estimates = []
        weights = []
        
        # Regression estimate weight based on R² score
        if target_symbol in self.model_scores:
            r2_score = self.model_scores[target_symbol]
            if r2_score > 0.1:  # Only use if reasonably predictive
                estimates.append(regression_est)
                weights.append(Decimal(str(r2_score)))
        
        # Correlation estimate (moderate weight)
        estimates.append(correlation_est)
        weights.append(Decimal('0.3'))
        
        # Factor estimate (lower weight, but always available)
        estimates.append(factor_est)
        weights.append(Decimal('0.2'))
        
        # Calculate weighted average
        if not estimates:
            return Decimal('0')
        
        weighted_sum = sum(est * weight for est, weight in zip(estimates, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else Decimal('0')
    
    def get_model_quality_metrics(self) -> Dict[str, Dict]:
        """Get quality metrics for the covariance models."""
        metrics = {}
        
        for symbol in self.model_scores:
            metrics[symbol] = {
                "r2_score": self.model_scores[symbol],
                "has_regression_model": symbol in self.regression_models,
                "avg_correlation": float(self.correlation_matrix.loc[symbol].abs().mean()) 
                                 if symbol in self.correlation_matrix.index else 0.0
            }
        
        return metrics


def create_covariance_estimator(lookback_days: int = 252) -> CovarianceBasedEstimator:
    """Create and return a CovarianceBasedEstimator instance."""
    return CovarianceBasedEstimator(lookback_days=lookback_days)
