"""
Volatility-Based Trading Strategies

Implementation of volatility-based position sizing and timing strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats


class VolatilityBasedSizing:
    """
    Volatility-based position sizing strategy.
    
    Adjusts position sizes based on predicted volatility levels.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize volatility-based sizing strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategy_config = self.config.get('strategies', {}).get('volatility_sizing', {})
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.target_volatility = self.strategy_config.get('target_volatility', 0.12)
        self.lookback_window = self.strategy_config.get('lookback_window', 63)
        self.min_position_size = self.config.get('portfolio', {}).get('min_position_size', 0.01)
        self.max_position_size = self.config.get('portfolio', {}).get('max_position_size', 0.10)
        
    def calculate_position_sizes(self, volatility_forecasts: np.ndarray,
                               base_weights: np.ndarray,
                               current_volatility: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate position sizes based on volatility forecasts.
        
        Args:
            volatility_forecasts: Predicted volatilities for each asset
            base_weights: Base portfolio weights (e.g., from GMV)
            current_volatility: Current realized volatilities
            
        Returns:
            Adjusted position sizes
        """
        n_assets = len(volatility_forecasts)
        
        # Calculate volatility scaling factors
        vol_scaling = self.target_volatility / volatility_forecasts
        
        # Apply scaling to base weights
        scaled_weights = base_weights * vol_scaling
        
        # Normalize to maintain budget constraint
        scaled_weights = scaled_weights / np.sum(np.abs(scaled_weights))
        
        # Apply position size limits
        scaled_weights = np.clip(scaled_weights, self.min_position_size, self.max_position_size)
        
        # Renormalize after clipping
        scaled_weights = scaled_weights / np.sum(scaled_weights)
        
        self.logger.debug(f"Volatility scaling factors: {vol_scaling}")
        self.logger.debug(f"Adjusted weights: {scaled_weights}")
        
        return scaled_weights
    
    def calculate_volatility_signals(self, volatility_forecasts: np.ndarray,
                                   historical_volatility: np.ndarray) -> np.ndarray:
        """
        Calculate volatility-based trading signals.
        
        Args:
            volatility_forecasts: Predicted volatilities
            historical_volatility: Historical volatility estimates
            
        Returns:
            Volatility signals (-1 to 1)
        """
        # Calculate volatility regime (high/low relative to history)
        vol_percentiles = np.zeros_like(volatility_forecasts)
        
        for i in range(len(volatility_forecasts)):
            if len(historical_volatility) > 0:
                vol_percentiles[i] = stats.percentileofscore(historical_volatility, volatility_forecasts[i]) / 100
            else:
                vol_percentiles[i] = 0.5  # Neutral if no history
        
        # Generate signals based on volatility regime
        # Low volatility -> increase positions (positive signal)
        # High volatility -> decrease positions (negative signal)
        signals = 2 * (1 - vol_percentiles) - 1  # Maps [0,1] to [1,-1]
        
        return signals
    
    def dynamic_rebalancing_trigger(self, current_volatility: np.ndarray,
                                  target_volatility: float,
                                  threshold: float = 0.2) -> bool:
        """
        Determine if portfolio should be rebalanced based on volatility changes.
        
        Args:
            current_volatility: Current portfolio volatility
            target_volatility: Target portfolio volatility
            threshold: Rebalancing threshold (relative change)
            
        Returns:
            True if rebalancing is triggered
        """
        portfolio_vol = np.sqrt(np.mean(current_volatility ** 2))  # Portfolio volatility
        vol_deviation = abs(portfolio_vol - target_volatility) / target_volatility
        
        return vol_deviation > threshold


class VolatilityTiming:
    """
    Volatility timing strategy for market exposure adjustment.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize volatility timing strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.low_vol_threshold = 0.10   # 10% annual volatility
        self.high_vol_threshold = 0.25  # 25% annual volatility
        self.max_exposure = 1.0         # 100% exposure in low vol
        self.min_exposure = 0.3         # 30% exposure in high vol
        
    def calculate_market_exposure(self, market_volatility: float,
                                vol_forecast: float) -> float:
        """
        Calculate optimal market exposure based on volatility.
        
        Args:
            market_volatility: Current market volatility
            vol_forecast: Volatility forecast
            
        Returns:
            Market exposure (0 to 1)
        """
        # Use the higher of current and forecasted volatility for conservative approach
        effective_vol = max(market_volatility, vol_forecast)
        
        if effective_vol <= self.low_vol_threshold:
            exposure = self.max_exposure
        elif effective_vol >= self.high_vol_threshold:
            exposure = self.min_exposure
        else:
            # Linear interpolation between thresholds
            vol_range = self.high_vol_threshold - self.low_vol_threshold
            exposure_range = self.max_exposure - self.min_exposure
            
            vol_position = (effective_vol - self.low_vol_threshold) / vol_range
            exposure = self.max_exposure - vol_position * exposure_range
        
        return np.clip(exposure, self.min_exposure, self.max_exposure)
    
    def generate_timing_signals(self, volatility_forecasts: pd.Series,
                              lookback_window: int = 63) -> pd.Series:
        """
        Generate market timing signals based on volatility forecasts.
        
        Args:
            volatility_forecasts: Time series of volatility forecasts
            lookback_window: Window for volatility regime detection
            
        Returns:
            Market timing signals
        """
        signals = pd.Series(index=volatility_forecasts.index, dtype=float)
        
        for i in range(lookback_window, len(volatility_forecasts)):
            # Current forecast
            current_vol = volatility_forecasts.iloc[i]
            
            # Historical volatility distribution
            hist_vol = volatility_forecasts.iloc[i-lookback_window:i]
            
            # Calculate percentile of current forecast
            vol_percentile = stats.percentileofscore(hist_vol, current_vol) / 100
            
            # Generate signal based on volatility regime
            if vol_percentile <= 0.2:  # Low volatility regime
                signal = 1.0  # Increase exposure
            elif vol_percentile >= 0.8:  # High volatility regime
                signal = -1.0  # Decrease exposure
            else:
                signal = 0.0  # Neutral
            
            signals.iloc[i] = signal
        
        return signals.fillna(0)


class VolatilityBreakout:
    """
    Volatility breakout strategy for detecting regime changes.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize volatility breakout strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.breakout_threshold = 2.0  # Standard deviations
        self.lookback_window = 126     # 6 months
        self.holding_period = 21       # 1 month
        
    def detect_volatility_breakouts(self, volatility_series: pd.Series) -> pd.Series:
        """
        Detect volatility breakouts using statistical thresholds.
        
        Args:
            volatility_series: Time series of volatility estimates
            
        Returns:
            Breakout signals (1 for breakout, 0 for normal)
        """
        signals = pd.Series(index=volatility_series.index, dtype=float)
        
        for i in range(self.lookback_window, len(volatility_series)):
            # Historical volatility statistics
            hist_vol = volatility_series.iloc[i-self.lookback_window:i]
            vol_mean = hist_vol.mean()
            vol_std = hist_vol.std()
            
            # Current volatility
            current_vol = volatility_series.iloc[i]
            
            # Z-score
            z_score = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
            
            # Breakout signal
            if abs(z_score) > self.breakout_threshold:
                signal = 1.0
            else:
                signal = 0.0
            
            signals.iloc[i] = signal
        
        return signals.fillna(0)
    
    def volatility_momentum(self, volatility_series: pd.Series,
                           short_window: int = 21,
                           long_window: int = 63) -> pd.Series:
        """
        Calculate volatility momentum signals.
        
        Args:
            volatility_series: Time series of volatility
            short_window: Short-term moving average window
            long_window: Long-term moving average window
            
        Returns:
            Momentum signals
        """
        # Calculate moving averages
        short_ma = volatility_series.rolling(window=short_window).mean()
        long_ma = volatility_series.rolling(window=long_window).mean()
        
        # Momentum signal
        momentum = (short_ma - long_ma) / long_ma
        
        # Normalize to [-1, 1] range
        momentum_signals = np.tanh(momentum * 2)  # Scaling factor of 2
        
        return momentum_signals.fillna(0)
