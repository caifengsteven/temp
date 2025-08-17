"""
Correlation-Based Trading Strategies

Implementation of correlation breakdown detection and pairs trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from itertools import combinations
import warnings


class CorrelationBreakdown:
    """
    Strategy for detecting correlation breakdown and regime changes.
    
    Identifies when asset correlations deviate significantly from model predictions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize correlation breakdown strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategy_config = self.config.get('strategies', {}).get('correlation_breakdown', {})
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.threshold = self.strategy_config.get('threshold', 2.0)  # Standard deviations
        self.lookback_window = self.strategy_config.get('lookback_window', 126)  # 6 months
        self.min_correlation = self.strategy_config.get('min_correlation', 0.3)
        
    def detect_correlation_breakdown(self, predicted_correlations: np.ndarray,
                                   realized_correlations: np.ndarray,
                                   asset_names: Optional[List[str]] = None) -> Dict:
        """
        Detect correlation breakdown events.
        
        Args:
            predicted_correlations: Model-predicted correlation matrix
            realized_correlations: Realized correlation matrix
            asset_names: Asset names for reporting
            
        Returns:
            Dictionary with breakdown information
        """
        n_assets = predicted_correlations.shape[0]
        asset_names = asset_names or [f"Asset_{i}" for i in range(n_assets)]
        
        # Calculate correlation differences
        correlation_diff = realized_correlations - predicted_correlations
        
        # Extract upper triangular part (unique pairs)
        triu_indices = np.triu_indices(n_assets, k=1)
        pred_corr_pairs = predicted_correlations[triu_indices]
        real_corr_pairs = realized_correlations[triu_indices]
        diff_pairs = correlation_diff[triu_indices]
        
        # Calculate z-scores for differences
        if len(diff_pairs) > 1:
            diff_std = np.std(diff_pairs)
            z_scores = diff_pairs / diff_std if diff_std > 0 else np.zeros_like(diff_pairs)
        else:
            z_scores = np.zeros_like(diff_pairs)
        
        # Identify breakdown events
        breakdown_mask = np.abs(z_scores) > self.threshold
        breakdown_pairs = []
        
        for idx, is_breakdown in enumerate(breakdown_mask):
            if is_breakdown:
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                breakdown_pairs.append({
                    'asset_i': asset_names[i],
                    'asset_j': asset_names[j],
                    'predicted_corr': pred_corr_pairs[idx],
                    'realized_corr': real_corr_pairs[idx],
                    'difference': diff_pairs[idx],
                    'z_score': z_scores[idx]
                })
        
        return {
            'breakdown_detected': len(breakdown_pairs) > 0,
            'num_breakdowns': len(breakdown_pairs),
            'breakdown_pairs': breakdown_pairs,
            'max_z_score': np.max(np.abs(z_scores)) if len(z_scores) > 0 else 0,
            'mean_abs_difference': np.mean(np.abs(diff_pairs))
        }
    
    def generate_breakdown_signals(self, correlation_history: List[np.ndarray],
                                 predicted_correlations: np.ndarray) -> np.ndarray:
        """
        Generate trading signals based on correlation breakdown.
        
        Args:
            correlation_history: Historical correlation matrices
            predicted_correlations: Current predicted correlations
            
        Returns:
            Breakdown signals for each asset pair
        """
        if len(correlation_history) < self.lookback_window:
            return np.zeros_like(predicted_correlations)
        
        # Use recent history for comparison
        recent_correlations = correlation_history[-self.lookback_window:]
        
        # Calculate historical mean and std of correlations
        hist_corr_stack = np.stack(recent_correlations)
        hist_mean = np.mean(hist_corr_stack, axis=0)
        hist_std = np.std(hist_corr_stack, axis=0)
        
        # Calculate z-scores for predicted vs historical
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_scores = np.divide(predicted_correlations - hist_mean, hist_std,
                               out=np.zeros_like(predicted_correlations),
                               where=hist_std != 0)
        
        # Generate signals based on z-scores
        signals = np.zeros_like(predicted_correlations)
        signals[np.abs(z_scores) > self.threshold] = np.sign(z_scores[np.abs(z_scores) > self.threshold])
        
        return signals
    
    def calculate_correlation_stability(self, correlation_history: List[np.ndarray]) -> Dict:
        """
        Calculate correlation stability metrics.
        
        Args:
            correlation_history: Historical correlation matrices
            
        Returns:
            Stability metrics
        """
        if len(correlation_history) < 2:
            return {'stability_score': 1.0, 'volatility_of_correlations': 0.0}
        
        # Stack correlations
        corr_stack = np.stack(correlation_history)
        
        # Calculate volatility of correlations (upper triangular)
        n_assets = corr_stack.shape[1]
        triu_indices = np.triu_indices(n_assets, k=1)
        
        corr_series = corr_stack[:, triu_indices[0], triu_indices[1]]  # (time, pairs)
        corr_volatility = np.std(corr_series, axis=0)
        
        # Overall stability score (inverse of mean volatility)
        mean_volatility = np.mean(corr_volatility)
        stability_score = 1 / (1 + mean_volatility)  # Bounded between 0 and 1
        
        return {
            'stability_score': stability_score,
            'volatility_of_correlations': mean_volatility,
            'max_pair_volatility': np.max(corr_volatility),
            'min_pair_volatility': np.min(corr_volatility)
        }


class PairsTrading:
    """
    Pairs trading strategy based on correlation breakdown.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pairs trading strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.entry_threshold = 2.0      # Z-score threshold for entry
        self.exit_threshold = 0.5       # Z-score threshold for exit
        self.lookback_window = 63       # Window for spread calculation
        self.max_holding_period = 21    # Maximum holding period
        
    def identify_pairs(self, correlation_matrix: np.ndarray,
                      asset_names: List[str],
                      min_correlation: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify potential trading pairs based on correlation.
        
        Args:
            correlation_matrix: Asset correlation matrix
            asset_names: List of asset names
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        n_assets = len(asset_names)
        pairs = []
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = correlation_matrix[i, j]
                if abs(corr) >= min_correlation:
                    pairs.append((asset_names[i], asset_names[j], corr))
        
        # Sort by correlation strength
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return pairs
    
    def calculate_spread(self, price_series_1: pd.Series,
                        price_series_2: pd.Series,
                        method: str = "ratio") -> pd.Series:
        """
        Calculate spread between two price series.
        
        Args:
            price_series_1: First asset price series
            price_series_2: Second asset price series
            method: Spread calculation method ('ratio' or 'difference')
            
        Returns:
            Spread series
        """
        if method == "ratio":
            spread = price_series_1 / price_series_2
        elif method == "difference":
            spread = price_series_1 - price_series_2
        else:
            raise ValueError(f"Unknown spread method: {method}")
        
        return spread
    
    def generate_pairs_signals(self, spread_series: pd.Series) -> pd.Series:
        """
        Generate pairs trading signals based on spread.
        
        Args:
            spread_series: Spread time series
            
        Returns:
            Trading signals (-1, 0, 1)
        """
        signals = pd.Series(index=spread_series.index, dtype=float)
        
        # Calculate rolling statistics
        rolling_mean = spread_series.rolling(window=self.lookback_window).mean()
        rolling_std = spread_series.rolling(window=self.lookback_window).std()
        
        # Calculate z-scores
        z_scores = (spread_series - rolling_mean) / rolling_std
        
        # Generate signals
        for i in range(len(z_scores)):
            z_score = z_scores.iloc[i]
            
            if pd.isna(z_score):
                signals.iloc[i] = 0
            elif z_score > self.entry_threshold:
                signals.iloc[i] = -1  # Short spread (short asset1, long asset2)
            elif z_score < -self.entry_threshold:
                signals.iloc[i] = 1   # Long spread (long asset1, short asset2)
            elif abs(z_score) < self.exit_threshold:
                signals.iloc[i] = 0   # Exit position
            else:
                # Hold previous signal if available
                signals.iloc[i] = signals.iloc[i-1] if i > 0 else 0
        
        return signals
    
    def backtest_pair(self, price_series_1: pd.Series,
                     price_series_2: pd.Series,
                     transaction_cost: float = 0.001) -> Dict:
        """
        Backtest pairs trading strategy for a single pair.
        
        Args:
            price_series_1: First asset prices
            price_series_2: Second asset prices
            transaction_cost: Transaction cost (as fraction)
            
        Returns:
            Backtest results
        """
        # Calculate spread and signals
        spread = self.calculate_spread(price_series_1, price_series_2)
        signals = self.generate_pairs_signals(spread)
        
        # Calculate returns
        returns_1 = price_series_1.pct_change()
        returns_2 = price_series_2.pct_change()
        
        # Calculate strategy returns
        strategy_returns = []
        position = 0
        
        for i in range(1, len(signals)):
            signal = signals.iloc[i]
            prev_signal = signals.iloc[i-1]
            
            # Check for position change
            if signal != prev_signal:
                # Transaction cost
                cost = abs(signal - prev_signal) * transaction_cost
            else:
                cost = 0
            
            # Calculate return based on position
            if signal != 0:
                # Pairs return: long asset1, short asset2 (or vice versa)
                pair_return = signal * (returns_1.iloc[i] - returns_2.iloc[i])
                strategy_returns.append(pair_return - cost)
            else:
                strategy_returns.append(-cost)  # Only transaction cost
        
        strategy_returns = pd.Series(strategy_returns, index=signals.index[1:])
        
        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': np.sum(signals.diff() != 0),
            'strategy_returns': strategy_returns
        }
