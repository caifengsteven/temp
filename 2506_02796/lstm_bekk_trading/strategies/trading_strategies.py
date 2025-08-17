"""
Trading Strategies Coordinator

Main class that coordinates all trading strategies using LSTM-BEKK model outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from .portfolio_optimization import GMVPortfolio, MeanVarianceOptimizer, RiskParityOptimizer
from .volatility_strategies import VolatilityBasedSizing, VolatilityTiming, VolatilityBreakout
from .correlation_strategies import CorrelationBreakdown, PairsTrading


class TradingStrategies:
    """
    Main trading strategies coordinator for the LSTM-BEKK system.
    
    Integrates multiple strategies and provides a unified interface for trading decisions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trading strategies coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy components
        self.gmv_optimizer = GMVPortfolio(config)
        self.mean_var_optimizer = MeanVarianceOptimizer(config=config)
        self.risk_parity_optimizer = RiskParityOptimizer(config)
        
        self.volatility_sizing = VolatilityBasedSizing(config)
        self.volatility_timing = VolatilityTiming(config)
        self.volatility_breakout = VolatilityBreakout(config)
        
        self.correlation_breakdown = CorrelationBreakdown(config)
        self.pairs_trading = PairsTrading(config)
        
        # Strategy state
        self.current_positions = {}
        self.position_history = []
        self.signal_history = []
        
        # Configuration
        self.rebalance_frequency = self.config.get('trading', {}).get('portfolio', {}).get('rebalance_frequency', 5)
        self.enabled_strategies = self.config.get('trading', {}).get('strategies', {})
        
    def generate_trading_signals(self, model_outputs: Dict,
                                market_data: pd.DataFrame,
                                current_date: datetime) -> Dict:
        """
        Generate comprehensive trading signals from all strategies.
        
        Args:
            model_outputs: LSTM-BEKK model outputs (covariance forecasts, etc.)
            market_data: Current market data
            current_date: Current trading date
            
        Returns:
            Dictionary of trading signals and recommendations
        """
        signals = {
            'date': current_date,
            'gmv_weights': None,
            'volatility_signals': None,
            'correlation_signals': None,
            'pairs_signals': None,
            'final_weights': None,
            'rebalance_required': False
        }
        
        # Extract model outputs
        covariance_forecast = model_outputs.get('covariance_forecast')
        volatility_forecast = model_outputs.get('volatility_forecast')
        correlation_forecast = model_outputs.get('correlation_forecast')
        
        if covariance_forecast is None:
            self.logger.warning("No covariance forecast available")
            return signals
        
        # 1. GMV Portfolio Optimization
        if self.enabled_strategies.get('gmv', {}).get('enabled', True):
            try:
                gmv_weights = self.gmv_optimizer.optimize_weights(
                    covariance_matrix=covariance_forecast
                )
                signals['gmv_weights'] = gmv_weights
                self.logger.debug(f"GMV weights calculated: {gmv_weights}")
            except Exception as e:
                self.logger.error(f"GMV optimization failed: {e}")
                signals['gmv_weights'] = self._get_equal_weights(covariance_forecast.shape[0])
        
        # 2. Volatility-Based Strategies
        if self.enabled_strategies.get('volatility_sizing', {}).get('enabled', True):
            try:
                if volatility_forecast is not None and signals['gmv_weights'] is not None:
                    vol_adjusted_weights = self.volatility_sizing.calculate_position_sizes(
                        volatility_forecasts=volatility_forecast,
                        base_weights=signals['gmv_weights']
                    )
                    signals['volatility_signals'] = {
                        'adjusted_weights': vol_adjusted_weights,
                        'volatility_forecast': volatility_forecast
                    }
            except Exception as e:
                self.logger.error(f"Volatility sizing failed: {e}")
        
        # 3. Correlation Breakdown Detection
        if self.enabled_strategies.get('correlation_breakdown', {}).get('enabled', True):
            try:
                if correlation_forecast is not None:
                    # Calculate realized correlations from recent data
                    recent_returns = market_data.tail(21)  # Last 21 days
                    realized_corr = recent_returns.corr().values
                    
                    breakdown_info = self.correlation_breakdown.detect_correlation_breakdown(
                        predicted_correlations=correlation_forecast,
                        realized_correlations=realized_corr,
                        asset_names=market_data.columns.tolist()
                    )
                    signals['correlation_signals'] = breakdown_info
            except Exception as e:
                self.logger.error(f"Correlation breakdown detection failed: {e}")
        
        # 4. Combine Signals and Generate Final Weights
        signals['final_weights'] = self._combine_signals(signals)
        
        # 5. Determine if Rebalancing is Required
        signals['rebalance_required'] = self._should_rebalance(
            new_weights=signals['final_weights'],
            current_date=current_date
        )
        
        # Store signal history
        self.signal_history.append(signals)
        
        return signals
    
    def _combine_signals(self, signals: Dict) -> np.ndarray:
        """
        Combine signals from different strategies into final portfolio weights.
        
        Args:
            signals: Dictionary of individual strategy signals
            
        Returns:
            Final portfolio weights
        """
        # Start with GMV weights as base
        base_weights = signals.get('gmv_weights')
        if base_weights is None:
            return None
        
        final_weights = base_weights.copy()
        
        # Apply volatility adjustments
        vol_signals = signals.get('volatility_signals')
        if vol_signals is not None:
            vol_adjusted = vol_signals.get('adjusted_weights')
            if vol_adjusted is not None:
                # Blend GMV and volatility-adjusted weights
                blend_factor = 0.7  # 70% volatility-adjusted, 30% GMV
                final_weights = blend_factor * vol_adjusted + (1 - blend_factor) * base_weights
        
        # Apply correlation breakdown adjustments
        corr_signals = signals.get('correlation_signals')
        if corr_signals is not None and corr_signals.get('breakdown_detected', False):
            # Reduce concentration when correlations break down
            concentration_reduction = 0.1  # Reduce by 10%
            final_weights = final_weights * (1 - concentration_reduction)
            final_weights = final_weights / np.sum(final_weights)  # Renormalize
        
        # Ensure weights sum to 1 and respect constraints
        final_weights = self._apply_weight_constraints(final_weights)
        
        return final_weights
    
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio weight constraints."""
        if weights is None:
            return None
        
        portfolio_config = self.config.get('trading', {}).get('portfolio', {})
        max_weight = portfolio_config.get('max_position_size', 0.1)
        min_weight = portfolio_config.get('min_position_size', 0.01)
        
        # Apply position size limits
        weights = np.clip(weights, min_weight, max_weight)
        
        # Renormalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def _should_rebalance(self, new_weights: np.ndarray, current_date: datetime) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            new_weights: New target weights
            current_date: Current date
            
        Returns:
            True if rebalancing is required
        """
        if new_weights is None:
            return False
        
        # Check if enough time has passed since last rebalance
        if len(self.position_history) > 0:
            last_rebalance = self.position_history[-1]['date']
            days_since_rebalance = (current_date - last_rebalance).days
            
            if days_since_rebalance < self.rebalance_frequency:
                return False
        
        # Check if weights have changed significantly
        if len(self.position_history) > 0:
            current_weights = self.position_history[-1]['weights']
            weight_change = np.sum(np.abs(new_weights - current_weights))
            
            # Rebalance if total weight change exceeds threshold
            rebalance_threshold = 0.05  # 5% total weight change
            if weight_change < rebalance_threshold:
                return False
        
        return True
    
    def execute_rebalancing(self, target_weights: np.ndarray,
                          current_prices: pd.Series,
                          current_date: datetime,
                          portfolio_value: float) -> Dict:
        """
        Execute portfolio rebalancing.
        
        Args:
            target_weights: Target portfolio weights
            current_prices: Current asset prices
            current_date: Current date
            portfolio_value: Current portfolio value
            
        Returns:
            Rebalancing execution details
        """
        if target_weights is None:
            return {'success': False, 'message': 'No target weights provided'}
        
        # Calculate target positions
        target_values = target_weights * portfolio_value
        target_shares = target_values / current_prices.values
        
        # Get current positions
        current_positions = self.current_positions.copy()
        
        # Calculate trades required
        trades = {}
        total_trade_value = 0
        
        for i, asset in enumerate(current_prices.index):
            current_shares = current_positions.get(asset, 0)
            target_shares_asset = target_shares[i]
            
            trade_shares = target_shares_asset - current_shares
            trade_value = abs(trade_shares * current_prices.iloc[i])
            
            if abs(trade_shares) > 1e-6:  # Minimum trade threshold
                trades[asset] = {
                    'shares': trade_shares,
                    'value': trade_value,
                    'direction': 'buy' if trade_shares > 0 else 'sell'
                }
                total_trade_value += trade_value
        
        # Update positions
        for asset, trade in trades.items():
            self.current_positions[asset] = self.current_positions.get(asset, 0) + trade['shares']
        
        # Record position history
        position_record = {
            'date': current_date,
            'weights': target_weights,
            'positions': self.current_positions.copy(),
            'portfolio_value': portfolio_value,
            'trades': trades,
            'total_trade_value': total_trade_value
        }
        self.position_history.append(position_record)
        
        self.logger.info(f"Rebalancing executed on {current_date}: {len(trades)} trades, "
                        f"total value: ${total_trade_value:,.2f}")
        
        return {
            'success': True,
            'trades': trades,
            'total_trade_value': total_trade_value,
            'new_positions': self.current_positions.copy()
        }
    
    def _get_equal_weights(self, n_assets: int) -> np.ndarray:
        """Get equal weights as fallback."""
        return np.ones(n_assets) / n_assets
    
    def get_strategy_performance(self) -> Dict:
        """Get performance statistics for all strategies."""
        if len(self.position_history) < 2:
            return {'message': 'Insufficient history for performance calculation'}
        
        # Calculate portfolio returns
        portfolio_values = [pos['portfolio_value'] for pos in self.position_history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Performance metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_rebalances': len(self.position_history),
            'total_trades': sum(len(pos['trades']) for pos in self.position_history)
        }
