"""
Performance Metrics

Implementation of comprehensive performance evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator.
    
    Implements all performance metrics mentioned in the LSTM-BEKK paper.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate basic return metrics.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary of return metrics
        """
        # Annualized return (AR)
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        periods_per_year = 252  # Trading days
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Annualized volatility (AV)
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate drawdown metrics including Maximum Drawdown (MDD).
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary of drawdown metrics
        """
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Drawdown series
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown (MDD)
        max_drawdown = drawdowns.min()
        
        # Drawdown duration
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Recovery time (time to recover from max drawdown)
        max_dd_end = drawdowns.idxmin()
        recovery_series = cumulative_returns[max_dd_end:]
        max_value_at_dd = running_max.loc[max_dd_end]
        
        recovery_time = 0
        for i, value in enumerate(recovery_series):
            if value >= max_value_at_dd:
                recovery_time = i
                break
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdowns,
            'num_drawdown_periods': len(drawdown_periods)
        }
    
    def calculate_risk_adjusted_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of risk-adjusted metrics
        """
        metrics = {}
        
        # Basic metrics
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        metrics['sharpe_ratio'] = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        drawdown_metrics = self.calculate_drawdown_metrics(returns)
        max_dd = abs(drawdown_metrics['max_drawdown'])
        metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None:
            # Align series
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 0:
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(252)
                metrics['information_ratio'] = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
                metrics['tracking_error'] = tracking_error
                
                # Beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha
                metrics['alpha'] = annual_return - (self.risk_free_rate + metrics['beta'] * (aligned_benchmark.mean() * 252 - self.risk_free_rate))
        
        return metrics
    
    def calculate_tail_risk_metrics(self, returns: pd.Series,
                                  confidence_levels: List[float] = [0.01, 0.05]) -> Dict:
        """
        Calculate tail risk metrics (VaR, ES).
        
        Args:
            returns: Return series
            confidence_levels: List of confidence levels for VaR/ES
            
        Returns:
            Dictionary of tail risk metrics
        """
        metrics = {}
        
        for alpha in confidence_levels:
            # Value at Risk
            var = -np.percentile(returns, alpha * 100)
            metrics[f'var_{int(alpha*100)}'] = var
            
            # Expected Shortfall
            tail_returns = returns[returns <= -var]
            es = -tail_returns.mean() if len(tail_returns) > 0 else var
            metrics[f'es_{int(alpha*100)}'] = es
            
            # Tail ratio
            tail_ratio = len(tail_returns) / len(returns)
            metrics[f'tail_ratio_{int(alpha*100)}'] = tail_ratio
        
        return metrics
    
    def calculate_stability_metrics(self, returns: pd.Series,
                                  window_size: int = 63) -> Dict:
        """
        Calculate stability metrics for rolling performance.
        
        Args:
            returns: Return series
            window_size: Rolling window size
            
        Returns:
            Dictionary of stability metrics
        """
        # Rolling Sharpe ratios
        rolling_returns = returns.rolling(window=window_size).mean() * 252
        rolling_vol = returns.rolling(window=window_size).std() * np.sqrt(252)
        rolling_sharpe = (rolling_returns - self.risk_free_rate) / rolling_vol
        
        # Stability metrics
        sharpe_stability = rolling_sharpe.std()
        sharpe_consistency = (rolling_sharpe > 0).mean()  # Percentage of positive Sharpe periods
        
        # Return stability
        return_stability = rolling_returns.std()
        
        # Volatility stability
        vol_stability = rolling_vol.std()
        
        return {
            'sharpe_stability': sharpe_stability,
            'sharpe_consistency': sharpe_consistency,
            'return_stability': return_stability,
            'volatility_stability': vol_stability,
            'rolling_sharpe': rolling_sharpe.dropna()
        }
    
    def calculate_comprehensive_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of all performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self.calculate_returns_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self.calculate_drawdown_metrics(returns))
        
        # Risk-adjusted metrics
        metrics.update(self.calculate_risk_adjusted_metrics(returns, benchmark_returns))
        
        # Tail risk metrics
        metrics.update(self.calculate_tail_risk_metrics(returns))
        
        # Stability metrics
        metrics.update(self.calculate_stability_metrics(returns))
        
        return metrics
    
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare multiple strategies using key metrics.
        
        Args:
            strategy_returns: Dictionary of strategy name -> return series
            
        Returns:
            DataFrame comparing strategies
        """
        comparison_metrics = []
        
        for strategy_name, returns in strategy_returns.items():
            metrics = self.calculate_comprehensive_metrics(returns)
            
            # Select key metrics for comparison
            key_metrics = {
                'Strategy': strategy_name,
                'Annualized Return': metrics['annualized_return'],
                'Annualized Volatility': metrics['annualized_volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'VaR 5%': metrics['var_5'],
                'ES 5%': metrics['es_5']
            }
            
            comparison_metrics.append(key_metrics)
        
        return pd.DataFrame(comparison_metrics).set_index('Strategy')
