"""
Benchmark Models

Implementation of benchmark models for comparison with LSTM-BEKK.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from arch import arch_model
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import warnings


class BenchmarkModels:
    """
    Collection of benchmark models for comparison with LSTM-BEKK.
    
    Implements traditional models like DCC, Scalar BEKK, and simple covariance estimators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize benchmark models.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def equal_weighted_portfolio(self, returns: pd.DataFrame) -> Dict:
        """
        Equal-weighted (1/N) portfolio benchmark.
        
        Args:
            returns: Return data
            
        Returns:
            Portfolio weights and performance
        """
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        return {
            'name': 'Equal Weighted',
            'weights': weights,
            'returns': portfolio_returns,
            'description': '1/N equal-weighted portfolio'
        }
    
    def sample_covariance_gmv(self, returns: pd.DataFrame,
                             lookback_window: int = 252) -> Dict:
        """
        GMV portfolio using sample covariance matrix.
        
        Args:
            returns: Return data
            lookback_window: Lookback window for covariance estimation
            
        Returns:
            Portfolio weights and performance
        """
        try:
            # Use last lookback_window observations
            recent_returns = returns.tail(lookback_window)
            
            # Sample covariance matrix
            cov_matrix = recent_returns.cov().values
            
            # GMV optimization: minimize w'Σw subject to w'1 = 1
            n_assets = cov_matrix.shape[0]
            ones = np.ones((n_assets, 1))
            
            # Analytical solution for GMV
            cov_inv = np.linalg.inv(cov_matrix)
            weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)
            weights = weights.flatten()
            
            portfolio_returns = (returns * weights).sum(axis=1)
            
            return {
                'name': 'Sample Covariance GMV',
                'weights': weights,
                'returns': portfolio_returns,
                'covariance_matrix': cov_matrix,
                'description': 'GMV using sample covariance matrix'
            }
            
        except Exception as e:
            self.logger.error(f"Sample covariance GMV failed: {e}")
            return self.equal_weighted_portfolio(returns)
    
    def ledoit_wolf_gmv(self, returns: pd.DataFrame,
                       lookback_window: int = 252) -> Dict:
        """
        GMV portfolio using Ledoit-Wolf shrinkage estimator.
        
        Args:
            returns: Return data
            lookback_window: Lookback window for covariance estimation
            
        Returns:
            Portfolio weights and performance
        """
        try:
            # Use last lookback_window observations
            recent_returns = returns.tail(lookback_window)
            
            # Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(recent_returns).covariance_
            
            # GMV optimization
            n_assets = cov_matrix.shape[0]
            ones = np.ones((n_assets, 1))
            
            cov_inv = np.linalg.inv(cov_matrix)
            weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)
            weights = weights.flatten()
            
            portfolio_returns = (returns * weights).sum(axis=1)
            
            return {
                'name': 'Ledoit-Wolf GMV',
                'weights': weights,
                'returns': portfolio_returns,
                'covariance_matrix': cov_matrix,
                'shrinkage': lw.shrinkage_,
                'description': 'GMV using Ledoit-Wolf shrinkage estimator'
            }
            
        except Exception as e:
            self.logger.error(f"Ledoit-Wolf GMV failed: {e}")
            return self.sample_covariance_gmv(returns, lookback_window)
    
    def scalar_bekk_model(self, returns: pd.DataFrame,
                         lookback_window: int = 252) -> Dict:
        """
        Scalar BEKK model implementation.
        
        Args:
            returns: Return data
            lookback_window: Lookback window for estimation
            
        Returns:
            Model results and covariance forecast
        """
        try:
            # Use recent data for estimation
            recent_returns = returns.tail(lookback_window)
            
            # Simple scalar BEKK implementation
            # H_t = ω + α * r_{t-1} * r_{t-1}' + β * H_{t-1}
            
            # Initialize parameters
            n_assets = len(returns.columns)
            
            # Estimate unconditional covariance
            unconditional_cov = recent_returns.cov().values
            
            # Simple parameter estimation (method of moments)
            # This is a simplified version - full MLE would be more complex
            alpha = 0.05  # Shock parameter
            beta = 0.90   # Persistence parameter
            omega = unconditional_cov * (1 - alpha - beta)
            
            # Forecast one-step ahead covariance
            last_return = recent_returns.iloc[-1].values.reshape(-1, 1)
            last_cov = unconditional_cov  # Simplified - would use actual H_{t-1}
            
            forecast_cov = omega + alpha * (last_return @ last_return.T) + beta * last_cov
            
            # GMV weights using forecasted covariance
            ones = np.ones((n_assets, 1))
            cov_inv = np.linalg.inv(forecast_cov)
            weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)
            weights = weights.flatten()
            
            portfolio_returns = (returns * weights).sum(axis=1)
            
            return {
                'name': 'Scalar BEKK',
                'weights': weights,
                'returns': portfolio_returns,
                'covariance_forecast': forecast_cov,
                'parameters': {'alpha': alpha, 'beta': beta},
                'description': 'Scalar BEKK model with GMV optimization'
            }
            
        except Exception as e:
            self.logger.error(f"Scalar BEKK model failed: {e}")
            return self.sample_covariance_gmv(returns, lookback_window)
    
    def dcc_garch_model(self, returns: pd.DataFrame,
                       lookback_window: int = 252) -> Dict:
        """
        DCC-GARCH model implementation.
        
        Args:
            returns: Return data
            lookback_window: Lookback window for estimation
            
        Returns:
            Model results and covariance forecast
        """
        try:
            # Use recent data
            recent_returns = returns.tail(lookback_window)
            n_assets = len(returns.columns)
            
            # Step 1: Fit univariate GARCH models
            garch_models = {}
            standardized_residuals = pd.DataFrame(index=recent_returns.index, 
                                                columns=recent_returns.columns)
            conditional_volatilities = pd.DataFrame(index=recent_returns.index,
                                                   columns=recent_returns.columns)
            
            for asset in recent_returns.columns:
                try:
                    # Fit GARCH(1,1) model
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = arch_model(recent_returns[asset] * 100, vol='Garch', p=1, q=1)
                        fitted_model = model.fit(disp='off')
                    
                    garch_models[asset] = fitted_model
                    
                    # Extract standardized residuals and conditional volatilities
                    standardized_residuals[asset] = fitted_model.std_resid
                    conditional_volatilities[asset] = fitted_model.conditional_volatility / 100
                    
                except Exception as e:
                    self.logger.warning(f"GARCH fitting failed for {asset}: {e}")
                    # Fallback to simple volatility
                    vol = recent_returns[asset].rolling(21).std().fillna(recent_returns[asset].std())
                    conditional_volatilities[asset] = vol
                    standardized_residuals[asset] = recent_returns[asset] / vol
            
            # Step 2: DCC estimation (simplified)
            # Q_t = (1-a-b)*S + a*z_{t-1}*z_{t-1}' + b*Q_{t-1}
            
            # Parameters (simplified estimation)
            a_dcc = 0.01  # DCC alpha
            b_dcc = 0.95  # DCC beta
            
            # Unconditional correlation
            S = standardized_residuals.corr().values
            
            # Last period standardized residuals
            last_z = standardized_residuals.iloc[-1].values.reshape(-1, 1)
            
            # Forecast correlation (simplified)
            Q_forecast = (1 - a_dcc - b_dcc) * S + a_dcc * (last_z @ last_z.T) + b_dcc * S
            
            # Standardize to get correlation matrix
            Q_diag_inv_sqrt = np.diag(1 / np.sqrt(np.diag(Q_forecast)))
            R_forecast = Q_diag_inv_sqrt @ Q_forecast @ Q_diag_inv_sqrt
            
            # Forecast volatilities (simple persistence)
            vol_forecast = conditional_volatilities.iloc[-1].values
            
            # Construct covariance matrix
            D_forecast = np.diag(vol_forecast)
            cov_forecast = D_forecast @ R_forecast @ D_forecast
            
            # GMV weights
            ones = np.ones((n_assets, 1))
            cov_inv = np.linalg.inv(cov_forecast)
            weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)
            weights = weights.flatten()
            
            portfolio_returns = (returns * weights).sum(axis=1)
            
            return {
                'name': 'DCC-GARCH',
                'weights': weights,
                'returns': portfolio_returns,
                'covariance_forecast': cov_forecast,
                'correlation_forecast': R_forecast,
                'volatility_forecast': vol_forecast,
                'parameters': {'a_dcc': a_dcc, 'b_dcc': b_dcc},
                'description': 'DCC-GARCH model with GMV optimization'
            }
            
        except Exception as e:
            self.logger.error(f"DCC-GARCH model failed: {e}")
            return self.scalar_bekk_model(returns, lookback_window)
    
    def rolling_window_gmv(self, returns: pd.DataFrame,
                          window_size: int = 63) -> Dict:
        """
        Rolling window GMV portfolio.
        
        Args:
            returns: Return data
            window_size: Rolling window size
            
        Returns:
            Portfolio performance with time-varying weights
        """
        portfolio_returns = []
        weights_history = []
        
        for i in range(window_size, len(returns)):
            # Rolling window data
            window_data = returns.iloc[i-window_size:i]
            
            # Sample covariance
            cov_matrix = window_data.cov().values
            
            try:
                # GMV weights
                n_assets = cov_matrix.shape[0]
                ones = np.ones((n_assets, 1))
                cov_inv = np.linalg.inv(cov_matrix)
                weights = (cov_inv @ ones) / (ones.T @ cov_inv @ ones)
                weights = weights.flatten()
                
                # Portfolio return for next period
                next_return = (returns.iloc[i] * weights).sum()
                
            except Exception:
                # Fallback to equal weights
                weights = np.ones(len(returns.columns)) / len(returns.columns)
                next_return = returns.iloc[i].mean()
            
            portfolio_returns.append(next_return)
            weights_history.append(weights)
        
        portfolio_returns = pd.Series(portfolio_returns, 
                                    index=returns.index[window_size:])
        
        return {
            'name': 'Rolling Window GMV',
            'weights': np.array(weights_history),
            'returns': portfolio_returns,
            'description': f'GMV with {window_size}-day rolling window'
        }
    
    def get_all_benchmarks(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get all benchmark models for comparison.
        
        Args:
            returns: Return data
            
        Returns:
            Dictionary of all benchmark results
        """
        benchmarks = {}
        
        self.logger.info("Calculating benchmark models...")
        
        # Equal weighted
        benchmarks['equal_weighted'] = self.equal_weighted_portfolio(returns)
        
        # Sample covariance GMV
        benchmarks['sample_cov_gmv'] = self.sample_covariance_gmv(returns)
        
        # Ledoit-Wolf GMV
        benchmarks['ledoit_wolf_gmv'] = self.ledoit_wolf_gmv(returns)
        
        # Scalar BEKK
        benchmarks['scalar_bekk'] = self.scalar_bekk_model(returns)
        
        # DCC-GARCH
        benchmarks['dcc_garch'] = self.dcc_garch_model(returns)
        
        # Rolling window GMV
        benchmarks['rolling_gmv'] = self.rolling_window_gmv(returns)
        
        self.logger.info(f"Calculated {len(benchmarks)} benchmark models")
        
        return benchmarks
