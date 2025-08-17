"""
Value-at-Risk and Expected Shortfall Models

Implementation of VaR and ES calculations as described in the LSTM-BEKK paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from scipy.optimize import minimize


class VaRCalculator:
    """
    Value-at-Risk calculator using various methods.
    
    Implements VaR calculation methods including parametric, historical simulation,
    and Monte Carlo approaches.
    """
    
    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: VaR confidence level (e.g., 0.05 for 5% VaR)
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def parametric_var(self, portfolio_returns: np.ndarray,
                      covariance_matrix: np.ndarray,
                      weights: np.ndarray,
                      holding_period: int = 1) -> float:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            portfolio_returns: Historical portfolio returns
            covariance_matrix: Portfolio covariance matrix
            weights: Portfolio weights
            holding_period: Holding period in days
            
        Returns:
            VaR value
        """
        # Portfolio volatility
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance * holding_period)
        
        # Portfolio expected return
        expected_return = np.mean(portfolio_returns) * holding_period
        
        # VaR calculation
        z_score = stats.norm.ppf(self.confidence_level)
        var = -(expected_return + z_score * portfolio_volatility)
        
        return var
    
    def historical_var(self, portfolio_returns: np.ndarray,
                      holding_period: int = 1) -> float:
        """
        Calculate historical simulation VaR.
        
        Args:
            portfolio_returns: Historical portfolio returns
            holding_period: Holding period in days
            
        Returns:
            Historical VaR value
        """
        # Scale returns for holding period
        scaled_returns = portfolio_returns * np.sqrt(holding_period)
        
        # Calculate VaR as percentile
        var = -np.percentile(scaled_returns, self.confidence_level * 100)
        
        return var
    
    def monte_carlo_var(self, expected_return: float,
                       covariance_matrix: np.ndarray,
                       weights: np.ndarray,
                       n_simulations: int = 10000,
                       holding_period: int = 1) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            expected_return: Expected portfolio return
            covariance_matrix: Portfolio covariance matrix
            weights: Portfolio weights
            n_simulations: Number of Monte Carlo simulations
            holding_period: Holding period in days
            
        Returns:
            Monte Carlo VaR value
        """
        # Portfolio parameters
        portfolio_return = expected_return * holding_period
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance * holding_period)
        
        # Generate random returns
        random_returns = np.random.normal(
            portfolio_return, portfolio_volatility, n_simulations
        )
        
        # Calculate VaR
        var = -np.percentile(random_returns, self.confidence_level * 100)
        
        return var
    
    def conditional_var(self, portfolio_returns: np.ndarray,
                       market_conditions: Optional[np.ndarray] = None) -> float:
        """
        Calculate conditional VaR based on market conditions.
        
        Args:
            portfolio_returns: Historical portfolio returns
            market_conditions: Market condition indicators (e.g., VIX levels)
            
        Returns:
            Conditional VaR value
        """
        if market_conditions is None:
            return self.historical_var(portfolio_returns)
        
        # Simple approach: calculate VaR for different market regimes
        high_stress_mask = market_conditions > np.percentile(market_conditions, 80)
        
        if np.any(high_stress_mask):
            stress_returns = portfolio_returns[high_stress_mask]
            return self.historical_var(stress_returns)
        else:
            return self.historical_var(portfolio_returns)


class ExpectedShortfall:
    """
    Expected Shortfall (Conditional VaR) calculator.
    
    Implements ES calculation methods as described in the LSTM-BEKK paper.
    """
    
    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize Expected Shortfall calculator.
        
        Args:
            confidence_level: ES confidence level
        """
        self.confidence_level = confidence_level
        self.var_calculator = VaRCalculator(confidence_level)
        self.logger = logging.getLogger(__name__)
    
    def parametric_es(self, portfolio_returns: np.ndarray,
                     covariance_matrix: np.ndarray,
                     weights: np.ndarray,
                     holding_period: int = 1) -> float:
        """
        Calculate parametric Expected Shortfall.
        
        Args:
            portfolio_returns: Historical portfolio returns
            covariance_matrix: Portfolio covariance matrix
            weights: Portfolio weights
            holding_period: Holding period in days
            
        Returns:
            Expected Shortfall value
        """
        # Portfolio parameters
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance * holding_period)
        expected_return = np.mean(portfolio_returns) * holding_period
        
        # ES calculation for normal distribution
        z_alpha = stats.norm.ppf(self.confidence_level)
        phi_z_alpha = stats.norm.pdf(z_alpha)
        
        es = -(expected_return - portfolio_volatility * phi_z_alpha / self.confidence_level)
        
        return es
    
    def historical_es(self, portfolio_returns: np.ndarray,
                     holding_period: int = 1) -> float:
        """
        Calculate historical Expected Shortfall.
        
        Args:
            portfolio_returns: Historical portfolio returns
            holding_period: Holding period in days
            
        Returns:
            Historical Expected Shortfall value
        """
        # Scale returns for holding period
        scaled_returns = portfolio_returns * np.sqrt(holding_period)
        
        # Calculate VaR threshold
        var_threshold = np.percentile(scaled_returns, self.confidence_level * 100)
        
        # Calculate ES as mean of returns below VaR
        tail_returns = scaled_returns[scaled_returns <= var_threshold]
        
        if len(tail_returns) > 0:
            es = -np.mean(tail_returns)
        else:
            es = -var_threshold  # Fallback to VaR if no tail observations
        
        return es
    
    def calculate_joint_loss(self, portfolio_returns: np.ndarray,
                           var_forecast: float,
                           es_forecast: float) -> float:
        """
        Calculate joint loss function for VaR and ES as in the paper.
        
        Args:
            portfolio_returns: Realized portfolio returns
            var_forecast: VaR forecast
            es_forecast: ES forecast
            
        Returns:
            Joint loss value
        """
        T = len(portfolio_returns)
        joint_loss = 0
        
        for t in range(T):
            return_t = portfolio_returns[t]
            
            # Indicator function
            indicator = 1 if return_t <= -var_forecast else 0
            
            # Joint loss component (Asymmetric Laplace loss)
            loss_component = -np.log((self.confidence_level - 1) / (self.confidence_level * es_forecast)) - \
                           (return_t + var_forecast) * (self.confidence_level - indicator) / (self.confidence_level * es_forecast)
            
            joint_loss += loss_component
        
        return joint_loss / T


class RiskDecomposition:
    """
    Risk decomposition and attribution analysis.
    """
    
    def __init__(self):
        """Initialize risk decomposition calculator."""
        self.logger = logging.getLogger(__name__)
    
    def component_var(self, weights: np.ndarray,
                     covariance_matrix: np.ndarray,
                     confidence_level: float = 0.05) -> Dict:
        """
        Calculate component VaR for portfolio assets.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            confidence_level: VaR confidence level
            
        Returns:
            Dictionary with component VaR analysis
        """
        n_assets = len(weights)
        
        # Portfolio variance and volatility
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal VaR (partial derivatives)
        marginal_var = (covariance_matrix @ weights) / portfolio_volatility
        
        # Component VaR
        component_var = weights * marginal_var
        
        # VaR multiplier for confidence level
        z_score = abs(stats.norm.ppf(confidence_level))
        
        return {
            'portfolio_var': z_score * portfolio_volatility,
            'marginal_var': marginal_var * z_score,
            'component_var': component_var * z_score,
            'percentage_contribution': component_var / np.sum(component_var) * 100
        }
    
    def risk_budgeting(self, target_risk_contributions: np.ndarray,
                      covariance_matrix: np.ndarray,
                      initial_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate portfolio weights for target risk contributions.
        
        Args:
            target_risk_contributions: Target risk contribution percentages
            covariance_matrix: Asset covariance matrix
            initial_weights: Initial weight guess
            
        Returns:
            Optimal weights for risk budgeting
        """
        n_assets = covariance_matrix.shape[0]
        
        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets
        
        # Normalize target contributions
        target_risk_contributions = target_risk_contributions / np.sum(target_risk_contributions)
        
        def objective(weights):
            """Objective function for risk budgeting optimization."""
            portfolio_variance = weights.T @ covariance_matrix @ weights
            
            if portfolio_variance <= 0:
                return 1e6  # Large penalty for invalid portfolio
            
            portfolio_volatility = np.sqrt(portfolio_variance)
            marginal_contrib = (covariance_matrix @ weights) / portfolio_volatility
            risk_contrib = weights * marginal_contrib
            
            # Normalize risk contributions
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
            
            # Minimize squared deviations from target
            return np.sum((risk_contrib_pct - target_risk_contributions) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget constraint
            {'type': 'ineq', 'fun': lambda w: w}  # Long-only constraint
        ]
        
        # Bounds
        bounds = [(0.001, 0.5) for _ in range(n_assets)]  # Min 0.1%, max 50%
        
        # Optimize
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            self.logger.warning(f"Risk budgeting optimization failed: {result.message}")
            return initial_weights
