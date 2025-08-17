"""
Portfolio Optimization Strategies

Implementation of portfolio optimization strategies using LSTM-BEKK covariance forecasts.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import minimize
import warnings


class PortfolioOptimizer:
    """
    Base class for portfolio optimization strategies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize portfolio optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def optimize_weights(self, covariance_matrix: np.ndarray,
                        expected_returns: Optional[np.ndarray] = None,
                        constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            constraints: Additional constraints
            
        Returns:
            Optimal weights
        """
        raise NotImplementedError("Subclasses must implement optimize_weights")
    
    def _validate_inputs(self, expected_returns: np.ndarray, 
                        covariance_matrix: np.ndarray) -> None:
        """Validate optimization inputs."""
        n_assets = len(expected_returns)
        
        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError("Covariance matrix dimensions don't match returns")
        
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(covariance_matrix)
        if np.any(eigenvals <= 0):
            self.logger.warning("Covariance matrix is not positive definite")


class GMVPortfolio(PortfolioOptimizer):
    """
    Global Minimum Variance Portfolio optimizer.
    
    Implements the GMV strategy as described in the LSTM-BEKK paper.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize GMV portfolio optimizer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.portfolio_config = self.config.get('portfolio', {})
        
    def optimize_weights(self, covariance_matrix: np.ndarray,
                        expected_returns: Optional[np.ndarray] = None,
                        constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimize GMV portfolio weights.
        
        Args:
            expected_returns: Not used for GMV (can be None)
            covariance_matrix: Covariance matrix forecast
            constraints: Additional constraints
            
        Returns:
            Optimal GMV weights
        """
        n_assets = covariance_matrix.shape[0]
        
        # Validate inputs
        self._validate_inputs(np.zeros(n_assets), covariance_matrix)
        
        # Setup optimization problem
        try:
            return self._solve_with_cvxpy(covariance_matrix, constraints)
        except Exception as e:
            self.logger.warning(f"CVXPY optimization failed: {e}. Trying scipy.")
            return self._solve_with_scipy(covariance_matrix, constraints)
    
    def _solve_with_cvxpy(self, covariance_matrix: np.ndarray, 
                         constraints: Optional[Dict] = None) -> np.ndarray:
        """Solve GMV optimization using CVXPY."""
        n_assets = covariance_matrix.shape[0]
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(weights, covariance_matrix))
        
        # Constraints
        constraint_list = [cp.sum(weights) == 1]  # Budget constraint
        
        # Position size constraints
        max_weight = self.portfolio_config.get('max_position_size', 0.5)
        min_weight = self.portfolio_config.get('min_position_size', -0.5)
        constraint_list.append(weights >= min_weight)
        constraint_list.append(weights <= max_weight)
        
        # Long-only constraint (common for GMV)
        if self.portfolio_config.get('long_only', True):
            constraint_list.append(weights >= 0)
        
        # Additional constraints
        if constraints:
            if 'sector_limits' in constraints:
                # Sector exposure limits (if sector mapping provided)
                sector_map = constraints['sector_limits']
                for sector, (assets, limit) in sector_map.items():
                    constraint_list.append(cp.sum(weights[assets]) <= limit)
            
            if 'turnover_limit' in constraints:
                # Turnover constraint (requires previous weights)
                prev_weights = constraints.get('previous_weights', np.zeros(n_assets))
                turnover_limit = constraints['turnover_limit']
                constraint_list.append(cp.norm(weights - prev_weights, 1) <= turnover_limit)
        
        # Solve problem
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            return weights.value
        else:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    def _solve_with_scipy(self, covariance_matrix: np.ndarray,
                         constraints: Optional[Dict] = None) -> np.ndarray:
        """Solve GMV optimization using scipy."""
        n_assets = covariance_matrix.shape[0]
        
        # Objective function
        def objective(weights):
            return weights.T @ covariance_matrix @ weights
        
        # Constraints
        constraint_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Budget
        
        # Bounds
        max_weight = self.portfolio_config.get('max_position_size', 0.5)
        min_weight = 0 if self.portfolio_config.get('long_only', True) else -0.5
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Solve
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraint_list)
        
        if result.success:
            return result.x
        else:
            raise RuntimeError(f"Scipy optimization failed: {result.message}")
    
    def calculate_portfolio_metrics(self, weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            
        Returns:
            Dictionary of metrics
        """
        portfolio_return = weights.T @ expected_returns
        portfolio_variance = weights.T @ covariance_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration metrics
        herfindahl_index = np.sum(weights ** 2)
        effective_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'herfindahl_index': herfindahl_index,
            'effective_assets': effective_assets,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'long_exposure': np.sum(weights[weights > 0]),
            'short_exposure': np.sum(weights[weights < 0])
        }


class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    Mean-Variance portfolio optimizer.
    """
    
    def __init__(self, risk_aversion: float = 1.0, config: Optional[Dict] = None):
        """
        Initialize mean-variance optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter
            config: Configuration dictionary
        """
        super().__init__(config)
        self.risk_aversion = risk_aversion
    
    def optimize_weights(self, covariance_matrix: np.ndarray,
                        expected_returns: np.ndarray,
                        constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimize mean-variance portfolio weights.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            constraints: Additional constraints
            
        Returns:
            Optimal weights
        """
        n_assets = len(expected_returns)
        
        # Validate inputs
        self._validate_inputs(expected_returns, covariance_matrix)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective: maximize utility = return - (risk_aversion/2) * variance
        portfolio_return = weights.T @ expected_returns
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
        
        objective = cp.Maximize(utility)
        
        # Constraints
        constraint_list = [cp.sum(weights) == 1]  # Budget constraint
        
        # Position limits
        portfolio_config = self.config.get('portfolio', {})
        max_weight = portfolio_config.get('max_position_size', 0.5)
        min_weight = portfolio_config.get('min_position_size', -0.5)
        constraint_list.append(weights >= min_weight)
        constraint_list.append(weights <= max_weight)
        
        # Long-only if specified
        if portfolio_config.get('long_only', False):
            constraint_list.append(weights >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            return weights.value
        else:
            raise RuntimeError(f"Mean-variance optimization failed: {problem.status}")


class RiskParityOptimizer(PortfolioOptimizer):
    """
    Risk Parity portfolio optimizer.
    """
    
    def optimize_weights(self, covariance_matrix: np.ndarray,
                        expected_returns: Optional[np.ndarray] = None,
                        constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimize risk parity portfolio weights.
        
        Args:
            expected_returns: Not used for risk parity
            covariance_matrix: Covariance matrix
            constraints: Additional constraints
            
        Returns:
            Risk parity weights
        """
        n_assets = covariance_matrix.shape[0]
        
        # Validate inputs
        self._validate_inputs(np.zeros(n_assets), covariance_matrix)
        
        # Use iterative algorithm for risk parity
        return self._solve_risk_parity(covariance_matrix)
    
    def _solve_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Solve risk parity using iterative algorithm."""
        n_assets = covariance_matrix.shape[0]
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative algorithm
        for _ in range(100):  # Max iterations
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Update weights to equalize risk contributions
            target_risk = np.sum(risk_contrib) / n_assets
            weights_new = weights * np.sqrt(target_risk / risk_contrib)
            weights_new = weights_new / np.sum(weights_new)  # Normalize
            
            # Check convergence
            if np.max(np.abs(weights_new - weights)) < 1e-6:
                break
            
            weights = weights_new
        
        return weights
