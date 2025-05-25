import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import the original tracker
import importlib.util
import sys

# Load the module from the file
spec = importlib.util.spec_from_file_location(
    "index_tracking", 
    "High-Dimensional Index Tracking Strategy with Adaptive Elastic Net.py"
)
index_tracking = importlib.util.module_from_spec(spec)
sys.modules["index_tracking"] = index_tracking
spec.loader.exec_module(index_tracking)

# Access the original tracker class
AdaptiveElasticNetIndexTracker = index_tracking.AdaptiveElasticNetIndexTracker

class EnhancedIndexTrackerOptimized(AdaptiveElasticNetIndexTracker):
    """
    Final optimized version of the Enhanced Index Tracker
    that aims to outperform the index by focusing on high-quality assets.
    """
    
    def __init__(self, lambda1=1e-5, lambda2=1e-2, lambda_c=1e-4, tau=1, 
                 index_weight=0.5, active_weight=0.5,
                 quality_threshold=0.0):
        """
        Initialize the Optimized Enhanced Index Tracker
        
        Parameters:
        -----------
        lambda1 : float
            Regularization parameter for L1 penalty
        lambda2 : float
            Regularization parameter for L2 penalty
        lambda_c : float
            Regularization parameter for turnover penalty
        tau : float
            Power parameter for adaptive weights
        index_weight : float
            Weight for the index-tracking component
        active_weight : float
            Weight for the active component
        quality_threshold : float
            Minimum quality score for assets to be included in active component
        """
        super().__init__(lambda1, lambda2, lambda_c, tau)
        self.index_weight = index_weight
        self.active_weight = active_weight
        self.quality_threshold = quality_threshold
        self.alpha_values = None
        self.momentum_values = None
        self.quality_scores = None
        self.betas = None
    
    def _calculate_alpha_values(self, X, y):
        """
        Calculate alpha values for each asset
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        alpha_values : array-like
            Alpha values for each asset
        betas : array-like
            Beta coefficients for each asset
        """
        n, p = X.shape
        
        # Calculate alpha and beta for each asset
        alpha_values = np.zeros(p)
        betas = np.zeros(p)
        
        for j in range(p):
            # Fit market model: asset_return = alpha + beta * market_return + error
            model = LinearRegression()
            model.fit(y.reshape(-1, 1), X[:, j])
            
            # Extract alpha and beta
            alpha = model.intercept_
            beta = model.coef_[0]
            
            alpha_values[j] = alpha
            betas[j] = beta
        
        return alpha_values, betas
    
    def _calculate_momentum_values(self, X):
        """
        Calculate momentum values for each asset
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
            
        Returns:
        --------
        momentum_values : array-like
            Momentum values for each asset
        """
        n, p = X.shape
        
        # Use different lookback periods for momentum
        lookback_periods = [20, 60, 120]  # Short, medium, and long-term momentum
        weights = [0.5, 0.3, 0.2]  # More weight to recent momentum
        
        momentum_values = np.zeros(p)
        
        for period, weight in zip(lookback_periods, weights):
            if n > period:
                # Calculate cumulative returns over the period
                period_returns = np.sum(X[-period:, :], axis=0)
                momentum_values += weight * period_returns
        
        return momentum_values
    
    def _calculate_quality_scores(self, X, y, alpha_values, momentum_values, betas):
        """
        Calculate quality scores for each asset
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        alpha_values : array-like
            Alpha values for each asset
        momentum_values : array-like
            Momentum values for each asset
        betas : array-like
            Beta coefficients for each asset
            
        Returns:
        --------
        quality_scores : array-like
            Quality scores for each asset
        """
        n, p = X.shape
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(X, axis=0)
        
        # Calculate Sharpe-like ratio (alpha / volatility)
        sharpe_ratio = alpha_values / (volatility + 1e-8)
        
        # Calculate downside deviation (std of negative returns only)
        downside_returns = np.copy(X)
        downside_returns[downside_returns > 0] = 0
        downside_deviation = np.std(downside_returns, axis=0)
        
        # Calculate Sortino-like ratio (alpha / downside deviation)
        sortino_ratio = alpha_values / (downside_deviation + 1e-8)
        
        # Normalize components
        norm_alpha = (alpha_values - np.mean(alpha_values)) / (np.std(alpha_values) + 1e-8)
        norm_momentum = (momentum_values - np.mean(momentum_values)) / (np.std(momentum_values) + 1e-8)
        norm_sharpe = (sharpe_ratio - np.mean(sharpe_ratio)) / (np.std(sharpe_ratio) + 1e-8)
        norm_sortino = (sortino_ratio - np.mean(sortino_ratio)) / (np.std(sortino_ratio) + 1e-8)
        
        # Calculate beta deviation from 1.0
        beta_deviation = np.abs(betas - 1.0)
        norm_beta = -(beta_deviation - np.mean(beta_deviation)) / (np.std(beta_deviation) + 1e-8)
        
        # Combine components into quality score
        quality_scores = (
            0.3 * norm_alpha +
            0.2 * norm_momentum +
            0.2 * norm_sharpe +
            0.2 * norm_sortino +
            0.1 * norm_beta
        )
        
        return quality_scores
    
    def fit(self, X, y, w_prev=None):
        """
        Fit the optimized enhanced index tracking model
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        w_prev : array-like, optional
            Previous weights for turnover penalty
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Calculate adaptive weights for index tracking (same as original)
        v_hat, beta_init = self._calculate_adaptive_weights(X, y)
        
        # Run original coordinate descent to find index-tracking weights
        index_tracking_weights = self._coordinate_descent(X, y, v_hat, w_prev)
        
        # Calculate alpha, momentum, and quality scores
        self.alpha_values, self.betas = self._calculate_alpha_values(X, y)
        self.momentum_values = self._calculate_momentum_values(X)
        self.quality_scores = self._calculate_quality_scores(
            X, y, self.alpha_values, self.momentum_values, self.betas
        )
        
        # Create active component based on quality scores
        active_weights = np.zeros_like(self.quality_scores)
        
        # Identify high-quality assets
        high_quality_assets = np.where(self.quality_scores > self.quality_threshold)[0]
        
        if len(high_quality_assets) > 0:
            # Weight proportional to quality score
            quality_weights = np.maximum(self.quality_scores[high_quality_assets], 0)
            
            if np.sum(quality_weights) > 0:
                active_weights[high_quality_assets] = quality_weights
                active_weights = active_weights / np.sum(active_weights)
            else:
                # If no positive quality scores, use equal weights for top assets by alpha
                top_assets = np.argsort(self.alpha_values)[-int(len(self.alpha_values)*0.2):]
                active_weights[top_assets] = 1.0 / len(top_assets)
        else:
            # If no high-quality assets, use equal weights for top assets by alpha
            top_assets = np.argsort(self.alpha_values)[-int(len(self.alpha_values)*0.2):]
            active_weights[top_assets] = 1.0 / len(top_assets)
        
        # Combine index-tracking and active components
        self.optimal_weights = self.index_weight * index_tracking_weights + self.active_weight * active_weights
        
        # Ensure full investment and no-short selling
        self.optimal_weights = np.maximum(self.optimal_weights, 0)
        if np.sum(self.optimal_weights) > 0:
            self.optimal_weights = self.optimal_weights / np.sum(self.optimal_weights)
        else:
            self.optimal_weights = np.ones_like(self.optimal_weights) / len(self.optimal_weights)
        
        # Identify active assets (non-zero weights)
        self.active_assets = np.where(self.optimal_weights > 1e-6)[0]
        
        # Calculate in-sample tracking error
        y_pred = X @ self.optimal_weights
        self.tracking_error = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return self
    
    def get_factor_exposures(self):
        """
        Return the factor exposures of the portfolio
        
        Returns:
        --------
        exposures : dict
            Dictionary of factor exposures
        """
        if self.alpha_values is None:
            return None
        
        # Calculate weighted average alpha
        alpha_exposure = np.sum(self.optimal_weights * self.alpha_values)
        
        # Calculate weighted average beta
        beta_exposure = np.sum(self.optimal_weights * self.betas)
        
        # Calculate weighted average momentum
        momentum_exposure = np.sum(self.optimal_weights * self.momentum_values)
        
        # Calculate weighted average quality score
        quality_exposure = np.sum(self.optimal_weights * self.quality_scores)
        
        # Calculate active weight (sum of weights of high-quality assets)
        high_quality_assets = np.where(self.quality_scores > self.quality_threshold)[0]
        active_weight = np.sum(self.optimal_weights[high_quality_assets])
        
        return {
            'alpha': alpha_exposure,
            'beta': beta_exposure,
            'momentum': momentum_exposure,
            'quality': quality_exposure,
            'active_weight': active_weight
        }
