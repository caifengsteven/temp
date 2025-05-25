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

class EnhancedIndexTrackerPositiveAlpha(AdaptiveElasticNetIndexTracker):
    """
    Enhanced version of the Adaptive Elastic Net Index Tracker
    that focuses specifically on positive alpha assets to outperform the index.
    """
    
    def __init__(self, lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1, 
                 index_weight=0.6, active_weight=0.4):
        """
        Initialize the Enhanced Index Tracker with Positive Alpha Focus
        
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
        """
        super().__init__(lambda1, lambda2, lambda_c, tau)
        self.index_weight = index_weight
        self.active_weight = active_weight
        self.alpha_values = None
        self.expected_returns = None
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
        expected_returns : array-like
            Expected returns for each asset
        betas : array-like
            Beta coefficients for each asset
        """
        n, p = X.shape
        
        # Calculate alpha, beta, and expected returns for each asset
        alpha_values = np.zeros(p)
        expected_returns = np.zeros(p)
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
            
            # Calculate expected return
            expected_returns[j] = alpha + beta * np.mean(y)
        
        return alpha_values, expected_returns, betas
    
    def fit(self, X, y, w_prev=None):
        """
        Fit the enhanced index tracking model with positive alpha focus
        
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
        
        # Calculate alpha values for active component
        self.alpha_values, self.expected_returns, self.betas = self._calculate_alpha_values(X, y)
        
        # Create active component based on positive alpha
        active_weights = np.zeros_like(self.alpha_values)
        
        # Identify assets with positive alpha
        positive_alpha_assets = np.where(self.alpha_values > 0)[0]
        
        if len(positive_alpha_assets) > 0:
            # Weight proportional to alpha value
            active_weights[positive_alpha_assets] = self.alpha_values[positive_alpha_assets]
            
            # Ensure non-negative weights
            active_weights = np.maximum(active_weights, 0)
            
            # Normalize to sum to 1
            if np.sum(active_weights) > 0:
                active_weights = active_weights / np.sum(active_weights)
            else:
                # If no positive alpha, use equal weights for top assets by expected return
                top_assets = np.argsort(self.expected_returns)[-int(len(self.expected_returns)*0.2):]
                active_weights[top_assets] = 1.0 / len(top_assets)
        else:
            # If no positive alpha, use equal weights for top assets by expected return
            top_assets = np.argsort(self.expected_returns)[-int(len(self.expected_returns)*0.2):]
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
        
        # Calculate weighted average expected return
        expected_return = np.sum(self.optimal_weights * self.expected_returns)
        
        # Calculate active weight (sum of weights of positive alpha assets)
        positive_alpha_assets = np.where(self.alpha_values > 0)[0]
        active_weight = np.sum(self.optimal_weights[positive_alpha_assets])
        
        return {
            'alpha': alpha_exposure,
            'beta': beta_exposure,
            'expected_return': expected_return,
            'active_weight': active_weight
        }
