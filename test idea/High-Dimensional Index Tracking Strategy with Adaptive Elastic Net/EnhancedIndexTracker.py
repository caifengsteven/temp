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

class EnhancedIndexTracker(AdaptiveElasticNetIndexTracker):
    """
    Enhanced version of the Adaptive Elastic Net Index Tracker
    that aims to outperform the index rather than just track it.
    """
    
    def __init__(self, lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1, 
                 alpha_weight=0.5, momentum_weight=0.3, vol_weight=0.2):
        """
        Initialize the Enhanced Index Tracker
        
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
        alpha_weight : float
            Weight for the alpha component in the objective function
        momentum_weight : float
            Weight for the momentum component in the objective function
        vol_weight : float
            Weight for the volatility component in the objective function
        """
        super().__init__(lambda1, lambda2, lambda_c, tau)
        self.alpha_weight = alpha_weight
        self.momentum_weight = momentum_weight
        self.vol_weight = vol_weight
        self.alpha_scores = None
        self.momentum_scores = None
        self.volatility_scores = None
    
    def _calculate_alpha_scores(self, X, y):
        """
        Calculate alpha scores for each asset based on residual returns
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        alpha_scores : array-like
            Alpha scores for each asset
        """
        n, p = X.shape
        
        # Use the last 20% of the data for alpha calculation
        split_idx = int(n * 0.8)
        X_train, X_recent = X[:split_idx], X[split_idx:]
        y_train, y_recent = y[:split_idx], y[split_idx:]
        
        # Fit a market model for each asset
        alpha_scores = np.zeros(p)
        
        for j in range(p):
            # Simple market model: asset_return = alpha + beta * market_return + error
            model = LinearRegression()
            model.fit(y_train.reshape(-1, 1), X_train[:, j])
            
            # Predict expected returns based on market model
            expected_returns = model.predict(y_recent.reshape(-1, 1))
            
            # Calculate residual returns (actual - expected)
            residual_returns = X_recent[:, j] - expected_returns
            
            # Alpha score is the average residual return
            alpha_scores[j] = np.mean(residual_returns)
        
        # Normalize alpha scores
        alpha_scores = (alpha_scores - np.mean(alpha_scores)) / (np.std(alpha_scores) + 1e-8)
        
        return alpha_scores
    
    def _calculate_momentum_scores(self, X):
        """
        Calculate momentum scores based on recent performance
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
            
        Returns:
        --------
        momentum_scores : array-like
            Momentum scores for each asset
        """
        n, p = X.shape
        
        # Use different lookback periods for momentum
        lookback_periods = [20, 60, 120]  # Approximately 1, 3, and 6 months
        weights = [0.5, 0.3, 0.2]  # More weight to recent momentum
        
        momentum_scores = np.zeros(p)
        
        for period, weight in zip(lookback_periods, weights):
            if n > period:
                # Calculate cumulative returns over the period
                period_returns = np.mean(X[-period:, :], axis=0)
                momentum_scores += weight * period_returns
        
        # Normalize momentum scores
        momentum_scores = (momentum_scores - np.mean(momentum_scores)) / (np.std(momentum_scores) + 1e-8)
        
        return momentum_scores
    
    def _calculate_volatility_scores(self, X):
        """
        Calculate volatility scores (lower volatility is better)
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
            
        Returns:
        --------
        volatility_scores : array-like
            Volatility scores for each asset (negative, so lower volatility gets higher score)
        """
        n, p = X.shape
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(X, axis=0)
        
        # Invert and normalize (lower volatility gets higher score)
        volatility_scores = -1 * volatility
        volatility_scores = (volatility_scores - np.mean(volatility_scores)) / (np.std(volatility_scores) + 1e-8)
        
        return volatility_scores
    
    def _calculate_combined_scores(self, X, y):
        """
        Calculate combined alpha, momentum, and volatility scores
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        combined_scores : array-like
            Combined scores for each asset
        """
        # Calculate individual scores
        self.alpha_scores = self._calculate_alpha_scores(X, y)
        self.momentum_scores = self._calculate_momentum_scores(X)
        self.volatility_scores = self._calculate_volatility_scores(X)
        
        # Combine scores using weights
        combined_scores = (
            self.alpha_weight * self.alpha_scores +
            self.momentum_weight * self.momentum_scores +
            self.vol_weight * self.volatility_scores
        )
        
        return combined_scores
    
    def _enhanced_coordinate_descent(self, X, y, v_hat, combined_scores, w_prev=None, max_iter=1000, tol=1e-6):
        """
        Enhanced coordinate descent algorithm that incorporates alpha signals
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        v_hat : array-like
            Adaptive weights
        combined_scores : array-like
            Combined alpha, momentum, and volatility scores
        w_prev : array-like, optional
            Previous weights for turnover penalty
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
            
        Returns:
        --------
        w : array-like
            Optimal weights
        """
        n, p = X.shape
        
        # Initialize weights
        w = np.ones(p) / p
        
        # If no previous weights, use equal weights
        if w_prev is None:
            w_prev = np.zeros(p)
        
        # Precompute X'X and X'y
        XtX = X.T @ X / n
        Xty = X.T @ y / n
        
        # Precompute squared sum of each column in X
        col_squared_sum = np.sum(X**2, axis=0) / n
        
        # Initialize gamma0 (Lagrange multiplier for full investment)
        gamma0 = v_hat.max() * self.lambda1 + self.lambda_c + 0.1
        
        # Scale combined scores to be in a similar range as dj
        score_scale = np.mean(np.abs(Xty)) / np.mean(np.abs(combined_scores))
        scaled_scores = combined_scores * score_scale
        
        converged = False
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # Update each weight
            for j in range(p):
                # Calculate dj (partial residual)
                Xj = X[:, j]
                y_partial = y - X @ w + Xj * w[j]
                dj = Xj.T @ y_partial / n
                
                # Add alpha component to dj
                enhanced_dj = dj + scaled_scores[j]
                
                # Case logic from paper, but with enhanced_dj
                if w_prev[j] < (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2):
                    w[j] = (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2)
                elif (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 - self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2) <= w_prev[j] <= (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2):
                    w[j] = w_prev[j]
                elif 0 <= (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2) < w_prev[j]:
                    w[j] = (enhanced_dj + gamma0 - v_hat[j] * self.lambda1 + self.lambda_c) / (col_squared_sum[j] + 2 * self.lambda2)
                else:
                    w[j] = 0
            
            # Identify sets Su, Sm, Sl for gamma0 update
            Su = np.where((w > w_prev) & (w_prev >= 0))[0]
            Sm = np.where((w == w_prev) & (w_prev > 0))[0]
            Sl = np.where((0 < w) & (w < w_prev))[0]
            
            # Update gamma0
            denominator = np.sum(1 / (col_squared_sum[Su] + 2 * self.lambda2)) + np.sum(1 / (col_squared_sum[Sl] + 2 * self.lambda2))
            if denominator > 0:
                gamma0_numerator = 1 - np.sum(w[Sm])
                gamma0_numerator -= np.sum((enhanced_dj - v_hat[j] * self.lambda1) / (col_squared_sum[j] + 2 * self.lambda2) for j in np.concatenate((Su, Sl)))
                gamma0_numerator -= np.sum(self.lambda_c / (col_squared_sum[j] + 2 * self.lambda2) for j in Sl)
                gamma0_numerator += np.sum(self.lambda_c / (col_squared_sum[j] + 2 * self.lambda2) for j in Su)
                
                gamma0 = gamma0_numerator / denominator
            
            # Project to ensure full investment and no-short selling
            w = np.maximum(w, 0)
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = np.ones(p) / p  # Equal weights if all weights are zero
            
            # Check convergence
            if np.linalg.norm(w - w_old) < tol:
                converged = True
                break
        
        if not converged:
            print(f"Warning: Enhanced coordinate descent did not converge after {max_iter} iterations")
        
        return w
    
    def fit(self, X, y, w_prev=None):
        """
        Fit the enhanced index tracking model
        
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
        # Calculate adaptive weights (same as original)
        v_hat, beta_init = self._calculate_adaptive_weights(X, y)
        
        # Calculate combined alpha, momentum, and volatility scores
        combined_scores = self._calculate_combined_scores(X, y)
        
        # Run enhanced coordinate descent to find optimal weights
        self.optimal_weights = self._enhanced_coordinate_descent(X, y, v_hat, combined_scores, w_prev)
        
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
        if self.alpha_scores is None:
            return None
        
        # Calculate weighted average of factor scores
        alpha_exposure = np.sum(self.optimal_weights * self.alpha_scores)
        momentum_exposure = np.sum(self.optimal_weights * self.momentum_scores)
        volatility_exposure = np.sum(self.optimal_weights * self.volatility_scores)
        
        return {
            'alpha': alpha_exposure,
            'momentum': momentum_exposure,
            'volatility': volatility_exposure
        }
