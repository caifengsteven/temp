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

class EnhancedIndexTrackerFinal(AdaptiveElasticNetIndexTracker):
    """
    Final enhanced version of the Adaptive Elastic Net Index Tracker
    that aims to outperform the index rather than just track it.
    
    Key improvements:
    1. Direct optimization for outperformance
    2. Predictive alpha model
    3. Aggressive tilting toward high-alpha assets
    4. Dynamic risk management
    5. Adaptive factor weighting
    """
    
    def __init__(self, lambda1=1e-5, lambda2=1e-3, lambda_c=0, tau=1, 
                 alpha_weight=1.0, momentum_weight=0.7, vol_weight=0.5,
                 target_active_return=0.03, max_tracking_error=0.04,
                 min_active_weight=0.4):
        """
        Initialize the Enhanced Index Tracker Final
        
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
        target_active_return : float
            Target annualized active return
        max_tracking_error : float
            Maximum acceptable tracking error
        min_active_weight : float
            Minimum weight to allocate to active bets
        """
        super().__init__(lambda1, lambda2, lambda_c, tau)
        self.alpha_weight = alpha_weight
        self.momentum_weight = momentum_weight
        self.vol_weight = vol_weight
        self.target_active_return = target_active_return
        self.max_tracking_error = max_tracking_error
        self.min_active_weight = min_active_weight
        self.alpha_scores = None
        self.momentum_scores = None
        self.volatility_scores = None
        self.sector_scores = None
        self.expected_returns = None
        self.betas = None
    
    def _calculate_alpha_scores(self, X, y):
        """
        Calculate improved alpha scores with predictive components
        
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
        
        # Use multiple lookback periods for alpha calculation
        lookback_periods = [
            int(n * 0.25),  # 25% of data
            int(n * 0.5),   # 50% of data
            int(n * 0.75)   # 75% of data
        ]
        weights = [0.5, 0.3, 0.2]  # More weight to recent data
        
        alpha_scores = np.zeros(p)
        
        for period, weight in zip(lookback_periods, weights):
            if n > period:
                # Use the most recent data for this period
                X_period = X[-period:, :]
                y_period = y[-period:]
                
                # Calculate residual returns for each asset
                period_alphas = np.zeros(p)
                
                for j in range(p):
                    # Simple market model: asset_return = alpha + beta * market_return + error
                    model = LinearRegression()
                    model.fit(y_period.reshape(-1, 1), X_period[:, j])
                    
                    # Extract alpha and beta
                    alpha = model.intercept_
                    beta = model.coef_[0]
                    
                    # Predict expected returns based on market model
                    expected_returns = model.predict(y_period.reshape(-1, 1))
                    
                    # Calculate residual returns (actual - expected)
                    residual_returns = X_period[:, j] - expected_returns
                    
                    # Calculate trend in residual returns
                    if len(residual_returns) > 20:
                        recent_residuals = residual_returns[-20:]
                        trend = np.polyfit(np.arange(len(recent_residuals)), recent_residuals, 1)[0]
                        
                        # Add trend component to alpha
                        alpha += trend * 10  # Scale up the trend effect
                    
                    # Alpha score is the intercept plus trend component
                    period_alphas[j] = alpha
                
                # Add weighted alpha scores
                alpha_scores += weight * period_alphas
        
        # Normalize alpha scores
        alpha_scores = (alpha_scores - np.mean(alpha_scores)) / (np.std(alpha_scores) + 1e-8)
        
        return alpha_scores
    
    def _calculate_momentum_scores(self, X):
        """
        Calculate improved momentum scores with multiple timeframes
        
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
        lookback_periods = [10, 20, 60, 120, 250]  # Multiple timeframes
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # More weight to recent momentum
        
        momentum_scores = np.zeros(p)
        
        for period, weight in zip(lookback_periods, weights):
            if n > period:
                # Calculate cumulative returns over the period
                period_returns = np.sum(X[-period:, :], axis=0)
                momentum_scores += weight * period_returns
        
        # Normalize momentum scores
        momentum_scores = (momentum_scores - np.mean(momentum_scores)) / (np.std(momentum_scores) + 1e-8)
        
        return momentum_scores
    
    def _calculate_volatility_scores(self, X):
        """
        Calculate improved volatility scores with risk-adjusted returns
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
            
        Returns:
        --------
        volatility_scores : array-like
            Volatility scores for each asset
        """
        n, p = X.shape
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(X, axis=0)
        
        # Calculate mean returns
        mean_returns = np.mean(X, axis=0)
        
        # Calculate Sharpe-like ratio (mean return / volatility)
        sharpe_ratio = mean_returns / (volatility + 1e-8)
        
        # Calculate downside deviation (std of negative returns only)
        downside_returns = np.copy(X)
        downside_returns[downside_returns > 0] = 0
        downside_deviation = np.std(downside_returns, axis=0)
        
        # Calculate Sortino-like ratio (mean return / downside deviation)
        sortino_ratio = mean_returns / (downside_deviation + 1e-8)
        
        # Combine Sharpe and Sortino ratios
        combined_ratio = 0.5 * sharpe_ratio + 0.5 * sortino_ratio
        
        # Normalize combined ratio
        volatility_scores = (combined_ratio - np.mean(combined_ratio)) / (np.std(combined_ratio) + 1e-8)
        
        return volatility_scores
    
    def _simulate_sectors(self, X, num_sectors=10):
        """
        Simulate sector assignments for assets based on return correlations
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        num_sectors : int
            Number of sectors to simulate
            
        Returns:
        --------
        sector_assignments : array-like
            Sector assignment for each asset
        """
        n, p = X.shape
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Simple clustering: assign assets to sectors based on correlation with random seeds
        np.random.seed(42)  # For reproducibility
        sector_seeds = np.random.choice(p, size=num_sectors, replace=False)
        sector_assignments = np.zeros(p, dtype=int)
        
        for i in range(p):
            # Find the sector seed with highest correlation
            correlations = corr_matrix[i, sector_seeds]
            sector_assignments[i] = np.argmax(correlations)
        
        return sector_assignments
    
    def _calculate_sector_scores(self, X, y):
        """
        Calculate sector rotation scores with momentum and mean-reversion
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        sector_scores : array-like
            Sector rotation scores for each asset
        """
        n, p = X.shape
        
        # Simulate sector assignments
        sector_assignments = self._simulate_sectors(X)
        num_sectors = len(np.unique(sector_assignments))
        
        # Calculate sector performance
        sector_returns = np.zeros((n, num_sectors))
        for i in range(num_sectors):
            sector_assets = np.where(sector_assignments == i)[0]
            if len(sector_assets) > 0:
                sector_returns[:, i] = np.mean(X[:, sector_assets], axis=1)
        
        # Calculate sector momentum (last 60 days)
        momentum_lookback = min(60, n)
        sector_momentum = np.sum(sector_returns[-momentum_lookback:, :], axis=0)
        
        # Calculate sector mean-reversion (last 250 days)
        reversion_lookback = min(250, n)
        if reversion_lookback > 0:
            # Calculate z-scores of cumulative returns
            cumulative_returns = np.sum(sector_returns[-reversion_lookback:, :], axis=0)
            mean_return = np.mean(cumulative_returns)
            std_return = np.std(cumulative_returns) + 1e-8
            z_scores = (cumulative_returns - mean_return) / std_return
            
            # Mean-reversion score (negative z-score)
            mean_reversion = -z_scores
        else:
            mean_reversion = np.zeros(num_sectors)
        
        # Combine momentum and mean-reversion with adaptive weighting
        # If momentum is strong, give it more weight
        momentum_strength = np.abs(np.corrcoef(
            np.arange(momentum_lookback), 
            np.mean(sector_returns[-momentum_lookback:, :], axis=1)
        )[0, 1])
        
        momentum_weight = 0.7 if momentum_strength > 0.5 else 0.3
        reversion_weight = 1.0 - momentum_weight
        
        sector_scores_by_sector = momentum_weight * sector_momentum + reversion_weight * mean_reversion
        
        # Normalize sector scores
        sector_scores_by_sector = (sector_scores_by_sector - np.mean(sector_scores_by_sector)) / (np.std(sector_scores_by_sector) + 1e-8)
        
        # Assign sector scores to assets
        sector_scores = np.zeros(p)
        for i in range(p):
            sector_scores[i] = sector_scores_by_sector[sector_assignments[i]]
        
        return sector_scores
    
    def _calculate_expected_returns(self, X, y):
        """
        Calculate expected returns for each asset with multiple models
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
            
        Returns:
        --------
        expected_returns : array-like
            Expected returns for each asset
        betas : array-like
            Beta coefficients for each asset
        """
        n, p = X.shape
        
        # Use the last 20% of data for expected return calculation
        split_idx = int(n * 0.8)
        X_train, X_recent = X[:split_idx], X[split_idx:]
        y_train, y_recent = y[:split_idx], y[split_idx:]
        
        # Calculate expected returns using market model
        expected_returns = np.zeros(p)
        betas = np.zeros(p)
        
        for j in range(p):
            # Fit market model
            model = LinearRegression()
            model.fit(y_train.reshape(-1, 1), X_train[:, j])
            
            # Extract alpha and beta
            alpha = model.intercept_
            beta = model.coef_[0]
            betas[j] = beta
            
            # Calculate expected return based on market model
            expected_returns[j] = alpha + beta * np.mean(y_recent)
        
        return expected_returns, betas
    
    def _calculate_combined_scores(self, X, y):
        """
        Calculate combined alpha, momentum, volatility, and sector scores
        with adaptive weighting
        
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
        self.sector_scores = self._calculate_sector_scores(X, y)
        
        # Calculate expected returns and betas
        self.expected_returns, self.betas = self._calculate_expected_returns(X, y)
        
        # Adaptive weighting based on market conditions
        # If momentum is strong, increase its weight
        momentum_strength = np.abs(np.corrcoef(
            np.arange(min(60, len(y))), 
            y[-min(60, len(y)):]
        )[0, 1])
        
        # Adjust weights based on market conditions
        if momentum_strength > 0.6:  # Strong trend
            alpha_w = self.alpha_weight * 0.8
            momentum_w = self.momentum_weight * 1.2
            vol_w = self.vol_weight
            sector_w = 0.4
        elif momentum_strength < 0.2:  # Weak trend/mean-reversion
            alpha_w = self.alpha_weight * 1.2
            momentum_w = self.momentum_weight * 0.6
            vol_w = self.vol_weight * 1.2
            sector_w = 0.6
        else:  # Moderate trend
            alpha_w = self.alpha_weight
            momentum_w = self.momentum_weight
            vol_w = self.vol_weight
            sector_w = 0.5
        
        # Combine scores using adaptive weights
        combined_scores = (
            alpha_w * self.alpha_scores +
            momentum_w * self.momentum_scores +
            vol_w * self.volatility_scores +
            sector_w * self.sector_scores
        )
        
        # Adjust for beta - penalize high beta assets
        beta_penalty = (self.betas - 1.0) * 0.3
        combined_scores -= beta_penalty
        
        return combined_scores
    
    def _enhanced_coordinate_descent_final(self, X, y, v_hat, combined_scores, w_prev=None, max_iter=1000, tol=1e-6):
        """
        Final enhanced coordinate descent algorithm with direct optimization for outperformance
        
        Parameters:
        -----------
        X : array-like
            Asset returns matrix
        y : array-like
            Index returns vector
        v_hat : array-like
            Adaptive weights
        combined_scores : array-like
            Combined alpha, momentum, volatility, and sector scores
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
        scaled_scores = combined_scores * score_scale * 2.0  # Double the impact
        
        # Boost top assets - increase their scores
        top_assets = np.argsort(combined_scores)[-int(p*0.2):]  # Top 20%
        scaled_scores[top_assets] *= 2.0  # Boost by 100%
        
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
        
        # Post-processing: Create a more aggressive active portfolio
        # Split into index-tracking and active components
        index_weight = 1.0 - self.min_active_weight
        active_weight = self.min_active_weight
        
        # Create index-tracking component (use beta_init from adaptive weights calculation)
        index_component = np.zeros(p)
        if 'beta_init' in locals():
            index_component = beta_init
        else:
            # If beta_init not available, use a simple approximation
            index_component = np.ones(p) / p
        
        # Create active component based on combined scores
        active_component = np.zeros(p)
        top_active_assets = np.argsort(combined_scores)[-int(p*0.2):]  # Top 20%
        active_component[top_active_assets] = combined_scores[top_active_assets]
        active_component = np.maximum(active_component, 0)  # Ensure non-negative
        if np.sum(active_component) > 0:
            active_component = active_component / np.sum(active_component)
        else:
            active_component = np.zeros(p)
            active_component[top_active_assets] = 1.0 / len(top_active_assets)
        
        # Combine components
        w_final = index_weight * w + active_weight * active_component
        
        # Ensure full investment and no-short selling
        w_final = np.maximum(w_final, 0)
        if np.sum(w_final) > 0:
            w_final = w_final / np.sum(w_final)
        else:
            w_final = np.ones(p) / p
        
        return w_final
    
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
        
        # Calculate combined alpha, momentum, volatility, and sector scores
        combined_scores = self._calculate_combined_scores(X, y)
        
        # Run enhanced coordinate descent to find optimal weights
        self.optimal_weights = self._enhanced_coordinate_descent_final(X, y, v_hat, combined_scores, w_prev)
        
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
        sector_exposure = np.sum(self.optimal_weights * self.sector_scores)
        
        # Calculate beta exposure
        if self.betas is not None:
            beta_exposure = np.sum(self.optimal_weights * self.betas)
        else:
            beta_exposure = 1.0
        
        return {
            'alpha': alpha_exposure,
            'momentum': momentum_exposure,
            'volatility': volatility_exposure,
            'sector': sector_exposure,
            'beta': beta_exposure
        }
