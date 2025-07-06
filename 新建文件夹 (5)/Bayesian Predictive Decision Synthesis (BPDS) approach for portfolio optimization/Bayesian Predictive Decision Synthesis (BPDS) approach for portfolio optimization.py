import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create simulated FX data for 9 currencies
def generate_simulated_fx_data(n_days=1000, n_assets=9, ar_coefs=None):
    """
    Generate simulated FX returns data with time-varying volatility.
    
    Parameters:
    - n_days: Number of days to simulate
    - n_assets: Number of assets (currencies)
    - ar_coefs: AR coefficients for each model (list of arrays)
    
    Returns:
    - df_returns: DataFrame of simulated returns
    - true_vols: True volatilities used in simulation
    """
    # Create date range
    start_date = datetime(2015, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Asset names (currencies)
    currencies = ['AUD', 'EUR', 'NZD', 'GBP', 'CAD', 'JPY', 'NOK', 'ZAR', 'CHF'][:n_assets]
    
    # Initialize returns and volatility matrices
    returns = np.zeros((n_days, n_assets))
    true_vols = np.zeros((n_days, n_assets))
    
    # Initial volatility
    vols = np.ones(n_assets) * 0.005  # Initial daily volatility (0.5%)
    
    # Create correlation matrix (fixed for simplicity)
    corr = np.eye(n_assets)
    # Add some realistic correlations
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Higher correlation between similar currency groups
            if i % 3 == j % 3:
                corr[i, j] = corr[j, i] = 0.7
            else:
                corr[i, j] = corr[j, i] = 0.3
    
    # Generate returns with AR(1) process and time-varying volatility
    for t in range(n_days):
        # Update volatility with some persistence and mean reversion
        vols = 0.95 * vols + 0.05 * (0.005 + 0.001 * np.random.randn(n_assets))
        vols = np.maximum(vols, 0.001)  # Ensure positive volatility
        
        # Create covariance matrix
        vol_matrix = np.diag(vols)
        cov = vol_matrix @ corr @ vol_matrix
        
        # Generate shocks
        shocks = np.random.multivariate_normal(np.zeros(n_assets), cov)
        
        # Add AR component if we're past the first day
        if t > 0:
            if ar_coefs is not None:
                # Add different AR components based on model
                ar_component = np.zeros(n_assets)
                returns[t, :] = ar_component + shocks
            else:
                # Simple AR(1) with coefficient 0.05
                returns[t, :] = 0.05 * returns[t-1, :] + shocks
        else:
            returns[t, :] = shocks
        
        true_vols[t, :] = vols
    
    # Convert to DataFrame
    df_returns = pd.DataFrame(returns, index=dates, columns=currencies)
    df_true_vols = pd.DataFrame(true_vols, index=dates, columns=currencies)
    
    return df_returns, df_true_vols

# Generate models with different AR orders and volatility discount factors
def create_model_set(returns_data, lookback=252):
    """
    Create a set of TVP-VAR models with different AR orders and discount factors.
    
    Parameters:
    - returns_data: DataFrame of asset returns
    - lookback: Lookback window for model estimation
    
    Returns:
    - models: List of model objects
    """
    class TVP_VAR_Model:
        def __init__(self, beta, p, r_star):
            """
            Initialize TVP-VAR model.
            
            Parameters:
            - beta: Volatility discount factor
            - p: AR order
            - r_star: Target return
            """
            self.beta = beta
            self.p = p
            self.r_star = r_star
            self.name = f"Model_beta{beta}_p{p}_r{r_star}"
            
            # Model parameters
            self.n_assets = returns_data.shape[1]
            self.coef = np.zeros((self.n_assets, self.n_assets * p))
            self.cov = np.eye(self.n_assets) * 0.0001
            
            # For cumulative performance tracking
            self.cumulative_return = 1.0
            self.daily_returns = []
            
        def update(self, t, window_size=lookback):
            """Update model parameters based on recent data"""
            if t < self.p:
                return
            
            # Get historical data window
            end_idx = t
            start_idx = max(0, end_idx - window_size)
            y = returns_data.iloc[start_idx:end_idx].values
            
            # Create design matrix with lagged returns
            X = np.zeros((len(y) - self.p, self.n_assets * self.p))
            Y = y[self.p:]
            
            for i in range(self.p):
                X[:, i*self.n_assets:(i+1)*self.n_assets] = y[self.p-i-1:-i-1]
            
            # Simple OLS estimation for coefficients
            try:
                self.coef = np.linalg.inv(X.T @ X) @ X.T @ Y
            except:
                # Fallback to ridge regression if matrix is singular
                ridge_lambda = 0.01
                self.coef = np.linalg.inv(X.T @ X + ridge_lambda * np.eye(X.shape[1])) @ X.T @ Y
            
            # Update covariance matrix with exponential weighting (discount factor)
            residuals = Y - X @ self.coef.T
            new_cov = residuals.T @ residuals / (len(residuals) - 1)
            self.cov = self.beta * self.cov + (1 - self.beta) * new_cov
            
        def forecast(self, t, horizon=1):
            """
            Generate forecast for t+horizon based on data up to t-1
            Returns mean and covariance of forecast
            """
            if t < self.p:
                # Not enough history, return zeros with high uncertainty
                return np.zeros(self.n_assets), np.eye(self.n_assets) * 0.01
            
            # Get last p observations
            last_obs = returns_data.iloc[t-self.p:t].values
            
            # Create forecast recursively for multi-step ahead
            forecast_mean = np.zeros(self.n_assets)
            forecast_path = np.zeros((horizon, self.n_assets))
            
            # First forecast step
            x = np.zeros(self.n_assets * self.p)
            for i in range(self.p):
                x[i*self.n_assets:(i+1)*self.n_assets] = last_obs[self.p-i-1]
            
            forecast_mean = self.coef @ x
            forecast_path[0] = forecast_mean
            
            # Multi-step forecasts if horizon > 1
            for h in range(1, horizon):
                # Shift lagged values
                for i in range(self.p-1, 0, -1):
                    x[i*self.n_assets:(i+1)*self.n_assets] = x[(i-1)*self.n_assets:i*self.n_assets]
                
                # Add latest forecast
                x[:self.n_assets] = forecast_path[h-1]
                
                # Generate new forecast
                forecast_path[h] = self.coef @ x
            
            # For simplicity, we assume the covariance grows linearly with horizon
            forecast_cov = self.cov * horizon
            
            if horizon == 1:
                return forecast_mean, forecast_cov
            else:
                # Return the forecast for the full horizon (h-day return)
                h_day_mean = np.sum(forecast_path, axis=0)
                return h_day_mean, forecast_cov * horizon
            
        def optimal_portfolio(self, forecast_mean, forecast_cov):
            """
            Compute optimal portfolio weights using mean-variance optimization
            with target return constraint
            """
            n = len(forecast_mean)
            
            # Ensure the target return is achievable
            # If not, use the maximum possible expected return
            target_return = min(max(np.mean(forecast_mean), 1e-6), self.r_star)
            
            # Define objective function (minimize portfolio variance)
            def portfolio_variance(weights):
                return weights.T @ forecast_cov @ weights
            
            # Constraints: weights sum to 1 and expected return meets target
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
                {'type': 'eq', 'fun': lambda x: x.T @ forecast_mean - target_return}  # Target return
            ]
            
            # Bounds: allow short selling but with limits
            bounds = [(-0.5, 1.5) for _ in range(n)]
            
            # Initial guess: equal weights
            initial_weights = np.ones(n) / n
            
            # Solve optimization problem
            try:
                result = minimize(portfolio_variance, initial_weights, method='SLSQP', 
                                 bounds=bounds, constraints=constraints)
                weights = result.x
            except:
                # Fallback to equal weights if optimization fails
                weights = initial_weights
            
            return weights
    
    # Create models with different parameters
    models = []
    
    # AR orders
    ar_orders = [1, 2, 3]
    
    # Volatility discount factors
    betas = [0.94, 0.98, 0.995]
    
    # Target returns
    r_stars = [0.05/252, 0.10/252, 0.15/252]  # Daily target returns (annualized divided by 252)
    
    # Create all combinations
    for beta in betas:
        for p in ar_orders:
            for r_star in r_stars:
                models.append(TVP_VAR_Model(beta, p, r_star))
    
    return models

# Functions for Bayesian Model Averaging (BMA)
def calculate_model_likelihoods(models, actual_returns, t):
    """Calculate likelihood of each model given the actual returns at time t"""
    likelihoods = np.zeros(len(models))
    
    for i, model in enumerate(models):
        forecast_mean, forecast_cov = model.forecast(t)
        
        # Calculate multivariate normal likelihood
        try:
            inv_cov = np.linalg.inv(forecast_cov)
            diff = actual_returns - forecast_mean
            exponent = -0.5 * diff.T @ inv_cov @ diff
            det_cov = np.linalg.det(forecast_cov)
            
            if det_cov > 0:
                norm_const = 1.0 / np.sqrt((2 * np.pi) ** len(forecast_mean) * det_cov)
                likelihoods[i] = norm_const * np.exp(exponent)
            else:
                likelihoods[i] = 1e-10  # Small positive value for numerical stability
        except:
            likelihoods[i] = 1e-10
    
    # Normalize to get probabilities
    if np.sum(likelihoods) > 0:
        return likelihoods / np.sum(likelihoods)
    else:
        return np.ones(len(models)) / len(models)

def update_model_probabilities(prev_probs, likelihoods, alpha=0.8):
    """Update model probabilities using discounted BMA"""
    updated = prev_probs ** alpha * likelihoods
    return updated / np.sum(updated)

# Functions for Bayesian Predictive Decision Synthesis (BPDS)
def calculate_score(returns, weights, r_star):
    """Calculate bivariate score function (return, -variance/2)"""
    portfolio_return = weights @ returns
    excess_return = portfolio_return - r_star
    return np.array([portfolio_return, -0.5 * excess_return**2])

def calculate_score_moments(models, model_probs, t, horizon=1):
    """Calculate mean and covariance of the score distribution"""
    n_models = len(models)
    score_dim = 2
    
    # Initialize score moments
    score_mean = np.zeros(score_dim)
    score_cov = np.zeros((score_dim, score_dim))
    
    # Calculate score mean and intermediate values for covariance
    model_scores = []
    for j, model in enumerate(models):
        forecast_mean, forecast_cov = model.forecast(t, horizon)
        weights = model.optimal_portfolio(forecast_mean, forecast_cov)
        
        # Expected portfolio return
        exp_return = weights @ forecast_mean
        
        # Expected portfolio variance
        exp_var = weights @ forecast_cov @ weights
        
        # Target return for this model
        r_star = model.r_star
        
        # Expected excess return
        exp_excess = exp_return - r_star
        
        # Expected score for this model
        model_score = np.array([exp_return, -0.5 * (exp_excess**2 + exp_var)])
        model_scores.append(model_score)
        
        # Weighted contribution to overall score mean
        score_mean += model_probs[j] * model_score
    
    # Calculate score covariance
    for j in range(n_models):
        # Contribution from model-specific uncertainty
        # For simplicity, we're using a diagonal approximation here
        model_forecast_mean, model_forecast_cov = models[j].forecast(t, horizon)
        weights = models[j].optimal_portfolio(model_forecast_mean, model_forecast_cov)
        portfolio_var = weights @ model_forecast_cov @ weights
        
        # Model-specific score covariance (simplified)
        model_score_cov = np.zeros((score_dim, score_dim))
        model_score_cov[0, 0] = portfolio_var
        model_score_cov[1, 1] = 0.5 * portfolio_var  # Approximation for variance of squared term
        
        # Add weighted contribution to score covariance
        score_cov += model_probs[j] * model_score_cov
        
        # Add between-model variation
        diff = model_scores[j] - score_mean
        score_cov += model_probs[j] * np.outer(diff, diff)
    
    return score_mean, score_cov

def calculate_bpds_weights(score_mean, score_cov, target_improvement):
    """
    Calculate BPDS tilting vector and weights using eigenscore approach
    
    Parameters:
    - score_mean: Mean of score distribution
    - score_cov: Covariance of score distribution
    - target_improvement: Percent improvement for first score element (return)
    
    Returns:
    - lambda: Tilting vector
    - risk_tolerance: Implied risk tolerance (lambda[0]/lambda[1])
    """
    # Eigendecomposition of score covariance
    eigvals, eigvecs = np.linalg.eigh(score_cov)
    D = np.sqrt(np.maximum(eigvals, 1e-10))  # Ensure positive eigenvalues
    E = eigvecs
    
    # Calculate standardized target score
    gamma = target_improvement * np.abs(score_mean[0]) / D[0]
    m_tilde = np.array([gamma, gamma])
    
    # Calculate tilting vector (approximate)
    lambda_approx = np.zeros(2)
    lambda_approx[0] = gamma / D[0]
    lambda_approx[1] = gamma / D[1]
    
    # Calculate risk tolerance
    risk_tolerance = lambda_approx[0] / lambda_approx[1]
    
    return lambda_approx, risk_tolerance

def bpds_update_probabilities(models, initial_probs, t, target_improvement, c=1.0, horizon=1):
    """
    Update model probabilities using BPDS
    
    Parameters:
    - models: List of models
    - initial_probs: Initial model probabilities
    - t: Current time
    - target_improvement: Percent improvement for expected return
    - c: Scaling factor for second score dimension (risk)
    - horizon: Forecast horizon
    
    Returns:
    - updated_probs: Updated model probabilities
    - lambda_vec: Tilting vector
    - risk_tolerance: Implied risk tolerance
    """
    # Calculate score moments
    score_mean, score_cov = calculate_score_moments(models, initial_probs, t, horizon)
    
    # Calculate tilting vector using eigenscore approach
    lambda_vec, risk_tolerance = calculate_bpds_weights(score_mean, score_cov, target_improvement)
    
    # Apply scaling to second dimension
    lambda_vec[1] *= c
    
    # Calculate model-specific weights
    weights = np.zeros(len(models))
    for j, model in enumerate(models):
        # Get model forecast
        forecast_mean, forecast_cov = model.forecast(t, horizon)
        
        # Get optimal portfolio
        portfolio_weights = model.optimal_portfolio(forecast_mean, forecast_cov)
        
        # Expected score for this model
        exp_return = portfolio_weights @ forecast_mean
        exp_var = portfolio_weights @ forecast_cov @ portfolio_weights
        r_star = model.r_star
        exp_excess = exp_return - r_star
        model_score = np.array([exp_return, -0.5 * (exp_excess**2 + exp_var)])
        
        # Calculate weight
        weights[j] = np.exp(lambda_vec @ model_score)
    
    # Calculate updated probabilities
    updated_probs = initial_probs * weights
    if np.sum(updated_probs) > 0:
        updated_probs = updated_probs / np.sum(updated_probs)
    else:
        # If numerical issues, fall back to initial probs
        updated_probs = initial_probs
    
    return updated_probs, lambda_vec, risk_tolerance

# Portfolio optimization and evaluation functions
def markowitz_portfolio(means, cov, target_return):
    """
    Calculate Markowitz optimal portfolio with target return constraint
    """
    n = len(means)
    
    # Ensure the target return is achievable
    target_return = min(max(np.mean(means), 1e-6), target_return)
    
    # Define objective function (minimize portfolio variance)
    def portfolio_variance(weights):
        return weights.T @ cov @ weights
    
    # Constraints: weights sum to 1 and expected return meets target
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        {'type': 'eq', 'fun': lambda x: x.T @ means - target_return}  # Target return
    ]
    
    # Bounds: allow short selling but with limits
    bounds = [(-0.5, 1.5) for _ in range(n)]
    
    # Initial guess: equal weights
    initial_weights = np.ones(n) / n
    
    # Solve optimization problem
    try:
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        weights = result.x
    except:
        # Fallback to equal weights if optimization fails
        weights = initial_weights
    
    return weights

def calculate_portfolio_performance(returns, weights):
    """Calculate portfolio return, variance, and Sharpe ratio"""
    port_return = weights @ returns
    
    # Annualize for Sharpe ratio (assuming daily returns)
    annual_return = port_return * 252
    annual_vol = np.sqrt(252 * np.var(port_return))
    
    if annual_vol > 0:
        sharpe_ratio = annual_return / annual_vol
    else:
        sharpe_ratio = 0
        
    return port_return, annual_vol, sharpe_ratio

# Main simulation function
def run_simulation(n_days=1000, test_start=500, n_assets=9):
    """
    Run a complete simulation to compare BMA and BPDS portfolio strategies
    
    Parameters:
    - n_days: Total number of simulated days
    - test_start: Day to start testing strategies (after model training)
    - n_assets: Number of assets
    
    Returns:
    - results: Dictionary with simulation results
    """
    # Generate simulated data
    returns_data, true_vols = generate_simulated_fx_data(n_days, n_assets)
    
    # Create model set
    models = create_model_set(returns_data)
    print(f"Created {len(models)} models")
    
    # Divide data into training and testing periods
    train_data = returns_data.iloc[:test_start]
    test_data = returns_data.iloc[test_start:]
    
    # Initialize model parameters on training data
    for t in range(test_start):
        for model in models:
            model.update(t)
    
    # Initialize tracking variables
    n_test_days = len(test_data)
    n_models = len(models)
    
    # Initialize arrays to store results
    bma_probs = np.ones(n_models) / n_models  # Start with equal probabilities
    bpds_probs_1day = np.ones(n_models) / n_models
    bpds_probs_5day = np.ones(n_models) / n_models
    
    # Portfolio returns
    bma_returns = np.zeros(n_test_days)
    bma2_returns = np.zeros(n_test_days)  # BMA with doubled target
    bpds_returns = np.zeros((6, n_test_days))  # For different target improvements
    bpds_5day_returns = np.zeros((6, n_test_days))  # For 5-day BPDS portfolios
    
    # Target improvements to test
    target_improvements = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    
    # Risk tolerance constraint
    d_star_1day = 0.05 / 252  # Daily equivalent of 5% annual
    d_star_5day = 0.05 * 5 / 252  # 5-day equivalent
    
    # Scaling factors for risk dimension (to be calibrated)
    c_1day = 7.4  # As used in the paper
    c_5day = 3.25  # As used in the paper
    
    # Run test period simulation
    for t in range(test_start, n_days):
        test_idx = t - test_start
        actual_returns = returns_data.iloc[t].values
        
        # Update models with new data
        for model in models:
            model.update(t)
        
        # Calculate model likelihoods
        likelihoods = calculate_model_likelihoods(models, actual_returns, t)
        
        # Update BMA probabilities
        bma_probs = update_model_probabilities(bma_probs, likelihoods)
        
        # Calculate BMA forecast and portfolio
        bma_mean = np.zeros(n_assets)
        bma_cov = np.zeros((n_assets, n_assets))
        bma_target = 0
        
        for j, model in enumerate(models):
            model_mean, model_cov = model.forecast(t+1)
            bma_mean += bma_probs[j] * model_mean
            bma_cov += bma_probs[j] * model_cov
            bma_target += bma_probs[j] * model.r_star
        
        # BMA portfolio
        bma_weights = markowitz_portfolio(bma_mean, bma_cov, bma_target)
        
        # BMA2 portfolio (doubled target)
        bma2_weights = markowitz_portfolio(bma_mean, bma_cov, min(2 * bma_target, 0.01))
        
        # Calculate BMA portfolio returns
        bma_returns[test_idx] = bma_weights @ actual_returns
        bma2_returns[test_idx] = bma2_weights @ actual_returns
        
        # Update BPDS probabilities and calculate portfolios for different target improvements
        for i, improvement in enumerate(target_improvements):
            # 1-day BPDS
            bpds_probs_1day, lambda_1day, risk_tolerance_1day = bpds_update_probabilities(
                models, bma_probs, t+1, improvement, c_1day, horizon=1)
            
            # Calculate BPDS forecast
            bpds_mean = np.zeros(n_assets)
            bpds_cov = np.zeros((n_assets, n_assets))
            bpds_target = 0
            
            for j, model in enumerate(models):
                model_mean, model_cov = model.forecast(t+1)
                bpds_mean += bpds_probs_1day[j] * model_mean
                bpds_cov += bpds_probs_1day[j] * model_cov
                bpds_target += bpds_probs_1day[j] * model.r_star
            
            # BPDS portfolio with constrained risk tolerance
            constrained_risk_tol = min(risk_tolerance_1day, d_star_1day)
            bpds_weights = markowitz_portfolio(bpds_mean, bpds_cov, bpds_target + constrained_risk_tol)
            
            # Calculate BPDS portfolio return
            bpds_returns[i, test_idx] = bpds_weights @ actual_returns
            
            # 5-day BPDS
            bpds_probs_5day, lambda_5day, risk_tolerance_5day = bpds_update_probabilities(
                models, bma_probs, t+1, improvement, c_5day, horizon=5)
            
            # Calculate 5-day BPDS forecast
            bpds_5day_mean = np.zeros(n_assets)
            bpds_5day_cov = np.zeros((n_assets, n_assets))
            bpds_5day_target = 0
            
            for j, model in enumerate(models):
                model_mean, model_cov = model.forecast(t+1, horizon=5)
                bpds_5day_mean += bpds_probs_5day[j] * model_mean
                bpds_5day_cov += bpds_probs_5day[j] * model_cov
                bpds_5day_target += bpds_probs_5day[j] * model.r_star
            
            # 5-day BPDS portfolio with constrained risk tolerance
            constrained_risk_tol_5day = min(risk_tolerance_5day, d_star_5day)
            bpds_5day_weights = markowitz_portfolio(bpds_5day_mean, bpds_5day_cov, bpds_5day_target + constrained_risk_tol_5day)
            
            # Calculate 5-day BPDS portfolio return (still evaluated on 1-day returns)
            bpds_5day_returns[i, test_idx] = bpds_5day_weights @ actual_returns
        
        # Print progress
        if test_idx % 100 == 0:
            print(f"Completed {test_idx} of {n_test_days} test days")
    
    # Calculate cumulative returns
    bma_cum_returns = np.cumprod(1 + bma_returns) - 1
    bma2_cum_returns = np.cumprod(1 + bma2_returns) - 1
    
    bpds_cum_returns = np.zeros_like(bpds_returns)
    bpds_5day_cum_returns = np.zeros_like(bpds_5day_returns)
    
    for i in range(len(target_improvements)):
        bpds_cum_returns[i] = np.cumprod(1 + bpds_returns[i]) - 1
        bpds_5day_cum_returns[i] = np.cumprod(1 + bpds_5day_returns[i]) - 1
    
    # Calculate Sharpe ratios
    bma_sharpe = np.sqrt(252) * np.mean(bma_returns) / np.std(bma_returns) if np.std(bma_returns) > 0 else 0
    bma2_sharpe = np.sqrt(252) * np.mean(bma2_returns) / np.std(bma2_returns) if np.std(bma2_returns) > 0 else 0
    
    bpds_sharpes = np.zeros(len(target_improvements))
    bpds_5day_sharpes = np.zeros(len(target_improvements))
    
    for i in range(len(target_improvements)):
        bpds_sharpes[i] = np.sqrt(252) * np.mean(bpds_returns[i]) / np.std(bpds_returns[i]) if np.std(bpds_returns[i]) > 0 else 0
        bpds_5day_sharpes[i] = np.sqrt(252) * np.mean(bpds_5day_returns[i]) / np.std(bpds_5day_returns[i]) if np.std(bpds_5day_returns[i]) > 0 else 0
    
    # Compile results
    results = {
        'returns_data': returns_data,
        'bma_returns': bma_returns,
        'bma_cum_returns': bma_cum_returns,
        'bma_sharpe': bma_sharpe,
        'bma2_returns': bma2_returns,
        'bma2_cum_returns': bma2_cum_returns,
        'bma2_sharpe': bma2_sharpe,
        'bpds_returns': bpds_returns,
        'bpds_cum_returns': bpds_cum_returns,
        'bpds_sharpes': bpds_sharpes,
        'bpds_5day_returns': bpds_5day_returns,
        'bpds_5day_cum_returns': bpds_5day_cum_returns,
        'bpds_5day_sharpes': bpds_5day_sharpes,
        'target_improvements': target_improvements
    }
    
    return results

# Plot results
def plot_results(results):
    """Plot cumulative returns and Sharpe ratios"""
    target_improvements = results['target_improvements']
    
    # Plot 1-day portfolio cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(results['bma_cum_returns'], label=f'BMA (SR={results["bma_sharpe"]:.2f})')
    plt.plot(results['bma2_cum_returns'], label=f'BMA2 (SR={results["bma2_sharpe"]:.2f})')
    
    for i, imp in enumerate(target_improvements):
        plt.plot(results['bpds_cum_returns'][i], 
                label=f'BPDS γ={imp} (SR={results["bpds_sharpes"][i]:.2f})')
    
    plt.title('Cumulative Returns - 1-day Portfolios')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 5-day portfolio cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(results['bma_cum_returns'], label=f'BMA (SR={results["bma_sharpe"]:.2f})')
    plt.plot(results['bma2_cum_returns'], label=f'BMA2 (SR={results["bma2_sharpe"]:.2f})')
    
    for i, imp in enumerate(target_improvements):
        plt.plot(results['bpds_5day_cum_returns'][i], 
                label=f'BPDS 5-day γ={imp} (SR={results["bpds_5day_sharpes"][i]:.2f})')
    
    plt.title('Cumulative Returns - 5-day Portfolios')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Sharpe ratios comparison
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(target_improvements))
    width = 0.35
    
    plt.bar(x - width/2, results['bpds_sharpes'], width, label='BPDS 1-day')
    plt.bar(x + width/2, results['bpds_5day_sharpes'], width, label='BPDS 5-day')
    
    plt.axhline(y=results['bma_sharpe'], color='r', linestyle='-', label='BMA')
    plt.axhline(y=results['bma2_sharpe'], color='g', linestyle='--', label='BMA2')
    
    plt.xlabel('Target Improvement γ')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')
    plt.xticks(x, [str(imp) for imp in target_improvements])
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()

# Run simulation
np.random.seed(42)
results = run_simulation(n_days=1000, test_start=500, n_assets=9)
plot_results(results)