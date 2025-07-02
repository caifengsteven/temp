import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import time
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------
# Phase 1: Data Simulation
# ------------------------------------------------------------

def simulate_financial_data(N_ASSETS=10, N_DAYS_TOTAL=5040, DT=1/252, 
                           S0=100, mu=None, sigma=None, corr_matrix=None,
                           cosine_term_coef=0.3, plot=False):
    """
    Simulate multi-asset price paths with improved dynamics.
    
    Parameters:
    -----------
    N_ASSETS: int
        Number of assets
    N_DAYS_TOTAL: int
        Total number of days to simulate
    DT: float
        Time step (1/252 for daily data with 252 trading days per year)
    S0: float or array
        Initial price(s) for all assets
    mu: array
        Drift parameters for each asset
    sigma: array
        Volatility parameters for each asset
    corr_matrix: array
        Correlation matrix for Brownian motions
    cosine_term_coef: float
        Coefficient for the cosine term (0.3 in the paper)
    plot: bool
        Whether to plot the simulated paths
        
    Returns:
    --------
    S: array (N_DAYS_TOTAL x N_ASSETS)
        Simulated price paths
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns calculated from prices
    """
    print("Simulating financial data with improved dynamics...")
    
    # Initialize parameters with positive bias
    if mu is None:
        mu = np.random.uniform(0.08, 0.18, N_ASSETS)  # More positive drift (8-18% vs 5-15%)
    
    if sigma is None:
        sigma = np.random.uniform(0.08, 0.25, N_ASSETS)  # Slightly lower minimum volatility
    
    if corr_matrix is None:
        # Create a more realistic correlation structure
        # Base correlation level between assets
        base_corr = 0.3
        # Start with a base correlation matrix
        corr_matrix = np.ones((N_ASSETS, N_ASSETS)) * base_corr
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Add some sector-like structure by creating blocks of higher correlation
        sector_size = 3  # Each "sector" has 3 assets
        for i in range(0, N_ASSETS, sector_size):
            end_idx = min(i + sector_size, N_ASSETS)
            corr_matrix[i:end_idx, i:end_idx] = 0.7  # Higher intra-sector correlation
    
    # Ensure S0 is an array
    if isinstance(S0, (int, float)):
        S0 = np.ones(N_ASSETS) * S0
    
    # Compute Cholesky decomposition of correlation matrix
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        print("Correlation matrix is not positive definite. Adjusting...")
        corr_matrix = corr_matrix + 1e-6 * np.eye(N_ASSETS)
        L = np.linalg.cholesky(corr_matrix)
    
    # Initialize price paths
    S = np.zeros((N_DAYS_TOTAL, N_ASSETS))
    S[0] = S0
    
    # Add a trending component to drift
    trend = np.zeros((N_DAYS_TOTAL, N_ASSETS))
    # Generate slow-moving trends for each asset
    for j in range(N_ASSETS):
        # Random walk with mean reversion for the trend
        trend[0, j] = 0
        for t in range(1, N_DAYS_TOTAL):
            trend[t, j] = 0.98 * trend[t-1, j] + 0.02 * np.random.normal(0, 0.01)
    
    # Simulate paths with both cosine term and trend
    for t in range(1, N_DAYS_TOTAL):
        # Generate independent normal increments
        dZ = np.sqrt(DT) * np.random.normal(0, 1, N_ASSETS)
        
        # Generate correlated Brownian increments
        dW = L @ dZ
        
        # Compute cosine term (market-wide effect)
        cosine_term = np.cos(cosine_term_coef * np.sum(S[t-1])/N_ASSETS/S0[0])
        
        # Update prices with both cosine term and trend component
        drift_t = mu * (cosine_term + trend[t])
        S[t] = S[t-1] + S[t-1] * drift_t * DT + S[t-1] * sigma * dW
    
    # Calculate log returns
    log_returns = np.log(S[1:]) - np.log(S[:-1])
    
    # True drift at each time point
    true_drift = np.zeros((N_DAYS_TOTAL, N_ASSETS))
    for t in range(N_DAYS_TOTAL):
        cosine_t = np.cos(cosine_term_coef * np.sum(S[t])/N_ASSETS/S0[0])
        true_drift[t] = mu * (cosine_t + trend[t])
    
    if plot:
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 2, 1)
        for i in range(min(N_ASSETS, 5)):
            plt.plot(S[:, i], label=f'Asset {i+1}')
        plt.title('Simulated Price Paths')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for i in range(min(N_ASSETS, 5)):
            plt.plot(log_returns[:, i], label=f'Asset {i+1}')
        plt.title('Log Returns')
        plt.xlabel('Days')
        plt.ylabel('Log Return')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for i in range(min(N_ASSETS, 5)):
            plt.plot(true_drift[:, i], label=f'Asset {i+1}')
        plt.title('True Drift')
        plt.xlabel('Days')
        plt.ylabel('Drift')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar()
        plt.title('Asset Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
    
    print(f"Simulated {N_DAYS_TOTAL} days of data for {N_ASSETS} assets with enhanced dynamics")
    return S, log_returns, true_drift[1:]  # Return drift from t=1 to match log_returns

# ------------------------------------------------------------
# Phase 2: Randomized Signature Method
# ------------------------------------------------------------

def compute_randomized_signature(log_returns_window, A_matrices, b_vectors, R_initial=None, 
                                sigma_act=np.tanh, dt_internal=1):
    """
    Compute the randomized signature for a window of log returns.
    
    Parameters:
    -----------
    log_returns_window: array (tw x n)
        Window of log returns for n assets over tw time steps
    A_matrices: list of arrays
        Random projection matrices for each input dimension
    b_vectors: list of arrays
        Random bias vectors for each input dimension
    R_initial: array
        Initial reservoir state (if None, random initialization)
    sigma_act: function
        Activation function
    dt_internal: float
        Time increment for dX_0t
        
    Returns:
    --------
    R_final: array
        Final reservoir state (randomized signature)
    """
    tw, n = log_returns_window.shape
    rd = A_matrices[0].shape[0]  # Reservoir dimension
    
    # Initialize reservoir state if not provided
    if R_initial is None:
        R_initial = np.random.normal(0, 1, rd)
    
    # Current reservoir state
    R_current = R_initial.copy()
    
    # Iterate through the time steps
    for k in range(tw):
        # Vector of increments for this step [dt_internal, LR_1,k, ..., LR_n,k]
        dX_k = np.zeros(n + 1)
        dX_k[0] = dt_internal
        dX_k[1:] = log_returns_window[k]
        
        # Initialize sum term
        sum_term = np.zeros(rd)
        
        # Compute sum term
        for i in range(n + 1):
            # sigma_act(A_i @ R_current + b_i) * dX_k[i]
            sum_term += sigma_act(A_matrices[i] @ R_current + b_vectors[i]) * dX_k[i]
        
        # Update reservoir state
        R_current = R_current + sum_term
    
    return R_current


def generate_random_matrices(d_input, rd, rm, rv):
    """
    Generate random matrices and biases for randomized signature.
    
    Parameters:
    -----------
    d_input: int
        Dimension of input path (including time)
    rd: int
        Reservoir dimension
    rm: float
        Mean for random projection matrices
    rv: float
        Variance for random projection matrices
        
    Returns:
    --------
    A_matrices: list of arrays
        Random projection matrices
    b_vectors: list of arrays
        Random bias vectors
    """
    A_matrices = []
    b_vectors = []
    
    for i in range(d_input + 1):  # +1 for time
        A_i = np.random.normal(rm, np.sqrt(rv), (rd, rd))
        b_i = np.random.normal(0, 1, rd)
        A_matrices.append(A_i)
        b_vectors.append(b_i)
    
    return A_matrices, b_vectors


def generate_randomized_signatures(log_returns, tw, rd, rm, rv, sigma_act=np.tanh, 
                                   normalize=True):
    """
    Generate randomized signature features for all time windows.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
    rd: int
        Reservoir dimension
    rm: float
        Mean for random projection matrices
    rv: float
        Variance for random projection matrices
    sigma_act: function
        Activation function
    normalize: bool
        Whether to normalize the input window
        
    Returns:
    --------
    signatures: array ((N_DAYS_TOTAL-tw) x rd)
        Randomized signature features
    """
    print("Generating randomized signatures...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Generate random matrices and biases
    A_matrices, b_vectors = generate_random_matrices(N_ASSETS, rd, rm, rv)
    
    # Initialize signatures array
    signatures = np.zeros((N_DAYS - tw + 1, rd))
    
    # Generate signatures for each window
    for t in range(tw - 1, N_DAYS):
        # Extract window
        window = log_returns[t - tw + 1:t + 1]
        
        # Normalize window if specified
        if normalize:
            scaler = StandardScaler()
            window = scaler.fit_transform(window)
        
        # Compute signature
        R_t = compute_randomized_signature(window, A_matrices, b_vectors)
        
        # Store signature
        signatures[t - tw + 1] = R_t
    
    print(f"Generated {signatures.shape[0]} randomized signatures")
    return signatures


# ------------------------------------------------------------
# Phase 3: Drift Estimation via Ridge Regression
# ------------------------------------------------------------

def estimate_drift(signatures, log_returns, tw, ts, r_alpha=1e-3):
    """
    Estimate drift using ridge regression on randomized signatures.
    
    Parameters:
    -----------
    signatures: array ((N_DAYS_TOTAL-tw) x rd)
        Randomized signature features
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
    ts: int
        Burn-in period
    r_alpha: float
        Ridge regression regularization parameter
        
    Returns:
    --------
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Estimated drift for each asset
    """
    print("Estimating drift using ridge regression...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Initialize drift estimates array
    drift_estimates = np.zeros((N_DAYS - tw - ts, N_ASSETS))
    
    # Expanding window regression for drift estimation
    for t_idx, t_pred in enumerate(range(ts + tw - 1, N_DAYS - 1)):
        # Extract training data - FIXED INDEXING
        # The signatures start at the tw-th day, so we need to adjust accordingly
        signature_idx_end = t_pred - tw + 2  # +1 for next signature, +1 for inclusive upper bound
        
        features_train = signatures[:signature_idx_end]  # signatures up to t_pred
        
        # The target returns need to be aligned with the signatures
        # A signature at index i corresponds to a window ending at day tw+i-1
        # We want the return on the following day, which is tw+i
        targets_idx_start = tw
        targets_idx_end = t_pred + 1  # +1 for inclusive upper bound
        
        targets_train = log_returns[targets_idx_start:targets_idx_end]
        
        # Make sure features and targets have the same number of samples
        if features_train.shape[0] != targets_train.shape[0]:
            print(f"Shape mismatch at t_pred={t_pred}: features={features_train.shape}, targets={targets_train.shape}")
            # Trim to the same length
            min_samples = min(features_train.shape[0], targets_train.shape[0])
            features_train = features_train[:min_samples]
            targets_train = targets_train[:min_samples]
        
        # Initialize models for each asset
        models = []
        
        # Fit ridge regression for each asset
        for j in range(N_ASSETS):
            model_j = Ridge(alpha=r_alpha)
            model_j.fit(features_train, targets_train[:, j])
            models.append(model_j)
        
        # Predict drift for t_pred+1
        for j in range(N_ASSETS):
            drift_estimates[t_idx, j] = models[j].predict(signatures[signature_idx_end-1].reshape(1, -1))[0]
    
    print(f"Estimated drift for {drift_estimates.shape[0]} days")
    return drift_estimates


def estimate_drift_linear(log_returns, tw, ts, r_alpha=1e-3):
    """
    Estimate drift using linear regression directly on past returns.
    This serves as a benchmark for the randomized signature method.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
    ts: int
        Burn-in period
    r_alpha: float
        Ridge regression regularization parameter
        
    Returns:
    --------
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Estimated drift for each asset
    """
    print("Estimating drift using linear regression (benchmark)...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Initialize drift estimates array
    drift_estimates = np.zeros((N_DAYS - tw - ts, N_ASSETS))
    
    # Expanding window regression for drift estimation
    for t_idx, t_pred in enumerate(range(ts + tw - 1, N_DAYS - 1)):
        # For each asset
        for j in range(N_ASSETS):
            # Extract training data - flatten past windows
            X_train = []
            y_train = []
            
            for i in range(tw, t_pred + 1):
                X_train.append(log_returns[i - tw:i, j])
                y_train.append(log_returns[i, j])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Fit linear regression
            model = Ridge(alpha=r_alpha)
            model.fit(X_train, y_train)
            
            # Predict drift
            X_pred = log_returns[t_pred - tw + 1:t_pred + 1, j].reshape(1, -1)
            drift_estimates[t_idx, j] = model.predict(X_pred)[0]
    
    print(f"Estimated drift using linear regression for {drift_estimates.shape[0]} days")
    return drift_estimates


def estimate_drift_momentum(log_returns, tw, ts):
    """
    Estimate drift using simple momentum (average of past returns).
    This serves as another benchmark.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
    ts: int
        Burn-in period
        
    Returns:
    --------
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Estimated drift for each asset
    """
    print("Estimating drift using momentum (benchmark)...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Initialize drift estimates array
    drift_estimates = np.zeros((N_DAYS - tw - ts, N_ASSETS))
    
    # Compute rolling mean for each asset
    for t_idx, t_pred in enumerate(range(ts + tw - 1, N_DAYS - 1)):
        for j in range(N_ASSETS):
            drift_estimates[t_idx, j] = np.mean(log_returns[t_pred - tw + 1:t_pred + 1, j])
    
    print(f"Estimated drift using momentum for {drift_estimates.shape[0]} days")
    return drift_estimates


# ------------------------------------------------------------
# Phase 4: Covariance Estimation
# ------------------------------------------------------------

def estimate_covariance(log_returns, tc, ts, tw):
    """
    Estimate covariance matrix using Ledoit-Wolf shrinkage.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tc: int
        Window size for covariance estimation
    ts: int
        Burn-in period
    tw: int
        Rolling window size for input path
        
    Returns:
    --------
    covariance_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS x N_ASSETS)
        Estimated covariance matrices
    """
    print("Estimating covariance matrices...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Initialize covariance estimates array
    covariance_estimates = np.zeros((N_DAYS - tw - ts, N_ASSETS, N_ASSETS))
    
    # Compute rolling covariance matrix for each day
    for t_idx, t_pred in enumerate(range(ts + tw - 1, N_DAYS - 1)):
        # Extract window
        window = log_returns[max(0, t_pred - tc + 1):t_pred + 1]
        
        # Estimate covariance using Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        lw.fit(window)
        covariance_estimates[t_idx] = lw.covariance_
    
    print(f"Estimated covariance matrices for {covariance_estimates.shape[0]} days")
    return covariance_estimates


# ------------------------------------------------------------
# Phase 5: Portfolio Optimization (Markowitz)
# ------------------------------------------------------------

def optimize_portfolio(mu_hat, sigma_hat, rf=0, max_weight=0.2):
    """
    Optimize portfolio with improved risk management.
    
    Parameters:
    -----------
    mu_hat: array (N_ASSETS)
        Estimated drift for each asset
    sigma_hat: array (N_ASSETS x N_ASSETS)
        Estimated covariance matrix
    rf: float
        Risk-free rate
    max_weight: float
        Maximum weight for any asset
        
    Returns:
    --------
    optimal_weights: array (N_ASSETS)
        Optimal portfolio weights
    """
    N_ASSETS = len(mu_hat)
    
    # Risk-adjusted returns - shrink extreme estimates
    mu_mean = np.mean(mu_hat)
    mu_std = np.std(mu_hat)
    # Shrink extreme values toward the mean
    mu_hat_adj = mu_mean + 0.7 * (mu_hat - mu_mean)  # 30% shrinkage to the mean
    # Clip at 3 standard deviations
    mu_hat_adj = np.clip(mu_hat_adj, mu_mean - 3*mu_std, mu_mean + 3*mu_std)
    
    # Improve numerical stability of covariance matrix
    sigma_hat_adj = sigma_hat + 1e-6 * np.eye(N_ASSETS)
    
    # Define objective function (negative Sharpe ratio)
    def objective(weights, mu, Sigma, rf):
        portfolio_return = np.dot(weights, mu)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
        if portfolio_volatility == 0:
            return -np.inf  # Avoid division by zero
        return -(portfolio_return - rf) / portfolio_volatility  # Minimize negative Sharpe
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum of weights = 1
    
    # Bounds
    bounds = tuple((0, max_weight) for _ in range(N_ASSETS))  # No short selling, max weight
    
    # Initial guess
    initial_weights = np.ones(N_ASSETS) / N_ASSETS  # Equal weights
    
    # Optimize with multiple starting points to avoid local minima
    best_result = None
    best_sharpe = -np.inf
    
    for _ in range(3):  # Try 3 different starting points
        try:
            # Random perturbation to initial weights
            perturbed_weights = initial_weights + np.random.normal(0, 0.05, N_ASSETS)
            perturbed_weights = np.clip(perturbed_weights, 0, max_weight)
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)  # Normalize
            
            result = minimize(
                objective,
                perturbed_weights,
                args=(mu_hat_adj, sigma_hat_adj, rf),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            # Calculate Sharpe ratio
            portfolio_return = np.dot(result.x, mu_hat_adj)
            portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(sigma_hat_adj, result.x)))
            sharpe_ratio = (portfolio_return - rf) / portfolio_volatility if portfolio_volatility > 0 else -np.inf
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_result = result
        except:
            continue
    
    # If optimization fails, return risk-weighted allocation
    if best_result is None or not best_result.success:
        print("Portfolio optimization failed. Using risk-weighted allocation.")
        # Calculate inverse volatility weights
        vols = np.sqrt(np.diag(sigma_hat_adj))
        inv_vols = 1/vols
        weights = inv_vols / np.sum(inv_vols)
        
        # If expected returns are mostly negative, put more weight on assets with less negative returns
        if np.mean(mu_hat_adj) < 0:
            # Rank-based weighting for negative return environment
            ranks = np.argsort(np.argsort(mu_hat_adj))  # Double argsort to get ranks
            rank_weights = ranks + 1  # Add 1 to avoid zero weights
            rank_weights = rank_weights / np.sum(rank_weights)
            
            # Combine risk and return ranks
            weights = 0.7 * weights + 0.3 * rank_weights
        
        # Ensure weight constraints
        weights = np.clip(weights, 0, max_weight)
        weights = weights / np.sum(weights)
        return weights
    
    return best_result.x

def calculate_optimal_weights(drift_estimates, covariance_estimates, rf=0, max_weight=0.2):
    """
    Calculate optimal portfolio weights for each day.
    
    Parameters:
    -----------
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Estimated drift for each asset
    covariance_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS x N_ASSETS)
        Estimated covariance matrices
    rf: float
        Risk-free rate
    max_weight: float
        Maximum weight for any asset
        
    Returns:
    --------
    optimal_weights: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Optimal portfolio weights for each day
    """
    print("Calculating optimal portfolio weights...")
    
    N_DAYS = drift_estimates.shape[0]
    N_ASSETS = drift_estimates.shape[1]
    
    # Initialize optimal weights array
    optimal_weights = np.zeros((N_DAYS, N_ASSETS))
    
    # Calculate optimal weights for each day
    for t in tqdm(range(N_DAYS)):
        optimal_weights[t] = optimize_portfolio(
            drift_estimates[t],
            covariance_estimates[t],
            rf,
            max_weight
        )
    
    print(f"Calculated optimal weights for {N_DAYS} days")
    return optimal_weights


def calculate_equal_weights(N_DAYS, N_ASSETS):
    """
    Calculate equal weights (1/n) for each day.
    
    Parameters:
    -----------
    N_DAYS: int
        Number of days
    N_ASSETS: int
        Number of assets
        
    Returns:
    --------
    equal_weights: array (N_DAYS x N_ASSETS)
        Equal weights for each day
    """
    print("Calculating equal weights (1/n)...")
    
    # Initialize weights array
    equal_weights = np.ones((N_DAYS, N_ASSETS)) / N_ASSETS
    
    print(f"Calculated equal weights for {N_DAYS} days")
    return equal_weights


# ------------------------------------------------------------
# Phase 6: Performance Evaluation
# ------------------------------------------------------------

def calculate_portfolio_returns(weights, log_returns, ts, tw):
    """
    Calculate portfolio returns.
    
    Parameters:
    -----------
    weights: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Portfolio weights
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    ts: int
        Burn-in period
    tw: int
        Rolling window size for input path
        
    Returns:
    --------
    portfolio_returns: array (N_DAYS_TOTAL-tw-ts-1)
        Portfolio returns
    """
    N_DAYS = weights.shape[0]
    returns = np.zeros(N_DAYS - 1)
    
    # Calculate portfolio returns
    for t in range(N_DAYS - 1):
        returns[t] = np.sum(weights[t] * log_returns[ts + tw + t])
    
    return returns


def calculate_performance_metrics(returns, days_per_year=252):
    """
    Calculate performance metrics.
    
    Parameters:
    -----------
    returns: array
        Portfolio returns
    days_per_year: int
        Number of trading days per year
        
    Returns:
    --------
    metrics: dict
        Performance metrics
    """
    # Annualized return
    ra = np.mean(returns) * days_per_year
    
    # Annualized volatility
    sigma_a = np.std(returns) * np.sqrt(days_per_year)
    
    # Annualized Sharpe ratio
    sr_a = ra / sigma_a if sigma_a > 0 else 0
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate monthly returns
    monthly_returns = []
    days_per_month = days_per_year // 12
    for i in range(0, len(returns), days_per_month):
        if i + days_per_month <= len(returns):
            monthly_return = np.prod(1 + returns[i:i + days_per_month]) - 1
            monthly_returns.append(monthly_return)
    
    monthly_returns = np.array(monthly_returns)
    
    return {
        'annualized_return': ra,
        'annualized_volatility': sigma_a,
        'annualized_sharpe_ratio': sr_a,
        'max_drawdown': max_drawdown,
        'monthly_returns': monthly_returns
    }


def calculate_information_coefficient(drift_estimates, true_returns, ts, tw, true_drift=None):
    """
    Calculate information coefficient (correlation between predicted and actual returns).
    
    Parameters:
    -----------
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Estimated drift for each asset
    true_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Actual log returns
    ts: int
        Burn-in period
    tw: int
        Rolling window size for input path
    true_drift: array ((N_DAYS_TOTAL-1) x N_ASSETS), optional
        True drift values (if available from simulation)
        
    Returns:
    --------
    ic: array (N_ASSETS)
        Information coefficient for each asset
    ic_drift: array (N_ASSETS), optional
        Information coefficient between estimated and true drift (if true_drift provided)
    """
    N_DAYS = drift_estimates.shape[0]
    N_ASSETS = drift_estimates.shape[1]
    
    # Initialize information coefficient arrays
    ic = np.zeros(N_ASSETS)
    ic_drift = np.zeros(N_ASSETS) if true_drift is not None else None
    
    # Calculate IC for each asset
    for j in range(N_ASSETS):
        # Correlation between predicted and actual returns
        returns_subset = true_returns[ts + tw:ts + tw + N_DAYS, j]
        if len(returns_subset) != len(drift_estimates[:, j]):
            # Make sure the lengths match
            min_len = min(len(returns_subset), len(drift_estimates[:, j]))
            returns_subset = returns_subset[:min_len]
            drift_subset = drift_estimates[:min_len, j]
            ic[j] = np.corrcoef(drift_subset, returns_subset)[0, 1]
        else:
            ic[j] = np.corrcoef(drift_estimates[:, j], returns_subset)[0, 1]
        
        # Correlation between predicted and true drift (if provided)
        if true_drift is not None:
            drift_true_subset = true_drift[ts + tw:ts + tw + N_DAYS, j]
            if len(drift_true_subset) != len(drift_estimates[:, j]):
                # Make sure the lengths match
                min_len = min(len(drift_true_subset), len(drift_estimates[:, j]))
                drift_true_subset = drift_true_subset[:min_len]
                drift_subset = drift_estimates[:min_len, j]
                ic_drift[j] = np.corrcoef(drift_subset, drift_true_subset)[0, 1]
            else:
                ic_drift[j] = np.corrcoef(drift_estimates[:, j], drift_true_subset)[0, 1]
    
    return ic, ic_drift


# ------------------------------------------------------------
# Phase 7: Transaction Costs
# ------------------------------------------------------------

def apply_transaction_costs(weights, prices, drift_estimates=None, lambda_tc=0.001, tau_threshold=0.01, k_smoothing=5):
    """
    Apply transaction costs with improved logic based on expected returns.
    
    Parameters:
    -----------
    weights: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Optimal portfolio weights
    prices: array ((N_DAYS_TOTAL) x N_ASSETS)
        Asset prices
    drift_estimates: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS), optional
        Estimated drift for each asset (for more intelligent trading decisions)
    lambda_tc: float
        Proportional transaction cost rate
    tau_threshold: float
        Threshold for change in predicted returns
    k_smoothing: int
        Number of days for moving average of shares
        
    Returns:
    --------
    adjusted_weights: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Adjusted portfolio weights
    transaction_costs: array (N_DAYS_TOTAL-tw-ts-1)
        Transaction costs
    """
    print("Applying improved transaction cost management...")
    
    N_DAYS = weights.shape[0]
    N_ASSETS = weights.shape[1]
    
    # Initialize portfolio value and shares
    P_0 = 1.0  # Initial portfolio value
    SH_0 = weights[0] * P_0 / prices[0]  # Initial shares
    
    # Initialize arrays
    P_t = np.zeros(N_DAYS)
    P_t[0] = P_0
    SH_t = np.zeros((N_DAYS, N_ASSETS))
    SH_t[0] = SH_0
    transaction_costs = np.zeros(N_DAYS - 1)
    
    # Apply transaction cost adjustments
    for t in range(1, N_DAYS):
        # Update portfolio value based on yesterday's shares and today's prices
        P_t_pre_trade = np.sum(SH_t[t-1] * prices[t])
        
        # Calculate target shares based on optimal weights
        SH_star_t = weights[t] * P_t_pre_trade / prices[t]
        
        # Calculate change in shares (as percentage of portfolio)
        change_pct = np.sum(np.abs(SH_star_t * prices[t] - SH_t[t-1] * prices[t])) / P_t_pre_trade
        
        # Intelligent trading decision: only trade if the expected return improvement 
        # outweighs transaction costs or if position drift is significant
        if drift_estimates is not None:
            # Estimated return of current portfolio
            current_ret = np.sum(SH_t[t-1] * prices[t] / P_t_pre_trade * drift_estimates[t-1])
            # Estimated return of target portfolio
            target_ret = np.sum(weights[t] * drift_estimates[t-1])
            
            # Only trade if expected return improvement exceeds transaction costs plus buffer
            expected_improvement = target_ret - current_ret
            tc_estimate = lambda_tc * change_pct
            
            # If the improvement doesn't justify costs, maintain current allocation
            if expected_improvement < tc_estimate * 1.5:  # 1.5x buffer for uncertainty
                SH_star_t = SH_t[t-1]
        else:
            # Fallback to threshold-based trading if drift estimates not provided
            if change_pct < tau_threshold:
                SH_star_t = SH_t[t-1]
        
        # Apply moving average smoothing with adaptive window size
        # Use shorter window in trending markets, longer in flat markets
        if drift_estimates is not None and t > 0 and np.abs(np.mean(drift_estimates[t-1])) > 0.01:
            # Trending market - less smoothing
            k = min(t + 1, max(2, k_smoothing - 2))
        else:
            # Flat market - more smoothing
            k = min(t + 1, k_smoothing)
            
        SH_double_star_k_t = np.mean(np.vstack([SH_star_t] + [SH_t[max(0, t-i)] for i in range(1, k)]), axis=0)
        
        # Renormalize to ensure full investment
        SH_final_t = SH_double_star_k_t * P_t_pre_trade / np.sum(SH_double_star_k_t * prices[t])
        
        # Calculate transaction costs
        TC_t = lambda_tc * np.sum(np.abs(SH_final_t * prices[t] - SH_t[t-1] * prices[t]))
        transaction_costs[t-1] = TC_t
        
        # Update portfolio value and shares
        P_t[t] = P_t_pre_trade - TC_t
        SH_t[t] = SH_final_t
    
    # Calculate adjusted weights
    adjusted_weights = np.zeros_like(weights)
    for t in range(N_DAYS):
        total_value = np.sum(SH_t[t] * prices[t])
        if total_value > 0:
            adjusted_weights[t] = SH_t[t] * prices[t] / total_value
        else:
            adjusted_weights[t] = weights[t]  # Fallback
    
    print(f"Applied transaction costs. Average cost: {np.mean(transaction_costs):.6f}")
    return adjusted_weights, transaction_costs


def calculate_tc_portfolio_returns(weights, log_returns, transaction_costs, ts, tw):
    """
    Calculate portfolio returns with transaction costs.
    
    Parameters:
    -----------
    weights: array ((N_DAYS_TOTAL-tw-ts) x N_ASSETS)
        Portfolio weights
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    transaction_costs: array (N_DAYS_TOTAL-tw-ts-1)
        Transaction costs
    ts: int
        Burn-in period
    tw: int
        Rolling window size for input path
        
    Returns:
    --------
    portfolio_returns: array (N_DAYS_TOTAL-tw-ts-1)
        Portfolio returns after transaction costs
    """
    N_DAYS = weights.shape[0]
    returns = np.zeros(N_DAYS - 1)
    
    # Calculate portfolio returns with transaction costs
    for t in range(N_DAYS - 1):
        gross_return = np.sum(weights[t] * log_returns[ts + tw + t])
        returns[t] = gross_return - transaction_costs[t]
    
    return returns


# ------------------------------------------------------------
# Phase 8: Randomized Signature with Additional Features
# ------------------------------------------------------------

def calculate_additional_features(log_returns, tw):
    """
    Calculate enhanced additional features for the randomized signature method.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
        
    Returns:
    --------
    additional_features: array ((N_DAYS_TOTAL-1) x N_FEATURES)
        Additional features
    """
    print("Calculating enhanced additional features...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Feature 1: Random portfolio returns with different horizons
    np.random.seed(42)
    w_rand_1 = np.random.random(N_ASSETS)
    w_rand_1 = w_rand_1 / np.sum(w_rand_1)
    w_rand_2 = np.random.random(N_ASSETS)
    w_rand_2 = w_rand_2 / np.sum(w_rand_2)
    
    # Short-term and long-term random portfolio returns
    rand_portfolio_returns_st = np.zeros(N_DAYS)
    rand_portfolio_returns_lt = np.zeros(N_DAYS)
    
    for t in range(N_DAYS):
        rand_portfolio_returns_st[t] = np.sum(w_rand_1 * log_returns[t])
        rand_portfolio_returns_lt[t] = np.sum(w_rand_2 * log_returns[t])
    
    # Feature 2: Mean return with multiple horizons
    mean_returns = np.mean(log_returns, axis=1)
    
    # Feature 3: Cross-sectional dispersion (max-min spread)
    dispersion = np.zeros(N_DAYS)
    for t in range(N_DAYS):
        dispersion[t] = np.max(log_returns[t]) - np.min(log_returns[t])
    
    # Feature 4: Rolling volatility of each asset at different horizons
    vol_stocks_short = np.zeros((N_DAYS, N_ASSETS))
    vol_stocks_long = np.zeros((N_DAYS, N_ASSETS))
    
    short_window = min(tw, 10)  # Short-term vol (10 days)
    long_window = tw            # Long-term vol (tw days)
    
    for t in range(long_window, N_DAYS):
        for j in range(N_ASSETS):
            vol_stocks_short[t, j] = np.std(log_returns[t-short_window:t, j])
            vol_stocks_long[t, j] = np.std(log_returns[t-long_window:t, j])
    
    # Fill initial values
    for j in range(N_ASSETS):
        vol_stocks_short[:long_window, j] = vol_stocks_short[long_window, j]
        vol_stocks_long[:long_window, j] = vol_stocks_long[long_window, j]
    
    # Feature 5: Rolling correlations (average pairwise correlation)
    avg_correlation = np.zeros(N_DAYS)
    for t in range(long_window, N_DAYS):
        corr_matrix_t = np.corrcoef(log_returns[t-long_window:t].T)
        # Get only the lower triangular part without diagonal
        mask = np.tril(np.ones_like(corr_matrix_t), -1).astype(bool)
        avg_correlation[t] = np.mean(corr_matrix_t[mask])
    
    # Fill initial values
    avg_correlation[:long_window] = avg_correlation[long_window]
    
    # Feature 6: Momentum signals at different horizons
    momentum_short = np.zeros((N_DAYS, N_ASSETS))
    momentum_medium = np.zeros((N_DAYS, N_ASSETS))
    momentum_long = np.zeros((N_DAYS, N_ASSETS))
    
    for t in range(tw, N_DAYS):
        for j in range(N_ASSETS):
            # 5-day momentum
            momentum_short[t, j] = np.sum(log_returns[t-min(5, t):t, j])
            # 10-day momentum
            momentum_medium[t, j] = np.sum(log_returns[t-min(10, t):t, j])
            # 20-day momentum
            momentum_long[t, j] = np.sum(log_returns[t-min(20, t):t, j])
    
    # Fill initial values
    momentum_short[:tw] = momentum_short[tw]
    momentum_medium[:tw] = momentum_medium[tw]
    momentum_long[:tw] = momentum_long[tw]
    
    # Feature 7: "Technical" indicators - RSI-like (Relative Strength Index)
    rsi = np.zeros((N_DAYS, N_ASSETS))
    for t in range(tw, N_DAYS):
        for j in range(N_ASSETS):
            gains = np.sum(np.maximum(log_returns[t-tw:t, j], 0))
            losses = np.sum(np.maximum(-log_returns[t-tw:t, j], 0))
            if losses == 0:
                rsi[t, j] = 100
            else:
                rs = gains / losses
                rsi[t, j] = 100 - (100 / (1 + rs))
    
    # Fill initial values
    rsi[:tw] = rsi[tw]
    
    # Combine all features
    additional_features = np.column_stack((
        rand_portfolio_returns_st,
        rand_portfolio_returns_lt,
        mean_returns,
        dispersion,
        avg_correlation,
        vol_stocks_short.reshape(N_DAYS, -1),  # Flatten array of arrays
        vol_stocks_long.reshape(N_DAYS, -1),   # Flatten array of arrays
        momentum_short.reshape(N_DAYS, -1),    # Flatten array of arrays
        momentum_medium.reshape(N_DAYS, -1),   # Flatten array of arrays
        momentum_long.reshape(N_DAYS, -1),     # Flatten array of arrays
        rsi.reshape(N_DAYS, -1)                # Flatten array of arrays
    ))
    
    print(f"Calculated {additional_features.shape[1]} enhanced additional features")
    return additional_features


def generate_enhanced_signatures(log_returns, tw, rd, rm, rv, additional_features=None, 
                                sigma_act=np.tanh, normalize=True):
    """
    Generate randomized signatures with additional features.
    
    Parameters:
    -----------
    log_returns: array ((N_DAYS_TOTAL-1) x N_ASSETS)
        Log returns
    tw: int
        Rolling window size for input path
    rd: int
        Reservoir dimension
    rm: float
        Mean for random projection matrices
    rv: float
        Variance for random projection matrices
    additional_features: array ((N_DAYS_TOTAL-1) x N_FEATURES), optional
        Additional features to include
    sigma_act: function
        Activation function
    normalize: bool
        Whether to normalize the input window
        
    Returns:
    --------
    signatures: array ((N_DAYS_TOTAL-tw) x rd)
        Randomized signature features
    """
    print("Generating enhanced randomized signatures...")
    
    N_DAYS = log_returns.shape[0]
    N_ASSETS = log_returns.shape[1]
    
    # Calculate number of input dimensions
    d_input = N_ASSETS
    if additional_features is not None:
        d_input += additional_features.shape[1]
    
    # Generate random matrices and biases
    A_matrices, b_vectors = generate_random_matrices(d_input, rd, rm, rv)
    
    # Initialize signatures array
    signatures = np.zeros((N_DAYS - tw + 1, rd))
    
    # Generate signatures for each window
    for t in range(tw - 1, N_DAYS):
        # Extract return window
        return_window = log_returns[t - tw + 1:t + 1]
        
        # Combine with additional features if provided
        if additional_features is not None:
            additional_window = additional_features[t - tw + 1:t + 1]
            window = np.hstack((return_window, additional_window))
        else:
            window = return_window
        
        # Normalize window if specified
        if normalize:
            scaler = StandardScaler()
            window = scaler.fit_transform(window)
        
        # Compute signature
        R_t = compute_randomized_signature(window, A_matrices, b_vectors)
        
        # Store signature
        signatures[t - tw + 1] = R_t
    
    print(f"Generated {signatures.shape[0]} enhanced randomized signatures")
    return signatures


# ------------------------------------------------------------
# Phase 9: Main Execution and Orchestration
# ------------------------------------------------------------

def run_simulation_experiment(config):
    """
    Run a complete simulation experiment.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
        
    Returns:
    --------
    results: dict
        Experiment results
    """
    start_time = time.time()
    np.random.seed(config['seed'])
    
    print(f"\n{'='*50}")
    print(f"Starting experiment: {config['experiment_name']}")
    print(f"{'='*50}")
    
    # Phase 1: Simulate financial data
    S, log_returns, true_drift = simulate_financial_data(
        N_ASSETS=config['N_ASSETS'],
        N_DAYS_TOTAL=config['N_DAYS_TOTAL'],
        DT=1/config['days_per_year'],
        S0=config['S0'],
        mu=config['mu'],
        sigma=config['sigma'],
        corr_matrix=config['corr_matrix'],
        cosine_term_coef=config['cosine_term_coef'],
        plot=config['plot_data']
    )
    
    # Calculate burn-in period
    ts = int(config['N_DAYS_TOTAL'] * config['tb'])
    
    # Calculate additional features if needed
    if config['use_additional_features']:
        additional_features = calculate_additional_features(log_returns, config['tw'])
        # Use enhanced signatures
        signatures = generate_enhanced_signatures(
            log_returns,
            config['tw'],
            config['rd'],
            config['rm'],
            config['rv'],
            additional_features=additional_features,
            sigma_act=config['sigma_act'],
            normalize=config['normalize_window']
        )
    else:
        additional_features = None
        # Use regular signatures
        signatures = generate_randomized_signatures(
            log_returns,
            config['tw'],
            config['rd'],
            config['rm'],
            config['rv'],
            sigma_act=config['sigma_act'],
            normalize=config['normalize_window']
        )
    
    # Phase 3: Estimate drift using randomized signatures
    drift_estimates_rs = estimate_drift(
        signatures,
        log_returns,
        config['tw'],
        ts,
        r_alpha=config['r_alpha']
    )
    
    # Also estimate drift using benchmarks
    drift_estimates_linear = estimate_drift_linear(
        log_returns,
        config['tw'],
        ts,
        r_alpha=config['r_alpha']
    )
    
    drift_estimates_momentum = estimate_drift_momentum(
        log_returns,
        config['tw'],
        ts
    )
    
    # Phase 4: Estimate covariance
    covariance_estimates = estimate_covariance(
        log_returns,
        config['tc'],
        ts,
        config['tw']
    )
    
    # Phase 5: Calculate optimal weights
    weights_rs = calculate_optimal_weights(
        drift_estimates_rs,
        covariance_estimates,
        rf=config['rf'],
        max_weight=config['max_weight']
    )
    
    weights_linear = calculate_optimal_weights(
        drift_estimates_linear,
        covariance_estimates,
        rf=config['rf'],
        max_weight=config['max_weight']
    )
    
    weights_momentum = calculate_optimal_weights(
        drift_estimates_momentum,
        covariance_estimates,
        rf=config['rf'],
        max_weight=config['max_weight']
    )
    
    weights_equal = calculate_equal_weights(
        drift_estimates_rs.shape[0],
        config['N_ASSETS']
    )
    
    # Phase 6: Calculate portfolio returns and performance metrics
    # Without transaction costs
    returns_rs = calculate_portfolio_returns(weights_rs, log_returns, ts, config['tw'])
    returns_linear = calculate_portfolio_returns(weights_linear, log_returns, ts, config['tw'])
    returns_momentum = calculate_portfolio_returns(weights_momentum, log_returns, ts, config['tw'])
    returns_equal = calculate_portfolio_returns(weights_equal, log_returns, ts, config['tw'])
    
    metrics_rs = calculate_performance_metrics(returns_rs, config['days_per_year'])
    metrics_linear = calculate_performance_metrics(returns_linear, config['days_per_year'])
    metrics_momentum = calculate_performance_metrics(returns_momentum, config['days_per_year'])
    metrics_equal = calculate_performance_metrics(returns_equal, config['days_per_year'])
    
    # Phase 7: Apply transaction costs
    if config['apply_transaction_costs']:
        prices_subset = S[ts + config['tw'] - 1:, :]
        
        weights_rs_tc, tc_rs = apply_transaction_costs(
            weights_rs,
            prices_subset,
            drift_estimates=drift_estimates_rs,
            lambda_tc=config['lambda_tc'],
            tau_threshold=config['tau_threshold'],
            k_smoothing=config['k_smoothing']
        )
        
        weights_linear_tc, tc_linear = apply_transaction_costs(
            weights_linear,
            prices_subset,
            drift_estimates=drift_estimates_linear,
            lambda_tc=config['lambda_tc'],
            tau_threshold=config['tau_threshold'],
            k_smoothing=config['k_smoothing']
        )
        
        weights_momentum_tc, tc_momentum = apply_transaction_costs(
            weights_momentum,
            prices_subset,
            drift_estimates=drift_estimates_momentum,
            lambda_tc=config['lambda_tc'],
            tau_threshold=config['tau_threshold'],
            k_smoothing=config['k_smoothing']
        )
        
        weights_equal_tc, tc_equal = apply_transaction_costs(
            weights_equal,
            prices_subset,
            lambda_tc=config['lambda_tc'],
            tau_threshold=config['tau_threshold'],
            k_smoothing=config['k_smoothing']
        )
        
        # Calculate returns with transaction costs
        returns_rs_tc = calculate_tc_portfolio_returns(weights_rs_tc, log_returns, tc_rs, ts, config['tw'])
        returns_linear_tc = calculate_tc_portfolio_returns(weights_linear_tc, log_returns, tc_linear, ts, config['tw'])
        returns_momentum_tc = calculate_tc_portfolio_returns(weights_momentum_tc, log_returns, tc_momentum, ts, config['tw'])
        returns_equal_tc = calculate_tc_portfolio_returns(weights_equal_tc, log_returns, tc_equal, ts, config['tw'])
        
        metrics_rs_tc = calculate_performance_metrics(returns_rs_tc, config['days_per_year'])
        metrics_linear_tc = calculate_performance_metrics(returns_linear_tc, config['days_per_year'])
        metrics_momentum_tc = calculate_performance_metrics(returns_momentum_tc, config['days_per_year'])
        metrics_equal_tc = calculate_performance_metrics(returns_equal_tc, config['days_per_year'])
    
    # Calculate information coefficients
    ic_rs, ic_rs_drift = calculate_information_coefficient(
        drift_estimates_rs,
        log_returns,
        ts,
        config['tw'],
        true_drift
    )
    
    ic_linear, ic_linear_drift = calculate_information_coefficient(
        drift_estimates_linear,
        log_returns,
        ts,
        config['tw'],
        true_drift
    )
    
    ic_momentum, ic_momentum_drift = calculate_information_coefficient(
        drift_estimates_momentum,
        log_returns,
        ts,
        config['tw'],
        true_drift
    )
    
    # Prepare results
    results = {
        'config': config,
        'execution_time': time.time() - start_time,
        'metrics': {
            'randomized_signature': metrics_rs,
            'linear_regression': metrics_linear,
            'momentum': metrics_momentum,
            'equal_weights': metrics_equal
        },
        'ic': {
            'randomized_signature': {
                'returns': np.mean(ic_rs),
                'drift': np.mean(ic_rs_drift) if ic_rs_drift is not None else None
            },
            'linear_regression': {
                'returns': np.mean(ic_linear),
                'drift': np.mean(ic_linear_drift) if ic_linear_drift is not None else None
            },
            'momentum': {
                'returns': np.mean(ic_momentum),
                'drift': np.mean(ic_momentum_drift) if ic_momentum_drift is not None else None
            }
        },
        'returns': {
            'randomized_signature': returns_rs,
            'linear_regression': returns_linear,
            'momentum': returns_momentum,
            'equal_weights': returns_equal
        }
    }
    
    if config['apply_transaction_costs']:
        results['metrics_tc'] = {
            'randomized_signature': metrics_rs_tc,
            'linear_regression': metrics_linear_tc,
            'momentum': metrics_momentum_tc,
            'equal_weights': metrics_equal_tc
        }
        results['returns_tc'] = {
            'randomized_signature': returns_rs_tc,
            'linear_regression': returns_linear_tc,
            'momentum': returns_momentum_tc,
            'equal_weights': returns_equal_tc
        }
        results['transaction_costs'] = {
            'randomized_signature': tc_rs,
            'linear_regression': tc_linear,
            'momentum': tc_momentum,
            'equal_weights': tc_equal
        }
    
    print(f"\nExperiment completed in {results['execution_time']:.2f} seconds")
    
    # Print performance metrics
    print("\nPerformance Metrics (Without Transaction Costs):")
    print(f"{'Strategy':<25} | {'Annualized Return':<20} | {'Annualized Vol':<20} | {'Sharpe Ratio':<15} | {'Max Drawdown':<15}")
    print("-" * 100)
    for strategy, metrics in results['metrics'].items():
        print(f"{strategy:<25} | {metrics['annualized_return']*100:18.2f}% | {metrics['annualized_volatility']*100:18.2f}% | {metrics['annualized_sharpe_ratio']:13.2f} | {metrics['max_drawdown']*100:13.2f}%")
    
    if config['apply_transaction_costs']:
        print("\nPerformance Metrics (With Transaction Costs):")
        print(f"{'Strategy':<25} | {'Annualized Return':<20} | {'Annualized Vol':<20} | {'Sharpe Ratio':<15} | {'Max Drawdown':<15}")
        print("-" * 100)
        for strategy, metrics in results['metrics_tc'].items():
            print(f"{strategy:<25} | {metrics['annualized_return']*100:18.2f}% | {metrics['annualized_volatility']*100:18.2f}% | {metrics['annualized_sharpe_ratio']:13.2f} | {metrics['max_drawdown']*100:13.2f}%")
    
    # Print information coefficients
    print("\nInformation Coefficients:")
    print(f"{'Strategy':<25} | {'IC Returns':<15} | {'IC Drift':<15}")
    print("-" * 60)
    for strategy, ic in results['ic'].items():
        ic_drift_val = ic['drift'] if ic['drift'] is not None else "N/A"
        print(f"{strategy:<25} | {ic['returns']:13.4f} | {ic_drift_val if isinstance(ic_drift_val, str) else ic_drift_val:13.4f}")
    
    return results


def plot_results(results, save_path=None):
    """
    Plot results from the experiment.
    
    Parameters:
    -----------
    results: dict
        Experiment results
    save_path: str, optional
        Path to save the plots
    """
    # Plot cumulative returns
    plt.figure(figsize=(15, 10))
    
    # Without transaction costs
    plt.subplot(2, 1, 1)
    for strategy, returns in results['returns'].items():
        cum_returns = np.cumprod(1 + returns)
        plt.plot(cum_returns, label=strategy)
    
    plt.title('Cumulative Portfolio Returns (Without Transaction Costs)')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # With transaction costs (if available)
    if 'returns_tc' in results:
        plt.subplot(2, 1, 2)
        for strategy, returns in results['returns_tc'].items():
            cum_returns = np.cumprod(1 + returns)
            plt.plot(cum_returns, label=strategy)
        
        plt.title('Cumulative Portfolio Returns (With Transaction Costs)')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_cumulative_returns.png")
    plt.show()
    
    # Plot monthly returns distribution
    plt.figure(figsize=(15, 10))
    
    # Without transaction costs
    plt.subplot(2, 1, 1)
    for strategy, metrics in results['metrics'].items():
        if 'monthly_returns' in metrics and len(metrics['monthly_returns']) > 0:
            sns.kdeplot(metrics['monthly_returns'] * 100, label=strategy)
    
    plt.title('Monthly Returns Distribution (Without Transaction Costs)')
    plt.xlabel('Monthly Return (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # With transaction costs (if available)
    if 'metrics_tc' in results:
        plt.subplot(2, 1, 2)
        for strategy, metrics in results['metrics_tc'].items():
            if 'monthly_returns' in metrics and len(metrics['monthly_returns']) > 0:
                sns.kdeplot(metrics['monthly_returns'] * 100, label=strategy)
        
        plt.title('Monthly Returns Distribution (With Transaction Costs)')
        plt.xlabel('Monthly Return (%)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_monthly_returns.png")
    plt.show()
    
    # Plot transaction costs impact (if available)
    if 'transaction_costs' in results:
        plt.figure(figsize=(15, 6))
        
        for strategy, tc in results['transaction_costs'].items():
            plt.plot(tc * 100, label=strategy)
        
        plt.title('Transaction Costs Over Time')
        plt.xlabel('Days')
        plt.ylabel('Transaction Costs (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_transaction_costs.png")
        plt.show()


def main():
    """
    Main function to run the simulation experiment with improved parameters.
    """
    # Improved configuration
    config = {
        'experiment_name': 'Improved Randomized Signature Portfolio Optimization',
        'seed': 42,
        'N_ASSETS': 10,
        'N_DAYS_TOTAL': 2520,  # 10 years
        'days_per_year': 252,
        'S0': 100,
        'mu': None,  # Will be randomly generated with higher positive bias
        'sigma': None,  # Will be randomly generated
        'corr_matrix': None,
        'cosine_term_coef': 0.1,  # Reduced from 0.3 to create less extreme non-linearity
        'tw': 30,  # Increased window size for better signal capture (from 22)
        'tc': 126,  # Shorter covariance estimation (from 252) - half year instead of full year
        'tb': 0.2,  # Increased burn-in period to allow model to stabilize (from 0.1)
        'rd': 100,  # Increased reservoir dimension for more capacity (from 50)
        'rm': 0,    # Mean for random matrices
        'rv': 0.05, # Reduced variance for more stable projections (from 0.1)
        'r_alpha': 0.01,  # Increased regularization to prevent overfitting (from 1e-3)
        'rf': 0,     # Risk-free rate
        'max_weight': 0.3,  # Increased from 0.2 to allow more concentration when beneficial
        'sigma_act': np.tanh,  # Keep tanh activation
        'normalize_window': True,
        'use_additional_features': True,  # Enable additional features for better signal capture
        'apply_transaction_costs': True,
        'lambda_tc': 0.0005,  # Reduced transaction costs (from 0.001)
        'tau_threshold': 0.02,  # Increased threshold to reduce unnecessary trading (from 0.01)
        'k_smoothing': 3,      # Reduced smoothing window (from 5)
        'plot_data': True
    }
    
    # Run experiment
    results = run_simulation_experiment(config)
    
    # Plot results
    plot_results(results, save_path="improved_rs_portfolio")


if __name__ == "__main__":
    main()