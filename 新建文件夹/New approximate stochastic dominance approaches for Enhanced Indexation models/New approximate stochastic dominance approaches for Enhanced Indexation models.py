import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate simulated return data
def generate_returns(n_assets, n_periods, mean_returns=None, cov_matrix=None):
    """
    Generate simulated asset returns based on multivariate normal distribution.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_periods : int
        Number of time periods
    mean_returns : numpy.ndarray, optional
        Mean returns for each asset (annualized)
    cov_matrix : numpy.ndarray, optional
        Covariance matrix of returns (annualized)
    
    Returns:
    --------
    numpy.ndarray
        Matrix of simulated returns with shape (n_periods, n_assets)
    """
    if mean_returns is None:
        # Default: random returns between 0% and 15% annualized
        mean_returns = np.random.uniform(0.0, 0.15, size=n_assets)
    
    if cov_matrix is None:
        # Generate a positive semi-definite covariance matrix
        A = np.random.normal(0, 1, size=(n_assets, n_assets))
        cov_matrix = np.dot(A, A.T) / (n_assets * 10)  # Scale down to get realistic values
        
        # Ensure diagonal elements are between 5% and 25% (annualized volatility)
        for i in range(n_assets):
            cov_matrix[i, i] = np.random.uniform(0.05**2, 0.25**2)
    
    # Convert annual parameters to daily
    daily_mean = mean_returns / 252
    daily_cov = cov_matrix / 252
    
    # Generate returns using multivariate normal distribution
    returns = np.random.multivariate_normal(daily_mean, daily_cov, size=n_periods)
    
    return returns

# Function to generate a market index
def generate_index(asset_returns, weights=None):
    """
    Generate a market index based on asset returns and weights.
    
    Parameters:
    -----------
    asset_returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    weights : numpy.ndarray, optional
        Weights for each asset in the index
    
    Returns:
    --------
    numpy.ndarray
        Array of index returns with shape (n_periods,)
    """
    n_assets = asset_returns.shape[1]
    
    if weights is None:
        # Default: market-cap weights (simulated as random weights)
        weights = np.random.uniform(0.5, 1.5, size=n_assets)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # Calculate index returns
    index_returns = np.dot(asset_returns, weights)
    
    return index_returns

# Function to calculate CVaR (Expected Shortfall)
def calculate_cvar(returns, alpha=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) at a given confidence level.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of returns
    alpha : float
        Confidence level (default: 0.05 for 95% CVaR)
    
    Returns:
    --------
    float
        CVaR value
    """
    sorted_returns = np.sort(returns)
    var_idx = int(np.ceil(alpha * len(returns)))
    return -np.mean(sorted_returns[:var_idx])

# Function to calculate CVaR for multiple probability levels
def calculate_cvar_levels(returns, levels):
    """
    Calculate CVaR for multiple probability levels.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of returns
    levels : list
        List of probability levels
    
    Returns:
    --------
    numpy.ndarray
        Array of CVaR values for each level
    """
    cvars = []
    for alpha in levels:
        cvars.append(calculate_cvar(returns, alpha))
    return np.array(cvars)

# Implementation of the OWA-based Enhanced Indexation model
def owa_enhanced_indexation(returns, index_returns, lambda_weights, cvar_levels=None):
    """
    Implement the OWA-based Enhanced Indexation model as described in the paper.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    index_returns : numpy.ndarray
        Array of index returns with shape (n_periods,)
    lambda_weights : numpy.ndarray
        Lambda weights for the OWA operator
    cvar_levels : numpy.ndarray, optional
        Probability levels for CVaR calculation
    
    Returns:
    --------
    numpy.ndarray
        Optimal portfolio weights
    """
    n_periods, n_assets = returns.shape
    n_lambda = len(lambda_weights)
    
    if cvar_levels is None:
        # Default: use 1/T, 2/T, ..., T/T as probability levels
        cvar_levels = np.arange(1, n_periods + 1) / n_periods
    
    n_levels = len(cvar_levels)
    
    # Calculate CVaR values for the index at different levels
    index_cvars = np.array([calculate_cvar(index_returns, alpha) for alpha in cvar_levels])
    
    # Define the optimization problem
    weights = cp.Variable(n_assets)
    u = cp.Variable(n_levels)
    v = cp.Variable(n_lambda)
    phi = cp.Variable(n_levels)
    d = cp.Variable((n_levels, n_periods))
    
    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Portfolio weights sum to 1
        weights >= 0,  # Long-only constraint
        d >= 0  # Non-negative d variables
    ]
    
    # CVaR constraints
    for t in range(n_levels):
        for k in range(n_periods):
            constraints.append(d[t, k] >= -phi[t] - returns[k] @ weights)
    
    # OWA constraints
    for j in range(n_lambda):
        for t in range(n_levels):
            constraints.append(u[t] + v[j] <= lambda_weights[j] * (index_cvars[t] - phi[t] - (1/cvar_levels[t]) * cp.sum(d[t])))
    
    # Objective: maximize the OWA of centered CVaRs
    objective = cp.Maximize(cp.sum(u) + cp.sum(v))
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}")
    
    return weights.value

# Function to implement the Roman et al. (2013) Minimax approach
def roman_minimax(returns, index_returns, cvar_levels=None):
    """
    Implement the Minimax approach from Roman et al. (2013).
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    index_returns : numpy.ndarray
        Array of index returns with shape (n_periods,)
    cvar_levels : numpy.ndarray, optional
        Probability levels for CVaR calculation
    
    Returns:
    --------
    numpy.ndarray
        Optimal portfolio weights
    """
    # This is equivalent to OWA with lambda = [1, 0, 0, ...]
    n_periods = returns.shape[0]
    if cvar_levels is None:
        cvar_levels = np.arange(1, n_periods + 1) / n_periods
    
    lambda_weights = np.zeros(len(cvar_levels))
    lambda_weights[0] = 1.0
    
    return owa_enhanced_indexation(returns, index_returns, lambda_weights, cvar_levels)

# Function to implement different variants of the OWA approach
def k_owa_cvar(returns, index_returns, beta, cumulative=False, cvar_levels=None):
    """
    Implement the k-OWA-CVaR-β or k-OWA-CumCVaR-β approach.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    index_returns : numpy.ndarray
        Array of index returns with shape (n_periods,)
    beta : float
        Parameter for determining k = round(β*T)
    cumulative : bool
        Whether to use cumulative weights (default: False)
    cvar_levels : numpy.ndarray, optional
        Probability levels for CVaR calculation
    
    Returns:
    --------
    numpy.ndarray
        Optimal portfolio weights
    """
    n_periods = returns.shape[0]
    if cvar_levels is None:
        cvar_levels = np.arange(1, n_periods + 1) / n_periods
    
    # Determine k based on beta
    k = int(np.round(beta * len(cvar_levels)))
    k = max(1, min(k, len(cvar_levels)))  # Ensure k is between 1 and T
    
    # Create lambda weights
    lambda_weights = np.zeros(len(cvar_levels))
    
    if cumulative:
        # k-OWA-CumCVaR-β: λ = (k, k-1, k-2, ..., 2, 1, 0, ...)
        lambda_weights[:k] = np.arange(k, 0, -1)
    else:
        # k-OWA-CVaR-β: λ = (1, 1, ..., 1, 0, ..., 0) with k ones
        lambda_weights[:k] = 1.0
    
    return owa_enhanced_indexation(returns, index_returns, lambda_weights, cvar_levels)

# Function to calculate minimum variance portfolio
def min_variance_portfolio(returns):
    """
    Calculate the minimum variance portfolio.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    
    Returns:
    --------
    numpy.ndarray
        Optimal portfolio weights
    """
    n_assets = returns.shape[1]
    
    # Calculate sample covariance matrix
    cov_matrix = np.cov(returns, rowvar=False)
    
    # Define the optimization problem
    weights = cp.Variable(n_assets)
    risk = cp.quad_form(weights, cov_matrix)
    
    # Define constraints
    constraints = [
        cp.sum(weights) == 1,  # Portfolio weights sum to 1
        weights >= 0  # Long-only constraint
    ]
    
    # Define objective function (minimize portfolio variance)
    objective = cp.Minimize(risk)
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return weights.value

# Function to calculate portfolio returns
def calculate_portfolio_returns(weights, returns):
    """
    Calculate portfolio returns given weights and asset returns.
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Portfolio weights with shape (n_assets,)
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    
    Returns:
    --------
    numpy.ndarray
        Array of portfolio returns with shape (n_periods,)
    """
    return np.dot(returns, weights)

# Function to calculate performance metrics
def calculate_performance_metrics(returns, benchmark_returns=None):
    """
    Calculate various performance metrics for a portfolio.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of portfolio returns
    benchmark_returns : numpy.ndarray, optional
        Array of benchmark returns
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Calculate average return (annualized)
    avg_return = np.mean(returns) * 252
    
    # Calculate standard deviation (annualized)
    volatility = np.std(returns) * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cum_returns = np.cumprod(1 + returns) - 1
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / (1 + running_max)
    max_drawdown = np.max(drawdowns)
    
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate VaR and CVaR at 95% confidence
    var_95 = -np.percentile(returns, 5)
    cvar_95 = calculate_cvar(returns, 0.05)
    
    # Calculate information ratio if benchmark is provided
    info_ratio = None
    alpha = None
    if benchmark_returns is not None:
        tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
        info_ratio = (np.mean(returns) - np.mean(benchmark_returns)) * 252 / tracking_error if tracking_error > 0 else 0
        
        # Calculate Jensen's Alpha
        beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = avg_return - beta * (np.mean(benchmark_returns) * 252)
    
    # Return performance metrics
    metrics = {
        'Annual Return': avg_return,
        'Annual Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95
    }
    
    if benchmark_returns is not None:
        metrics['Information Ratio'] = info_ratio
        metrics['Alpha'] = alpha
        metrics['Beta'] = beta
    
    return metrics

# Function to backtest portfolio strategies
def backtest_strategies(returns, index_returns, window_size=125, out_sample_size=20):
    """
    Backtest portfolio strategies using a rolling window approach.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns with shape (n_periods, n_assets)
    index_returns : numpy.ndarray
        Array of index returns with shape (n_periods,)
    window_size : int
        Size of the in-sample window (default: 125 days ≈ 6 months)
    out_sample_size : int
        Size of the out-of-sample window (default: 20 days ≈ 1 month)
    
    Returns:
    --------
    dict
        Dictionary of out-of-sample returns for each strategy
    """
    n_periods, n_assets = returns.shape
    
    # Initialize dictionaries to store results
    portfolio_returns = {
        'Index': [],
        'Roman-CVaR': [],
        'k-OWA-CVaR-5%': [],
        'k-OWA-CVaR-10%': [],
        'k-OWA-CVaR-25%': [],
        'k-OWA-CVaR-50%': [],
        'k-OWA-CVaR-75%': [],
        'k-OWA-CVaR-100%': [],
        'k-OWA-CumCVaR-5%': [],
        'k-OWA-CumCVaR-10%': [],
        'k-OWA-CumCVaR-25%': [],
        'k-OWA-CumCVaR-50%': [],
        'k-OWA-CumCVaR-75%': [],
        'k-OWA-CumCVaR-100%': [],
        'MinV': []
    }
    
    portfolio_weights = {key: [] for key in portfolio_returns.keys()}
    
    # Determine the number of rolling windows
    n_windows = (n_periods - window_size) // out_sample_size
    
    # Perform rolling window backtesting
    for i in tqdm(range(n_windows), desc="Backtesting"):
        # Define in-sample and out-of-sample periods
        in_sample_start = i * out_sample_size
        in_sample_end = in_sample_start + window_size
        out_sample_start = in_sample_end
        out_sample_end = min(out_sample_start + out_sample_size, n_periods)
        
        # Get in-sample and out-of-sample data
        in_sample_returns = returns[in_sample_start:in_sample_end]
        in_sample_index = index_returns[in_sample_start:in_sample_end]
        out_sample_returns = returns[out_sample_start:out_sample_end]
        out_sample_index = index_returns[out_sample_start:out_sample_end]
        
        # Store index returns
        portfolio_returns['Index'].extend(out_sample_index)
        
        # Calculate CVaR levels
        cvar_levels = np.arange(1, window_size + 1) / window_size
        
        # Compute optimal weights for each strategy
        # 1. Roman-CVaR (Minimax)
        roman_weights = roman_minimax(in_sample_returns, in_sample_index, cvar_levels)
        portfolio_weights['Roman-CVaR'].append(roman_weights)
        
        # 2. k-OWA-CVaR-β
        for beta in [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
            key = f'k-OWA-CVaR-{int(beta*100)}%'
            weights = k_owa_cvar(in_sample_returns, in_sample_index, beta, False, cvar_levels)
            portfolio_weights[key].append(weights)
        
        # 3. k-OWA-CumCVaR-β
        for beta in [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
            key = f'k-OWA-CumCVaR-{int(beta*100)}%'
            weights = k_owa_cvar(in_sample_returns, in_sample_index, beta, True, cvar_levels)
            portfolio_weights[key].append(weights)
        
        # 4. MinV (Minimum Variance)
        minv_weights = min_variance_portfolio(in_sample_returns)
        portfolio_weights['MinV'].append(minv_weights)
        
        # Calculate out-of-sample returns for each strategy
        for key in portfolio_weights.keys():
            if key == 'Index':
                continue
            
            weights = portfolio_weights[key][-1]
            if weights is not None:  # Check if optimization was successful
                out_returns = calculate_portfolio_returns(weights, out_sample_returns)
                portfolio_returns[key].extend(out_returns)
            else:
                # Fill with benchmark returns if optimization failed
                portfolio_returns[key].extend(out_sample_index)
    
    # Convert lists to numpy arrays
    for key in portfolio_returns.keys():
        portfolio_returns[key] = np.array(portfolio_returns[key])
    
    return portfolio_returns, portfolio_weights

# Function to compare and visualize performance
def compare_performance(portfolio_returns):
    """
    Compare and visualize the performance of different strategies.
    
    Parameters:
    -----------
    portfolio_returns : dict
        Dictionary of portfolio returns for each strategy
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame of performance metrics for each strategy
    """
    # Calculate performance metrics for each strategy
    metrics = {}
    index_returns = portfolio_returns['Index']
    
    for key, returns in portfolio_returns.items():
        metrics[key] = calculate_performance_metrics(returns, index_returns if key != 'Index' else None)
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    for key, returns in portfolio_returns.items():
        plt.plot(np.cumprod(1 + returns) - 1, label=key)
    
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot risk-return tradeoff
    plt.figure(figsize=(10, 8))
    for key, metric in metrics.items():
        plt.scatter(metric['Annual Volatility'], metric['Annual Return'], label=key, s=100)
    
    plt.title('Risk-Return Tradeoff')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot maximum drawdowns
    drawdowns = {}
    plt.figure(figsize=(12, 8))
    
    for key, returns in portfolio_returns.items():
        cum_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (running_max - cum_returns) / (1 + running_max)
        drawdowns[key] = drawdown
        plt.plot(drawdown, label=key)
    
    plt.title('Drawdowns')
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return metrics_df

# Function to evaluate 3-year ROI
def evaluate_roi(portfolio_returns, horizon=750):
    """
    Evaluate Return on Investment (ROI) over a specified time horizon.
    
    Parameters:
    -----------
    portfolio_returns : dict
        Dictionary of portfolio returns for each strategy
    horizon : int
        Time horizon in days (default: 750 days ≈ 3 years)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame of ROI statistics for each strategy
    """
    roi_stats = {}
    
    for key, returns in portfolio_returns.items():
        if len(returns) <= horizon:
            continue
        
        # Calculate ROI for each time point
        roi_values = []
        for t in range(horizon, len(returns)):
            # Calculate wealth at time t based on returns from t-horizon to t
            period_returns = returns[t-horizon:t]
            wealth_start = 1.0
            wealth_end = wealth_start * np.prod(1 + period_returns)
            roi = (wealth_end - wealth_start) / wealth_start
            roi_values.append(roi)
        
        # Calculate statistics
        roi_values = np.array(roi_values)
        roi_stats[key] = {
            'Mean': np.mean(roi_values),
            'Std Dev': np.std(roi_values),
            '5% Percentile': np.percentile(roi_values, 5),
            '25% Percentile': np.percentile(roi_values, 25),
            '50% Percentile': np.percentile(roi_values, 50),
            '75% Percentile': np.percentile(roi_values, 75),
            '95% Percentile': np.percentile(roi_values, 95)
        }
    
    # Convert to DataFrame
    roi_df = pd.DataFrame(roi_stats).T
    
    return roi_df

# Main function to run the simulation and analysis
def main():
    # Parameters
    n_assets = 50          # Number of assets
    n_periods = 2000       # Number of time periods
    in_sample_window = 125 # In-sample window size (≈ 6 months)
    out_sample_window = 20 # Out-of-sample window size (≈ 1 month)
    
    print("Generating simulated data...")
    # Generate simulated returns
    returns = generate_returns(n_assets, n_periods)
    
    # Generate market index
    index_returns = generate_index(returns)
    
    print("Backtesting strategies...")
    # Backtest strategies
    portfolio_returns, portfolio_weights = backtest_strategies(
        returns, index_returns, in_sample_window, out_sample_window
    )
    
    print("Comparing performance...")
    # Compare performance
    metrics_df = compare_performance(portfolio_returns)
    print("\nPerformance Metrics:")
    print(metrics_df)
    
    print("\nCalculating 3-year ROI statistics...")
    # Calculate 3-year ROI statistics
    roi_df = evaluate_roi(portfolio_returns)
    print("\nROI Statistics (3-year horizon):")
    print(roi_df)
    
    # Analyze weights stability
    print("\nAnalyzing portfolio weights stability...")
    # Calculate average number of assets used (with weight > 0.01)
    avg_assets_used = {}
    for key, weights_list in portfolio_weights.items():
        if key == 'Index':
            continue
        
        assets_used = []
        for weights in weights_list:
            if weights is not None:
                count = np.sum(weights > 0.01)
                assets_used.append(count)
        
        avg_assets_used[key] = np.mean(assets_used)
    
    print("\nAverage number of assets used (weight > 1%):")
    for key, avg in avg_assets_used.items():
        print(f"{key}: {avg:.2f}")
    
    # Analyze portfolio turnover
    turnover = {}
    for key, weights_list in portfolio_weights.items():
        if key == 'Index' or len(weights_list) < 2:
            continue
        
        turnovers = []
        for i in range(1, len(weights_list)):
            if weights_list[i] is not None and weights_list[i-1] is not None:
                # Calculate absolute changes in weights
                turnover_i = np.sum(np.abs(weights_list[i] - weights_list[i-1]))
                turnovers.append(turnover_i)
        
        turnover[key] = np.mean(turnovers) if turnovers else 0
    
    print("\nAverage portfolio turnover:")
    for key, avg in turnover.items():
        print(f"{key}: {avg:.4f}")
    
    # Plot efficient frontier with all strategies
    plt.figure(figsize=(12, 8))
    
    # Calculate risk-return for different portfolios
    risks = []
    returns_annual = []
    labels = []
    
    for key, rets in portfolio_returns.items():
        annual_return = np.mean(rets) * 252
        annual_risk = np.std(rets) * np.sqrt(252)
        returns_annual.append(annual_return)
        risks.append(annual_risk)
        labels.append(key)
    
    # Plot strategies
    plt.scatter(risks, returns_annual, s=150)
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (risks[i], returns_annual[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title('Risk-Return Profile of Different Strategies')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()