import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.covariance import empirical_covariance
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

#-----------------------------------------------------------------
# Data Simulation Functions
#-----------------------------------------------------------------

def simulate_stock_prices(n_stocks=7, n_days=230, market_impact=0.5, volatility=0.02, 
                          correlation_baseline=0.3, crisis_correlation_increase=0.3, 
                          crisis_start=150, crisis_end=180):
    """
    Simulate stock price data with correlation structure and market factor.
    
    Parameters:
    -----------
    n_stocks: int
        Number of stocks to simulate
    n_days: int
        Number of days to simulate
    market_impact: float
        Influence of market factor on stocks
    volatility: float
        Base volatility of stocks
    correlation_baseline: float
        Base correlation between stocks
    crisis_correlation_increase: float
        Increase in correlation during crisis
    crisis_start: int
        Day when crisis starts
    crisis_end: int
        Day when crisis ends
    
    Returns:
    --------
    prices: numpy array
        Simulated stock prices
    true_correlation: numpy array
        True correlation structure over time
    """
    # Initialize prices at 100
    prices = np.zeros((n_days, n_stocks))
    prices[0] = 100
    
    # Generate correlation matrix (baseline)
    corr = np.ones((n_stocks, n_stocks)) * correlation_baseline
    np.fill_diagonal(corr, 1.0)
    
    # Cholesky decomposition for correlated random variables
    chol = np.linalg.cholesky(corr)
    
    # Simulate market factor
    market_returns = np.random.normal(0.0005, 0.01, n_days-1)
    
    # Decrease market during crisis
    market_returns[crisis_start:crisis_end] = np.random.normal(-0.003, 0.02, crisis_end-crisis_start)
    
    # Store true correlation matrices for each time period
    true_correlation = np.zeros((n_days-1, n_stocks, n_stocks))
    
    # Generate returns
    for t in range(1, n_days):
        # Increase correlation during crisis
        if crisis_start <= t < crisis_end:
            temp_corr = np.ones((n_stocks, n_stocks)) * (correlation_baseline + crisis_correlation_increase)
            np.fill_diagonal(temp_corr, 1.0)
            chol = np.linalg.cholesky(temp_corr)
            true_correlation[t-1] = temp_corr
        else:
            true_correlation[t-1] = corr
        
        # Generate correlated noise
        z = np.random.normal(0, 1, n_stocks)
        epsilon = chol @ z
        
        # Generate returns with market factor
        stock_returns = market_impact * market_returns[t-1] + (1 - market_impact) * volatility * epsilon
        
        # Update prices
        prices[t] = prices[t-1] * (1 + stock_returns)
    
    return prices, true_correlation

def compute_log_returns(prices):
    """
    Compute log returns from price data
    """
    return np.diff(np.log(prices), axis=0)

#-----------------------------------------------------------------
# Laplacian Matrix Optimization Functions
#-----------------------------------------------------------------

def project_to_laplacian_constraints(L):
    """
    Project matrix L to Laplacian constraints:
    - Symmetric
    - Zero row/column sum
    - Non-positive off-diagonal elements
    """
    # Make symmetric
    L = (L + L.T) / 2
    
    # Make off-diagonal elements non-positive
    L = L - np.diag(np.diag(L))
    L = np.minimum(L, 0)
    
    # Set diagonal to negative sum of off-diagonal elements
    diag = -np.sum(L, axis=1)
    L = L + np.diag(diag)
    
    return L

def gdet(L):
    """
    Compute pseudo determinant of L (product of non-zero eigenvalues)
    """
    evals = np.linalg.eigvalsh(L)
    # Get positive eigenvalues (excluding the smallest one which should be ~0)
    positive_evals = evals[1:]
    return np.prod(positive_evals)

def log_gdet(L):
    """
    Compute log of pseudo determinant
    """
    evals = np.linalg.eigvalsh(L)
    # Get positive eigenvalues (excluding the smallest one which should be ~0)
    positive_evals = evals[1:]
    return np.sum(np.log(positive_evals + 1e-10))

def optimize_laplacian(S, alpha=0.01, max_iter=100, tol=1e-4):
    """
    Simple gradient descent to optimize the Laplacian matrix
    
    Min tr(LS) - log(gdet(L))
    s.t. L is a Laplacian matrix
    
    Parameters:
    -----------
    S: numpy array
        Similarity matrix (e.g., correlation matrix)
    alpha: float
        Step size
    max_iter: int
        Maximum number of iterations
    tol: float
        Tolerance for convergence
        
    Returns:
    --------
    L: numpy array
        Optimized Laplacian matrix
    """
    n = S.shape[0]
    
    # Initialize with simple Laplacian from similarity matrix
    W = np.maximum(0, -S)
    np.fill_diagonal(W, 0)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Optimize
    for i in range(max_iter):
        # Compute gradient (approximate)
        L_inv = np.linalg.pinv(L)
        grad = S - L_inv
        
        # Update L
        L_new = L - alpha * grad
        
        # Project to Laplacian constraints
        L_new = project_to_laplacian_constraints(L_new)
        
        # Check convergence
        if np.linalg.norm(L - L_new) < tol:
            break
            
        L = L_new
    
    return L

def learn_time_varying_graph(correlation_matrices, window_size=30, time_consistency=100, max_iter=100):
    """
    Learn time-varying graphs based on rolling window approach
    with time consistency regularization
    
    Parameters:
    -----------
    correlation_matrices: numpy array
        Time series of correlation matrices (n_timesteps, n_stocks, n_stocks)
    window_size: int
        Size of rolling window
    time_consistency: float
        Weight of time consistency regularization
    max_iter: int
        Maximum number of iterations for optimization
        
    Returns:
    --------
    laplacians: list
        List of estimated Laplacian matrices
    """
    n_timesteps = correlation_matrices.shape[0]
    n_stocks = correlation_matrices.shape[1]
    
    # Initialize output
    laplacians = []
    
    # Initialize first Laplacian
    S_init = np.mean(correlation_matrices[:window_size], axis=0)
    L_prev = optimize_laplacian(S_init, max_iter=max_iter)
    laplacians.append(L_prev)
    
    # Estimate remaining Laplacians with time consistency
    for t in tqdm(range(window_size, n_timesteps), desc="Learning time-varying graphs"):
        # Get correlation matrix for current window
        S_t = np.mean(correlation_matrices[t-window_size:t], axis=0)
        
        # Initialize with previous Laplacian
        L_t = L_prev.copy()
        
        # Simple gradient descent with time consistency
        for i in range(max_iter):
            # Compute gradient for likelihood term
            L_inv = np.linalg.pinv(L_t)
            grad_likelihood = S_t - L_inv
            
            # Compute gradient for time consistency term
            grad_consistency = L_t - L_prev
            
            # Combined gradient
            grad = grad_likelihood + time_consistency * grad_consistency
            
            # Update L
            L_new = L_t - 0.01 * grad
            
            # Project to Laplacian constraints
            L_new = project_to_laplacian_constraints(L_new)
            
            # Check convergence
            if np.linalg.norm(L_t - L_new) < 1e-4:
                break
                
            L_t = L_new
        
        laplacians.append(L_t)
        L_prev = L_t
    
    return laplacians

def compute_algebraic_connectivity(laplacians):
    """
    Compute algebraic connectivity (second smallest eigenvalue)
    for each Laplacian matrix
    """
    connectivity = []
    
    for L in laplacians:
        eigvals = np.linalg.eigvalsh(L)
        # The algebraic connectivity is the second smallest eigenvalue
        # (the smallest should be close to zero for a valid Laplacian)
        connectivity.append(eigvals[1])
    
    return np.array(connectivity)

#-----------------------------------------------------------------
# Trading Strategies
#-----------------------------------------------------------------

def basic_connectivity_strategy(prices, laplacians, threshold=1.0, window_size=30):
    """
    Basic strategy (S2) from the paper: invest uniformly when algebraic 
    connectivity is below threshold
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    threshold: float
        Threshold for algebraic connectivity
    window_size: int
        Window size used for graph estimation
        
    Returns:
    --------
    returns_s2: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Compute algebraic connectivity
    connectivity = compute_algebraic_connectivity(laplacians)
    
    # Initialize strategy returns
    n_timesteps = len(connectivity)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Strategy: Invest when algebraic connectivity < threshold
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # Set weights
    for t in range(n_timesteps):
        if connectivity[t] < threshold:
            portfolio_weights[t] = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def enhanced_connectivity_strategy(prices, laplacians, threshold=1.0, entry_lookback=3, 
                                  exit_lookback=5, window_size=30):
    """
    Enhanced strategy using algebraic connectivity trends
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    threshold: float
        Threshold for algebraic connectivity
    entry_lookback: int
        Number of days to confirm entry signal
    exit_lookback: int
        Number of days to confirm exit signal
    window_size: int
        Window size used for graph estimation
        
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Compute algebraic connectivity
    connectivity = compute_algebraic_connectivity(laplacians)
    
    # Initialize strategy returns
    n_timesteps = len(connectivity)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Initialize portfolio weights
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # Track position state
    in_position = False
    
    # Loop through time
    for t in range(entry_lookback, n_timesteps):
        # Entry signal: connectivity below threshold for entry_lookback days
        if not in_position:
            entry_signal = True
            for i in range(entry_lookback):
                if connectivity[t-i] >= threshold:
                    entry_signal = False
                    break
            
            if entry_signal:
                in_position = True
                portfolio_weights[t] = np.ones(n_stocks) / n_stocks
        
        # Exit signal: connectivity above threshold for exit_lookback days
        elif in_position:
            exit_signal = True
            for i in range(min(exit_lookback, t)):
                if connectivity[t-i] < threshold:
                    exit_signal = False
                    break
            
            if exit_signal:
                in_position = False
            else:
                portfolio_weights[t] = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def centrality_weighted_strategy(prices, laplacians, threshold=1.0, centrality_method='eigenvector', window_size=30):
    """
    Strategy that weights stocks based on centrality measures from the graph
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    threshold: float
        Threshold for algebraic connectivity
    centrality_method: str
        Method to compute centrality ('eigenvector', 'degree', 'closeness')
    window_size: int
        Window size used for graph estimation
    
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Compute algebraic connectivity
    connectivity = compute_algebraic_connectivity(laplacians)
    
    # Initialize strategy returns
    n_timesteps = len(connectivity)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Initialize portfolio weights
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # For each time step
    for t in range(n_timesteps):
        # Only trade when connectivity is below threshold
        if connectivity[t] < threshold:
            # Create adjacency matrix from Laplacian
            L = laplacians[t]
            W = -L.copy()
            np.fill_diagonal(W, 0)
            
            # Create graph
            G = nx.from_numpy_array(W)
            
            # Compute centrality
            try:
                if centrality_method == 'eigenvector':
                    centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                elif centrality_method == 'degree':
                    centrality = nx.degree_centrality(G)
                elif centrality_method == 'closeness':
                    centrality = nx.closeness_centrality(G, distance='weight')
                else:
                    centrality = {i: 1/n_stocks for i in range(n_stocks)}
                
                # Convert to array
                centrality_values = np.array([centrality[i] for i in range(n_stocks)])
                
                # Normalize weights
                if np.sum(centrality_values) > 0:
                    portfolio_weights[t] = centrality_values / np.sum(centrality_values)
            except:
                # Fallback if centrality calculation fails
                portfolio_weights[t] = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def community_strategy(prices, laplacians, threshold=1.0, window_size=30):
    """
    Strategy that detects communities in the graph and diversifies across them
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    threshold: float
        Threshold for algebraic connectivity
    window_size: int
        Window size used for graph estimation
    
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Compute algebraic connectivity
    connectivity = compute_algebraic_connectivity(laplacians)
    
    # Initialize strategy returns
    n_timesteps = len(connectivity)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Initialize portfolio weights
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # For each time step
    for t in range(n_timesteps):
        # Only invest when connectivity is below threshold
        if connectivity[t] < threshold:
            # Create adjacency matrix from Laplacian
            L = laplacians[t]
            W = -L.copy()
            np.fill_diagonal(W, 0)
            
            # Create graph
            G = nx.from_numpy_array(W)
            
            try:
                # Detect communities
                communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
                
                # Allocate weights equally between communities, then equally within communities
                if communities:
                    weight_per_community = 1.0 / len(communities)
                    for community in communities:
                        community_list = list(community)
                        weight_per_stock = weight_per_community / len(community_list)
                        for stock_idx in community_list:
                            portfolio_weights[t, stock_idx] = weight_per_stock
                else:
                    # Fallback to equal weighting
                    portfolio_weights[t] = np.ones(n_stocks) / n_stocks
            except:
                # Fallback if community detection fails
                portfolio_weights[t] = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def graph_evolution_strategy(prices, laplacians, window_size=30, lookback=5):
    """
    Strategy based on graph evolution over time
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    window_size: int
        Window size used for graph estimation
    lookback: int
        Number of days to look back for graph evolution
    
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Initialize strategy returns
    n_timesteps = len(laplacians)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Initialize portfolio weights
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # Calculate graph evolution metrics
    evolution_metrics = np.zeros(n_timesteps)
    
    for t in range(lookback, n_timesteps):
        # Calculate Frobenius norm of difference between current and previous graphs
        diff_norm = 0
        for i in range(lookback):
            if t-i-1 >= 0:
                diff_norm += np.linalg.norm(laplacians[t] - laplacians[t-i-1], 'fro')
        
        evolution_metrics[t] = diff_norm / lookback
    
    # Normalize metrics
    if n_timesteps > lookback:
        valid_metrics = evolution_metrics[lookback:]
        min_metric = np.min(valid_metrics)
        max_metric = np.max(valid_metrics)
        normalized_metrics = (valid_metrics - min_metric) / (max_metric - min_metric + 1e-10)
        evolution_metrics[lookback:] = normalized_metrics
    
    # Set portfolio weights based on rate of change of graph structure
    # Higher rate of change (more unstable) -> more conservative position
    for t in range(lookback, n_timesteps):
        # Invest less when graph is changing rapidly
        if evolution_metrics[t] > 0.7:  # High rate of change
            # Either don't invest or invest minimally
            continue
        elif evolution_metrics[t] > 0.3:  # Moderate rate of change
            # Invest in the most stable stocks (lowest degree change)
            degrees_t = np.diag(laplacians[t])
            degrees_prev = np.diag(laplacians[t-1])
            degree_changes = np.abs(degrees_t - degrees_prev)
            
            # Invest in the 30% most stable stocks
            n_selected = max(1, int(0.3 * n_stocks))
            selected_indices = np.argsort(degree_changes)[:n_selected]
            for idx in selected_indices:
                portfolio_weights[t, idx] = 1.0 / n_selected
        else:  # Low rate of change - stable graph
            # Invest equally
            portfolio_weights[t] = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def spectral_gap_strategy(prices, laplacians, window_size=30):
    """
    Strategy based on spectral gap of the Laplacian matrix
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    window_size: int
        Window size used for graph estimation
    
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Initialize strategy returns
    n_timesteps = len(laplacians)
    n_stocks = prices.shape[1]
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    if len(adjusted_returns) > n_timesteps:
        adjusted_returns = adjusted_returns[:n_timesteps]
    
    # Initialize portfolio weights
    portfolio_weights = np.zeros((n_timesteps, n_stocks))
    
    # Calculate spectral gaps
    spectral_gaps = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        eigvals = np.linalg.eigvalsh(laplacians[t])
        eigvals = np.sort(eigvals)
        # Spectral gap is difference between smallest non-zero eigenvalue and next eigenvalue
        if len(eigvals) >= 3:  # Make sure we have at least 3 eigenvalues
            spectral_gaps[t] = eigvals[2] - eigvals[1]
    
    # Normalize spectral gaps
    min_gap = np.min(spectral_gaps)
    max_gap = np.max(spectral_gaps)
    if max_gap > min_gap:  # Avoid division by zero
        normalized_gaps = (spectral_gaps - min_gap) / (max_gap - min_gap)
    else:
        normalized_gaps = np.zeros_like(spectral_gaps)
    
    # Bin the normalized gaps into quantiles
    quantiles = np.percentile(normalized_gaps, [20, 40, 60, 80])
    
    for t in range(n_timesteps):
        # Different allocation strategies based on spectral gap quantile
        if normalized_gaps[t] <= quantiles[0]:  # Bottom 20%
            # Large spectral gap - invest in high centrality stocks
            L = laplacians[t]
            W = -L.copy()
            np.fill_diagonal(W, 0)
            G = nx.from_numpy_array(W)
            
            try:
                centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
                centrality_values = np.array([centrality[i] for i in range(n_stocks)])
                
                # Invest in top 40% by centrality
                n_selected = max(1, int(0.4 * n_stocks))
                selected_indices = np.argsort(-centrality_values)[:n_selected]
                
                for idx in selected_indices:
                    portfolio_weights[t, idx] = 1.0 / n_selected
            except:
                # Fallback if eigenvector centrality fails
                portfolio_weights[t] = np.ones(n_stocks) / n_stocks
        
        elif normalized_gaps[t] <= quantiles[1]:  # 20-40%
            # Slightly selective - invest in top 60%
            portfolio_weights[t] = np.ones(n_stocks) / n_stocks
            
        elif normalized_gaps[t] <= quantiles[2]:  # 40-60%
            # Neutral - invest equally
            portfolio_weights[t] = np.ones(n_stocks) / n_stocks
            
        elif normalized_gaps[t] <= quantiles[3]:  # 60-80%
            # Slightly defensive - reduce exposure
            portfolio_weights[t] = np.ones(n_stocks) * 0.7 / n_stocks
            
        else:  # Top 20%
            # Very defensive - minimal exposure
            portfolio_weights[t] = np.ones(n_stocks) * 0.3 / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        if t < len(adjusted_returns):
            strategy_returns[t] = np.sum(portfolio_weights[t] * adjusted_returns[t])
    
    return strategy_returns

def meta_strategy(prices, laplacians, window_size=30):
    """
    Meta-strategy that combines multiple graph-based strategies
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    window_size: int
        Window size used for graph estimation
    
    Returns:
    --------
    strategy_returns: numpy array
        Returns of the meta-strategy
    """
    # Compute returns for individual strategies
    connectivity_returns = enhanced_connectivity_strategy(prices, laplacians, window_size=window_size)
    centrality_returns = centrality_weighted_strategy(prices, laplacians, window_size=window_size)
    community_returns = community_strategy(prices, laplacians, window_size=window_size)
    evolution_returns = graph_evolution_strategy(prices, laplacians, window_size=window_size)
    spectral_returns = spectral_gap_strategy(prices, laplacians, window_size=window_size)
    
    # Combine strategies with equal weight
    n_strategies = 5
    combined_returns = (connectivity_returns + centrality_returns + community_returns + 
                        evolution_returns + spectral_returns) / n_strategies
    
    return combined_returns

#-----------------------------------------------------------------
# Evaluation and Visualization Functions
#-----------------------------------------------------------------

def uniform_investment_strategy(prices, window_size=30):
    """
    Baseline strategy (S1) from the paper: invest uniformly throughout period
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    window_size: int
        Window size to make comparable with other strategies
        
    Returns:
    --------
    returns_s1: numpy array
        Returns of the uniform investment strategy
    """
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Adjust for window size
    adjusted_returns = log_returns[window_size-1:]
    
    # Equal weight portfolio
    n_stocks = prices.shape[1]
    weights = np.ones(n_stocks) / n_stocks
    
    # Compute strategy returns
    strategy_returns = np.zeros(len(adjusted_returns))
    
    for t in range(len(adjusted_returns)):
        strategy_returns[t] = np.sum(weights * adjusted_returns[t])
    
    return strategy_returns

def evaluate_strategies(prices, laplacians, window_size=30, threshold=1.0):
    """
    Evaluate and compare all trading strategies
    
    Parameters:
    -----------
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    window_size: int
        Window size used for graph estimation
    threshold: float
        Threshold for algebraic connectivity
        
    Returns:
    --------
    results: dict
        Dictionary with strategy returns and performance metrics
    """
    # Compute returns for all strategies
    returns_uniform = uniform_investment_strategy(prices, window_size)
    returns_basic = basic_connectivity_strategy(prices, laplacians, threshold, window_size)
    returns_enhanced = enhanced_connectivity_strategy(prices, laplacians, threshold, window_size=window_size)
    returns_centrality = centrality_weighted_strategy(prices, laplacians, threshold, window_size=window_size)
    returns_community = community_strategy(prices, laplacians, threshold, window_size=window_size)
    returns_evolution = graph_evolution_strategy(prices, laplacians, window_size=window_size)
    returns_spectral = spectral_gap_strategy(prices, laplacians, window_size=window_size)
    returns_meta = meta_strategy(prices, laplacians, window_size=window_size)
    
    # Ensure all return arrays have the same length
    min_length = min(len(returns_uniform), len(returns_basic), len(returns_enhanced),
                     len(returns_centrality), len(returns_community), len(returns_evolution),
                     len(returns_spectral), len(returns_meta))
    
    returns_uniform = returns_uniform[:min_length]
    returns_basic = returns_basic[:min_length]
    returns_enhanced = returns_enhanced[:min_length]
    returns_centrality = returns_centrality[:min_length]
    returns_community = returns_community[:min_length]
    returns_evolution = returns_evolution[:min_length]
    returns_spectral = returns_spectral[:min_length]
    returns_meta = returns_meta[:min_length]
    
    # Calculate cumulative returns
    cum_returns_uniform = np.cumprod(1 + returns_uniform) - 1
    cum_returns_basic = np.cumprod(1 + returns_basic) - 1
    cum_returns_enhanced = np.cumprod(1 + returns_enhanced) - 1
    cum_returns_centrality = np.cumprod(1 + returns_centrality) - 1
    cum_returns_community = np.cumprod(1 + returns_community) - 1
    cum_returns_evolution = np.cumprod(1 + returns_evolution) - 1
    cum_returns_spectral = np.cumprod(1 + returns_spectral) - 1
    cum_returns_meta = np.cumprod(1 + returns_meta) - 1
    
    # Calculate performance metrics
    def calc_metrics(returns, cum_returns):
        total_return = cum_returns[-1]
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(returns)
        ann_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        ann_vol = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = running_max - cum_returns
        max_drawdown = np.max(drawdowns)
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    # Compile results
    results = {
        'returns': {
            'uniform': returns_uniform,
            'basic': returns_basic,
            'enhanced': returns_enhanced,
            'centrality': returns_centrality,
            'community': returns_community,
            'evolution': returns_evolution,
            'spectral': returns_spectral,
            'meta': returns_meta
        },
        'cumulative_returns': {
            'uniform': cum_returns_uniform,
            'basic': cum_returns_basic,
            'enhanced': cum_returns_enhanced,
            'centrality': cum_returns_centrality,
            'community': cum_returns_community,
            'evolution': cum_returns_evolution,
            'spectral': cum_returns_spectral,
            'meta': cum_returns_meta
        },
        'metrics': {
            'uniform': calc_metrics(returns_uniform, cum_returns_uniform),
            'basic': calc_metrics(returns_basic, cum_returns_basic),
            'enhanced': calc_metrics(returns_enhanced, cum_returns_enhanced),
            'centrality': calc_metrics(returns_centrality, cum_returns_centrality),
            'community': calc_metrics(returns_community, cum_returns_community),
            'evolution': calc_metrics(returns_evolution, cum_returns_evolution),
            'spectral': calc_metrics(returns_spectral, cum_returns_spectral),
            'meta': calc_metrics(returns_meta, cum_returns_meta)
        }
    }
    
    return results

def plot_strategy_comparison(results, prices, laplacians, window_size=30, threshold=1.0, stock_names=None):
    """
    Plot comparison of all trading strategies
    
    Parameters:
    -----------
    results: dict
        Results from evaluate_strategies function
    prices: numpy array
        Stock prices
    laplacians: list
        Estimated Laplacian matrices
    window_size: int
        Window size used for graph estimation
    threshold: float
        Threshold for algebraic connectivity
    stock_names: list
        Names of stocks
    """
    if stock_names is None:
        stock_names = [f"Stock {i+1}" for i in range(prices.shape[1])]
    
    # Create dates for plotting
    start_date = datetime(2019, 6, 1)
    dates = [start_date + timedelta(days=i) for i in range(prices.shape[0])]
    
    # Compute algebraic connectivity
    connectivity = compute_algebraic_connectivity(laplacians)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: Market index
    market_index = np.mean(prices, axis=1)
    axes[0].plot(dates, np.log(market_index), 'b-')
    axes[0].set_title('Market Index (log-price)', fontsize=14)
    axes[0].grid(True)
    
    # Adjust dates for window
    plot_dates = dates[window_size:]
    if len(plot_dates) > len(connectivity):
        plot_dates = plot_dates[:len(connectivity)]
    elif len(plot_dates) < len(connectivity):
        connectivity = connectivity[:len(plot_dates)]
        
    # Plot 2: Algebraic connectivity
    axes[1].plot(plot_dates, connectivity, 'g-')
    axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    axes[1].set_title('Algebraic Connectivity Indicator', fontsize=14)
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot 3: Cumulative returns of trading strategies
    cum_returns = results['cumulative_returns']
    
    # Find the minimum length among return arrays to ensure consistent plotting
    min_length = min(len(returns) for returns in cum_returns.values())
    min_length = min(min_length, len(plot_dates))
    
    # Use consistent plot dates and return arrays
    plot_data = plot_dates[:min_length]
    
    for strategy, returns in cum_returns.items():
        returns_to_plot = returns[:min_length]
        
        if strategy == 'uniform':
            axes[2].plot(plot_data, returns_to_plot, 'k-', label=f'S1: Uniform ({results["metrics"][strategy]["sharpe_ratio"]:.2f})')
        elif strategy == 'basic':
            axes[2].plot(plot_data, returns_to_plot, 'b-', label=f'S2: Basic ({results["metrics"][strategy]["sharpe_ratio"]:.2f})')
        elif strategy == 'meta':
            axes[2].plot(plot_data, returns_to_plot, 'r-', linewidth=2, label=f'Meta ({results["metrics"][strategy]["sharpe_ratio"]:.2f})')
        else:
            axes[2].plot(plot_data, returns_to_plot, '--', label=f'{strategy.capitalize()} ({results["metrics"][strategy]["sharpe_ratio"]:.2f})')
    
    axes[2].set_title('Cumulative Returns of Trading Strategies (Sharpe in parentheses)', fontsize=14)
    axes[2].grid(True)
    axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Create a table of performance metrics
    metrics = results['metrics']
    
    strategy_names = ['Uniform', 'Basic', 'Enhanced', 'Centrality', 'Community', 'Evolution', 'Spectral', 'Meta']
    metrics_keys = ['total_return', 'annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']
    metrics_names = ['Total Return', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    # Create a new figure for the table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    
    # Create the data for the table
    table_data = []
    for metric_key, metric_name in zip(metrics_keys, metrics_names):
        row = [metric_name]
        for strategy in ['uniform', 'basic', 'enhanced', 'centrality', 'community', 'evolution', 'spectral', 'meta']:
            if metric_key in ['total_return', 'annualized_return', 'max_drawdown']:
                # Format as percentage
                row.append(f"{metrics[strategy][metric_key]*100:.2f}%")
            elif metric_key in ['annualized_volatility']:
                row.append(f"{metrics[strategy][metric_key]*100:.2f}%")
            else:
                row.append(f"{metrics[strategy][metric_key]:.2f}")
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data, 
                    colLabels=['Metric'] + strategy_names,
                    cellLoc='center', 
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title
    plt.title('Performance Metrics of Trading Strategies', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Plot network visualization at crisis onset and during crisis
    # Find crisis period (lowest market prices)
    market_returns = np.diff(np.log(market_index))
    cumulative_returns = np.cumsum(market_returns)
    crisis_idx = np.argmin(cumulative_returns) + window_size
    
    # Before crisis (30 days before bottom)
    before_crisis_idx = max(0, crisis_idx - 30)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, (idx, title) in enumerate([(before_crisis_idx, "Before Crisis"), 
                                     (crisis_idx, "During Crisis")]):
        if idx < len(laplacians):
            L = laplacians[idx]
            # Create adjacency matrix from Laplacian
            W = -L.copy()
            np.fill_diagonal(W, 0)
            
            # Create graph
            G = nx.from_numpy_array(W)
            
            # Set node labels
            node_labels = {i: name for i, name in enumerate(stock_names)}
            
            # Set edge weights for thickness
            edge_weights = [W[u][v] * 5 for u, v in G.edges()]
            
            # Draw graph
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', ax=axes[i])
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=axes[i])
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, ax=axes[i])
            
            axes[i].set_title(f"Network {title} ({dates[idx].strftime('%Y-%m-%d')})")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------
# Main Function
#-----------------------------------------------------------------

def main():
    # Define FAAMUNG stocks for simulation
    stock_names = ['FB', 'AAPL', 'AMZN', 'MSFT', 'UBER', 'NFLX', 'GOOG']
    
    # Simulate stock price data
    print("Simulating stock price data...")
    prices, true_correlation = simulate_stock_prices(
        n_stocks=len(stock_names), 
        n_days=230,
        market_impact=0.5,
        volatility=0.02,
        correlation_baseline=0.3,
        crisis_correlation_increase=0.3,
        crisis_start=150,
        crisis_end=180
    )
    
    # Compute log returns
    log_returns = compute_log_returns(prices)
    
    # Compute rolling window correlation matrices
    print("Computing rolling window correlation matrices...")
    window_size = 30
    n_timesteps = log_returns.shape[0]
    correlation_matrices = np.zeros((n_timesteps - window_size + 1, len(stock_names), len(stock_names)))
    
    for t in range(n_timesteps - window_size + 1):
        window_data = log_returns[t:t+window_size]
        correlation_matrices[t] = np.corrcoef(window_data.T)
    
    # Learn time-varying graphs
    print("Learning time-varying graphs...")
    laplacians = learn_time_varying_graph(
        correlation_matrices,
        window_size=window_size,
        time_consistency=100,
        max_iter=50
    )
    
    # Evaluate all strategies
    print("Evaluating trading strategies...")
    results = evaluate_strategies(
        prices,
        laplacians,
        window_size=window_size,
        threshold=1.0
    )
    
    # Plot results
    print("Plotting strategy comparison...")
    plot_strategy_comparison(
        results,
        prices,
        laplacians,
        window_size=window_size,
        threshold=1.0,
        stock_names=stock_names
    )

if __name__ == "__main__":
    main()