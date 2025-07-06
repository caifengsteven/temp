import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
n_assets = 50  # Number of assets
n_days = 1000  # Number of trading days
n_pairs_baseline = 25  # Number of pairs to include in baseline portfolio
transaction_cost = 0.0005  # 0.05% per transaction

# Create clusters of assets that will be cointegrated
n_clusters = 10
assets_per_cluster = n_assets // n_clusters
cluster_assignments = np.repeat(np.arange(n_clusters), assets_per_cluster)

# Simulation parameters for asset prices
mu = 0.0005  # Daily drift term
sigma = 0.01  # Daily volatility
mean_reversion_strength = 0.05  # Mean reversion parameter for cointegrated pairs
spread_volatility = 0.005  # Volatility of the spread between cointegrated pairs

# Function to simulate asset prices
def simulate_asset_prices(n_assets, n_days, mu, sigma, cluster_assignments, mean_reversion_strength, spread_volatility):
    """
    Simulate asset prices where assets within the same cluster are cointegrated.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_days : int
        Number of trading days
    mu : float
        Daily drift term
    sigma : float
        Daily volatility
    cluster_assignments : numpy.ndarray
        Array indicating which cluster each asset belongs to
    mean_reversion_strength : float
        Strength of mean reversion for cointegrated pairs
    spread_volatility : float
        Volatility of the spread between cointegrated pairs
        
    Returns:
    --------
    prices : pandas.DataFrame
        Simulated prices for each asset
    """
    # Initialize price array
    prices = np.zeros((n_days, n_assets))
    
    # Set initial prices
    prices[0, :] = 100
    
    # Cluster factors - these drive the common movements within clusters
    cluster_factors = np.zeros((n_days, n_clusters))
    cluster_factors[0, :] = 100
    
    # Simulate cluster factors (random walks)
    for t in range(1, n_days):
        cluster_factors[t, :] = cluster_factors[t-1, :] * np.exp(mu + sigma * np.random.randn(n_clusters))
    
    # Simulate asset prices with cointegration within clusters
    for t in range(1, n_days):
        for i in range(n_assets):
            cluster = cluster_assignments[i]
            
            # Asset follows its cluster factor with some idiosyncratic movement
            # and mean-reverts to maintain cointegration with other assets in the cluster
            if t == 1:
                prices[t, i] = prices[t-1, i] * np.exp(mu + sigma * np.random.randn())
            else:
                # Calculate spread with other assets in the same cluster
                cluster_indices = np.where(cluster_assignments == cluster)[0]
                cluster_indices = cluster_indices[cluster_indices != i]  # Exclude current asset
                
                # Mean reverting component based on average spread
                if len(cluster_indices) > 0:
                    avg_spread = np.mean(np.log(prices[t-1, i]) - np.log(prices[t-1, cluster_indices]))
                    mean_reversion = -mean_reversion_strength * avg_spread
                else:
                    mean_reversion = 0
                
                # Update price
                prices[t, i] = prices[t-1, i] * np.exp(
                    mu + mean_reversion + sigma * np.random.randn() + spread_volatility * np.random.randn()
                )
    
    # Convert to DataFrame
    prices_df = pd.DataFrame(prices, columns=[f"Asset_{i}" for i in range(n_assets)])
    
    return prices_df

# Simulate asset prices
print("Simulating asset prices...")
prices = simulate_asset_prices(n_assets, n_days, mu, sigma, cluster_assignments, 
                              mean_reversion_strength, spread_volatility)

# Function to test for cointegration between two assets
def test_cointegration(series1, series2, max_lag=1):
    """
    Test for cointegration between two price series using the Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    series1, series2 : numpy.ndarray
        Price series to test
    max_lag : int
        Maximum lag for the ADF test
        
    Returns:
    --------
    t_stat : float
        The t-statistic from the ADF test
    p_value : float
        The p-value from the ADF test
    """
    # Log prices
    log_series1 = np.log(series1)
    log_series2 = np.log(series2)
    
    # Linear regression to find the cointegration relationship
    X = log_series1
    y = log_series2
    beta = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X))**2)
    alpha = np.mean(y) - beta * np.mean(X)
    
    # Calculate residuals (spread)
    spread = y - (alpha + beta * X)
    
    # Run ADF test on the spread
    result = adfuller(spread, maxlag=max_lag)
    
    return result[0], result[1]  # t-statistic and p-value

# Function to build a pairs graph based on cointegration tests
def build_pairs_graph(prices, training_window=500):
    """
    Build a pairs graph where nodes are assets and edge weights are the negative t-statistics
    from cointegration tests.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    training_window : int
        Number of days to use for cointegration testing
        
    Returns:
    --------
    G : networkx.Graph
        Graph where nodes are assets and edge weights are -t_statistics
    t_stats : dict
        Dictionary mapping pairs of assets to their t-statistics
    p_values : dict
        Dictionary mapping pairs of assets to their p-values
    """
    # Use the most recent 'training_window' days for testing
    price_data = prices.iloc[-training_window:]
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes (assets)
    for asset in price_data.columns:
        G.add_node(asset)
    
    # Test all pairs of assets for cointegration
    t_stats = {}
    p_values = {}
    
    print("Testing all asset pairs for cointegration...")
    for i, asset1 in enumerate(tqdm(price_data.columns)):
        for j, asset2 in enumerate(price_data.columns):
            if i < j:  # Avoid duplicate pairs and self-pairs
                t_stat, p_value = test_cointegration(
                    price_data[asset1].values, price_data[asset2].values
                )
                
                # Store t-statistic and p-value
                t_stats[(asset1, asset2)] = t_stat
                p_values[(asset1, asset2)] = p_value
                
                # Add edge with weight = -t_statistic (more negative t-stat means stronger cointegration)
                # The negative sign ensures that maximum weight matching selects the most cointegrated pairs
                G.add_edge(asset1, asset2, weight=-t_stat)
    
    return G, t_stats, p_values

# Function to select pairs using baseline method (based on p-values)
def select_baseline_pairs(p_values, n_pairs):
    """
    Select pairs based on the lowest p-values from cointegration tests.
    
    Parameters:
    -----------
    p_values : dict
        Dictionary mapping pairs of assets to their p-values
    n_pairs : int
        Number of pairs to select
        
    Returns:
    --------
    selected_pairs : list
        List of selected pairs
    """
    # Sort pairs by p-value
    sorted_pairs = sorted(p_values.items(), key=lambda x: x[1])
    
    # Select top n_pairs
    selected_pairs = [pair for pair, _ in sorted_pairs[:n_pairs]]
    
    return selected_pairs

# Function to select pairs using maximum weight matching
def select_matching_pairs(G):
    """
    Select pairs using maximum weight matching.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph where nodes are assets and edge weights are -t_statistics
        
    Returns:
    --------
    matching_pairs : list
        List of selected pairs
    """
    # Find maximum weight matching
    matching = nx.max_weight_matching(G, maxcardinality=True)
    
    # Convert matching to list of pairs
    matching_pairs = list(matching)
    
    # Ensure consistent order of pairs
    matching_pairs = [tuple(sorted(pair)) for pair in matching_pairs]
    
    return matching_pairs

# Function to create baseline and matching portfolios
def create_portfolios(prices, training_window=500):
    """
    Create baseline and matching portfolios based on the most recent data.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    training_window : int
        Number of days to use for cointegration testing
        
    Returns:
    --------
    baseline_pairs : list
        List of pairs in the baseline portfolio
    matching_pairs : list
        List of pairs in the matching portfolio
    """
    # Build pairs graph
    G, t_stats, p_values = build_pairs_graph(prices, training_window)
    
    # Select pairs for baseline portfolio
    baseline_pairs = select_baseline_pairs(p_values, n_pairs_baseline)
    
    # Select pairs for matching portfolio
    matching_pairs = select_matching_pairs(G)
    
    return baseline_pairs, matching_pairs

# Function to calculate trading signals using z-score
def calculate_z_score_signals(prices, pairs, lookback=60, z_threshold=2.0):
    """
    Calculate trading signals for each pair based on z-scores of the spread.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    pairs : list
        List of pairs to trade
    lookback : int
        Lookback period for z-score calculation
    z_threshold : float
        Z-score threshold for trading signals
        
    Returns:
    --------
    signals : pandas.DataFrame
        DataFrame with trading signals for each pair
    """
    signals = pd.DataFrame(index=prices.index)
    
    for pair in pairs:
        asset1, asset2 = pair
        
        # Get log prices
        log_price1 = np.log(prices[asset1])
        log_price2 = np.log(prices[asset2])
        
        # Calculate spread and z-score using rolling window
        signals[f"{asset1}_{asset2}_beta"] = 0.0
        signals[f"{asset1}_{asset2}_spread"] = 0.0
        signals[f"{asset1}_{asset2}_zscore"] = 0.0
        signals[f"{asset1}_{asset2}_signal"] = 0.0
        
        for t in range(lookback, len(prices)):
            # Linear regression on the lookback window
            X = log_price1.iloc[t-lookback:t]
            y = log_price2.iloc[t-lookback:t]
            beta = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X))**2)
            alpha = np.mean(y) - beta * np.mean(X)
            
            # Calculate spread
            spread = log_price2.iloc[t] - (alpha + beta * log_price1.iloc[t])
            
            # Calculate z-score
            spread_mean = np.mean(log_price2.iloc[t-lookback:t] - 
                                  (alpha + beta * log_price1.iloc[t-lookback:t]))
            spread_std = np.std(log_price2.iloc[t-lookback:t] - 
                               (alpha + beta * log_price1.iloc[t-lookback:t]))
            
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
            else:
                z_score = 0
                
            # Winsorize z-score at -3 and 3
            z_score = max(min(z_score, 3), -3)
            
            # Generate signal
            if z_score >= z_threshold:
                signal = -1  # Sell asset2, buy asset1
            elif z_score <= -z_threshold:
                signal = 1   # Buy asset2, sell asset1
            else:
                signal = 0   # No position
            
            # Store values
            signals.loc[prices.index[t], f"{asset1}_{asset2}_beta"] = beta
            signals.loc[prices.index[t], f"{asset1}_{asset2}_spread"] = spread
            signals.loc[prices.index[t], f"{asset1}_{asset2}_zscore"] = z_score
            signals.loc[prices.index[t], f"{asset1}_{asset2}_signal"] = signal
    
    return signals

# Function to calculate trading signals using q-score
def calculate_q_score_signals(prices, pairs, lookback=60):
    """
    Calculate trading signals for each pair based on q-scores (quantile-based) of the spread.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    pairs : list
        List of pairs to trade
    lookback : int
        Lookback period for q-score calculation
        
    Returns:
    --------
    signals : pandas.DataFrame
        DataFrame with trading signals for each pair
    """
    signals = pd.DataFrame(index=prices.index)
    
    for pair in pairs:
        asset1, asset2 = pair
        
        # Get log prices
        log_price1 = np.log(prices[asset1])
        log_price2 = np.log(prices[asset2])
        
        # Calculate spread and q-score using rolling window
        signals[f"{asset1}_{asset2}_beta"] = 0.0
        signals[f"{asset1}_{asset2}_spread"] = 0.0
        signals[f"{asset1}_{asset2}_qscore"] = 0.0
        signals[f"{asset1}_{asset2}_signal"] = 0.0
        
        for t in range(lookback, len(prices)):
            # Linear regression on the lookback window
            X = log_price1.iloc[t-lookback:t]
            y = log_price2.iloc[t-lookback:t]
            beta = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X))**2)
            alpha = np.mean(y) - beta * np.mean(X)
            
            # Calculate spread
            spread = log_price2.iloc[t] - (alpha + beta * log_price1.iloc[t])
            
            # Calculate historical spreads
            historical_spreads = log_price2.iloc[t-lookback:t] - (alpha + beta * log_price1.iloc[t-lookback:t])
            
            # Calculate q-score
            median = np.median(historical_spreads)
            q25 = np.percentile(historical_spreads, 25)
            q75 = np.percentile(historical_spreads, 75)
            
            if q75 > q25:
                q_score = (spread - median) / (q75 - q25)
            else:
                q_score = 0
                
            # Generate signal using rounded q-score as defined in the paper
            signal = -1 if q_score < 0 else 1
            signal = signal * round(abs(q_score))
            
            # Store values
            signals.loc[prices.index[t], f"{asset1}_{asset2}_beta"] = beta
            signals.loc[prices.index[t], f"{asset1}_{asset2}_spread"] = spread
            signals.loc[prices.index[t], f"{asset1}_{asset2}_qscore"] = q_score
            signals.loc[prices.index[t], f"{asset1}_{asset2}_signal"] = signal
    
    return signals

# Function to calculate portfolio returns
def calculate_returns(prices, signals, pairs, method='z'):
    """
    Calculate returns for a portfolio of pairs.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    signals : pandas.DataFrame
        DataFrame with trading signals for each pair
    pairs : list
        List of pairs to trade
    method : str
        Method used for signals ('z' for z-score, 'q' for q-score)
        
    Returns:
    --------
    returns : pandas.Series
        Daily returns of the portfolio
    """
    # Initialize daily returns
    daily_returns = pd.Series(index=prices.index[1:], data=0.0)
    
    # Calculate returns for each pair
    for pair in pairs:
        asset1, asset2 = pair
        
        # Get daily price returns
        returns1 = prices[asset1].pct_change()
        returns2 = prices[asset2].pct_change()
        
        # Get signals and betas
        if method == 'z':
            signal_col = f"{asset1}_{asset2}_signal"
        elif method == 'q':
            signal_col = f"{asset1}_{asset2}_signal"
        
        beta_col = f"{asset1}_{asset2}_beta"
        
        # Calculate pair returns
        for t in range(1, len(prices)):
            if t < lookback:
                continue
                
            signal = signals.loc[prices.index[t-1], signal_col]
            beta = signals.loc[prices.index[t-1], beta_col]
            
            if signal != 0:
                # Position sizing according to the paper
                if signal > 0:  # Buy asset2, sell asset1
                    pair_return = returns2.iloc[t] - beta * returns1.iloc[t]
                elif signal < 0:  # Sell asset2, buy asset1
                    pair_return = beta * returns1.iloc[t] - returns2.iloc[t]
                
                # For q-score, adjust position size based on the signal magnitude
                if method == 'q':
                    pair_return = pair_return * abs(signal)
                
                # Apply transaction costs
                # Assuming we trade on signal changes
                previous_signal = signals.loc[prices.index[t-2], signal_col] if t > 1 else 0
                if signal != previous_signal:
                    # Apply transaction costs for both legs of the trade
                    pair_return -= 2 * transaction_cost
                
                # Add to daily returns
                daily_returns.iloc[t-1] += pair_return / len(pairs)
    
    return daily_returns

# Function to calculate performance metrics
def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a return series.
    
    Parameters:
    -----------
    returns : pandas.Series
        Daily returns of the portfolio
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Annualized return
    metrics['annualized_return'] = returns.mean() * 252
    
    # Annualized volatility
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
    
    # Sortino ratio (using negative returns only)
    negative_returns = returns[returns < 0]
    metrics['sortino_ratio'] = metrics['annualized_return'] / (negative_returns.std() * np.sqrt(252))
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    metrics['max_drawdown'] = drawdown.min()
    
    # Total return
    metrics['total_return'] = (1 + returns).prod() - 1
    
    # Min and max daily return
    metrics['min_daily_return'] = returns.min()
    metrics['max_daily_return'] = returns.max()
    
    # Skewness
    metrics['skew'] = returns.skew()
    
    return metrics

# Function to visualize the portfolio graphs
def visualize_portfolio_graphs(prices, baseline_pairs, matching_pairs):
    """
    Visualize the baseline and matching portfolio graphs.
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Price data for all assets
    baseline_pairs : list
        List of pairs in the baseline portfolio
    matching_pairs : list
        List of pairs in the matching portfolio
    """
    # Create baseline portfolio graph
    G_baseline = nx.Graph()
    for pair in baseline_pairs:
        asset1, asset2 = pair
        G_baseline.add_edge(asset1, asset2)
    
    # Create matching portfolio graph
    G_matching = nx.Graph()
    for pair in matching_pairs:
        asset1, asset2 = pair
        G_matching.add_edge(asset1, asset2)
    
    # Create figure with two subplots
    plt.figure(figsize=(18, 8))
    
    # Plot baseline portfolio graph
    plt.subplot(1, 2, 1)
    pos_baseline = nx.spring_layout(G_baseline, seed=42)
    nx.draw(G_baseline, pos_baseline, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, edge_color='gray')
    plt.title('Baseline Portfolio Graph')
    
    # Plot matching portfolio graph
    plt.subplot(1, 2, 2)
    pos_matching = nx.spring_layout(G_matching, seed=42)
    nx.draw(G_matching, pos_matching, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=8, edge_color='gray')
    plt.title('Matching Portfolio Graph')
    
    plt.tight_layout()
    plt.show()

# Function to calculate concentration of portfolios
def calculate_concentration(pairs):
    """
    Calculate the maximum number of pairs containing any single asset.
    
    Parameters:
    -----------
    pairs : list
        List of pairs in the portfolio
        
    Returns:
    --------
    concentration : int
        Maximum number of pairs containing any single asset
    """
    # Count occurrences of each asset
    asset_counts = {}
    for pair in pairs:
        asset1, asset2 = pair
        asset_counts[asset1] = asset_counts.get(asset1, 0) + 1
        asset_counts[asset2] = asset_counts.get(asset2, 0) + 1
    
    # Maximum count
    concentration = max(asset_counts.values()) if asset_counts else 0
    
    return concentration

# Function to calculate portfolio turnover
def calculate_turnover(old_pairs, new_pairs):
    """
    Calculate turnover between two portfolios.
    
    Parameters:
    -----------
    old_pairs : list
        List of pairs in the old portfolio
    new_pairs : list
        List of pairs in the new portfolio
        
    Returns:
    --------
    turnover : float
        Turnover as a percentage
    """
    old_set = set(old_pairs)
    new_set = set(new_pairs)
    
    # Calculate number of pairs that changed
    pairs_removed = len(old_set - new_set)
    pairs_added = len(new_set - old_set)
    
    # Total change relative to average portfolio size
    avg_size = (len(old_pairs) + len(new_pairs)) / 2
    turnover = 100 * (pairs_removed + pairs_added) / (2 * avg_size) if avg_size > 0 else 0
    
    return turnover

# Function to calculate Jaccard retention
def calculate_retention(old_pairs, new_pairs):
    """
    Calculate Jaccard retention between two portfolios.
    
    Parameters:
    -----------
    old_pairs : list
        List of pairs in the old portfolio
    new_pairs : list
        List of pairs in the new portfolio
        
    Returns:
    --------
    retention : float
        Retention as a percentage
    """
    old_set = set(old_pairs)
    new_set = set(new_pairs)
    
    # Calculate Jaccard index
    if not old_set and not new_set:
        return 100.0
    
    intersection = len(old_set.intersection(new_set))
    union = len(old_set.union(new_set))
    
    retention = 100 * intersection / union if union > 0 else 0
    
    return retention

# Main simulation
print("Starting main simulation...")

# Trading parameters
lookback = 60  # Lookback period for signal calculation
rebalance_period = 21  # Rebalance portfolio every 21 days (approximately monthly)
training_window = 500  # Use 500 days of data for cointegration testing

# Set up indices for portfolio rebalancing
rebalance_indices = list(range(training_window, n_days, rebalance_period))
if rebalance_indices[-1] != n_days - 1:
    rebalance_indices.append(n_days - 1)

# Create DataFrame to store portfolio performance metrics
performance_metrics = pd.DataFrame(
    columns=['baseline_z_sharpe', 'baseline_q_sharpe', 'matching_z_sharpe', 'matching_q_sharpe'],
    index=range(len(rebalance_indices) - 1)
)

# Create DataFrames to store concentration, turnover, and retention metrics
concentration_metrics = pd.DataFrame(
    columns=['baseline', 'matching'],
    index=range(len(rebalance_indices))
)

turnover_metrics = pd.DataFrame(
    columns=['baseline', 'matching'],
    index=range(len(rebalance_indices) - 1)
)

retention_metrics = pd.DataFrame(
    columns=['baseline', 'matching'],
    index=range(len(rebalance_indices) - 1)
)

# Create empty lists to store returns
baseline_z_returns = []
baseline_q_returns = []
matching_z_returns = []
matching_q_returns = []

# Create empty lists to store pairs
baseline_pairs_history = []
matching_pairs_history = []

# Loop through each rebalancing period
for i in range(len(rebalance_indices)):
    print(f"Rebalancing period {i+1}/{len(rebalance_indices)}")
    
    # Get data up to the current rebalance index
    current_idx = rebalance_indices[i]
    current_prices = prices.iloc[:current_idx+1]
    
    # Create portfolios
    baseline_pairs, matching_pairs = create_portfolios(current_prices, training_window)
    
    # Store pairs
    baseline_pairs_history.append(baseline_pairs)
    matching_pairs_history.append(matching_pairs)
    
    # Calculate concentration
    concentration_metrics.loc[i, 'baseline'] = calculate_concentration(baseline_pairs)
    concentration_metrics.loc[i, 'matching'] = calculate_concentration(matching_pairs)
    
    # Calculate turnover and retention only if not the first period
    if i > 0:
        turnover_metrics.loc[i-1, 'baseline'] = calculate_turnover(
            baseline_pairs_history[i-1], baseline_pairs)
        turnover_metrics.loc[i-1, 'matching'] = calculate_turnover(
            matching_pairs_history[i-1], matching_pairs)
        
        retention_metrics.loc[i-1, 'baseline'] = calculate_retention(
            baseline_pairs_history[i-1], baseline_pairs)
        retention_metrics.loc[i-1, 'matching'] = calculate_retention(
            matching_pairs_history[i-1], matching_pairs)
    
    # If not the last rebalancing period, calculate returns
    if i < len(rebalance_indices) - 1:
        next_idx = rebalance_indices[i+1]
        period_prices = prices.iloc[current_idx-lookback:next_idx+1]
        
        # Calculate signals
        baseline_z_signals = calculate_z_score_signals(period_prices, baseline_pairs, lookback)
        baseline_q_signals = calculate_q_score_signals(period_prices, baseline_pairs, lookback)
        matching_z_signals = calculate_z_score_signals(period_prices, matching_pairs, lookback)
        matching_q_signals = calculate_q_score_signals(period_prices, matching_pairs, lookback)
        
        # Calculate returns
        period_baseline_z_returns = calculate_returns(period_prices, baseline_z_signals, baseline_pairs, 'z')
        period_baseline_q_returns = calculate_returns(period_prices, baseline_q_signals, baseline_pairs, 'q')
        period_matching_z_returns = calculate_returns(period_prices, matching_z_signals, matching_pairs, 'z')
        period_matching_q_returns = calculate_returns(period_prices, matching_q_signals, matching_pairs, 'q')
        
        # Keep only returns for the current period
        start_return_idx = lookback
        end_return_idx = len(period_prices) - 1
        period_baseline_z_returns = period_baseline_z_returns.iloc[start_return_idx:end_return_idx]
        period_baseline_q_returns = period_baseline_q_returns.iloc[start_return_idx:end_return_idx]
        period_matching_z_returns = period_matching_z_returns.iloc[start_return_idx:end_return_idx]
        period_matching_q_returns = period_matching_q_returns.iloc[start_return_idx:end_return_idx]
        
        # Append to return lists
        baseline_z_returns.append(period_baseline_z_returns)
        baseline_q_returns.append(period_baseline_q_returns)
        matching_z_returns.append(period_matching_z_returns)
        matching_q_returns.append(period_matching_q_returns)
        
        # Calculate performance metrics
        baseline_z_metrics = calculate_performance_metrics(period_baseline_z_returns)
        baseline_q_metrics = calculate_performance_metrics(period_baseline_q_returns)
        matching_z_metrics = calculate_performance_metrics(period_matching_z_returns)
        matching_q_metrics = calculate_performance_metrics(period_matching_q_returns)
        
        # Store Sharpe ratios
        performance_metrics.loc[i, 'baseline_z_sharpe'] = baseline_z_metrics['sharpe_ratio']
        performance_metrics.loc[i, 'baseline_q_sharpe'] = baseline_q_metrics['sharpe_ratio']
        performance_metrics.loc[i, 'matching_z_sharpe'] = matching_z_metrics['sharpe_ratio']
        performance_metrics.loc[i, 'matching_q_sharpe'] = matching_q_metrics['sharpe_ratio']

# Combine returns
baseline_z_returns_combined = pd.concat(baseline_z_returns)
baseline_q_returns_combined = pd.concat(baseline_q_returns)
matching_z_returns_combined = pd.concat(matching_z_returns)
matching_q_returns_combined = pd.concat(matching_q_returns)

# Calculate overall performance metrics
baseline_z_metrics = calculate_performance_metrics(baseline_z_returns_combined)
baseline_q_metrics = calculate_performance_metrics(baseline_q_returns_combined)
matching_z_metrics = calculate_performance_metrics(matching_z_returns_combined)
matching_q_metrics = calculate_performance_metrics(matching_q_returns_combined)

# Print performance metrics
print("\nOverall Performance Metrics:")
print("\nBaseline Z-Score Strategy:")
for key, value in baseline_z_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nBaseline Q-Score Strategy:")
for key, value in baseline_q_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMatching Z-Score Strategy:")
for key, value in matching_z_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nMatching Q-Score Strategy:")
for key, value in matching_q_metrics.items():
    print(f"{key}: {value:.4f}")

# Create a DataFrame for cumulative returns
cumulative_returns = pd.DataFrame({
    'Baseline (Z-Score)': (1 + baseline_z_returns_combined).cumprod() - 1,
    'Baseline (Q-Score)': (1 + baseline_q_returns_combined).cumprod() - 1,
    'Matching (Z-Score)': (1 + matching_z_returns_combined).cumprod() - 1,
    'Matching (Q-Score)': (1 + matching_q_returns_combined).cumprod() - 1
})

# Plot cumulative returns
plt.figure(figsize=(14, 7))
cumulative_returns.plot()
plt.title('Cumulative Returns')
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.show()

# Plot Sharpe ratios by period
plt.figure(figsize=(14, 7))
performance_metrics.plot(kind='bar')
plt.title('Sharpe Ratio by Rebalancing Period')
plt.xlabel('Period')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot concentration, turnover, and retention metrics
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

# Concentration
concentration_metrics.plot(ax=axes[0])
axes[0].set_title('Concentration (Maximum Number of Pairs Containing Any Single Asset)')
axes[0].set_xlabel('Period')
axes[0].set_ylabel('Concentration')
axes[0].grid(True)
axes[0].legend()

# Turnover
turnover_metrics.plot(ax=axes[1])
axes[1].set_title('Turnover')
axes[1].set_xlabel('Period')
axes[1].set_ylabel('Turnover (%)')
axes[1].grid(True)
axes[1].legend()

# Retention
retention_metrics.plot(ax=axes[2])
axes[2].set_title('Retention (Jaccard Index)')
axes[2].set_xlabel('Period')
axes[2].set_ylabel('Retention (%)')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()

# Visualize portfolio graphs for the last period
visualize_portfolio_graphs(prices, baseline_pairs_history[-1], matching_pairs_history[-1])

# Calculate correlation matrix of returns
correlation_matrix = pd.DataFrame({
    'Baseline (Z-Score)': baseline_z_returns_combined,
    'Baseline (Q-Score)': baseline_q_returns_combined,
    'Matching (Z-Score)': matching_z_returns_combined,
    'Matching (Q-Score)': matching_q_returns_combined
}).corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Strategy Returns')
plt.tight_layout()
plt.show()

# Create a table comparing performance metrics
metrics_comparison = pd.DataFrame({
    'Baseline (Z-Score)': [
        baseline_z_metrics['annualized_return'],
        baseline_z_metrics['annualized_volatility'],
        baseline_z_metrics['sharpe_ratio'],
        baseline_z_metrics['sortino_ratio'],
        baseline_z_metrics['max_drawdown'],
        baseline_z_metrics['total_return'],
        baseline_z_metrics['min_daily_return'],
        baseline_z_metrics['max_daily_return'],
        baseline_z_metrics['skew'],
        turnover_metrics['baseline'].mean(),
        concentration_metrics['baseline'].mean()
    ],
    'Baseline (Q-Score)': [
        baseline_q_metrics['annualized_return'],
        baseline_q_metrics['annualized_volatility'],
        baseline_q_metrics['sharpe_ratio'],
        baseline_q_metrics['sortino_ratio'],
        baseline_q_metrics['max_drawdown'],
        baseline_q_metrics['total_return'],
        baseline_q_metrics['min_daily_return'],
        baseline_q_metrics['max_daily_return'],
        baseline_q_metrics['skew'],
        turnover_metrics['baseline'].mean(),
        concentration_metrics['baseline'].mean()
    ],
    'Matching (Z-Score)': [
        matching_z_metrics['annualized_return'],
        matching_z_metrics['annualized_volatility'],
        matching_z_metrics['sharpe_ratio'],
        matching_z_metrics['sortino_ratio'],
        matching_z_metrics['max_drawdown'],
        matching_z_metrics['total_return'],
        matching_z_metrics['min_daily_return'],
        matching_z_metrics['max_daily_return'],
        matching_z_metrics['skew'],
        turnover_metrics['matching'].mean(),
        concentration_metrics['matching'].mean()
    ],
    'Matching (Q-Score)': [
        matching_q_metrics['annualized_return'],
        matching_q_metrics['annualized_volatility'],
        matching_q_metrics['sharpe_ratio'],
        matching_q_metrics['sortino_ratio'],
        matching_q_metrics['max_drawdown'],
        matching_q_metrics['total_return'],
        matching_q_metrics['min_daily_return'],
        matching_q_metrics['max_daily_return'],
        matching_q_metrics['skew'],
        turnover_metrics['matching'].mean(),
        concentration_metrics['matching'].mean()
    ]
}, index=[
    'Annualized Return',
    'Annualized Volatility',
    'Sharpe Ratio',
    'Sortino Ratio',
    'Maximum Drawdown',
    'Total Return',
    'Minimum Daily Return',
    'Maximum Daily Return',
    'Skewness',
    'Average Turnover (%)',
    'Average Concentration'
])

# Print metrics comparison
print("\nPerformance Metrics Comparison:")
print(metrics_comparison)

# Save metrics comparison to CSV
metrics_comparison.to_csv('pairs_trading_metrics_comparison.csv')

print("\nSimulation complete!")