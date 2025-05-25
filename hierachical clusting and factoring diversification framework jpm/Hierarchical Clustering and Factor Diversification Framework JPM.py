import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.optimize as sco
from sklearn.metrics import pairwise_distances
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style - use a simpler style name that works in older versions
try:
    plt.style.use('seaborn-whitegrid')
except:
    # If that doesn't work, just use the default style
    pass

# Use a consistent color palette
sns.set_palette("colorblind")

# Define the asset classes and styles for factor generation
ASSET_CLASSES = ["Equity", "Credit", "Rates", "FX", "Commodity"]
STYLES = ["Value", "Quality", "Carry", "Momentum", "Volatility", "Low Vol", "Size"]

def create_factor_list():
    """Create a list of factors based on asset classes and styles."""
    factors = []
    
    # Equity factors
    factors.append({"name": "Equity_Value", "asset_class": "Equity", "style": "Value"})
    factors.append({"name": "Equity_Quality", "asset_class": "Equity", "style": "Quality"})
    factors.append({"name": "Equity_Carry", "asset_class": "Equity", "style": "Carry"})
    factors.append({"name": "Equity_Momentum", "asset_class": "Equity", "style": "Momentum"})
    factors.append({"name": "Equity_Volatility", "asset_class": "Equity", "style": "Volatility"})
    factors.append({"name": "Equity_Low_Vol", "asset_class": "Equity", "style": "Low Vol"})
    factors.append({"name": "Equity_Size", "asset_class": "Equity", "style": "Size"})
    
    # Different varieties of the equity factors
    for i in range(1, 10):
        factors.append({"name": f"Equity_Value_{i}", "asset_class": "Equity", "style": "Value"})
        factors.append({"name": f"Equity_Quality_{i}", "asset_class": "Equity", "style": "Quality"})
        factors.append({"name": f"Equity_Momentum_{i}", "asset_class": "Equity", "style": "Momentum"})
        factors.append({"name": f"Equity_Low_Vol_{i}", "asset_class": "Equity", "style": "Low Vol"})
    
    # Credit factors
    factors.append({"name": "Credit_Value", "asset_class": "Credit", "style": "Value"})
    factors.append({"name": "Credit_Carry", "asset_class": "Credit", "style": "Carry"})
    factors.append({"name": "Credit_Momentum", "asset_class": "Credit", "style": "Momentum"})
    factors.append({"name": "Credit_Volatility", "asset_class": "Credit", "style": "Volatility"})
    
    # Rates factors
    factors.append({"name": "Rates_Value", "asset_class": "Rates", "style": "Value"})
    factors.append({"name": "Rates_Carry", "asset_class": "Rates", "style": "Carry"})
    factors.append({"name": "Rates_Momentum", "asset_class": "Rates", "style": "Momentum"})
    factors.append({"name": "Rates_Volatility", "asset_class": "Rates", "style": "Volatility"})
    
    # FX factors
    factors.append({"name": "FX_Value", "asset_class": "FX", "style": "Value"})
    factors.append({"name": "FX_Carry", "asset_class": "FX", "style": "Carry"})
    factors.append({"name": "FX_Momentum", "asset_class": "FX", "style": "Momentum"})
    factors.append({"name": "FX_Volatility", "asset_class": "FX", "style": "Volatility"})
    
    # Commodity factors
    factors.append({"name": "Commodity_Value", "asset_class": "Commodity", "style": "Value"})
    factors.append({"name": "Commodity_Carry", "asset_class": "Commodity", "style": "Carry"})
    factors.append({"name": "Commodity_Momentum", "asset_class": "Commodity", "style": "Momentum"})
    factors.append({"name": "Commodity_Volatility", "asset_class": "Commodity", "style": "Volatility"})
    
    return factors

def generate_synthetic_factor_returns(factors, n_periods=1000, is_long_only=True, include_hedges=False):
    """
    Generate synthetic daily returns for factors.
    
    Parameters:
    -----------
    factors : list of dict
        List of factor dictionaries with name, asset_class, and style
    n_periods : int
        Number of periods to generate
    is_long_only : bool
        If True, add a market component to all returns
    include_hedges : bool
        If True, include hedge factors (short positions)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic returns for each factor
    """
    np.random.seed(42)  # Reset seed for reproducibility
    
    # Create a date range as the index
    start_date = datetime(2000, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_periods)]
    
    # Asset class and style factor returns
    asset_class_factors = {}
    style_factors = {}
    
    # Generate asset class factors
    for asset_class in ASSET_CLASSES:
        asset_class_factors[asset_class] = np.random.normal(0.0001, 0.01, n_periods)
    
    # Generate style factors
    for style in STYLES:
        style_factors[style] = np.random.normal(0.0001, 0.01, n_periods)
    
    # Market factor for long-only strategies
    market_factor = np.random.normal(0.0004, 0.01, n_periods)
    
    # Initialize returns DataFrame
    returns = pd.DataFrame(index=dates)
    
    # Add AR(1) process to create autocorrelation in factors
    for factor in factors:
        name = factor["name"]
        asset_class = factor["asset_class"]
        style = factor["style"]
        
        # Base parameters
        mean = 0.0001  # Small positive mean
        vol = 0.01     # Daily volatility
        
        # Different weights for different component types
        if is_long_only:
            market_weight = 0.7
            asset_weight = 0.15
            style_weight = 0.15
        else:
            market_weight = 0.0
            asset_weight = 0.5
            style_weight = 0.5
        
        # Add an idiosyncratic component
        idiosyncratic = np.random.normal(0, 0.006, n_periods)
        
        # Initialize series with some autocorrelation
        returns_series = np.zeros(n_periods)
        returns_series[0] = np.random.normal(mean, vol)
        
        # Add AR(1) process - different for different style types
        ar_param = 0.3  # Default AR parameter
        
        # Style-specific autocorrelation
        if style == "Momentum":
            ar_param = 0.6  # Stronger momentum in Momentum factors
        elif style == "Value":
            ar_param = 0.2  # Mean-reversion in Value factors
        
        # Generate the time series
        for t in range(1, n_periods):
            # AR(1) component
            ar_component = ar_param * returns_series[t-1]
            
            # Style and asset class influences
            style_influence = style_weight * style_factors[style][t]
            asset_influence = asset_weight * asset_class_factors[asset_class][t]
            market_influence = market_weight * market_factor[t]
            
            # Combine components
            returns_series[t] = mean + ar_component + style_influence + asset_influence + market_influence + idiosyncratic[t]
        
        # Add to returns DataFrame
        returns[name] = returns_series
    
    # Add hedge factors if requested
    if include_hedges:
        # Create market hedge - negatively correlated with market
        returns['Market_Hedge'] = -market_factor
        
        # Create style hedges - negatively correlated with specific styles
        for style in ['Value', 'Momentum', 'Quality', 'Volatility']:
            returns[f'{style}_Hedge'] = -style_factors[style] - market_factor * 0.7
    
    return returns

def compute_correlation_matrix(returns, lookback=250):
    """
    Compute correlation matrix using the last 'lookback' periods of data.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    lookback : int
        Number of periods to use for correlation calculation
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return returns.iloc[-lookback:].corr()

def correlation_to_distance(corr_matrix):
    """
    Convert correlation matrix to distance matrix.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    
    Returns:
    --------
    pd.DataFrame
        Distance matrix
    """
    # Convert correlation to distance: d = sqrt(2 * (1 - corr))
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    return distance_matrix

def plot_correlation_matrix_with_clusters(corr_matrix, cluster_labels=None, title="Correlation Matrix with Clusters"):
    """
    Plot correlation matrix with cluster labels shown on the axis.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    cluster_labels : array-like, optional
        Cluster assignments for each asset
    title : str
        Plot title
    """
    plt.figure(figsize=(14, 12))
    
    # If cluster labels are provided, reorder the correlation matrix
    if cluster_labels is not None:
        # Create a DataFrame with assets and their cluster labels
        clusters_df = pd.DataFrame({
            'asset': corr_matrix.index,
            'cluster': cluster_labels
        })
        
        # Sort by cluster
        clusters_df = clusters_df.sort_values('cluster')
        ordered_assets = clusters_df['asset'].tolist()
        
        # Reorder correlation matrix
        corr_matrix = corr_matrix.loc[ordered_assets, ordered_assets]
    
    # Plot the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=.5, annot=False)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def perform_hierarchical_clustering(distance_matrix, method='average', plot=True):
    """
    Perform hierarchical clustering on the distance matrix.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Distance matrix
    method : str
        Linkage method ('single', 'complete', 'average', 'ward')
    plot : bool
        If True, plot the dendrogram
    
    Returns:
    --------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix
    """
    # Convert distance matrix to condensed form for linkage
    if isinstance(distance_matrix, pd.DataFrame):
        labels = distance_matrix.index
        dist_array = distance_matrix.values
    else:
        labels = range(len(distance_matrix))
        dist_array = distance_matrix
    
    # Get condensed distance matrix
    condensed_dist = squareform(dist_array)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method=method)
    
    if plot:
        plt.figure(figsize=(15, 7))
        dendrogram(
            Z,
            labels=labels,
            orientation='top',
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True
        )
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
        plt.xlabel('Factors')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    return Z

def get_clusters_at_distance(Z, labels, distance_threshold):
    """
    Get clusters at a specific distance threshold.
    
    Parameters:
    -----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix
    labels : list or array-like
        Labels for each asset
    distance_threshold : float
        Distance threshold for forming clusters
    
    Returns:
    --------
    dict
        Dictionary mapping cluster labels to lists of assets
    """
    cluster_assignments = fcluster(Z, distance_threshold, criterion='distance')
    
    clusters = {}
    for i, asset in enumerate(labels):
        cluster_id = cluster_assignments[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(asset)
    
    return clusters

def compute_cluster_returns(returns, clusters, method='average'):
    """
    Compute returns for each cluster based on a specific method.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    clusters : dict
        Dictionary mapping cluster labels to lists of assets
    method : str
        Method for computing cluster returns ('average', 'best_1m', 'best_12m')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with returns for each cluster
    """
    cluster_returns = pd.DataFrame(index=returns.index)
    
    for cluster_id, assets in clusters.items():
        if method == 'average':
            # Simple average of all assets in the cluster
            cluster_returns[f'Cluster_{cluster_id}'] = returns[assets].mean(axis=1)
        
        elif method == 'best_1m':
            # Select asset with best 1-month return
            if len(returns) > 21:  # Ensure we have at least 1 month of data
                last_month_returns = returns[assets].iloc[-21:].mean()
                best_asset = last_month_returns.idxmax()
                cluster_returns[f'Cluster_{cluster_id}'] = returns[best_asset]
            else:
                cluster_returns[f'Cluster_{cluster_id}'] = returns[assets].mean(axis=1)
        
        elif method == 'best_12m':
            # Select asset with best 12-month return
            if len(returns) > 252:  # Ensure we have at least 12 months of data
                last_year_returns = returns[assets].iloc[-252:].mean()
                best_asset = last_year_returns.idxmax()
                cluster_returns[f'Cluster_{cluster_id}'] = returns[best_asset]
            else:
                cluster_returns[f'Cluster_{cluster_id}'] = returns[assets].mean(axis=1)
    
    return cluster_returns

def equal_weight(returns):
    """
    Equal weight allocation.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    
    Returns:
    --------
    np.array
        Array of weights
    """
    n = len(returns.columns)
    return np.ones(n) / n

def inverse_volatility(returns, lookback=252):
    """
    Inverse volatility weight allocation.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    lookback : int
        Number of periods to use for volatility calculation
    
    Returns:
    --------
    np.array
        Array of weights
    """
    # Calculate volatility
    vols = returns.iloc[-lookback:].std().values
    # Inverse volatility weights
    weights = 1 / vols
    # Normalize to sum to 1
    return weights / np.sum(weights)

def hierarchical_risk_parity(returns, Z, lookback=252):
    """
    Hierarchical Risk Parity allocation.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix
    lookback : int
        Number of periods to use for covariance calculation
    
    Returns:
    --------
    np.array
        Array of weights
    """
    # Get covariance matrix
    cov_matrix = returns.iloc[-lookback:].cov().values
    n = len(cov_matrix)
    
    # Initialize weights
    weights = np.ones(n)
    
    # Helper function to calculate cluster variance
    def calculate_cluster_variance(cluster, cov_matrix):
        """
        Calculate variance of a cluster based on covariance matrix.
        
        Parameters:
        -----------
        cluster : list
            List of asset indices in the cluster
        cov_matrix : ndarray
            Covariance matrix
        
        Returns:
        --------
        float
            Variance of the cluster
        """
        if len(cluster) == 0:
            return 0
        
        # Equal weight within cluster
        weights = np.ones(len(cluster)) / len(cluster)
        
        # Extract cluster covariance matrix
        cluster_cov = cov_matrix[np.ix_(cluster, cluster)]
        
        # Calculate variance
        variance = weights.T @ cluster_cov @ weights
        
        return variance
    
    # Function to recursively bisect clusters
    def bisect(cluster, weights, cov_matrix):
        n = len(cluster)
        
        if n == 1:
            return
        
        # Calculate variances for split
        split = int(n / 2)
        cluster_1 = cluster[:split]
        cluster_2 = cluster[split:]
        
        # Calculate variance of each cluster
        var_1 = calculate_cluster_variance(cluster_1, cov_matrix)
        var_2 = calculate_cluster_variance(cluster_2, cov_matrix)
        
        # Assign weights inversely proportional to variances
        weight_factor = 1 / (var_1 + var_2)
        weights[cluster_1] *= weight_factor * var_2
        weights[cluster_2] *= weight_factor * var_1
        
        # Recurse
        bisect(cluster_1, weights, cov_matrix)
        bisect(cluster_2, weights, cov_matrix)
    
    # Initial ordering of assets
    n_assets = len(returns.columns)
    ordering = list(range(n_assets))
    
    # Apply bisection algorithm
    bisect(ordering, weights, cov_matrix)
    
    # Normalize weights to sum to 1
    return weights / np.sum(weights)

def hierarchical_cluster_parity(clusters, Z):
    """
    Hierarchical Cluster Parity allocation.
    
    Parameters:
    -----------
    clusters : dict
        Dictionary mapping cluster labels to lists of assets
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix
    
    Returns:
    --------
    dict
        Dictionary with weights for each asset
    """
    n_assets = sum(len(assets) for assets in clusters.values())
    n_clusters = len(clusters)
    
    # Equal weight to each cluster
    cluster_weights = {cluster_id: 1/n_clusters for cluster_id in clusters}
    
    # Equal weight within each cluster
    asset_weights = {}
    for cluster_id, assets in clusters.items():
        cluster_weight = cluster_weights[cluster_id]
        asset_weight = cluster_weight / len(assets)
        for asset in assets:
            asset_weights[asset] = asset_weight
    
    return asset_weights

def cluster_risk_parity(returns, clusters, lookback=252):
    """
    Cluster Risk Parity allocation - combines HCP with volatility adjustment.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    clusters : dict
        Dictionary mapping cluster labels to lists of assets
    lookback : int
        Number of periods to use for volatility calculation
    
    Returns:
    --------
    dict
        Dictionary with weights for each asset
    """
    n_clusters = len(clusters)
    
    # Calculate cluster volatilities
    cluster_vols = {}
    for cluster_id, assets in clusters.items():
        cluster_returns = returns[assets].mean(axis=1)
        cluster_vols[cluster_id] = cluster_returns.iloc[-lookback:].std()
    
    # Inverse volatility weights for clusters
    total_inv_vol = sum(1/vol for vol in cluster_vols.values())
    cluster_weights = {cluster_id: (1/vol)/total_inv_vol for cluster_id, vol in cluster_vols.items()}
    
    # Equal weight within each cluster
    asset_weights = {}
    for cluster_id, assets in clusters.items():
        cluster_weight = cluster_weights[cluster_id]
        asset_weight = cluster_weight / len(assets)
        for asset in assets:
            asset_weights[asset] = asset_weight
    
    return asset_weights

def apply_weights(returns, weights):
    """
    Apply weights to returns to get portfolio returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    weights : dict or array-like
        Weights for each asset
    
    Returns:
    --------
    pd.Series
        Portfolio returns
    """
    if isinstance(weights, dict):
        # Convert dict to array in the correct order
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
    else:
        weight_array = weights
    
    # Apply weights to returns
    portfolio_returns = returns.dot(weight_array)
    
    return portfolio_returns

def calculate_performance_metrics(returns, annualization_factor=252):
    """
    Calculate performance metrics for a return series.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    annualization_factor : int
        Factor to annualize returns (252 for daily, 12 for monthly)
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Calculate key metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1
    volatility = returns.std() * np.sqrt(annualization_factor)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Hit rate (% of positive returns)
    hit_rate = (returns > 0).mean()
    
    # Calculate t-statistic for Sharpe ratio
    t_stat = sharpe_ratio * np.sqrt(len(returns) / annualization_factor)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        't_stat': t_stat
    }

def run_backtest(returns, weighting_scheme, distance_thresholds, cluster_method='average', return_method='average', lookback=252):
    """
    Run backtest for a given weighting scheme and range of distance thresholds.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    weighting_scheme : str
        Weighting scheme to use ('EW', 'IV', 'HRP', 'HCP', 'CRP')
    distance_thresholds : list
        List of distance thresholds to test
    cluster_method : str
        Method for clustering ('single', 'complete', 'average', 'ward')
    return_method : str
        Method for computing cluster returns ('average', 'best_1m', 'best_12m')
    lookback : int
        Number of periods to use for calculations
    
    Returns:
    --------
    dict
        Dictionary of backtest results
    """
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(returns, lookback)
    
    # Convert to distance matrix
    distance_matrix = correlation_to_distance(corr_matrix)
    
    # Perform hierarchical clustering
    Z = perform_hierarchical_clustering(distance_matrix, method=cluster_method, plot=False)
    
    results = {}
    
    for threshold in distance_thresholds:
        # Get clusters at this threshold
        clusters = get_clusters_at_distance(Z, returns.columns, threshold)
        
        # Compute cluster returns
        cluster_returns = compute_cluster_returns(returns, clusters, method=return_method)
        
        # Apply weighting scheme
        if weighting_scheme == 'EW':
            weights = equal_weight(cluster_returns)
        elif weighting_scheme == 'IV':
            weights = inverse_volatility(cluster_returns, lookback)
        elif weighting_scheme == 'HRP':
            # Recompute clustering on cluster returns
            cluster_corr = compute_correlation_matrix(cluster_returns, lookback)
            cluster_dist = correlation_to_distance(cluster_corr)
            cluster_Z = perform_hierarchical_clustering(cluster_dist, method=cluster_method, plot=False)
            weights = hierarchical_risk_parity(cluster_returns, cluster_Z, lookback)
        elif weighting_scheme == 'HCP':
            # Get cluster weights based on dendrogram structure
            asset_weights = hierarchical_cluster_parity(clusters, Z)
            # Convert to array for consistent interface
            weights = np.array([asset_weights.get(col, 0) for col in cluster_returns.columns])
        elif weighting_scheme == 'CRP':
            # Combine HCP with volatility adjustment
            asset_weights = cluster_risk_parity(returns, clusters, lookback)
            # Convert to array
            weights = np.array([asset_weights.get(col, 0) for col in cluster_returns.columns])
        else:
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        
        # Calculate portfolio returns
        portfolio_returns = apply_weights(cluster_returns, weights)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(portfolio_returns)
        
        # Store results
        results[threshold] = {
            'n_clusters': len(clusters),
            'avg_cluster_size': sum(len(assets) for assets in clusters.values()) / len(clusters),
            'metrics': metrics,
            'portfolio_returns': portfolio_returns,
            'weights': weights
        }
    
    return results

def run_experiment():
    """Run a focused experiment to demonstrate the key findings of the paper."""
    print("Starting experiment to test hierarchical clustering for factor diversification...")
    
    # Create factors
    factors = create_factor_list()
    print(f"Created {len(factors)} factors")
    
    # Generate long-only returns with market beta
    print("Generating long-only returns...")
    lo_returns = generate_synthetic_factor_returns(factors, n_periods=2000, is_long_only=True)
    
    # Generate long-only with hedges
    print("Generating long-only returns with hedges...")
    lo_hedge_returns = generate_synthetic_factor_returns(factors, n_periods=2000, is_long_only=True, include_hedges=True)
    
    # Calculate correlation matrices
    lo_corr = compute_correlation_matrix(lo_returns)
    lo_hedge_corr = compute_correlation_matrix(lo_hedge_returns)
    
    # Plot correlation matrices
    print("Plotting correlation matrices...")
    plot_correlation_matrix_with_clusters(lo_corr, title="Long-Only Factor Correlations")
    plot_correlation_matrix_with_clusters(lo_hedge_corr, title="Long-Only with Hedges Factor Correlations")
    
    # Convert to distance matrices
    lo_dist = correlation_to_distance(lo_corr)
    lo_hedge_dist = correlation_to_distance(lo_hedge_corr)
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    Z_lo = perform_hierarchical_clustering(lo_dist, method='average')
    Z_lo_hedge = perform_hierarchical_clustering(lo_hedge_dist, method='average')
    
    # Define distance thresholds to test
    distance_thresholds = np.linspace(0.1, 1.0, 10)
    
    # Test different weighting schemes at different distance thresholds
    weighting_schemes = ['EW', 'IV', 'HRP', 'HCP', 'CRP']
    
    print("\nRunning backtests for long-only factors...")
    lo_results = {}
    for scheme in weighting_schemes:
        print(f"  Testing {scheme} weighting scheme...")
        lo_results[scheme] = run_backtest(lo_returns, scheme, distance_thresholds)
    
    print("\nRunning backtests for long-only factors with hedges...")
    lo_hedge_results = {}
    for scheme in weighting_schemes:
        print(f"  Testing {scheme} weighting scheme...")
        lo_hedge_results[scheme] = run_backtest(lo_hedge_returns, scheme, distance_thresholds)
    
    # Plot Sharpe ratios for different weighting schemes
    plt.figure(figsize=(14, 7))
    
    # Long-only results
    plt.subplot(1, 2, 1)
    for scheme in weighting_schemes:
        thresholds = sorted(lo_results[scheme].keys())
        sharpe_ratios = [lo_results[scheme][t]['metrics']['sharpe_ratio'] for t in thresholds]
        plt.plot(thresholds, sharpe_ratios, 'o-', label=scheme)
    
    plt.title('Long-Only: Sharpe Ratio vs Distance Threshold')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Long-only with hedges results
    plt.subplot(1, 2, 2)
    for scheme in weighting_schemes:
        thresholds = sorted(lo_hedge_results[scheme].keys())
        sharpe_ratios = [lo_hedge_results[scheme][t]['metrics']['sharpe_ratio'] for t in thresholds]
        plt.plot(thresholds, sharpe_ratios, 'o-', label=scheme)
    
    plt.title('Long-Only with Hedges: Sharpe Ratio vs Distance Threshold')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compare performance at a specific threshold (e.g., 0.5)
    threshold = 0.5
    
    # Long-only comparison
    lo_comparison = pd.DataFrame({
        scheme: {
            'n_clusters': lo_results[scheme][threshold]['n_clusters'],
            'sharpe_ratio': lo_results[scheme][threshold]['metrics']['sharpe_ratio'],
            'annual_return': lo_results[scheme][threshold]['metrics']['annual_return'],
            'volatility': lo_results[scheme][threshold]['metrics']['volatility'],
            'max_drawdown': lo_results[scheme][threshold]['metrics']['max_drawdown'],
            'hit_rate': lo_results[scheme][threshold]['metrics']['hit_rate'],
            't_stat': lo_results[scheme][threshold]['metrics']['t_stat']
        } for scheme in weighting_schemes
    }).T
    
    print("\nLong-Only Performance Comparison (Distance Threshold = 0.5):")
    print(lo_comparison)
    
    # Long-only with hedges comparison
    lo_hedge_comparison = pd.DataFrame({
        scheme: {
            'n_clusters': lo_hedge_results[scheme][threshold]['n_clusters'],
            'sharpe_ratio': lo_hedge_results[scheme][threshold]['metrics']['sharpe_ratio'],
            'annual_return': lo_hedge_results[scheme][threshold]['metrics']['annual_return'],
            'volatility': lo_hedge_results[scheme][threshold]['metrics']['volatility'],
            'max_drawdown': lo_hedge_results[scheme][threshold]['metrics']['max_drawdown'],
            'hit_rate': lo_hedge_results[scheme][threshold]['metrics']['hit_rate'],
            't_stat': lo_hedge_results[scheme][threshold]['metrics']['t_stat']
        } for scheme in weighting_schemes
    }).T
    
    print("\nLong-Only with Hedges Performance Comparison (Distance Threshold = 0.5):")
    print(lo_hedge_comparison)
    
    # Plot cumulative returns for different weighting schemes
    plt.figure(figsize=(14, 7))
    
    # Long-only cumulative returns
    plt.subplot(1, 2, 1)
    for scheme in weighting_schemes:
        returns = lo_results[scheme][threshold]['portfolio_returns']
        cum_returns = (1 + returns).cumprod() - 1
        plt.plot(cum_returns.index, cum_returns, label=scheme)
    
    plt.title('Long-Only: Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Long-only with hedges cumulative returns
    plt.subplot(1, 2, 2)
    for scheme in weighting_schemes:
        returns = lo_hedge_results[scheme][threshold]['portfolio_returns']
        cum_returns = (1 + returns).cumprod() - 1
        plt.plot(cum_returns.index, cum_returns, label=scheme)
    
    plt.title('Long-Only with Hedges: Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test best-in-cluster methods
    print("\nTesting best-in-cluster methods...")
    
    # Define distance thresholds for BIC test
    bic_thresholds = [0.3, 0.5, 0.7]
    
    # Test methods
    bic_methods = ['average', 'best_1m', 'best_12m']
    
    # Run backtests
    bic_results = {}
    for threshold in bic_thresholds:
        bic_results[threshold] = {}
        for method in bic_methods:
            bic_results[threshold][method] = run_backtest(lo_returns, 'EW', [threshold], return_method=method)[threshold]
    
    # Create comparison dataframe
    bic_comparison = pd.DataFrame({
        (threshold, method): {
            'n_clusters': bic_results[threshold][method]['n_clusters'],
            'sharpe_ratio': bic_results[threshold][method]['metrics']['sharpe_ratio'],
            'annual_return': bic_results[threshold][method]['metrics']['annual_return'],
            'volatility': bic_results[threshold][method]['metrics']['volatility'],
            'max_drawdown': bic_results[threshold][method]['metrics']['max_drawdown']
        }
        for threshold in bic_thresholds
        for method in bic_methods
    }).T
    
    print("Best-in-Cluster Comparison:")
    print(bic_comparison)
    
    # Plot cumulative returns for best-in-cluster methods
    plt.figure(figsize=(14, 7))
    
    threshold = 0.5  # Use a specific threshold for comparison
    
    for method in bic_methods:
        returns = bic_results[threshold][method]['portfolio_returns']
        cum_returns = (1 + returns).cumprod() - 1
        plt.plot(cum_returns.index, cum_returns, label=f"{method}")
    
    plt.title(f'Cumulative Returns for Different Cluster Return Methods (Distance Threshold = {threshold})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nExperiment completed.")

if __name__ == "__main__":
    run_experiment()