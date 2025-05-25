import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import datetime
import pdblp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set the plotting style
plt.style.use('seaborn-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Bloomberg connection setup
try:
    con = pdblp.BCon(timeout=60000)
    con.start()
    print("Connected to Bloomberg")
    has_bloomberg = True
except Exception as e:
    print(f"Could not connect to Bloomberg: {e}")
    print("Using simulated data instead")
    has_bloomberg = False

# Function to fetch data from Bloomberg
def fetch_data_from_bloomberg(tickers, start_date, end_date, field='PX_LAST'):
    """
    Fetch data from Bloomberg for the given tickers and date range
    """
    if not has_bloomberg:
        print("Bloomberg connection not available. Using simulated data.")
        return simulate_market_data(tickers, start_date, end_date)
    
    try:
        print(f"Fetching {len(tickers)} tickers from Bloomberg...")
        
        # Format dates for Bloomberg query
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        
        # Request data from Bloomberg
        data = con.bdh(tickers=tickers, 
                       flds=[field], 
                       start_date=start_date_fmt, 
                       end_date=end_date_fmt)
        
        # Create DataFrame for prices
        prices_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Extract data for each ticker
        valid_tickers = []
        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, axis=1, level=0)
                if field in ticker_data.columns:
                    prices_df[ticker] = ticker_data[field]
                    valid_tickers.append(ticker)
                    print(f"Successfully extracted data for {ticker}: {len(ticker_data)} rows")
                else:
                    print(f"No '{field}' data found for {ticker}")
            except Exception as e:
                print(f"Error extracting data for {ticker}: {e}")
        
        # Drop rows with missing values
        prices_df = prices_df[valid_tickers].dropna()
        
        print(f"Successfully downloaded data with shape: {prices_df.shape}")
        
        # If we have too few tickers, supplement with simulated data
        if len(valid_tickers) < 10:
            print("Not enough valid tickers. Supplementing with simulated data.")
            simulated_data = simulate_market_data([f"SIM_{i}" for i in range(10-len(valid_tickers))], start_date, end_date)
            prices_df = pd.concat([prices_df, simulated_data], axis=1)
            print(f"Final data shape after supplementing: {prices_df.shape}")
        
        return prices_df
        
    except Exception as e:
        print(f"Error retrieving data from Bloomberg: {e}")
        print("Using simulated data instead.")
        return simulate_market_data(tickers, start_date, end_date)

# Function to simulate market data if Bloomberg isn't available
def simulate_market_data(tickers, start_date, end_date):
    """
    Simulate market data for testing
    """
    print("Simulating market data...")
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Number of tickers
    n_tickers = len(tickers)
    
    # Create correlation matrix with block structure to simulate asset classes
    n_blocks = 3  # Equities, Fixed Income, Commodities
    block_size = n_tickers // n_blocks
    remainder = n_tickers % n_blocks
    
    # Create block sizes
    block_sizes = [block_size] * n_blocks
    for i in range(remainder):
        block_sizes[i] += 1
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_tickers, n_tickers))
    
    # Fill correlation matrix with blocks
    start_idx = 0
    for i, size in enumerate(block_sizes):
        end_idx = start_idx + size
        # Within block correlation (higher for assets in same class)
        corr_matrix[start_idx:end_idx, start_idx:end_idx] = 0.7 + 0.3 * np.random.rand(size, size)
        # Ensure diagonal is 1
        np.fill_diagonal(corr_matrix[start_idx:end_idx, start_idx:end_idx], 1.0)
        
        # Between block correlation (lower for assets in different classes)
        for j in range(i+1, n_blocks):
            start_idx_j = sum(block_sizes[:j])
            end_idx_j = start_idx_j + block_sizes[j]
            # Lower correlation between blocks
            block_corr = 0.2 + 0.3 * np.random.rand(size, block_sizes[j])
            corr_matrix[start_idx:end_idx, start_idx_j:end_idx_j] = block_corr
            corr_matrix[start_idx_j:end_idx_j, start_idx:end_idx] = block_corr.T
        
        start_idx = end_idx
    
    # Ensure the matrix is positive definite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix -= 1.1 * min_eig * np.eye(n_tickers)
    
    # Generate daily returns from multivariate normal distribution
    n_days = len(date_range)
    daily_returns = np.random.multivariate_normal(
        mean=np.zeros(n_tickers) + 0.0002,  # slight positive drift
        cov=corr_matrix * (0.01**2),  # annualized volatility around 15%
        size=n_days
    )
    
    # Convert returns to price series
    prices = 100 * np.cumprod(1 + daily_returns, axis=0)
    
    # Create DataFrame
    df_prices = pd.DataFrame(
        prices, 
        index=date_range, 
        columns=tickers
    )
    
    print(f"Simulated data for {len(tickers)} tickers over {len(date_range)} days.")
    return df_prices

# Standard HRP implementation
def hrp(returns, corr_matrix=None, method='single', risk_measure='variance'):
    """
    Hierarchical Risk Parity portfolio construction
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns
    corr_matrix : DataFrame, optional
        Pre-computed correlation matrix
    method : str, default 'single'
        Linkage method for hierarchical clustering
    risk_measure : str, default 'variance'
        Risk measure for weighting ('variance' or 'std_dev')
    
    Returns:
    --------
    dict with portfolio weights and additional information
    """
    if corr_matrix is None:
        corr_matrix = returns.corr()
    
    # Step 1: Clustering
    # Distance matrix
    distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # Hierarchical clustering
    if isinstance(method, str):
        link = linkage(squareform(distance_matrix), method=method)
        clustered_indices = leaves_list(link)
    else:
        # Accept pre-computed clustered indices
        clustered_indices = method
    
    # Sort the correlation matrix
    sorted_indices = corr_matrix.index[clustered_indices]
    sorted_corr_matrix = corr_matrix.loc[sorted_indices, sorted_indices]
    
    # Step 2: Quasi-diagonal
    sorted_returns = returns[sorted_corr_matrix.index]
    
    # Compute portfolio variance
    if risk_measure == 'variance':
        risk = sorted_returns.var(axis=0) 
    elif risk_measure == 'std_dev':
        risk = sorted_returns.std(axis=0)
    else:
        raise ValueError(f"Unsupported risk measure: {risk_measure}")
    
    # Step 3: Recursive bisection
    weights = pd.Series(1, index=sorted_returns.columns)
    cluster_indices = [list(range(len(sorted_returns.columns)))]
    
    while len(cluster_indices) > 0:
        cluster = cluster_indices.pop(0)
        
        # Stop if cluster contains only one asset
        if len(cluster) == 1:
            continue
        
        # Split cluster in half
        mid_point = len(cluster) // 2
        left_cluster = cluster[:mid_point]
        right_cluster = cluster[mid_point:]
        
        # Add new clusters to the queue
        if len(left_cluster) > 0:
            cluster_indices.append(left_cluster)
        if len(right_cluster) > 0:
            cluster_indices.append(right_cluster)
        
        # Compute cluster variances
        left_indices = weights.index[left_cluster]
        right_indices = weights.index[right_cluster]
        
        left_risk = risk[left_indices].sum()
        right_risk = risk[right_indices].sum()
        
        # Compute allocation factor (a)
        a_left = 1 - (left_risk / (left_risk + right_risk))
        a_right = 1 - (right_risk / (left_risk + right_risk))
        
        # Update weights
        weights[left_indices] *= a_left
        weights[right_indices] *= a_right
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return {
        'weights': weights,
        'clustered_indices': clustered_indices,
        'distance_matrix': distance_matrix,
        'sorted_corr_matrix': sorted_corr_matrix
    }

# Tree-based clustering methods for HRP
def get_tree_based_clustering(distance_matrix, method='average'):
    """
    Get clustered indices using various tree-based clustering methods
    
    Parameters:
    -----------
    distance_matrix : ndarray
        Distance matrix
    method : str
        Clustering method:
            - standard scipy methods: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
            - custom methods: 'ward.D', 'ward.D2', 'mcquitty', 'geometric', 'harmonic'
            
    Returns:
    --------
    ndarray of clustered indices
    """
    # Standard scipy methods
    if method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
        link = linkage(squareform(distance_matrix), method=method)
        return leaves_list(link)
    
    # Custom implementations for methods not directly available in scipy
    n = distance_matrix.shape[0]
    
    if method in ['ward.D', 'ward.D2']:
        # Both are approximated by ward in scipy
        link = linkage(squareform(distance_matrix), method='ward')
        return leaves_list(link)
    
    elif method == 'mcquitty':
        # Approximated by weighted method in scipy
        link = linkage(squareform(distance_matrix), method='weighted')
        return leaves_list(link)
    
    elif method == 'geometric':
        # Custom implementation of geometric linkage
        # For demo, we'll approximate with average linkage
        link = linkage(squareform(distance_matrix), method='average')
        return leaves_list(link)
    
    elif method == 'harmonic':
        # Custom implementation of harmonic linkage
        # For demo, we'll approximate with average linkage
        link = linkage(squareform(distance_matrix), method='average')
        return leaves_list(link)
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

# Function to implement minimum spanning tree (MST) based clustering
def get_mst_based_clustering(distance_matrix, method='edge_betweenness'):
    """
    Get clustered indices using MST-based community detection
    
    Parameters:
    -----------
    distance_matrix : ndarray
        Distance matrix
    method : str
        Community detection method:
            - 'edge_betweenness'
            - 'fastgreedy'
            - 'walktrap'
            
    Returns:
    --------
    ndarray of clustered indices
    """
    # Create a graph from the distance matrix
    G = nx.Graph()
    n = distance_matrix.shape[0]
    
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=distance_matrix[i, j])
    
    # Find the MST
    T = nx.minimum_spanning_tree(G, weight='weight')
    
    # Community detection
    if method == 'edge_betweenness':
        # Edge betweenness community detection
        # For simplicity, we'll use a greedy approach based on edge betweenness
        betweenness = nx.edge_betweenness_centrality(T, weight='weight')
        edges_to_remove = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        # Remove edges to form communities
        T_copy = T.copy()
        num_communities = max(1, min(5, n // 3))  # Aim for approximately 3-5 communities
        
        for _ in range(min(num_communities - 1, len(edges_to_remove))):
            if edges_to_remove:
                edge, _ = edges_to_remove.pop(0)
                T_copy.remove_edge(*edge)
        
        # Get communities
        communities = list(nx.connected_components(T_copy))
        
    elif method == 'fastgreedy':
        # Fast greedy community detection
        # Approximate with a simpler approach
        communities = nx.algorithms.community.greedy_modularity_communities(T, weight='weight')
        
    elif method == 'walktrap':
        # Walktrap community detection
        # Approximate with a simpler approach
        try:
            communities = nx.algorithms.community.louvain_communities(T, weight='weight')
        except AttributeError:
            # For older NetworkX versions
            communities = [nx.connected_components(T)]
            print("Using connected components as communities (older NetworkX version)")
        
    else:
        raise ValueError(f"Unsupported community detection method: {method}")
    
    # Order nodes by community (to keep related nodes together)
    ordered_nodes = []
    for community in communities:
        ordered_nodes.extend(list(community))
    
    return np.array(ordered_nodes)

# Function to implement seriation-based clustering
def get_seriation_based_ordering(distance_matrix, method='spectral'):
    """
    Get indices ordering using seriation methods
    
    Parameters:
    -----------
    distance_matrix : ndarray
        Distance matrix
    method : str
        Seriation method:
            - 'spectral': Spectral seriation
            - 'TSP': Traveling salesperson approximation
            - 'R2E': Rank-two ellipse seriation
            - 'VAT': Visual assessment of clustering tendency
            
    Returns:
    --------
    ndarray of ordered indices
    """
    n = distance_matrix.shape[0]
    
    if method == 'spectral':
        # Spectral seriation
        # Convert distance to similarity
        similarity = np.exp(-distance_matrix / distance_matrix.max())
        
        # Compute the Laplacian
        D = np.diag(similarity.sum(axis=1))
        L = D - similarity
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Get the second smallest eigenvector (Fiedler vector)
        fiedler = eigenvectors[:, 1]
        
        # Order by Fiedler vector
        return np.argsort(fiedler)
    
    elif method == 'TSP':
        # TSP-based seriation
        # Create a graph
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(i, j, weight=distance_matrix[i, j])
        
        # Custom implementation of TSP using nearest neighbor
        # This avoids the need for the approximation module
        
        # Start from a random node
        path = [0]
        unvisited = set(range(1, n))
        
        # Greedy algorithm - always go to the nearest unvisited node
        while unvisited:
            current = path[-1]
            # Find the nearest unvisited node
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            path.append(nearest)
            unvisited.remove(nearest)
            
        return np.array(path)
    
    elif method == 'R2E':
        # Rank-two ellipse seriation
        # For simplicity, we'll use spectral as an approximation
        return get_seriation_based_ordering(distance_matrix, 'spectral')
    
    elif method == 'VAT':
        # Visual assessment of clustering tendency
        # This uses Prim's algorithm to order the nodes
        
        # Start with a random node
        ordered_indices = [0]
        unordered = list(range(1, n))
        
        # Iteratively add the closest node
        while unordered:
            last = ordered_indices[-1]
            closest = min(unordered, key=lambda x: distance_matrix[last, x])
            ordered_indices.append(closest)
            unordered.remove(closest)
            
        return np.array(ordered_indices)
    
    elif method == 'MDS_angle':
        # Multidimensional scaling (angle)
        # For simplicity, we'll use spectral as an approximation
        return get_seriation_based_ordering(distance_matrix, 'spectral')
    
    elif method == 'SPIN_STS':
        # Sorting points into neighborhoods (side-to-side)
        # For simplicity, we'll use VAT as an approximation
        return get_seriation_based_ordering(distance_matrix, 'VAT')
    
    elif method == 'SPIN_NH':
        # Sorting points into neighborhoods (neighborhood algorithm)
        # For simplicity, we'll use spectral as an approximation
        return get_seriation_based_ordering(distance_matrix, 'spectral')
    
    else:
        # Default to spectral
        print(f"Seriation method {method} not fully implemented, using spectral as a fallback")
        return get_seriation_based_ordering(distance_matrix, 'spectral')

# Function to determine the best clustering method adaptively
def get_adaptive_clustering(distance_matrix, methods, criterion='cophenetic'):
    """
    Choose the best clustering method adaptively based on a criterion
    
    Parameters:
    -----------
    distance_matrix : ndarray
        Distance matrix
    methods : list
        List of clustering methods to consider
    criterion : str
        Criterion to use for selection:
            - 'cophenetic': Cophenetic correlation
            - 'gamma': Goodman-Kruskal's gamma
            - 'euclidean': Euclidean distance between ultrametric and original distance
            
    Returns:
    --------
    ndarray of clustered indices from the best method
    """
    best_score = -np.inf
    best_indices = None
    best_method = None
    
    for method in methods:
        # Get clustered indices
        if method.startswith('mst_'):
            clustered_indices = get_mst_based_clustering(distance_matrix, method[4:])
        elif method in ['ward.D', 'ward.D2', 'mcquitty', 'geometric', 'harmonic', 
                      'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            clustered_indices = get_tree_based_clustering(distance_matrix, method)
        else:
            # Assume it's a seriation method
            clustered_indices = get_seriation_based_ordering(distance_matrix, method)
        
        # Compute score
        if criterion == 'cophenetic':
            # Compute ultrametric distance
            link = linkage(squareform(distance_matrix), method='single')
            ultrametric = np.zeros_like(distance_matrix)
            n = distance_matrix.shape[0]
            
            for i in range(n):
                for j in range(i+1, n):
                    # Find height where i and j merge
                    # This is a simplified approach
                    cluster_i = {i}
                    cluster_j = {j}
                    for k in range(len(link)):
                        a, b, height, _ = link[k]
                        if a < n and a in cluster_i or b < n and b in cluster_i:
                            cluster_i.add(a)
                            cluster_i.add(b)
                        if a < n and a in cluster_j or b < n and b in cluster_j:
                            cluster_j.add(a)
                            cluster_j.add(b)
                        if cluster_i.intersection(cluster_j):
                            ultrametric[i, j] = ultrametric[j, i] = height
                            break
            
            # Compute cophenetic correlation
            upper_tri_idx = np.triu_indices(n, k=1)
            orig_distances = distance_matrix[upper_tri_idx]
            ultra_distances = ultrametric[upper_tri_idx]
            
            score = np.corrcoef(orig_distances, ultra_distances)[0, 1]
            
        elif criterion == 'gamma':
            # Goodman-Kruskal's gamma
            # Simplified approach: count concordant vs discordant pairs
            n = distance_matrix.shape[0]
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    idx_i = np.where(clustered_indices == i)[0][0]
                    idx_j = np.where(clustered_indices == j)[0][0]
                    
                    # Position difference in ordered sequence
                    pos_diff = abs(idx_i - idx_j)
                    
                    for k in range(n):
                        for l in range(k+1, n):
                            if k == i and l == j:
                                continue
                                
                            idx_k = np.where(clustered_indices == k)[0][0]
                            idx_l = np.where(clustered_indices == l)[0][0]
                            
                            pos_diff_kl = abs(idx_k - idx_l)
                            
                            if (pos_diff < pos_diff_kl and distance_matrix[i, j] < distance_matrix[k, l]) or \
                               (pos_diff > pos_diff_kl and distance_matrix[i, j] > distance_matrix[k, l]):
                                concordant += 1
                            else:
                                discordant += 1
            
            score = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
            
        elif criterion == 'euclidean':
            # Euclidean distance between ultrametric and original
            # Compute ultrametric as above
            # Then compute Euclidean distance
            # Lower distance is better, so negate for comparison
            score = -np.linalg.norm(ultrametric - distance_matrix)
            
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
        
        # Update best if better
        if score > best_score:
            best_score = score
            best_indices = clustered_indices
            best_method = method
    
    print(f"Selected method: {best_method} with score {best_score:.4f}")
    return best_indices

# Function to implement ASRP
def asrp(returns, method_type='tree_based_static', method='average', criterion='cophenetic', risk_measure='variance'):
    """
    Adaptive Seriational Risk Parity
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns
    method_type : str
        Type of method to use:
            - 'tree_based_static': Use a static tree-based method
            - 'tree_based_adaptive': Adaptively select the best tree-based method
            - 'seriation_based': Use a seriation-based method
    method : str or list
        Method to use (for static) or list of methods to consider (for adaptive)
    criterion : str
        Criterion to use for adaptive method selection
    risk_measure : str
        Risk measure for weighting ('variance' or 'std_dev')
        
    Returns:
    --------
    dict with portfolio weights and additional information
    """
    # Compute correlation matrix
    corr_matrix = returns.corr()
    
    # Compute distance matrix
    distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # Get clustered indices based on method type
    if method_type == 'tree_based_static':
        if method.startswith('mst_'):
            clustered_indices = get_mst_based_clustering(distance_matrix.values, method[4:])
        else:
            clustered_indices = get_tree_based_clustering(distance_matrix.values, method)
            
    elif method_type == 'tree_based_adaptive':
        if not isinstance(method, list):
            method = ['single', 'complete', 'average', 'ward']
        clustered_indices = get_adaptive_clustering(distance_matrix.values, method, criterion)
        
    elif method_type == 'seriation_based':
        clustered_indices = get_seriation_based_ordering(distance_matrix.values, method)
        
    else:
        raise ValueError(f"Unsupported method type: {method_type}")
    
    # Run HRP with the clustered indices
    result = hrp(returns, corr_matrix, clustered_indices, risk_measure)
    
    # Add method info to result
    result['method_type'] = method_type
    result['method'] = method
    result['criterion'] = criterion if method_type == 'tree_based_adaptive' else None
    
    return result

# Function to implement backtest for HRP and ASRP
def backtest_strategy(price_data, window_size=252, rebalance_freq=21, vol_target=0.05, 
                      method_type='tree_based_static', method='average', 
                      criterion='cophenetic', risk_measure='variance', 
                      transaction_cost=0.0002):
    """
    Backtest a HRP or ASRP strategy
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data for assets
    window_size : int
        Lookback window size for returns estimation
    rebalance_freq : int
        Rebalancing frequency in days
    vol_target : float
        Target volatility (annualized)
    method_type : str
        Type of method to use
    method : str or list
        Method to use (for static) or list of methods to consider (for adaptive)
    criterion : str
        Criterion to use for adaptive method selection
    risk_measure : str
        Risk measure for weighting
    transaction_cost : float
        Transaction cost per half-turn (as a percentage)
        
    Returns:
    --------
    DataFrame with backtest results
    """
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Initialize portfolio
    portfolio = pd.DataFrame(index=returns.index)
    portfolio['portfolio_return'] = 0
    
    # Track weights over time separately
    weights_history = {}
    
    # Backtest
    last_weights = None
    
    for i in tqdm(range(window_size, len(returns), rebalance_freq)):
        # Get estimation window
        est_window = returns.iloc[i - window_size:i]
        
        # Skip if not enough data
        if len(est_window) < window_size / 2:
            continue
        
        # Calculate portfolio weights
        if method_type == 'hrp':
            result = hrp(est_window, method=method, risk_measure=risk_measure)
        else:
            result = asrp(est_window, method_type=method_type, method=method, 
                          criterion=criterion, risk_measure=risk_measure)
            
        weights = result['weights']
        
        # Store weights
        rebalance_date = returns.index[i]
        weights_history[rebalance_date] = weights
        
        # Calculate leverage based on volatility target
        portfolio_vol = (weights * est_window.std() * np.sqrt(252)).sum()
        leverage = vol_target / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Scale weights by leverage
        scaled_weights = weights * leverage
        
        # Apply weights for the next period
        for j in range(i, min(i + rebalance_freq, len(returns))):
            if j >= len(returns):
                break
                
            # Get portfolio return for this day
            day_return = returns.iloc[j]
            
            # Apply transaction costs on rebalancing day
            if j == i and i > window_size and last_weights is not None:
                # Calculate turnover - we need to align indices first
                common_assets = last_weights.index.intersection(scaled_weights.index)
                
                if len(common_assets) > 0:
                    old_weights_aligned = last_weights.reindex(common_assets, fill_value=0)
                    new_weights_aligned = scaled_weights.reindex(common_assets, fill_value=0)
                    
                    turnover = np.sum(np.abs(new_weights_aligned - old_weights_aligned)) / 2
                    # Apply transaction costs
                    transaction_cost_day = turnover * transaction_cost
                else:
                    transaction_cost_day = 0
            else:
                transaction_cost_day = 0
            
            # Calculate portfolio return (using assets present on this day)
            common_assets = day_return.index.intersection(scaled_weights.index)
            
            if len(common_assets) > 0:
                day_return_aligned = day_return.reindex(common_assets)
                weights_aligned = scaled_weights.reindex(common_assets, fill_value=0)
                
                # Normalize weights to sum to 1
                weights_aligned = weights_aligned / weights_aligned.sum() if weights_aligned.sum() > 0 else weights_aligned
                
                portfolio_return = (day_return_aligned * weights_aligned).sum() - transaction_cost_day
            else:
                portfolio_return = -transaction_cost_day
            
            # Store in portfolio DataFrame
            portfolio.loc[returns.index[j], 'portfolio_return'] = portfolio_return
        
        # Update last_weights for the next rebalance
        last_weights = scaled_weights
            
    # Calculate cumulative returns
    portfolio['cumulative'] = (1 + portfolio['portfolio_return']).cumprod()
    
    # Calculate drawdowns
    portfolio['drawdown'] = portfolio['cumulative'] / portfolio['cumulative'].cummax() - 1
    
    # Annual performance metrics
    annual_return = ((1 + portfolio['portfolio_return']).prod()) ** (252 / len(portfolio)) - 1
    annual_vol = portfolio['portfolio_return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    max_drawdown = portfolio['drawdown'].min()
    
    # Store metrics
    portfolio.attrs = {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'method_type': method_type,
        'method': method,
        'criterion': criterion,
        'weights_history': weights_history
    }
    
    return portfolio

# Function to run multiple backtests
def run_multi_strategy_backtest(price_data, strategies, window_size=252, rebalance_freq=21, 
                               vol_target=0.05, transaction_cost=0.0002):
    """
    Run multiple strategy backtests
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data for assets
    strategies : list of dict
        List of strategy configurations
    other parameters : same as for backtest_strategy
        
    Returns:
    --------
    dict of DataFrames with backtest results
    """
    results = {}
    
    for strategy in strategies:
        strategy_name = strategy.get('name', f"{strategy['method_type']}_{strategy['method']}")
        print(f"Running backtest for {strategy_name}...")
        
        result = backtest_strategy(
            price_data=price_data,
            window_size=window_size,
            rebalance_freq=rebalance_freq,
            vol_target=vol_target,
            method_type=strategy['method_type'],
            method=strategy['method'],
            criterion=strategy.get('criterion', 'cophenetic'),
            risk_measure=strategy.get('risk_measure', 'variance'),
            transaction_cost=transaction_cost
        )
        
        results[strategy_name] = result
        
        # Print performance
        print(f"  Annual Return: {result.attrs['annual_return']:.4f}")
        print(f"  Annual Vol: {result.attrs['annual_vol']:.4f}")
        print(f"  Sharpe Ratio: {result.attrs['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {result.attrs['max_drawdown']:.4f}\n")
        
    return results

# Function to plot results
def plot_results(results):
    """
    Plot backtest results
    
    Parameters:
    -----------
    results : dict
        Dict of DataFrames with backtest results
    """
    # Plot cumulative returns
    plt.figure(figsize=(14, 8))
    for name, result in results.items():
        plt.plot(result['cumulative'], label=f"{name} (SR: {result.attrs['sharpe_ratio']:.2f})")
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(14, 8))
    for name, result in results.items():
        plt.plot(result['drawdown'], label=name)
    
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.savefig('drawdowns.png')
    plt.show()
    
    # Plot Sharpe Ratios
    plt.figure(figsize=(14, 8))
    sharpe_ratios = [(name, result.attrs['sharpe_ratio']) for name, result in results.items()]
    sharpe_ratios = sorted(sharpe_ratios, key=lambda x: x[1])
    
    names = [x[0] for x in sharpe_ratios]
    values = [x[1] for x in sharpe_ratios]
    
    plt.barh(names, values)
    plt.title('Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.grid(True)
    plt.savefig('sharpe_ratios.png')
    plt.show()
    
    # Calculate correlation matrix of strategy returns
    returns_df = pd.DataFrame({name: result['portfolio_return'] for name, result in results.items()})
    corr_matrix = returns_df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Strategy Return Correlations')
    plt.savefig('strategy_correlations.png')
    plt.show()

    # Plot dendrogram of strategy correlations
    plt.figure(figsize=(14, 8))
    distance_matrix = np.sqrt((1 - corr_matrix) / 2)
    linkage_matrix = linkage(squareform(distance_matrix), method='single')
    dendrogram(linkage_matrix, labels=corr_matrix.index)
    plt.title('Hierarchical Clustering of Strategy Returns')
    plt.xlabel('Strategy')
    plt.ylabel('Distance')
    plt.savefig('strategy_dendrogram.png')
    plt.show()

# Main function
def main():
    # Define alternative tickers that are more likely to be available
    futures_tickers = [
        'CL1 Comdty',  # WTI Crude Oil futures
        'GC1 Comdty',  # Gold futures
        'SI1 Comdty',  # Silver futures
        'ES1 Index',   # E-mini S&P 500 futures
        'NQ1 Index',   # E-mini NASDAQ-100 futures
        'VG1 Index',   # EURO STOXX 50 futures
        'TY1 Comdty',  # 10Y US T-Note futures
        'RX1 Comdty',  # Euro-Bund futures
        'DX1 Curncy',  # US Dollar Index futures
        'XP1 Index',   # SPI 200 futures
        'NK1 Index',   # Nikkei 225 futures
        'UX1 Index',   # VIX futures
        'XB1 Comdty',  # RBOB Gasoline futures
        'HG1 Comdty',  # Copper futures
        'C 1 Comdty',  # Corn futures
        'W 1 Comdty',  # Wheat futures
        'S 1 Comdty'   # Soybean futures
    ]
    
    # Define time range (we'll use a shorter time period for demonstration)
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    
    # Fetch or simulate price data
    price_data = fetch_data_from_bloomberg(futures_tickers, start_date, end_date)
    
    # Define strategies to test
    strategies = [
        {
            'name': 'HRP_single',
            'method_type': 'hrp',
            'method': 'single',
            'risk_measure': 'variance'
        },
        {
            'name': 'HRP_average',
            'method_type': 'hrp',
            'method': 'average',
            'risk_measure': 'variance'
        },
        {
            'name': 'HRP_complete',
            'method_type': 'hrp',
            'method': 'complete',
            'risk_measure': 'variance'
        },
        {
            'name': 'HRP_ward',
            'method_type': 'hrp',
            'method': 'ward',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_tree_static_average',
            'method_type': 'tree_based_static',
            'method': 'average',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_tree_static_ward',
            'method_type': 'tree_based_static',
            'method': 'ward',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_tree_adaptive',
            'method_type': 'tree_based_adaptive',
            'method': ['single', 'complete', 'average', 'ward'],
            'criterion': 'cophenetic',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_seriation_spectral',
            'method_type': 'seriation_based',
            'method': 'spectral',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_seriation_VAT',
            'method_type': 'seriation_based',
            'method': 'VAT',
            'risk_measure': 'variance'
        },
        {
            'name': 'ASRP_seriation_TSP',
            'method_type': 'seriation_based',
            'method': 'TSP',
            'risk_measure': 'variance'
        }
    ]
    
    # Run backtests with shortened window for demonstration
    results = run_multi_strategy_backtest(
        price_data=price_data,
        strategies=strategies,
        window_size=126,  # ~6 months (shortened for demonstration)
        rebalance_freq=21,  # Monthly
        vol_target=0.05,  # 5% target volatility
        transaction_cost=0.0002  # 2 bps per half-turn
    )
    
    # Plot results
    plot_results(results)
    
    # Get top-performing strategies
    sharpe_ratios = [(name, result.attrs['sharpe_ratio']) for name, result in results.items()]
    top_strategies = sorted(sharpe_ratios, key=lambda x: x[1], reverse=True)
    
    print("\nTop Performing Strategies:")
    for name, sharpe in top_strategies:
        print(f"{name}: Sharpe Ratio = {sharpe:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()