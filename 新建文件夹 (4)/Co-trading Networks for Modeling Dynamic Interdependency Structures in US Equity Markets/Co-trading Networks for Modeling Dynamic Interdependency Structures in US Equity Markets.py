import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import cvxpy as cp
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import pearma
import warnings
warnings.filterwarnings('ignore')

class CoTradingNetworks:
    """
    Implementation of the co-trading networks approach for modeling dynamic interdependency 
    structures and estimating high-dimensional covariances in equity markets.
    """
    
    def __init__(self, delta=500):
        """
        Initialize co-trading networks.
        
        Parameters:
        -----------
        delta : int
            Time threshold in microseconds for defining co-occurrence of trades.
        """
        self.delta = delta  # in microseconds
        self.stocks = None
        self.n_stocks = 0
        self.sectors = None
        self.daily_co_trading_matrices = {}
        self.daily_clusters = {}
        self.daily_covariance_matrices = {}
        
    def set_stocks(self, stocks, sectors=None):
        """
        Set the universe of stocks.
        
        Parameters:
        -----------
        stocks : list
            List of stock symbols.
        sectors : dict, optional
            Dictionary mapping stock symbols to their sectors.
        """
        self.stocks = stocks
        self.n_stocks = len(stocks)
        self.sectors = sectors if sectors is not None else {s: 'Unknown' for s in stocks}
        self.stock_to_idx = {s: i for i, s in enumerate(stocks)}
        
    def compute_co_trading_matrix(self, trades_data, date):
        """
        Compute co-trading matrix for a specific date.
        
        Parameters:
        -----------
        trades_data : pandas.DataFrame
            DataFrame containing trade information with columns:
            - timestamp (in microseconds)
            - stock
            - direction ('buy' or 'sell')
            - volume
        date : str
            Date string in format 'YYYY-MM-DD'.
        
        Returns:
        --------
        co_trading_matrix : numpy.ndarray
            Co-trading matrix for the given date.
        """
        # Filter trades for the given date
        if isinstance(date, str):
            date_trades = trades_data[trades_data['date'] == date]
        else:
            date_trades = trades_data
        
        # Initialize co-trading matrix
        co_trading_matrix = np.zeros((self.n_stocks, self.n_stocks))
        
        # Compute co-trading scores
        for i, stock_i in enumerate(self.stocks):
            # Trades for stock i
            stock_i_trades = date_trades[date_trades['stock'] == stock_i]
            n_trades_i = len(stock_i_trades)
            
            if n_trades_i == 0:
                continue
            
            for j, stock_j in enumerate(self.stocks):
                if i == j:
                    continue
                    
                # Trades for stock j
                stock_j_trades = date_trades[date_trades['stock'] == stock_j]
                n_trades_j = len(stock_j_trades)
                
                if n_trades_j == 0:
                    continue
                
                # Count co-occurrences
                co_occurrences = 0
                
                # For each trade of stock i, find co-occurring trades of stock j
                for _, trade_i in stock_i_trades.iterrows():
                    timestamp_i = trade_i['timestamp']
                    
                    # Find trades of stock j that co-occur with trade i
                    co_occurring_trades = stock_j_trades[
                        (stock_j_trades['timestamp'] >= timestamp_i - self.delta) & 
                        (stock_j_trades['timestamp'] <= timestamp_i + self.delta)
                    ]
                    
                    co_occurrences += len(co_occurring_trades)
                
                # For each trade of stock j, find co-occurring trades of stock i
                for _, trade_j in stock_j_trades.iterrows():
                    timestamp_j = trade_j['timestamp']
                    
                    # Find trades of stock i that co-occur with trade j
                    co_occurring_trades = stock_i_trades[
                        (stock_i_trades['timestamp'] >= timestamp_j - self.delta) & 
                        (stock_i_trades['timestamp'] <= timestamp_j + self.delta)
                    ]
                    
                    co_occurrences += len(co_occurring_trades)
                
                # Normalize co-occurrences
                co_trading_matrix[i, j] = co_occurrences / np.sqrt(n_trades_i * n_trades_j)
        
        # Store the co-trading matrix
        self.daily_co_trading_matrices[date] = co_trading_matrix
        
        return co_trading_matrix
    
    def spectral_clustering(self, similarity_matrix, n_clusters):
        """
        Apply spectral clustering to a similarity matrix.
        
        Parameters:
        -----------
        similarity_matrix : numpy.ndarray
            Similarity matrix (co-trading matrix).
        n_clusters : int
            Number of clusters to detect.
        
        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each stock.
        """
        # Compute degree matrix
        D = np.diag(np.sum(similarity_matrix, axis=1))
        
        # Compute Laplacian matrix
        L = D - similarity_matrix
        
        # Compute normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        
        # Compute eigenvectors of normalized Laplacian
        eigenvalues, eigenvectors = eigsh(L_sym, k=n_clusters, which='SM')
        
        # Normalize rows of eigenvectors
        U = eigenvectors
        U_normalized = U / np.sqrt(np.sum(U**2, axis=1, keepdims=True) + 1e-10)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(U_normalized)
        
        return labels
    
    def detect_clusters(self, date, n_clusters):
        """
        Detect clusters for a specific date.
        
        Parameters:
        -----------
        date : str
            Date string in format 'YYYY-MM-DD'.
        n_clusters : int
            Number of clusters to detect.
            
        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each stock.
        """
        # Get co-trading matrix for the date
        if date not in self.daily_co_trading_matrices:
            raise ValueError(f"Co-trading matrix for {date} not found.")
        
        co_trading_matrix = self.daily_co_trading_matrices[date]
        
        # Apply spectral clustering
        labels = self.spectral_clustering(co_trading_matrix, n_clusters)
        
        # Store the clusters
        self.daily_clusters[date] = labels
        
        return labels
    
    def compute_realized_covariance(self, returns_data, date, sampling_interval=5):
        """
        Compute realized covariance matrix for a specific date.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing returns data with columns for each stock.
        date : str
            Date string in format 'YYYY-MM-DD'.
        sampling_interval : int, optional
            Sampling interval in minutes for computing realized covariance.
            
        Returns:
        --------
        realized_cov : numpy.ndarray
            Realized covariance matrix.
        """
        # Filter returns for the given date
        date_returns = returns_data[returns_data.index.date == pd.to_datetime(date).date()]
        
        # Resample returns to desired frequency
        resampled_returns = date_returns.resample(f'{sampling_interval}min').last()
        
        # Compute returns
        log_returns = np.log(resampled_returns / resampled_returns.shift(1)).dropna()
        
        # Compute realized covariance
        realized_cov = log_returns.cov() * (len(log_returns) - 1)
        
        # Store the covariance matrix
        self.daily_covariance_matrices[date] = realized_cov.values
        
        return realized_cov.values
    
    def decompose_covariance(self, cov_matrix, n_factors):
        """
        Decompose covariance matrix into factor and idiosyncratic components.
        
        Parameters:
        -----------
        cov_matrix : numpy.ndarray
            Covariance matrix.
        n_factors : int
            Number of factors to extract.
            
        Returns:
        --------
        factor_cov : numpy.ndarray
            Factor component of covariance.
        idiosyncratic_cov : numpy.ndarray
            Idiosyncratic component of covariance.
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute factor component
        factor_cov = np.zeros_like(cov_matrix)
        for k in range(n_factors):
            factor_cov += eigenvalues[k] * np.outer(eigenvectors[:, k], eigenvectors[:, k])
        
        # Compute idiosyncratic component
        idiosyncratic_cov = cov_matrix - factor_cov
        
        return factor_cov, idiosyncratic_cov
    
    def estimate_robust_covariance(self, date, n_factors, n_clusters, previous_date=None):
        """
        Estimate robust covariance matrix using co-trading networks.
        
        Parameters:
        -----------
        date : str
            Date string in format 'YYYY-MM-DD'.
        n_factors : int
            Number of factors to extract.
        n_clusters : int
            Number of clusters to use.
        previous_date : str, optional
            Previous date string for cluster labels.
            
        Returns:
        --------
        robust_cov : numpy.ndarray
            Robust covariance matrix estimate.
        """
        # Get covariance matrix
        if date not in self.daily_covariance_matrices:
            raise ValueError(f"Covariance matrix for {date} not found.")
        
        cov_matrix = self.daily_covariance_matrices[date]
        
        # Get cluster labels
        cluster_date = previous_date if previous_date is not None else date
        if cluster_date not in self.daily_clusters:
            # Try to detect clusters
            if cluster_date in self.daily_co_trading_matrices:
                self.detect_clusters(cluster_date, n_clusters)
            else:
                raise ValueError(f"Clusters for {cluster_date} not found.")
        
        labels = self.daily_clusters[cluster_date]
        
        # Decompose covariance matrix
        factor_cov, idiosyncratic_cov = self.decompose_covariance(cov_matrix, n_factors)
        
        # Apply block structure to idiosyncratic covariance
        sparse_idiosyncratic_cov = np.zeros_like(idiosyncratic_cov)
        
        for i in range(self.n_stocks):
            for j in range(self.n_stocks):
                if labels[i] == labels[j]:  # Same cluster
                    sparse_idiosyncratic_cov[i, j] = idiosyncratic_cov[i, j]
        
        # Compute robust covariance estimate
        robust_cov = factor_cov + sparse_idiosyncratic_cov
        
        return robust_cov
    
    def compute_optimal_portfolio_weights(self, cov_matrix, leverage_constraint=float('inf')):
        """
        Compute optimal portfolio weights using mean-variance optimization.
        
        Parameters:
        -----------
        cov_matrix : numpy.ndarray
            Covariance matrix.
        leverage_constraint : float, optional
            Constraint on the L1 norm of the weights.
            
        Returns:
        --------
        weights : numpy.ndarray
            Optimal portfolio weights.
        """
        n = cov_matrix.shape[0]
        
        # Check if covariance matrix is well-conditioned
        condition_number = np.linalg.cond(cov_matrix)
        if condition_number > 1e9:
            return np.zeros(n)
        
        # Define optimization variables
        w = cp.Variable(n)
        
        # Define objective
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        
        # Define constraints
        constraints = [cp.sum(w) == 1]
        
        # Add leverage constraint if specified
        if leverage_constraint < float('inf'):
            constraints.append(cp.norm(w, 1) <= leverage_constraint)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Return optimal weights
        return w.value
    
    def backtest_portfolio(self, returns_data, start_date, end_date, n_factors, n_clusters, leverage_constraint=float('inf')):
        """
        Backtest a portfolio using robust covariance estimates.
        
        Parameters:
        -----------
        returns_data : pandas.DataFrame
            DataFrame containing daily returns data with columns for each stock.
        start_date : str
            Start date string in format 'YYYY-MM-DD'.
        end_date : str
            End date string in format 'YYYY-MM-DD'.
        n_factors : int
            Number of factors to extract.
        n_clusters : int
            Number of clusters to use.
        leverage_constraint : float, optional
            Constraint on the L1 norm of the weights.
            
        Returns:
        --------
        portfolio_returns : pandas.Series
            Daily portfolio returns.
        """
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Get list of trading days
        trading_days = returns_data.index[(returns_data.index >= start_date) & (returns_data.index <= end_date)]
        
        # Initialize portfolio weights and returns
        weights = np.zeros(self.n_stocks)
        portfolio_returns = pd.Series(index=trading_days[1:])
        
        # Loop through trading days
        for i in range(1, len(trading_days)):
            current_date = trading_days[i].strftime('%Y-%m-%d')
            previous_date = trading_days[i-1].strftime('%Y-%m-%d')
            
            # Estimate robust covariance using previous day's data
            if previous_date in self.daily_covariance_matrices and previous_date in self.daily_clusters:
                cov_matrix = self.estimate_robust_covariance(previous_date, n_factors, n_clusters)
                weights = self.compute_optimal_portfolio_weights(cov_matrix, leverage_constraint)
            
            # Compute portfolio return
            daily_returns = returns_data.loc[current_date].values
            portfolio_return = np.dot(weights, daily_returns)
            portfolio_returns[current_date] = portfolio_return
        
        return portfolio_returns
    
    def evaluate_portfolio(self, portfolio_returns):
        """
        Evaluate portfolio performance.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Daily portfolio returns.
            
        Returns:
        --------
        metrics : dict
            Dictionary containing performance metrics.
        """
        # Calculate annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak - 1)
        max_drawdown = drawdown.min()
        
        # Return metrics
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        
        return metrics
    
    def plot_network(self, date, threshold=None):
        """
        Plot co-trading network for a specific date.
        
        Parameters:
        -----------
        date : str
            Date string in format 'YYYY-MM-DD'.
        threshold : float, optional
            Threshold value for filtering edges.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        # Get co-trading matrix for the date
        if date not in self.daily_co_trading_matrices:
            raise ValueError(f"Co-trading matrix for {date} not found.")
        
        co_trading_matrix = self.daily_co_trading_matrices[date]
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for i, stock in enumerate(self.stocks):
            G.add_node(stock, sector=self.sectors[stock])
        
        # Add edges
        for i in range(self.n_stocks):
            for j in range(i + 1, self.n_stocks):
                weight = co_trading_matrix[i, j]
                if threshold is None or weight > threshold:
                    G.add_edge(self.stocks[i], self.stocks[j], weight=weight)
        
        # Plot network
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use spring layout for node positions
        pos = nx.spring_layout(G, seed=42)
        
        # Get unique sectors and assign colors
        unique_sectors = list(set(self.sectors.values()))
        sector_colors = {sector: plt.cm.tab10(i / len(unique_sectors)) for i, sector in enumerate(unique_sectors)}
        
        # Node colors based on sectors
        node_colors = [sector_colors[self.sectors[node]] for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
        
        # Draw edges with varying width based on weight
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3)
        
        # Add labels to some nodes
        if len(G.nodes()) <= 50:  # Only add labels if there are not too many nodes
            nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Add legend for sectors
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                      markersize=10, label=sector) 
                          for sector, color in sector_colors.items()]
        ax.legend(handles=legend_elements, title="Sectors")
        
        plt.title(f"Co-trading Network on {date}")
        plt.axis('off')
        
        return fig
    
    def plot_clusters(self, date, n_clusters):
        """
        Plot clusters for a specific date.
        
        Parameters:
        -----------
        date : str
            Date string in format 'YYYY-MM-DD'.
        n_clusters : int
            Number of clusters to detect.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        # Detect clusters if not already done
        if date not in self.daily_clusters or len(self.daily_clusters[date]) != self.n_stocks:
            self.detect_clusters(date, n_clusters)
        
        labels = self.daily_clusters[date]
        
        # Get co-trading matrix
        co_trading_matrix = self.daily_co_trading_matrices[date]
        
        # Reorder matrix based on clusters
        idx = np.argsort(labels)
        reordered_matrix = co_trading_matrix[idx, :][:, idx]
        
        # Create cluster boundaries
        boundaries = []
        current_label = labels[idx[0]]
        for i, label in enumerate(labels[idx]):
            if label != current_label:
                boundaries.append(i)
                current_label = label
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(reordered_matrix, cmap='viridis', ax=ax)
        
        # Add cluster boundaries
        for b in boundaries:
            ax.axhline(b, color='red', linestyle='-', linewidth=1)
            ax.axvline(b, color='red', linestyle='-', linewidth=1)
        
        plt.title(f"Co-trading Matrix Clustered on {date} (K={n_clusters})")
        
        return fig
    
    def plot_correlation_heatmap(self, date):
        """
        Plot correlation heatmap for a specific date.
        
        Parameters:
        -----------
        date : str
            Date string in format 'YYYY-MM-DD'.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        # Get covariance matrix
        if date not in self.daily_covariance_matrices:
            raise ValueError(f"Covariance matrix for {date} not found.")
        
        cov_matrix = self.daily_covariance_matrices[date]
        
        # Compute correlation matrix
        diag_sqrt = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        plt.title(f"Correlation Matrix on {date}")
        
        return fig
    
    def plot_cumulative_returns(self, portfolio_returns, benchmark_returns=None):
        """
        Plot cumulative portfolio returns.
        
        Parameters:
        -----------
        portfolio_returns : pandas.Series
            Daily portfolio returns.
        benchmark_returns : pandas.Series, optional
            Daily benchmark returns for comparison.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object.
        """
        # Calculate cumulative returns
        portfolio_cum_returns = (1 + portfolio_returns).cumprod()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio returns
        ax.plot(portfolio_cum_returns.index, portfolio_cum_returns, label='Portfolio')
        
        # Plot benchmark returns if provided
        if benchmark_returns is not None:
            benchmark_cum_returns = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='Benchmark')
        
        # Add labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Portfolio Returns')
        ax.legend()
        ax.grid(True)
        
        return fig


# Function to simulate trading data
def simulate_trading_data(n_stocks=50, n_days=100, trades_per_day=1000, sectors=None, seed=42):
    """
    Simulate trading data for testing the co-trading networks approach.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate.
    n_days : int
        Number of days to simulate.
    trades_per_day : int
        Average number of trades per day per stock.
    sectors : list, optional
        List of sector names. If None, random sectors are generated.
    seed : int
        Random seed for reproducibility.
    
    Returns:
    --------
    trades_data : pandas.DataFrame
        Simulated trading data.
    returns_data : pandas.DataFrame
        Simulated returns data.
    stock_sectors : dict
        Dictionary mapping stocks to sectors.
    """
    np.random.seed(seed)
    
    # Generate stock symbols
    stocks = [f'STOCK{i:03d}' for i in range(n_stocks)]
    
    # Generate sectors
    if sectors is None:
        sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # Assign stocks to sectors
    stock_sectors = {stock: np.random.choice(sectors) for stock in stocks}
    
    # Define time period
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate correlated stock returns using factor model
    n_factors = 3
    factor_loadings = np.random.normal(0, 1, (n_stocks, n_factors))
    
    # Add sector effect to factor loadings
    sector_to_idx = {sector: i for i, sector in enumerate(set(stock_sectors.values()))}
    for i, stock in enumerate(stocks):
        sector_idx = sector_to_idx[stock_sectors[stock]]
        factor_loadings[i, sector_idx % n_factors] += 2  # Stronger loading on sector-related factor
    
    # Generate factor returns
    factor_returns = np.random.normal(0, 0.01, (n_days, n_factors))
    
    # Generate idiosyncratic returns
    idiosyncratic_returns = np.random.normal(0, 0.02, (n_days, n_stocks))
    
    # Combine to get stock returns
    returns = np.dot(factor_returns, factor_loadings.T) + idiosyncratic_returns
    
    # Convert to DataFrame
    returns_data = pd.DataFrame(returns, index=dates, columns=stocks)
    
    # Generate trading data with co-trading patterns
    trades_list = []
    
    for day_idx, date in enumerate(dates):
        # Base timestamp for this day (market open at 9:30 AM)
        base_timestamp = int(date.timestamp() * 1_000_000) + 9 * 3600 * 1_000_000 + 30 * 60 * 1_000_000
        
        # Trading window (6.5 hours = 23,400 seconds)
        trading_window = 6.5 * 3600 * 1_000_000
        
        # Generate trades for each stock
        for stock_idx, stock in enumerate(stocks):
            # Number of trades for this stock on this day
            n_trades = np.random.poisson(trades_per_day)
            
            # Generate random timestamps throughout the trading day
            timestamps = np.sort(np.random.randint(0, trading_window, n_trades)) + base_timestamp
            
            # Generate trades
            for timestamp in timestamps:
                # Determine if this trade is part of a co-trading event
                is_co_trading = np.random.random() < 0.3
                
                if is_co_trading:
                    # Select other stocks that will co-trade
                    # Stocks in the same sector are more likely to co-trade
                    co_trading_probs = np.array([
                        0.7 if stock_sectors[other_stock] == stock_sectors[stock] else 0.2
                        for other_stock in stocks
                    ])
                    co_trading_probs[stock_idx] = 0  # Don't co-trade with self
                    co_trading_probs = co_trading_probs / co_trading_probs.sum()
                    
                    # Number of stocks to co-trade with
                    n_co_trade = np.random.randint(1, 5)
                    co_trade_stocks = np.random.choice(stocks, size=n_co_trade, p=co_trading_probs, replace=False)
                    
                    # Generate trades for co-trading stocks within a small time window
                    for co_stock in co_trade_stocks:
                        co_timestamp = timestamp + np.random.randint(-400, 400)  # Within 400 microseconds
                        direction = np.random.choice(['buy', 'sell'])
                        volume = np.random.randint(100, 10000)
                        
                        trades_list.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'timestamp': co_timestamp,
                            'stock': co_stock,
                            'direction': direction,
                            'volume': volume
                        })
                
                # Generate the original trade
                direction = np.random.choice(['buy', 'sell'])
                volume = np.random.randint(100, 10000)
                
                trades_list.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'timestamp': timestamp,
                    'stock': stock,
                    'direction': direction,
                    'volume': volume
                })
    
    # Convert to DataFrame
    trades_data = pd.DataFrame(trades_list)
    
    return trades_data, returns_data, stock_sectors


# Function to test the co-trading networks approach
def test_co_trading_networks():
    """
    Test the co-trading networks approach with simulated data.
    """
    print("Simulating trading data...")
    trades_data, returns_data, stock_sectors = simulate_trading_data(n_stocks=30, n_days=60, seed=42)
    
    # Initialize co-trading networks
    co_trading = CoTradingNetworks(delta=500)
    co_trading.set_stocks(list(returns_data.columns), stock_sectors)
    
    print("Processing trading data...")
    
    # Process first 10 days for visualization and analysis
    for date in returns_data.index[:10]:
        date_str = date.strftime('%Y-%m-%d')
        
        # Compute co-trading matrix
        co_trading.compute_co_trading_matrix(trades_data, date_str)
        
        # Compute realized covariance
        co_trading.compute_realized_covariance(returns_data, date_str)
        
        # Detect clusters
        co_trading.detect_clusters(date_str, n_clusters=5)
    
    print("Visualizing co-trading network...")
    
    # Visualize co-trading network for a single day
    sample_date = returns_data.index[5].strftime('%Y-%m-%d')
    fig = co_trading.plot_network(sample_date, threshold=0.1)
    plt.savefig('co_trading_network.png')
    plt.close(fig)
    
    # Visualize clusters
    fig = co_trading.plot_clusters(sample_date, n_clusters=5)
    plt.savefig('co_trading_clusters.png')
    plt.close(fig)
    
    # Visualize correlation heatmap
    fig = co_trading.plot_correlation_heatmap(sample_date)
    plt.savefig('correlation_heatmap.png')
    plt.close(fig)
    
    print("Backtesting portfolio strategies...")
    
    # Process all days for backtesting
    for date in returns_data.index:
        date_str = date.strftime('%Y-%m-%d')
        
        # Compute co-trading matrix
        co_trading.compute_co_trading_matrix(trades_data, date_str)
        
        # Compute realized covariance
        co_trading.compute_realized_covariance(returns_data, date_str)
        
        # Detect clusters
        co_trading.detect_clusters(date_str, n_clusters=5)
    
    # Backtest portfolio strategies
    start_date = returns_data.index[10].strftime('%Y-%m-%d')
    end_date = returns_data.index[-1].strftime('%Y-%m-%d')
    
    # Test different strategies
    strategies = []
    
    # 1. Co-trading clusters with different number of factors
    for n_factors in [1, 3, 5]:
        for n_clusters in [5, 10]:
            portfolio_returns = co_trading.backtest_portfolio(
                returns_data, start_date, end_date, n_factors, n_clusters)
            
            metrics = co_trading.evaluate_portfolio(portfolio_returns)
            strategies.append({
                'name': f'Co-trading (F={n_factors}, C={n_clusters})',
                'returns': portfolio_returns,
                'metrics': metrics
            })
    
    # 2. Benchmark strategy (using sectors as clusters)
    # Create dummy sector clusters
    sector_to_idx = {sector: i for i, sector in enumerate(set(stock_sectors.values()))}
    sector_clusters = np.array([sector_to_idx[stock_sectors[stock]] for stock in co_trading.stocks])
    
    for date in returns_data.index:
        date_str = date.strftime('%Y-%m-%d')
        co_trading.daily_clusters[date_str] = sector_clusters
    
    for n_factors in [1, 3, 5]:
        portfolio_returns = co_trading.backtest_portfolio(
            returns_data, start_date, end_date, n_factors, n_clusters=len(set(stock_sectors.values())))
        
        metrics = co_trading.evaluate_portfolio(portfolio_returns)
        strategies.append({
            'name': f'Sector-based (F={n_factors})',
            'returns': portfolio_returns,
            'metrics': metrics
        })
    
    # 3. Equally weighted benchmark
    equal_weight_returns = returns_data.loc[start_date:end_date].mean(axis=1)
    metrics = co_trading.evaluate_portfolio(equal_weight_returns)
    strategies.append({
        'name': 'Equal Weight',
        'returns': equal_weight_returns,
        'metrics': metrics
    })
    
    # Print results
    print("\nPortfolio Strategy Results:")
    print("-" * 80)
    print(f"{'Strategy':30} {'Annual Return':15} {'Annual Vol':15} {'Sharpe Ratio':15} {'Max Drawdown':15}")
    print("-" * 80)
    
    for strategy in strategies:
        name = strategy['name']
        metrics = strategy['metrics']
        
        print(f"{name:30} {metrics['Annual Return']:15.4f} {metrics['Annual Volatility']:15.4f} "
              f"{metrics['Sharpe Ratio']:15.4f} {metrics['Max Drawdown']:15.4f}")
    
    # Plot cumulative returns for all strategies
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for strategy in strategies:
        name = strategy['name']
        returns = strategy['returns']
        cum_returns = (1 + returns).cumprod()
        ax.plot(cum_returns.index, cum_returns, label=name)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Comparison of Portfolio Strategies')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('portfolio_comparison.png')
    plt.close(fig)
    
    print("\nResults saved to portfolio_comparison.png")
    
    # Compare the best co-trading strategy with the best sector-based strategy
    best_co_trading = max([s for s in strategies if 'Co-trading' in s['name']], 
                         key=lambda x: x['metrics']['Sharpe Ratio'])
    best_sector = max([s for s in strategies if 'Sector-based' in s['name']], 
                     key=lambda x: x['metrics']['Sharpe Ratio'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cum_returns_co = (1 + best_co_trading['returns']).cumprod()
    cum_returns_sector = (1 + best_sector['returns']).cumprod()
    
    ax.plot(cum_returns_co.index, cum_returns_co, label=f"Best Co-trading ({best_co_trading['name']})")
    ax.plot(cum_returns_sector.index, cum_returns_sector, label=f"Best Sector-based ({best_sector['name']})")
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Best Co-trading vs. Best Sector-based Strategy')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('best_strategies_comparison.png')
    plt.close(fig)
    
    print("Best strategies comparison saved to best_strategies_comparison.png")
    
    return co_trading, strategies


if __name__ == "__main__":
    co_trading, strategies = test_co_trading_networks()