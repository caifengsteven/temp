import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class GraphCentralityPortfolio:
    """
    A class to implement portfolio management using graph centrality measures
    as described in the paper "Portfolio Management Using Graph Centralities"
    """
    
    def __init__(self, window_size=125, shrinkage=False):
        """
        Initialize the portfolio strategy.
        
        Parameters:
        -----------
        window_size : int
            The size of the rolling window for correlation computation
        shrinkage : bool
            Whether to apply shrinkage to the correlation matrix
        """
        self.window_size = window_size
        self.shrinkage = shrinkage
    
    def simulate_stock_data(self, n_stocks=50, n_days=1000, n_clusters=5):
        """
        Simulate stock price data with correlation structure.
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        n_days : int
            Number of days to simulate
        n_clusters : int
            Number of clusters for correlation structure
        
        Returns:
        --------
        prices : pd.DataFrame
            Simulated stock prices
        returns : pd.DataFrame
            Simulated stock returns
        """
        # Assign stocks to clusters
        stocks_per_cluster = n_stocks // n_clusters
        clusters = []
        for i in range(n_clusters):
            clusters.extend([i] * stocks_per_cluster)
        
        # Add any remaining stocks to the last cluster
        remaining = n_stocks - len(clusters)
        if remaining > 0:
            clusters.extend([n_clusters-1] * remaining)
        
        # Create correlation matrix with cluster structure
        correlation = np.zeros((n_stocks, n_stocks))
        
        # Set diagonal to 1
        np.fill_diagonal(correlation, 1)
        
        # Set within-cluster correlation
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                if clusters[i] == clusters[j]:
                    # Higher correlation within cluster
                    correlation[i, j] = 0.6 + 0.2 * np.random.rand()
                else:
                    # Lower correlation between clusters
                    correlation[i, j] = 0.2 * np.random.rand()
                
                # Make correlation matrix symmetric
                correlation[j, i] = correlation[i, j]
        
        # Mean and standard deviation for returns
        mu = np.random.normal(0.0005, 0.0002, n_stocks)  # Daily mean returns around 0.05% (12.5% annual)
        sigma = np.random.uniform(0.01, 0.02, n_stocks)  # Daily volatility between 1% and 2%
        
        # Compute covariance matrix from correlation and volatilities
        cov_matrix = np.outer(sigma, sigma) * correlation
        
        # Generate multivariate normal returns
        returns = np.random.multivariate_normal(mu, cov_matrix, n_days)
        
        # Convert to DataFrame
        stock_names = [f'Stock_{i}' for i in range(n_stocks)]
        returns_df = pd.DataFrame(returns, columns=stock_names)
        
        # Convert returns to prices starting at 100
        prices_df = 100 * (1 + returns_df).cumprod()
        
        # Add dates as index
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        returns_df.index = dates
        prices_df.index = dates
        
        return prices_df, returns_df
    
    def compute_correlation_matrix(self, returns, exponential_weighting=True):
        """
        Compute the correlation matrix with optional exponential weighting.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        exponential_weighting : bool
            Whether to use exponential weighting for the correlation
        
        Returns:
        --------
        C : np.ndarray
            Correlation matrix
        """
        if len(returns) < self.window_size:
            raise ValueError(f"Not enough data points. Need at least {self.window_size}.")
        
        if exponential_weighting:
            # Compute exponentially weighted correlation matrix as in the paper
            correlations = []
            tau = self.window_size
            
            # Compute weights
            weights = np.zeros(tau)
            for t in range(1, tau + 1):
                weights[t-1] = np.exp((t - tau) / tau)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Compute correlation for each window and sum weighted correlations
            weighted_corr = np.zeros((returns.shape[1], returns.shape[1]))
            
            for t in range(1, tau + 1):
                window_returns = returns.iloc[-tau-t:-t] if t < tau else returns.iloc[:-t]
                corr = window_returns.corr().values
                weighted_corr += weights[t-1] * corr
            
            C = weighted_corr
        else:
            # Simple correlation matrix
            C = returns.corr().values
        
        # Apply shrinkage if requested
        if self.shrinkage:
            lw = LedoitWolf()
            cov = lw.fit(returns).covariance_
            
            # Convert covariance to correlation
            std = np.sqrt(np.diag(cov))
            C = cov / np.outer(std, std)
        
        return C
    
    def create_adjacency_matrix(self, C, method=1, threshold=0.5):
        """
        Create an adjacency matrix from the correlation matrix.
        
        Parameters:
        -----------
        C : np.ndarray
            Correlation matrix
        method : int
            Method to transform correlation to adjacency (1-8 as in the paper)
        threshold : float
            Threshold value for filtering correlations
        
        Returns:
        --------
        A : np.ndarray
            Adjacency matrix
        """
        # Create identity matrix of same size as C
        I = np.eye(C.shape[0])
        
        # Apply different methods as described in the paper
        if method == 1:
            # C > θ
            A = (C > threshold).astype(float)
        elif method == 2:
            # |C| > θ
            A = (np.abs(C) > threshold).astype(float)
        elif method == 3:
            # (C - I) > θ
            A = ((C - I) > threshold).astype(float)
        elif method == 4:
            # (|C| - I) > θ
            A = ((np.abs(C) - I) > threshold).astype(float)
        elif method == 5:
            # [C > θ] ◦ C
            A = np.where(C > threshold, C, 0)
        elif method == 6:
            # [|C| > θ] ◦ |C|
            A = np.where(np.abs(C) > threshold, np.abs(C), 0)
        elif method == 7:
            # [(C - I) > θ] ◦ (C - I)
            A = np.where((C - I) > threshold, (C - I), 0)
        elif method == 8:
            # [(|C| - I) > θ] ◦ (|C| - I)
            A = np.where((np.abs(C) - I) > threshold, (np.abs(C) - I), 0)
        else:
            raise ValueError(f"Method {method} not implemented")
            
        return A
    
    def compute_centrality(self, A, method='degree', alpha_ratio=0.5):
        """
        Compute centrality measures for the graph.
        
        Parameters:
        -----------
        A : np.ndarray
            Adjacency matrix
        method : str
            Centrality method ('degree', 'katz', 'eigenvector', etc.)
        alpha_ratio : float
            Value of α as a ratio of maximum allowed value (0 to 1)
        
        Returns:
        --------
        centrality : np.ndarray
            Vector of centrality values for each node
        """
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(A)
        
        if method == 'degree':
            # Degree centrality
            centrality_dict = nx.degree_centrality(G)
            centrality = np.array([centrality_dict[i] for i in range(len(A))])
            
        elif method == 'eigenvector':
            # Eigenvector centrality
            centrality_dict = nx.eigenvector_centrality_numpy(G)
            centrality = np.array([centrality_dict[i] for i in range(len(A))])
            
        elif method == 'katz':
            # Katz centrality
            # Compute maximum alpha value
            if nx.is_connected(G):
                alpha_max = 1.0 / max(nx.adjacency_spectrum(G))
            else:
                # For disconnected graphs, use power iteration to find largest eigenvalue
                eigen_values = np.linalg.eigvals(A)
                alpha_max = 1.0 / max(abs(eigen_values))
            
            alpha = alpha_ratio * alpha_max
            
            try:
                centrality_dict = nx.katz_centrality_numpy(G, alpha=alpha)
                centrality = np.array([centrality_dict[i] for i in range(len(A))])
            except:
                # Fallback to eigenvector centrality if Katz fails
                centrality_dict = nx.eigenvector_centrality_numpy(G)
                centrality = np.array([centrality_dict[i] for i in range(len(A))])
                
        elif method == 'katz_min':
            # Katz centrality with special alpha value as in the paper
            if nx.is_connected(G):
                alpha_max = 1.0 / max(nx.adjacency_spectrum(G))
            else:
                eigen_values = np.linalg.eigvals(A)
                alpha_max = 1.0 / max(abs(eigen_values))
            
            alpha = (1 - np.exp(-alpha_max)) / alpha_max
            
            try:
                centrality_dict = nx.katz_centrality_numpy(G, alpha=alpha)
                centrality = np.array([centrality_dict[i] for i in range(len(A))])
            except:
                # Fallback to eigenvector centrality if Katz fails
                centrality_dict = nx.eigenvector_centrality_numpy(G)
                centrality = np.array([centrality_dict[i] for i in range(len(A))])
        
        elif method == 'betweenness':
            # Betweenness centrality
            centrality_dict = nx.betweenness_centrality(G)
            centrality = np.array([centrality_dict[i] for i in range(len(A))])
            
        elif method == 'exponential':
            # Exponential centrality
            alpha = alpha_ratio
            exp_A = np.exp(alpha * A)
            centrality = exp_A @ np.ones(len(A))
            
        elif method == 'exponential_subgraph':
            # Exponential subgraph centrality
            alpha = alpha_ratio
            exp_A = np.exp(alpha * A)
            centrality = np.diag(exp_A)
            
        elif method == 'subgraph':
            # Subgraph centrality (Katz variant)
            if nx.is_connected(G):
                alpha_max = 1.0 / max(nx.adjacency_spectrum(G))
            else:
                eigen_values = np.linalg.eigvals(A)
                alpha_max = 1.0 / max(abs(eigen_values))
            
            alpha = alpha_ratio * alpha_max
            
            try:
                I = np.eye(len(A))
                katz_matrix = np.linalg.inv(I - alpha * A)
                centrality = np.diag(katz_matrix)
            except:
                # Fallback
                centrality = np.ones(len(A))
                
        elif method == 'nbtw':
            # NBTW centrality as described in the paper
            # This is a simplified version as the full implementation is complex
            # In a real implementation, you would use the formulae from the paper
            if nx.is_connected(G):
                alpha_max = 1.0 / max(nx.adjacency_spectrum(G))
            else:
                eigen_values = np.linalg.eigvals(A)
                alpha_max = 1.0 / max(abs(eigen_values))
            
            # For NBTW, alpha_max is larger than for Katz
            nbtw_alpha_max = alpha_max * 1.2
            alpha = alpha_ratio * nbtw_alpha_max
            
            # Simplified NBTW for undirected unweighted graphs
            D = np.diag(np.sum(A, axis=1))
            I = np.eye(len(A))
            
            try:
                # (1 - α²)(I - αA + α²(D - I))^(-1)1
                M = (1 - alpha**2) * np.linalg.inv(I - alpha * A + alpha**2 * (D - I))
                centrality = M @ np.ones(len(A))
            except:
                # Fallback
                centrality = np.ones(len(A))
                
        elif method == 'nbtw_subgraph':
            # NBTW subgraph centrality
            if nx.is_connected(G):
                alpha_max = 1.0 / max(nx.adjacency_spectrum(G))
            else:
                eigen_values = np.linalg.eigvals(A)
                alpha_max = 1.0 / max(abs(eigen_values))
            
            # For NBTW, alpha_max is larger than for Katz
            nbtw_alpha_max = alpha_max * 1.2
            alpha = alpha_ratio * nbtw_alpha_max
            
            # Simplified NBTW for undirected unweighted graphs
            D = np.diag(np.sum(A, axis=1))
            I = np.eye(len(A))
            
            try:
                # (1 - α²)(I - αA + α²(D - I))^(-1)
                M = (1 - alpha**2) * np.linalg.inv(I - alpha * A + alpha**2 * (D - I))
                centrality = np.diag(M)
            except:
                # Fallback
                centrality = np.ones(len(A))
        
        else:
            raise ValueError(f"Centrality method {method} not implemented")
            
        return centrality
    
    def select_stocks(self, centrality, num_stocks=10, central=True):
        """
        Select stocks based on centrality.
        
        Parameters:
        -----------
        centrality : np.ndarray
            Vector of centrality values
        num_stocks : int
            Number of stocks to select
        central : bool
            If True, select most central stocks, otherwise most peripheral
        
        Returns:
        --------
        selected : list
            Indices of selected stocks
        """
        if central:
            # Select most central stocks
            selected = np.argsort(centrality)[-num_stocks:]
        else:
            # Select most peripheral stocks
            selected = np.argsort(centrality)[:num_stocks]
            
        return selected
    
    def construct_portfolio(self, returns, selected_stocks, method='equal_weight', 
                           allow_short=False, max_weight=0.25, target_return=None):
        """
        Construct portfolio using selected stocks.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        selected_stocks : list
            Indices of selected stocks
        method : str
            Portfolio construction method ('equal_weight', 'min_var', 'mean_var')
        allow_short : bool
            Whether to allow short selling
        max_weight : float
            Maximum weight for any asset
        target_return : float
            Target return for mean-variance optimization
            
        Returns:
        --------
        weights : np.ndarray
            Portfolio weights
        """
        # Extract returns of selected stocks
        selected_returns = returns.iloc[:, selected_stocks]
        
        n = len(selected_stocks)
        
        if method == 'equal_weight':
            # Equal-weighted portfolio
            weights = np.ones(n) / n
            
        elif method == 'min_var':
            # Minimum variance portfolio
            cov_matrix = selected_returns.cov().values
            
            # Optimization problem
            def objective(w):
                return w @ cov_matrix @ w
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # Bounds
            if allow_short:
                bounds = [(-max_weight, max_weight) for _ in range(n)]
            else:
                bounds = [(0, max_weight) for _ in range(n)]
                
            # Initial guess
            x0 = np.ones(n) / n
            
            # Solve
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            weights = result.x
            
        elif method == 'mean_var':
            # Mean-variance portfolio
            cov_matrix = selected_returns.cov().values
            mean_returns = selected_returns.mean().values
            
            if target_return is None:
                target_return = np.mean(mean_returns)
            
            # Optimization problem
            def objective(w):
                return w @ cov_matrix @ w
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}
            ]
            
            # Bounds
            if allow_short:
                bounds = [(-max_weight, max_weight) for _ in range(n)]
            else:
                bounds = [(0, max_weight) for _ in range(n)]
                
            # Initial guess
            x0 = np.ones(n) / n
            
            try:
                # Solve
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                weights = result.x
                
                # If optimization fails (e.g., target return not achievable), fallback to min var
                if not result.success:
                    return self.construct_portfolio(returns, selected_stocks, 'min_var', allow_short, max_weight)
            except:
                # Fallback to min var if optimization fails
                return self.construct_portfolio(returns, selected_stocks, 'min_var', allow_short, max_weight)
            
        else:
            raise ValueError(f"Portfolio method {method} not implemented")
            
        return weights
    
    def calculate_portfolio_return(self, weights, returns, selected_stocks):
        """
        Calculate portfolio return.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame
            Asset returns
        selected_stocks : list
            Indices of selected stocks
            
        Returns:
        --------
        portfolio_return : float
            Portfolio return
        """
        # Extract returns of selected stocks
        selected_returns = returns.iloc[:, selected_stocks]
        
        # Calculate portfolio return
        portfolio_return = selected_returns.values @ weights
        
        return portfolio_return
    
    def calculate_performance_metrics(self, portfolio_returns, risk_free_rate=0.0):
        """
        Calculate performance metrics for a portfolio.
        
        Parameters:
        -----------
        portfolio_returns : np.ndarray
            Time series of portfolio returns
        risk_free_rate : float
            Risk-free rate (annualized)
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate metrics
        mean_return = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        sharpe_ratio = (mean_return - daily_rf) / std_dev * np.sqrt(252)
        
        # Cumulative return
        cum_return = np.prod(1 + portfolio_returns) - 1
        
        # Annualized return
        annual_return = (1 + cum_return) ** (252 / len(portfolio_returns)) - 1
        
        # Maximum drawdown
        cum_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Downside deviation (for Sortino ratio)
        negative_returns = portfolio_returns[portfolio_returns < daily_rf]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = (mean_return - daily_rf) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else np.nan
        
        # Omega ratio
        threshold = daily_rf
        gains = portfolio_returns[portfolio_returns > threshold] - threshold
        losses = threshold - portfolio_returns[portfolio_returns < threshold]
        omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
        
        # Upside potential
        upside_returns = portfolio_returns[portfolio_returns > threshold] - threshold
        upside_potential = np.mean(upside_returns) if len(upside_returns) > 0 else 0
        upside_potential_ratio = upside_potential / downside_deviation if downside_deviation > 0 else np.nan
        
        # Create metrics dictionary
        metrics = {
            'Expected Return (Daily)': mean_return,
            'Expected Return (Annual)': annual_return,
            'Standard Deviation (Daily)': std_dev,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Cumulative Return': cum_return,
            'Maximum Drawdown': max_drawdown,
            'Value at Risk (95%)': var_95,
            'Conditional VaR (95%)': cvar_95,
            'Omega Ratio': omega_ratio,
            'Upside Potential Ratio': upside_potential_ratio
        }
        
        return metrics
    
    def backtest(self, returns, training_window=252, rebalance_freq=21, num_stocks=10,
                centrality_method='exponential_subgraph', adjacency_method=7, threshold=0.5,
                alpha_ratio=0.5, portfolio_method='equal_weight', central=False, 
                allow_short=False, risk_free_rate=0.02):
        """
        Backtest the strategy.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        training_window : int
            Number of days for training
        rebalance_freq : int
            Rebalance frequency in days
        num_stocks : int
            Number of stocks to select
        centrality_method : str
            Centrality method
        adjacency_method : int
            Method to transform correlation to adjacency
        threshold : float
            Threshold value for filtering correlations
        alpha_ratio : float
            Alpha ratio for centrality measures
        portfolio_method : str
            Portfolio construction method
        central : bool
            If True, select most central stocks, otherwise most peripheral
        allow_short : bool
            Whether to allow short selling
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        portfolio_returns : pd.Series
            Time series of portfolio returns
        metrics : dict
            Performance metrics
        """
        # Check if there's enough data
        if len(returns) < training_window:
            raise ValueError(f"Not enough data. Need at least {training_window} days.")
        
        # Initialize variables
        portfolio_returns = []
        weights_history = []
        selected_stocks_history = []
        
        # Loop through time
        for t in range(training_window, len(returns), rebalance_freq):
            # Get training data
            train_returns = returns.iloc[t-training_window:t]
            
            # Compute correlation matrix
            C = self.compute_correlation_matrix(train_returns)
            
            # Create adjacency matrix
            A = self.create_adjacency_matrix(C, method=adjacency_method, threshold=threshold)
            
            # Compute centrality
            centrality = self.compute_centrality(A, method=centrality_method, alpha_ratio=alpha_ratio)
            
            # Select stocks
            selected_stocks = self.select_stocks(centrality, num_stocks=num_stocks, central=central)
            selected_stocks_history.append(selected_stocks)
            
            # Construct portfolio
            weights = self.construct_portfolio(train_returns, selected_stocks, 
                                             method=portfolio_method, allow_short=allow_short)
            weights_history.append(weights)
            
            # Get test data (next rebalance_freq days or remaining days)
            test_end = min(t + rebalance_freq, len(returns))
            test_returns = returns.iloc[t:test_end]
            
            # Calculate portfolio returns
            for i in range(len(test_returns)):
                daily_return = self.calculate_portfolio_return(weights, test_returns.iloc[[i]], selected_stocks)
                portfolio_returns.append(daily_return[0])
        
        # Convert to Series
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index[training_window:training_window+len(portfolio_returns)])
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio_returns.values, risk_free_rate)
        
        return portfolio_returns, metrics, weights_history, selected_stocks_history

# Function to run multiple portfolio strategies and compare results
def run_multiple_strategies(returns, strategies):
    """
    Run multiple portfolio strategies and compare results.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    strategies : list
        List of strategy dictionaries with parameters
        
    Returns:
    --------
    results : pd.DataFrame
        DataFrame of performance metrics for each strategy
    portfolio_returns : dict
        Dictionary of portfolio returns for each strategy
    """
    # Initialize results storage
    results = []
    portfolio_returns = {}
    
    # Create strategy instance
    strategy = GraphCentralityPortfolio()
    
    # Run each strategy
    for i, params in enumerate(strategies):
        print(f"Running strategy {i+1}/{len(strategies)}: {params['name']}")
        
        # Backtest the strategy
        returns_series, metrics, weights_history, selected_stocks_history = strategy.backtest(
            returns,
            training_window=params.get('training_window', 252),
            rebalance_freq=params.get('rebalance_freq', 21),
            num_stocks=params.get('num_stocks', 10),
            centrality_method=params.get('centrality_method', 'exponential_subgraph'),
            adjacency_method=params.get('adjacency_method', 7),
            threshold=params.get('threshold', 0.5),
            alpha_ratio=params.get('alpha_ratio', 0.5),
            portfolio_method=params.get('portfolio_method', 'equal_weight'),
            central=params.get('central', False),
            allow_short=params.get('allow_short', False)
        )
        
        # Store results
        metrics['Strategy'] = params['name']
        results.append(metrics)
        portfolio_returns[params['name']] = returns_series
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, portfolio_returns

# Function to plot portfolio performance
def plot_portfolio_performance(portfolio_returns, benchmark_returns=None):
    """
    Plot portfolio performance.
    
    Parameters:
    -----------
    portfolio_returns : dict
        Dictionary of portfolio returns for each strategy
    benchmark_returns : pd.Series
        Benchmark returns (e.g., market index)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns for each strategy
    for name, returns in portfolio_returns.items():
        cum_returns = (1 + returns).cumprod()
        plt.plot(cum_returns.index, cum_returns, label=name)
    
    # Plot benchmark if provided
    if benchmark_returns is not None:
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, 'k--', label='Benchmark')
    
    # Add labels and legend
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create instance of the strategy
    strategy = GraphCentralityPortfolio()
    
    # Simulate stock data
    print("Simulating stock data...")
    prices, returns = strategy.simulate_stock_data(n_stocks=50, n_days=1000, n_clusters=5)
    
    # Define strategies to test
    strategies = [
        {
            'name': 'Peripheral - Equal Weight',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Central - Equal Weight',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'equal_weight',
            'central': True
        },
        {
            'name': 'Peripheral - Min Var',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'min_var',
            'central': False
        },
        {
            'name': 'Central - Min Var',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'min_var',
            'central': True
        },
        {
            'name': 'Peripheral - Mean Var',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'mean_var',
            'central': False
        },
        {
            'name': 'Central - Mean Var',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'alpha_ratio': 0.5,
            'portfolio_method': 'mean_var',
            'central': True
        },
        {
            'name': 'Peripheral - Katz',
            'centrality_method': 'katz',
            'adjacency_method': 4,
            'threshold': 0.6,
            'alpha_ratio': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Central - NBTW Subgraph',
            'centrality_method': 'nbtw_subgraph',
            'adjacency_method': 2,
            'threshold': 0.7,
            'alpha_ratio': 0.3,
            'portfolio_method': 'mean_var',
            'central': True,
            'allow_short': True
        }
    ]
    
    # Run strategies
    results_df, portfolio_returns = run_multiple_strategies(returns, strategies)
    
    # Create equal-weight benchmark
    benchmark_returns = returns.mean(axis=1)
    
    # Plot results
    plot_portfolio_performance(portfolio_returns, benchmark_returns)
    
    # Print performance metrics
    pd.set_option('display.precision', 4)
    print("\nPerformance Metrics:")
    print(results_df[['Strategy', 'Sharpe Ratio', 'Expected Return (Annual)', 
                     'Maximum Drawdown', 'Sortino Ratio']])
    
    # Additional analysis: comparing centrality methods
    print("\nComparing different centrality methods on peripheral stocks with equal weighting...")
    
    centrality_strategies = [
        {
            'name': 'Degree',
            'centrality_method': 'degree',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Eigenvector',
            'centrality_method': 'eigenvector',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Katz',
            'centrality_method': 'katz',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Katz Min',
            'centrality_method': 'katz_min',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Betweenness',
            'centrality_method': 'betweenness',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Exponential',
            'centrality_method': 'exponential',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'Exponential Subgraph',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'NBTW',
            'centrality_method': 'nbtw',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        },
        {
            'name': 'NBTW Subgraph',
            'centrality_method': 'nbtw_subgraph',
            'adjacency_method': 7,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        }
    ]
    
    # Run centrality comparison
    centrality_results, centrality_returns = run_multiple_strategies(returns, centrality_strategies)
    
    # Plot centrality comparison
    plot_portfolio_performance(centrality_returns, benchmark_returns)
    
    # Print centrality comparison metrics
    print("\nCentrality Methods Comparison:")
    print(centrality_results[['Strategy', 'Sharpe Ratio', 'Expected Return (Annual)', 
                            'Maximum Drawdown', 'Sortino Ratio']])
    
    # Compare different threshold values
    print("\nComparing different threshold values...")
    
    threshold_strategies = []
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        threshold_strategies.append({
            'name': f'Threshold {threshold}',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': 7,
            'threshold': threshold,
            'portfolio_method': 'equal_weight',
            'central': False
        })
    
    # Run threshold comparison
    threshold_results, threshold_returns = run_multiple_strategies(returns, threshold_strategies)
    
    # Plot threshold comparison
    plot_portfolio_performance(threshold_returns)
    
    # Print threshold comparison metrics
    print("\nThreshold Values Comparison:")
    print(threshold_results[['Strategy', 'Sharpe Ratio', 'Expected Return (Annual)', 
                           'Maximum Drawdown', 'Sortino Ratio']])
    
    # Compare different adjacency matrix methods
    print("\nComparing different adjacency matrix methods...")
    
    adjacency_strategies = []
    for method in range(1, 9):
        adjacency_strategies.append({
            'name': f'Adjacency Method {method}',
            'centrality_method': 'exponential_subgraph',
            'adjacency_method': method,
            'threshold': 0.5,
            'portfolio_method': 'equal_weight',
            'central': False
        })
    
    # Run adjacency comparison
    adjacency_results, adjacency_returns = run_multiple_strategies(returns, adjacency_strategies)
    
    # Plot adjacency comparison
    plot_portfolio_performance(adjacency_returns)
    
    # Print adjacency comparison metrics
    print("\nAdjacency Matrix Methods Comparison:")
    print(adjacency_results[['Strategy', 'Sharpe Ratio', 'Expected Return (Annual)', 
                           'Maximum Drawdown', 'Sortino Ratio']])