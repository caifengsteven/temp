import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class GroupDiscoveryIndexTracker:
    """
    Trading strategy that discovers asset groups and applies group-based tracking.
    """
    
    def __init__(self, n_groups=None, group_method='correlation', min_group_size=3):
        """
        Initialize the group discovery tracker.
        
        Parameters:
        -----------
        n_groups : int or None
            Number of groups to discover. If None, determine automatically.
        group_method : str
            Method for group discovery: 'correlation', 'returns', 'factor', 'hybrid'
        min_group_size : int
            Minimum number of assets per group
        """
        self.n_groups = n_groups
        self.group_method = group_method
        self.min_group_size = min_group_size
        self.groups_ = None
        self.group_weights_ = None
        self.asset_weights_ = None
        
    def discover_groups(self, X):
        """
        Discover asset groups using various methods.
        """
        n_samples, n_assets = X.shape
        
        if self.group_method == 'correlation':
            # Use correlation clustering
            corr_matrix = np.corrcoef(X.T)
            distance_matrix = 1 - np.abs(corr_matrix)
            
            if self.n_groups is None:
                # Determine optimal number of groups using silhouette score
                from sklearn.metrics import silhouette_score
                scores = []
                for k in range(2, min(20, n_assets // self.min_group_size)):
                    clusters = AgglomerativeClustering(n_clusters=k, linkage='average')
                    labels = clusters.fit_predict(distance_matrix)
                    try:
                        score = silhouette_score(distance_matrix, labels, metric='precomputed')
                        scores.append(score)
                    except:
                        scores.append(-1)
                
                if scores and max(scores) > 0:
                    self.n_groups = np.argmax(scores) + 2
                else:
                    self.n_groups = 5  # Default
            
            clustering = AgglomerativeClustering(
                n_clusters=self.n_groups, 
                linkage='average',
                metric='precomputed'
            )
            labels = clustering.fit_predict(distance_matrix)
            
        elif self.group_method == 'returns':
            # Cluster based on return patterns
            if self.n_groups is None:
                self.n_groups = self._estimate_n_groups(X.T)
            
            kmeans = KMeans(n_clusters=self.n_groups, random_state=42)
            labels = kmeans.fit_predict(X.T)
            
        elif self.group_method == 'factor':
            # Use factor analysis
            n_components = min(10, n_assets // 3, n_samples // 2)
            pca = PCA(n_components=n_components)
            factors = pca.fit_transform(X)
            loadings = pca.components_.T
            
            # Cluster based on factor loadings
            if self.n_groups is None:
                self.n_groups = self._estimate_n_groups(loadings)
            
            kmeans = KMeans(n_clusters=self.n_groups, random_state=42)
            labels = kmeans.fit_predict(loadings)
            
        elif self.group_method == 'hybrid':
            # Combine correlation and return patterns
            corr_matrix = np.corrcoef(X.T)
            
            # First principal component
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(X)
            pc1_corr = np.array([np.corrcoef(X[:, i], pc1.ravel())[0, 1] for i in range(n_assets)])
            
            # Create feature matrix
            features = np.column_stack([
                corr_matrix.mean(axis=1),  # Average correlation
                X.mean(axis=0),  # Mean returns
                X.std(axis=0),  # Volatility
                pc1_corr  # Correlation with first PC
            ])
            
            if self.n_groups is None:
                self.n_groups = self._estimate_n_groups(features)
            
            kmeans = KMeans(n_clusters=self.n_groups, random_state=42)
            labels = kmeans.fit_predict(features)
        
        # Store groups
        self.groups_ = {}
        group_id = 0
        for i in range(self.n_groups):
            group_assets = np.where(labels == i)[0]
            if len(group_assets) >= self.min_group_size:
                self.groups_[group_id] = group_assets
                group_id += 1
        
        return self.groups_
    
    def _estimate_n_groups(self, data, max_groups=20):
        """Estimate optimal number of groups using elbow method."""
        max_k = min(max_groups, data.shape[0] // self.min_group_size, 10)
        if max_k < 2:
            return 3
            
        inertias = []
        K = range(2, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow = np.argmax(diffs2) + 2
            return min(elbow, max_k)
        else:
            return 3
    
    def fit(self, X, y, method='group_lasso', alpha=0.001):
        """
        Fit the group-based tracking model.
        
        Parameters:
        -----------
        X : np.ndarray
            Asset returns (T x N)
        y : np.ndarray
            Index returns (T x 1)
        method : str
            'group_lasso', 'group_ols', 'group_ridge'
        alpha : float
            Regularization parameter
        """
        # Discover groups
        self.groups_ = self.discover_groups(X)
        n_groups = len(self.groups_)
        
        if n_groups == 0:
            # Fallback to equal weights if no groups found
            n_assets = X.shape[1]
            self.asset_weights_ = np.ones(n_assets) / n_assets
            return self
        
        # Create group return matrix
        X_grouped = np.zeros((X.shape[0], n_groups))
        for i, group_assets in self.groups_.items():
            # Use mean return of group
            X_grouped[:, i] = X[:, group_assets].mean(axis=1)
        
        # Fit group weights
        if method == 'group_lasso':
            model = Lasso(alpha=alpha, positive=True, fit_intercept=False)
            model.fit(X_grouped, y)
            self.group_weights_ = model.coef_
            
        elif method == 'group_ols':
            # OLS with non-negativity
            try:
                self.group_weights_ = np.linalg.lstsq(X_grouped, y, rcond=None)[0]
                self.group_weights_ = np.maximum(0, self.group_weights_)
            except:
                self.group_weights_ = np.ones(n_groups) / n_groups
                
        elif method == 'group_ridge':
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_grouped, y)
            self.group_weights_ = np.maximum(0, model.coef_)
        
        # Normalize group weights
        if self.group_weights_.sum() > 0:
            self.group_weights_ = self.group_weights_ / self.group_weights_.sum()
        else:
            self.group_weights_ = np.ones(n_groups) / n_groups
        
        # Distribute to individual assets
        self.asset_weights_ = np.zeros(X.shape[1])
        for i, group_assets in self.groups_.items():
            weight_per_asset = self.group_weights_[i] / len(group_assets)
            self.asset_weights_[group_assets] = weight_per_asset
        
        return self
    
    def get_weights(self):
        """Get the final asset weights."""
        return self.asset_weights_


class AdaptiveGroupTracker:
    """
    Adaptive trading strategy that updates groups over time.
    """
    
    def __init__(self, lookback_window=120, rebalance_frequency=20, 
                 min_group_correlation=0.7):
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.min_group_correlation = min_group_correlation
        self.tracker = None
        
    def backtest(self, prices, index_prices, start_date=None):
        """
        Backtest the adaptive group tracking strategy.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset prices (dates x assets)
        index_prices : pd.Series
            Index prices
        start_date : str or pd.Timestamp
            Start date for backtest
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        index_returns = index_prices.pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(index_returns.index)
        returns = returns.loc[common_dates]
        index_returns = index_returns.loc[common_dates]
        
        if start_date:
            returns = returns[returns.index >= start_date]
            index_returns = index_returns[index_returns.index >= start_date]
        
        # Initialize results
        portfolio_values = [1.0]
        weights_history = []
        groups_history = []
        dates = []
        turnover_history = []
        
        # Previous weights for turnover calculation
        prev_weights = None
        
        # Backtest loop
        n_periods = len(returns)
        n_rebalances = 0
        
        for i in range(self.lookback_window, n_periods, self.rebalance_frequency):
            # Get historical window
            hist_returns = returns.iloc[i-self.lookback_window:i]
            hist_index = index_returns.iloc[i-self.lookback_window:i]
            
            # Fit model
            self.tracker = GroupDiscoveryIndexTracker(
                n_groups=None,  # Auto-discover
                group_method='hybrid'
            )
            
            try:
                self.tracker.fit(hist_returns.values, hist_index.values)
                weights = self.tracker.get_weights()
            except Exception as e:
                print(f"Warning: Model fitting failed at period {i}, using equal weights. Error: {e}")
                weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # Calculate turnover
            if prev_weights is not None:
                turnover = np.sum(np.abs(weights - prev_weights))
                turnover_history.append(turnover)
            
            prev_weights = weights.copy()
            
            # Store weights and groups
            weights_history.append(weights)
            if hasattr(self.tracker, 'groups_'):
                groups_history.append(self.tracker.groups_)
            else:
                groups_history.append({})
            
            n_rebalances += 1
            
            # Calculate returns for next period
            if i + self.rebalance_frequency <= n_periods:
                period_returns = returns.iloc[i:min(i+self.rebalance_frequency, n_periods)]
                
                for j in range(len(period_returns)):
                    daily_return = (period_returns.iloc[j].values @ weights)
                    portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
                    dates.append(period_returns.index[j])
            else:
                # Handle last partial period
                period_returns = returns.iloc[i:]
                for j in range(len(period_returns)):
                    daily_return = (period_returns.iloc[j].values @ weights)
                    portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
                    dates.append(period_returns.index[j])
        
        print(f"Completed {n_rebalances} rebalances")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_values[1:],
            'date': dates
        }).set_index('date')
        
        # Add benchmark
        benchmark_values = (1 + index_returns).cumprod()
        results['benchmark'] = benchmark_values.loc[results.index]
        results['benchmark'] = results['benchmark'] / results['benchmark'].iloc[0]
        
        # Calculate performance metrics
        results['returns'] = results['portfolio_value'].pct_change()
        results['benchmark_returns'] = results['benchmark'].pct_change()
        
        # Performance statistics
        total_return = (results['portfolio_value'].iloc[-1] - 1) * 100
        annual_factor = 252 / len(results)
        annual_return = ((results['portfolio_value'].iloc[-1] / 1) ** annual_factor - 1) * 100
        volatility = results['returns'].std() * np.sqrt(252) * 100
        sharpe = (annual_return - 2) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Tracking error
        tracking_error = (results['returns'] - results['benchmark_returns']).std() * np.sqrt(252) * 100
        
        # Information ratio
        excess_returns = results['returns'] - results['benchmark_returns']
        info_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        cummax = results['portfolio_value'].expanding().max()
        drawdown = (results['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Average turnover
        avg_turnover = np.mean(turnover_history) * 100 if turnover_history else 0
        
        print("\nBacktest Results:")
        print("="*50)
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annual Return: {annual_return:.2f}%")
        print(f"Volatility: {volatility:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Tracking Error: {tracking_error:.2f}%")
        print(f"Information Ratio: {info_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Average Turnover: {avg_turnover:.2f}%")
        
        return results, weights_history, groups_history


def simulate_market_data(n_assets=100, n_periods=1000, n_groups=5):
    """
    Simulate market data with group structure.
    """
    np.random.seed(42)
    
    # Create dates
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='B')
    
    # Simulate group factors with different characteristics
    group_factors = np.zeros((n_periods, n_groups))
    
    # Different volatility and trend for each group
    for g in range(n_groups):
        trend = 0.0001 * (g - n_groups/2)  # Some groups trend up, others down
        vol = 0.01 * (1 + 0.2 * g)  # Increasing volatility
        group_factors[:, g] = np.cumsum(trend + vol * np.random.randn(n_periods))
    
    # Add market factor
    market_factor = np.cumsum(0.0002 + 0.015 * np.random.randn(n_periods))
    
    # Assign assets to groups
    assets_per_group = n_assets // n_groups
    asset_groups = np.repeat(range(n_groups), assets_per_group)
    if len(asset_groups) < n_assets:
        asset_groups = np.append(asset_groups, [n_groups-1] * (n_assets - len(asset_groups)))
    
    # Generate asset returns with group structure
    asset_returns = np.zeros((n_periods, n_assets))
    
    for i in range(n_assets):
        group = asset_groups[i]
        # Asset return = market factor + group factor + idiosyncratic
        beta_market = 0.5 + 0.5 * np.random.rand()  # Market beta between 0.5 and 1
        beta_group = 0.7 + 0.3 * np.random.rand()   # Group beta between 0.7 and 1
        
        asset_returns[:, i] = (
            beta_market * np.diff(np.append(0, market_factor)) +
            beta_group * np.diff(np.append(0, group_factors[:, group])) +
            0.005 * np.random.randn(n_periods)  # Idiosyncratic risk
        )
    
    # Convert to prices
    prices = pd.DataFrame(
        index=dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Initialize prices
    for i in range(n_assets):
        prices.iloc[0, i] = 100
        for t in range(1, n_periods):
            prices.iloc[t, i] = prices.iloc[t-1, i] * (1 + asset_returns[t, i])
    
    # Create index as weighted average (higher weights for some groups)
    index_weights = np.zeros(n_assets)
    
    # Give higher weights to groups 2 and 3 (middle volatility/return)
    for i in range(n_assets):
        if asset_groups[i] in [2, 3]:
            index_weights[i] = 2.0
        elif asset_groups[i] in [0, 4]:
            index_weights[i] = 0.5
        else:
            index_weights[i] = 1.0
    
    index_weights = index_weights / index_weights.sum()
    
    # Calculate index prices
    index_returns = asset_returns @ index_weights
    index_prices = pd.Series(index=dates, name='Index')
    index_prices.iloc[0] = 100
    for t in range(1, n_periods):
        index_prices.iloc[t] = index_prices.iloc[t-1] * (1 + index_returns[t])
    
    return prices, index_prices, asset_groups


def run_trading_strategy():
    """
    Run the complete group-based trading strategy.
    """
    print("Group-Based Index Tracking Strategy")
    print("="*60)
    
    # Generate or load market data
    print("\n1. Generating market data with group structure...")
    prices, index_prices, true_groups = simulate_market_data(
        n_assets=100, 
        n_periods=1000, 
        n_groups=5
    )
    
    # Split into train/test
    split_date = prices.index[int(len(prices) * 0.6)]
    print(f"Training data: {prices.index[0]} to {split_date}")
    print(f"Testing data: {split_date} to {prices.index[-1]}")
    
    # Run backtest
    print("\n2. Running adaptive group tracking strategy...")
    strategy = AdaptiveGroupTracker(
        lookback_window=120,
        rebalance_frequency=20
    )
    
    results, weights_history, groups_history = strategy.backtest(
        prices, 
        index_prices, 
        start_date=split_date
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value
    ax = axes[0, 0]
    ax.plot(results.index, results['portfolio_value'], label='Portfolio', linewidth=2)
    ax.plot(results.index, results['benchmark'], label='Benchmark', linewidth=2, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Portfolio Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rolling tracking error
    ax = axes[0, 1]
    rolling_te = (results['returns'] - results['benchmark_returns']).rolling(60).std() * np.sqrt(252) * 100
    ax.plot(results.index, rolling_te)
    ax.set_ylabel('Tracking Error (%)')
    ax.set_title('60-Day Rolling Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(rolling_te.dropna()) * 1.2)
    
    # Group discovery analysis
    ax = axes[1, 0]
    n_groups_over_time = [len(g) for g in groups_history]
    rebalance_dates = results.index[::strategy.rebalance_frequency][:len(n_groups_over_time)]
    if len(rebalance_dates) > len(n_groups_over_time):
        rebalance_dates = rebalance_dates[:len(n_groups_over_time)]
    ax.plot(rebalance_dates, n_groups_over_time, marker='o')
    ax.set_ylabel('Number of Groups')
    ax.set_title('Discovered Groups Over Time')
    ax.grid(True, alpha=0.3)
    
    # Weight concentration
    ax = axes[1, 1]
    weight_concentration = []
    for w in weights_history:
        sorted_weights = np.sort(w)[::-1]
        top_20_weight = sorted_weights[:20].sum()
        weight_concentration.append(top_20_weight)
    
    concentration_dates = rebalance_dates[:len(weight_concentration)]
    ax.plot(concentration_dates, weight_concentration)
    ax.set_ylabel('Weight in Top 20 Assets')
    ax.set_title('Portfolio Concentration')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze group stability
    print("\n3. Group Stability Analysis:")
    if len(groups_history) > 1:
        group_similarities = []
        for i in range(1, len(groups_history)):
            prev_groups = groups_history[i-1]
            curr_groups = groups_history[i]
            
            if len(prev_groups) == 0 or len(curr_groups) == 0:
                continue
            
            # Calculate Jaccard similarity for each group
            similarities = []
            for g1 in prev_groups.values():
                best_similarity = 0
                for g2 in curr_groups.values():
                    intersection = len(set(g1) & set(g2))
                    union = len(set(g1) | set(g2))
                    similarity = intersection / union if union > 0 else 0
                    best_similarity = max(best_similarity, similarity)
                similarities.append(best_similarity)
            
            if similarities:
                group_similarities.append(np.mean(similarities))
        
        if group_similarities:
            print(f"Average group stability: {np.mean(group_similarities):.2%}")
            print(f"Min group stability: {np.min(group_similarities):.2%}")
            print(f"Max group stability: {np.max(group_similarities):.2%}")
    
    # Compare with simple strategies
    print("\n4. Comparison with Simple Strategies:")
    
    # Equal weight
    test_prices = prices.loc[results.index]
    equal_returns = test_prices.pct_change().dropna().mean(axis=1)
    equal_cumulative = (1 + equal_returns).cumprod()
    equal_return = (equal_cumulative.iloc[-1] - 1) * 100
    
    # Random selection
    np.random.seed(42)
    n_random = 30
    random_assets = np.random.choice(prices.columns, n_random, replace=False)
    random_returns = test_prices[random_assets].pct_change().dropna().mean(axis=1)
    random_cumulative = (1 + random_returns).cumprod()
    random_return = (random_cumulative.iloc[-1] - 1) * 100
    
    # Index buy-and-hold
    index_return = (results['benchmark'].iloc[-1] - 1) * 100
    
    print(f"Group Tracking Return: {(results['portfolio_value'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Index Return: {index_return:.2f}%")
    print(f"Equal Weight Return: {equal_return:.2f}%")
    print(f"Random Selection Return: {random_return:.2f}%")
    
    # Additional statistics
    print("\n5. Portfolio Statistics:")
    avg_groups = np.mean([len(g) for g in groups_history])
    avg_assets_per_group = np.mean([np.mean([len(group) for group in g.values()]) 
                                   for g in groups_history if g])
    
    print(f"Average number of groups: {avg_groups:.1f}")
    print(f"Average assets per group: {avg_assets_per_group:.1f}")
    print(f"Total rebalances: {len(weights_history)}")
    
    return results, strategy


if __name__ == "__main__":
    # Run the trading strategy
    results, strategy = run_trading_strategy()
    
    # Additional analysis
    print("\n6. Strategy Insights:")
    print("- The strategy automatically discovers asset groups using multiple signals")
    print("- Groups are rebalanced periodically to adapt to market changes")
    print("- Within-group equal weighting provides stable diversification")
    print("- Group selection (via LASSO) provides the tracking performance")
    print("- No need for prior knowledge of true group structure")