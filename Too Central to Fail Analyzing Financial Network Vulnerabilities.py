import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import itertools
import warnings
from datetime import datetime, timedelta
from pandas.tseries.offsets import BMonthEnd
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class NetworkBasedTradingStrategies:
    """
    Class to implement and backtest trading strategies based on 
    financial network vulnerabilities
    """
    
    def __init__(self, n_firms=100, n_periods=120, training_periods=60, test_periods=60):
        """
        Initialize the trading strategy simulation
        
        Parameters:
        -----------
        n_firms: int
            Number of firms in the simulation
        n_periods: int
            Number of time periods (e.g., months)
        training_periods: int
            Number of periods used for training the network model
        test_periods: int
            Number of periods used for testing the strategies
        """
        self.n_firms = n_firms
        self.n_periods = n_periods
        self.training_periods = training_periods
        self.test_periods = test_periods
        self.stock_returns = None
        self.bond_returns = None
        self.firm_characteristics = None
        
        # Strategy performance metrics
        self.strategy_returns = {}
        self.cumulative_returns = {}
        self.sharpe_ratios = {}
        self.max_drawdowns = {}
        
    def generate_firm_characteristics(self):
        """
        Generate firm characteristics (market cap, beta, etc.)
        """
        # Generate firm sizes following log-normal distribution
        log_market_cap = np.random.normal(16, 1.5, self.n_firms)
        market_cap = np.exp(log_market_cap)
        
        # Generate betas for stock market (CAPM)
        betas = np.random.normal(1, 0.3, self.n_firms)
        
        # Generate other firm characteristics
        earnings_per_share = np.random.normal(3, 2, self.n_firms)
        dividends_per_share = np.random.normal(1, 0.5, self.n_firms) * (np.random.rand(self.n_firms) > 0.2)
        
        # Higher leverage for smaller firms (negative correlation with market cap)
        leverage_ranks = self.n_firms - pd.Series(market_cap).rank()
        leverage = (leverage_ranks / self.n_firms) * 0.5 + 0.2 + np.random.normal(0, 0.1, self.n_firms)
        leverage = np.clip(leverage, 0.1, 0.9)
        
        # Create dataframe of firm characteristics
        self.firm_characteristics = pd.DataFrame({
            'market_cap': market_cap,
            'log_market_cap': log_market_cap,
            'beta': betas,
            'earnings_per_share': earnings_per_share,
            'dividends_per_share': dividends_per_share,
            'leverage': leverage,
            'firm_id': range(self.n_firms)
        })
        
        # Create firm names
        self.firm_characteristics['firm_name'] = ['Firm_' + str(i) for i in range(self.n_firms)]
        
        # Time-varying characteristics
        self.market_cap_time_series = pd.DataFrame(
            np.zeros((self.n_periods, self.n_firms)),
            columns=range(self.n_firms)
        )
        
        # Initialize with base market cap
        for i in range(self.n_firms):
            self.market_cap_time_series.iloc[0, i] = market_cap[i]
        
        return self.firm_characteristics
    
    def simulate_returns(self, stock_vol=0.08, bond_vol=0.03, market_impact=0.6, 
                        idiosyncratic_vol_stock=0.15, idiosyncratic_vol_bond=0.05, 
                        bond_market_impact=0.3, fundamental_link=0.5):
        """
        Simulate stock and bond returns for all firms
        
        Parameters:
        -----------
        stock_vol: float
            Stock market volatility
        bond_vol: float
            Bond market volatility
        market_impact: float
            Impact of market factor on stock returns
        idiosyncratic_vol_stock: float
            Idiosyncratic volatility for stocks
        idiosyncratic_vol_bond: float
            Idiosyncratic volatility for bonds
        bond_market_impact: float
            Impact of bond market factor on bond returns
        fundamental_link: float
            Strength of the link between a firm's stock and bond returns
        """
        # Market factors
        stock_market_factor = np.random.normal(0.005, stock_vol, self.n_periods)
        bond_market_factor = np.random.normal(0.002, bond_vol, self.n_periods)
        
        # Firm-specific factors
        firm_factors = np.random.normal(0, 0.05, (self.n_firms, self.n_periods))
        
        # Initialize return matrices
        stock_returns = np.zeros((self.n_firms, self.n_periods))
        bond_returns = np.zeros((self.n_firms, self.n_periods))
        
        # Generate correlated returns
        for i in range(self.n_firms):
            beta = self.firm_characteristics.loc[i, 'beta']
            leverage = self.firm_characteristics.loc[i, 'leverage']
            size = self.firm_characteristics.loc[i, 'log_market_cap']
            
            # Size effect - smaller firms are more volatile
            size_effect = np.exp(-size/20)
            
            # Stock returns
            stock_idiosyncratic = np.random.normal(0, idiosyncratic_vol_stock * size_effect, self.n_periods)
            stock_returns[i] = market_impact * beta * stock_market_factor + stock_idiosyncratic + firm_factors[i]
            
            # Bond returns (influenced by leverage and firm-specific factors)
            # Bonds of firms with higher leverage are more sensitive to market conditions
            bond_idiosyncratic = np.random.normal(0, idiosyncratic_vol_bond * leverage * size_effect, self.n_periods)
            
            # Bond returns are influenced by:
            # 1. Bond market factor
            # 2. Stock returns of the same firm (fundamental link)
            # 3. Idiosyncratic factors
            bond_returns[i] = (bond_market_impact * leverage * bond_market_factor + 
                             fundamental_link * stock_returns[i] + 
                             bond_idiosyncratic)
            
        # Convert to dataframes
        self.stock_returns = pd.DataFrame(stock_returns, index=range(self.n_firms), columns=range(self.n_periods)).T
        self.bond_returns = pd.DataFrame(bond_returns, index=range(self.n_firms), columns=range(self.n_periods)).T
        
        # Add lagged effects to create more complex dependencies (Granger causality)
        self.add_network_effects()
        
        # Update market caps based on returns
        for t in range(1, self.n_periods):
            for i in range(self.n_firms):
                # Market cap grows with stock returns (plus some noise)
                self.market_cap_time_series.iloc[t, i] = (
                    self.market_cap_time_series.iloc[t-1, i] * 
                    (1 + self.stock_returns.iloc[t, i])
                )
        
        return self.stock_returns, self.bond_returns
    
    def add_network_effects(self, network_density=0.05, effect_strength=0.2):
        """
        Add network effects to returns to create Granger causality relationships
        
        Parameters:
        -----------
        network_density: float
            Density of connections in the network
        effect_strength: float
            Strength of the network effects
        """
        # Create random directed networks for both markets
        stock_network = np.random.rand(self.n_firms, self.n_firms) < network_density
        bond_network = np.random.rand(self.n_firms, self.n_firms) < network_density
        
        # Ensure no self-loops
        np.fill_diagonal(stock_network, 0)
        np.fill_diagonal(bond_network, 0)
        
        # Larger firms have more outgoing connections (they affect others more)
        for i in range(self.n_firms):
            size_percentile = np.percentile(self.firm_characteristics['log_market_cap'], 
                                          100 * i / self.n_firms)
            size_effect = size_percentile / 20
            
            # Adjust network connections based on size
            if np.random.rand() < size_effect:
                # Add more outgoing connections for larger firms
                targets = np.random.choice(self.n_firms, size=int(3 * network_density * self.n_firms), replace=False)
                for t in targets:
                    if t != i:
                        stock_network[i, t] = 1
                        bond_network[i, t] = 1
        
        # Add lagged effects
        for t in range(1, self.n_periods):
            # Stock network effects
            for i in range(self.n_firms):
                influencers = np.where(stock_network[:, i])[0]  # Firms that influence firm i
                if len(influencers) > 0:
                    influence = np.mean(self.stock_returns.iloc[t-1, influencers])
                    self.stock_returns.iloc[t, i] += effect_strength * influence
            
            # Bond network effects
            for i in range(self.n_firms):
                influencers = np.where(bond_network[:, i])[0]  # Firms that influence firm i
                if len(influencers) > 0:
                    influence = np.mean(self.bond_returns.iloc[t-1, influencers])
                    self.bond_returns.iloc[t, i] += effect_strength * influence
        
        # Store the networks for reference
        self.stock_network = stock_network
        self.bond_network = bond_network
    
    def construct_granger_causal_network(self, returns_data, start_period, end_period, maxlag=2, significance=0.05):
        """
        Construct Granger causal network for given returns data
        
        Parameters:
        -----------
        returns_data: DataFrame
            Returns data for all firms
        start_period: int
            Start period for the estimation window
        end_period: int
            End period for the estimation window
        maxlag: int
            Maximum lag for Granger causality test
        significance: float
            Significance level for Granger causality test
            
        Returns:
        --------
        adjacency_matrix: numpy array
            Adjacency matrix of the Granger causal network
        """
        # Extract relevant time period
        data = returns_data.iloc[start_period:end_period]
        
        # Initialize adjacency matrix
        adjacency = np.zeros((self.n_firms, self.n_firms))
        
        # Loop through all pairs of firms
        for i, j in itertools.product(range(self.n_firms), range(self.n_firms)):
            if i != j:
                # Test Granger causality from firm i to firm j
                y = data[j]
                x = data[i]
                combined_data = pd.concat([y, x], axis=1)
                combined_data.columns = ['y', 'x']
                
                try:
                    result = grangercausalitytests(combined_data, maxlag=maxlag, verbose=False)
                    # Check if any lag shows significance
                    for lag in range(1, maxlag+1):
                        if result[lag][0]['ssr_ftest'][1] < significance:
                            adjacency[i, j] = 1
                            break
                except:
                    # If test fails (e.g., due to constant series), no causality
                    pass
        
        return adjacency
    
    def calculate_pagerank(self, adjacency_matrix, damping_factor=0.85):
        """
        Calculate PageRank centrality for a network
        
        Parameters:
        -----------
        adjacency_matrix: numpy array
            Adjacency matrix of the network
        damping_factor: float
            Damping factor for PageRank algorithm
            
        Returns:
        --------
        pagerank: dict
            PageRank values for each node
        """
        # Create networkx graph
        G = nx.DiGraph(adjacency_matrix)
        
        # Calculate PageRank
        pagerank = nx.pagerank(G, alpha=damping_factor)
        
        return pagerank
    
    def analyze_network_at_time(self, t, window_size=60):
        """
        Analyze networks at a specific time period
        
        Parameters:
        -----------
        t: int
            Current time period
        window_size: int
            Size of the rolling window for network estimation
            
        Returns:
        --------
        results: dict
            Dictionary containing network analysis results
        """
        # Ensure we have enough data
        if t < window_size:
            return None
        
        # Define estimation window
        start = max(0, t - window_size)
        end = t
        
        # Construct networks
        stock_adjacency = self.construct_granger_causal_network(
            self.stock_returns, start, end
        )
        bond_adjacency = self.construct_granger_causal_network(
            self.bond_returns, start, end
        )
        
        # Calculate PageRank centrality
        stock_pagerank = self.calculate_pagerank(stock_adjacency)
        bond_pagerank = self.calculate_pagerank(bond_adjacency)
        
        # Convert to dataframes
        stock_pr_df = pd.DataFrame(list(stock_pagerank.items()), 
                                 columns=['firm_id', 'stock_pagerank'])
        bond_pr_df = pd.DataFrame(list(bond_pagerank.items()), 
                                columns=['firm_id', 'bond_pagerank'])
        
        # Take log of PageRank for better scaling
        stock_pr_df['log_stock_pagerank'] = np.log(stock_pr_df['stock_pagerank'])
        bond_pr_df['log_bond_pagerank'] = np.log(bond_pr_df['bond_pagerank'])
        
        # Merge with firm characteristics
        merged_df = pd.merge(stock_pr_df, bond_pr_df, on='firm_id')
        
        # Add market cap at time t
        market_caps = self.market_cap_time_series.iloc[t]
        market_cap_df = pd.DataFrame({
            'firm_id': range(self.n_firms),
            'market_cap': market_caps.values,
            'log_market_cap': np.log(market_caps.values)
        })
        merged_df = pd.merge(merged_df, market_cap_df, on='firm_id')
        
        # Create firm rankings
        merged_df['stock_pagerank_rank'] = merged_df['stock_pagerank'].rank(ascending=False)
        merged_df['bond_pagerank_rank'] = merged_df['bond_pagerank'].rank(ascending=False)
        merged_df['market_cap_rank'] = merged_df['market_cap'].rank(ascending=True)
        
        # Store results
        results = {
            'stock_adjacency': stock_adjacency,
            'bond_adjacency': bond_adjacency,
            'stock_pagerank': stock_pagerank,
            'bond_pagerank': bond_pagerank,
            'merged_df': merged_df
        }
        
        return results
    
    def implement_strategy_1(self, results, top_pct=0.2, bottom_pct=0.2):
        """
        Strategy 1: Exploit the relationship between stock and bond vulnerabilities
        Long stocks of firms with low vulnerability and short stocks of firms with high vulnerability
        
        Parameters:
        -----------
        results: dict
            Results from network analysis
        top_pct: float
            Percentage of firms to include in the top portfolio
        bottom_pct: float
            Percentage of firms to include in the bottom portfolio
            
        Returns:
        --------
        positions: DataFrame
            DataFrame containing portfolio positions
        """
        if results is None:
            return pd.DataFrame()
        
        # Get merged data
        merged_df = results['merged_df']
        
        # Sort by stock PageRank (vulnerability)
        sorted_df = merged_df.sort_values('stock_pagerank', ascending=True)
        
        # Number of firms to include in each portfolio
        n_firms_portfolio = int(self.n_firms * top_pct)
        
        # Create positions dataframe
        positions = pd.DataFrame(index=range(self.n_firms), columns=['weight'])
        positions['weight'] = 0.0
        
        # Long positions in least vulnerable firms
        low_vulnerable_firms = sorted_df.head(n_firms_portfolio)['firm_id'].values
        positions.loc[low_vulnerable_firms, 'weight'] = 1.0 / n_firms_portfolio
        
        # Short positions in most vulnerable firms
        high_vulnerable_firms = sorted_df.tail(n_firms_portfolio)['firm_id'].values
        positions.loc[high_vulnerable_firms, 'weight'] = -1.0 / n_firms_portfolio
        
        return positions
    
    def implement_strategy_2(self, results, top_pct=0.2, bottom_pct=0.2):
        """
        Strategy 2: Exploit the relationship between market cap and vulnerability
        Long stocks of large firms (low vulnerability) and short stocks of small firms (high vulnerability)
        
        Parameters:
        -----------
        results: dict
            Results from network analysis
        top_pct: float
            Percentage of firms to include in the top portfolio
        bottom_pct: float
            Percentage of firms to include in the bottom portfolio
            
        Returns:
        --------
        positions: DataFrame
            DataFrame containing portfolio positions
        """
        if results is None:
            return pd.DataFrame()
        
        # Get merged data
        merged_df = results['merged_df']
        
        # Sort by market cap
        sorted_df = merged_df.sort_values('market_cap', ascending=False)
        
        # Number of firms to include in each portfolio
        n_firms_portfolio = int(self.n_firms * top_pct)
        
        # Create positions dataframe
        positions = pd.DataFrame(index=range(self.n_firms), columns=['weight'])
        positions['weight'] = 0.0
        
        # Long positions in large firms
        large_firms = sorted_df.head(n_firms_portfolio)['firm_id'].values
        positions.loc[large_firms, 'weight'] = 1.0 / n_firms_portfolio
        
        # Short positions in small firms
        small_firms = sorted_df.tail(n_firms_portfolio)['firm_id'].values
        positions.loc[small_firms, 'weight'] = -1.0 / n_firms_portfolio
        
        return positions
    
    def implement_strategy_3(self, results, threshold=0.3):
        """
        Strategy 3: Exploit differences between bond and stock vulnerabilities
        Long stocks of firms where bond vulnerability > stock vulnerability
        Short stocks of firms where stock vulnerability > bond vulnerability
        
        Parameters:
        -----------
        results: dict
            Results from network analysis
        threshold: float
            Threshold for the vulnerability difference
            
        Returns:
        --------
        positions: DataFrame
            DataFrame containing portfolio positions
        """
        if results is None:
            return pd.DataFrame()
        
        # Get merged data
        merged_df = results['merged_df']
        
        # Calculate vulnerability difference (in log scale)
        merged_df['vulnerability_diff'] = merged_df['log_bond_pagerank'] - merged_df['log_stock_pagerank']
        
        # Identify firms with significant differences
        bond_more_vulnerable = merged_df[merged_df['vulnerability_diff'] > threshold]['firm_id'].values
        stock_more_vulnerable = merged_df[merged_df['vulnerability_diff'] < -threshold]['firm_id'].values
        
        # Number of firms in each group
        n_bond = len(bond_more_vulnerable)
        n_stock = len(stock_more_vulnerable)
        
        # Create positions dataframe
        positions = pd.DataFrame(index=range(self.n_firms), columns=['weight'])
        positions['weight'] = 0.0
        
        # Assign weights (ensuring the portfolio is dollar neutral)
        if n_bond > 0:
            positions.loc[bond_more_vulnerable, 'weight'] = 1.0 / max(n_bond, 1)
        if n_stock > 0:
            positions.loc[stock_more_vulnerable, 'weight'] = -1.0 / max(n_stock, 1)
        
        return positions
    
    def implement_strategy_4(self, results, n_clusters=4):
        """
        Strategy 4: Cluster-based strategy
        Group firms based on their vulnerabilities and size, and create a long-short portfolio
        
        Parameters:
        -----------
        results: dict
            Results from network analysis
        n_clusters: int
            Number of clusters for k-means
            
        Returns:
        --------
        positions: DataFrame
            DataFrame containing portfolio positions
        """
        if results is None:
            return pd.DataFrame()
        
        # Get merged data
        merged_df = results['merged_df']
        
        # Prepare data for clustering
        X = merged_df[['log_stock_pagerank', 'log_bond_pagerank', 'log_market_cap']].values
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        merged_df['cluster'] = kmeans.fit_predict(X)
        
        # Calculate cluster centroids
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate average vulnerability for each cluster
        cluster_summary = merged_df.groupby('cluster').agg({
            'log_stock_pagerank': 'mean',
            'log_bond_pagerank': 'mean',
            'log_market_cap': 'mean',
            'firm_id': 'count'
        }).rename(columns={'firm_id': 'count'})
        
        # Sort clusters by average stock vulnerability
        cluster_summary = cluster_summary.sort_values('log_stock_pagerank')
        
        # Identify low and high vulnerability clusters
        low_vuln_cluster = cluster_summary.index[0]
        high_vuln_cluster = cluster_summary.index[-1]
        
        # Get firms in each cluster
        low_vuln_firms = merged_df[merged_df['cluster'] == low_vuln_cluster]['firm_id'].values
        high_vuln_firms = merged_df[merged_df['cluster'] == high_vuln_cluster]['firm_id'].values
        
        # Number of firms in each group
        n_low = len(low_vuln_firms)
        n_high = len(high_vuln_firms)
        
        # Create positions dataframe
        positions = pd.DataFrame(index=range(self.n_firms), columns=['weight'])
        positions['weight'] = 0.0
        
        # Assign weights
        if n_low > 0:
            positions.loc[low_vuln_firms, 'weight'] = 1.0 / n_low
        if n_high > 0:
            positions.loc[high_vuln_firms, 'weight'] = -1.0 / n_high
        
        return positions
    
    def implement_strategy_5(self, results, window_size=3):
        """
        Strategy 5: Predict vulnerability changes
        Based on recent changes in vulnerability, predict which firms will become more/less vulnerable
        
        Parameters:
        -----------
        results: dict
            Results from network analysis
        window_size: int
            Number of periods to look back for trend calculation
            
        Returns:
        --------
        positions: DataFrame
            DataFrame containing portfolio positions
        """
        if results is None or 'vulnerability_history' not in self.__dict__:
            # Initialize vulnerability history if it doesn't exist
            if 'vulnerability_history' not in self.__dict__:
                self.vulnerability_history = []
            
            # Store current vulnerabilities
            if results is not None:
                self.vulnerability_history.append(results['merged_df'][['firm_id', 'log_stock_pagerank']])
            
            # Not enough history to implement strategy
            return pd.DataFrame(index=range(self.n_firms), columns=['weight']).fillna(0)
        
        # Store current vulnerabilities
        self.vulnerability_history.append(results['merged_df'][['firm_id', 'log_stock_pagerank']])
        
        # Keep only the most recent windows
        if len(self.vulnerability_history) > window_size + 1:
            self.vulnerability_history = self.vulnerability_history[-(window_size+1):]
        
        # Not enough history
        if len(self.vulnerability_history) <= window_size:
            return pd.DataFrame(index=range(self.n_firms), columns=['weight']).fillna(0)
        
        # Calculate vulnerability changes
        current = self.vulnerability_history[-1].set_index('firm_id')['log_stock_pagerank']
        previous = self.vulnerability_history[-window_size-1].set_index('firm_id')['log_stock_pagerank']
        
        # Merge to calculate changes
        vuln_changes = pd.DataFrame({
            'firm_id': current.index,
            'current_vulnerability': current.values,
            'previous_vulnerability': previous.values
        })
        vuln_changes['change'] = vuln_changes['current_vulnerability'] - vuln_changes['previous_vulnerability']
        
        # Identify firms with significant changes
        increasing_vuln = vuln_changes[vuln_changes['change'] > 0].sort_values('change', ascending=False)
        decreasing_vuln = vuln_changes[vuln_changes['change'] < 0].sort_values('change', ascending=True)
        
        # Take top 20% of firms in each category
        n_firms_portfolio = int(self.n_firms * 0.1)
        increasing_vuln = increasing_vuln.head(n_firms_portfolio)
        decreasing_vuln = decreasing_vuln.head(n_firms_portfolio)
        
        # Create positions dataframe
        positions = pd.DataFrame(index=range(self.n_firms), columns=['weight'])
        positions['weight'] = 0.0
        
        # Assign weights - short firms with increasing vulnerability, long firms with decreasing vulnerability
        positions.loc[increasing_vuln['firm_id'], 'weight'] = -1.0 / len(increasing_vuln)
        positions.loc[decreasing_vuln['firm_id'], 'weight'] = 1.0 / len(decreasing_vuln)
        
        return positions
    
    def backtest_strategies(self):
        """
        Backtest all strategies
        
        Returns:
        --------
        strategy_returns: dict
            Dictionary containing strategy returns
        """
        # Initialize strategy returns
        self.strategy_returns = {
            'Strategy_1': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Strategy_2': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Strategy_3': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Strategy_4': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Strategy_5': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Equal_Weight': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
            'Market': pd.Series(index=range(self.training_periods, self.n_periods), dtype=float),
        }
        
        # Store positions for analysis
        self.strategy_positions = {
            'Strategy_1': [],
            'Strategy_2': [],
            'Strategy_3': [],
            'Strategy_4': [],
            'Strategy_5': [],
        }
        
        # Implement strategies at each time point
        for t in range(self.training_periods, self.n_periods):
            # Analyze networks
            results = self.analyze_network_at_time(t, window_size=self.training_periods)
            
            if results is not None:
                # Implement strategies
                positions_1 = self.implement_strategy_1(results)
                positions_2 = self.implement_strategy_2(results)
                positions_3 = self.implement_strategy_3(results)
                positions_4 = self.implement_strategy_4(results)
                positions_5 = self.implement_strategy_5(results)
                
                # Store positions
                self.strategy_positions['Strategy_1'].append(positions_1)
                self.strategy_positions['Strategy_2'].append(positions_2)
                self.strategy_positions['Strategy_3'].append(positions_3)
                self.strategy_positions['Strategy_4'].append(positions_4)
                self.strategy_positions['Strategy_5'].append(positions_5)
                
                # Next period returns
                if t + 1 < self.n_periods:
                    next_returns = self.stock_returns.iloc[t+1]
                    
                    # Calculate strategy returns
                    self.strategy_returns['Strategy_1'].loc[t] = (positions_1['weight'] * next_returns).sum()
                    self.strategy_returns['Strategy_2'].loc[t] = (positions_2['weight'] * next_returns).sum()
                    self.strategy_returns['Strategy_3'].loc[t] = (positions_3['weight'] * next_returns).sum()
                    self.strategy_returns['Strategy_4'].loc[t] = (positions_4['weight'] * next_returns).sum()
                    self.strategy_returns['Strategy_5'].loc[t] = (positions_5['weight'] * next_returns).sum()
                    
                    # Equal weight benchmark
                    equal_weights = pd.Series(1.0 / self.n_firms, index=range(self.n_firms))
                    self.strategy_returns['Equal_Weight'].loc[t] = (equal_weights * next_returns).sum()
                    
                    # Market benchmark (weighted by market cap)
                    market_caps = self.market_cap_time_series.iloc[t]
                    market_weights = market_caps / market_caps.sum()
                    self.strategy_returns['Market'].loc[t] = (market_weights * next_returns).sum()
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        return self.strategy_returns
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for all strategies
        """
        # Fill NA values
        for strategy in self.strategy_returns:
            self.strategy_returns[strategy] = self.strategy_returns[strategy].fillna(0)
        
        # Calculate cumulative returns
        self.cumulative_returns = {}
        for strategy in self.strategy_returns:
            self.cumulative_returns[strategy] = (1 + self.strategy_returns[strategy]).cumprod() - 1
        
        # Calculate Sharpe ratios (annualized)
        self.sharpe_ratios = {}
        for strategy in self.strategy_returns:
            returns = self.strategy_returns[strategy]
            if returns.std() > 0:
                self.sharpe_ratios[strategy] = returns.mean() / returns.std() * np.sqrt(12)  # Assuming monthly returns
            else:
                self.sharpe_ratios[strategy] = 0
        
        # Calculate maximum drawdowns
        self.max_drawdowns = {}
        for strategy in self.cumulative_returns:
            cum_returns = self.cumulative_returns[strategy]
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / (1 + running_max)
            self.max_drawdowns[strategy] = drawdown.min()
        
        # Create summary dataframe
        self.performance_summary = pd.DataFrame({
            'Annualized Return': {s: (1 + self.strategy_returns[s].mean()) ** 12 - 1 for s in self.strategy_returns},
            'Annualized Volatility': {s: self.strategy_returns[s].std() * np.sqrt(12) for s in self.strategy_returns},
            'Sharpe Ratio': self.sharpe_ratios,
            'Maximum Drawdown': self.max_drawdowns,
            'Final Cumulative Return': {s: self.cumulative_returns[s].iloc[-1] for s in self.cumulative_returns}
        })
    
    def plot_strategy_performance(self):
        """
        Plot strategy performance
        """
        # Create date index for better visualization
        start_date = datetime(2015, 1, 1)
        date_index = [start_date + BMonthEnd(i) for i in range(len(self.strategy_returns['Strategy_1']))]
        
        # Plot cumulative returns
        plt.figure(figsize=(15, 8))
        for strategy in self.cumulative_returns:
            cum_returns = self.cumulative_returns[strategy]
            plt.plot(date_index, cum_returns.values, label=strategy)
        
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Strategy Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('strategy_performance.png', dpi=300)
        plt.show()
        
        # Plot strategy returns
        plt.figure(figsize=(15, 8))
        for strategy in ['Strategy_1', 'Strategy_2', 'Strategy_3', 'Strategy_4', 'Strategy_5']:
            returns = self.strategy_returns[strategy]
            plt.plot(date_index, returns.values, label=strategy, alpha=0.7)
        
        plt.xlabel('Date')
        plt.ylabel('Monthly Return')
        plt.title('Strategy Monthly Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('strategy_returns.png', dpi=300)
        plt.show()
        
        # Plot performance metrics
        plt.figure(figsize=(15, 8))
        metrics = ['Annualized Return', 'Sharpe Ratio']
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 2, i+1)
            self.performance_summary[metric].plot(kind='bar')
            plt.title(metric)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('strategy_metrics.png', dpi=300)
        plt.show()
        
        # Plot drawdowns
        plt.figure(figsize=(15, 8))
        
        for strategy in ['Strategy_1', 'Strategy_2', 'Strategy_3', 'Strategy_4', 'Strategy_5', 'Market']:
            cum_returns = self.cumulative_returns[strategy]
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / (1 + running_max)
            plt.plot(date_index, drawdown.values, label=strategy)
        
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.title('Strategy Drawdowns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('strategy_drawdowns.png', dpi=300)
        plt.show()
    
    def analyze_portfolios(self):
        """
        Analyze portfolio composition and turnover
        """
        # Calculate average positions for each strategy
        avg_positions = {}
        for strategy in self.strategy_positions:
            positions_df = pd.concat(self.strategy_positions[strategy], axis=1)
            avg_positions[strategy] = positions_df.mean(axis=1)
        
        # Calculate portfolio turnover
        turnover = {}
        for strategy in self.strategy_positions:
            if len(self.strategy_positions[strategy]) > 1:
                turnover_sum = 0
                for i in range(1, len(self.strategy_positions[strategy])):
                    prev_pos = self.strategy_positions[strategy][i-1]['weight']
                    curr_pos = self.strategy_positions[strategy][i]['weight']
                    turnover_sum += np.abs(curr_pos - prev_pos).sum() / 2  # Divide by 2 to avoid double counting
                
                turnover[strategy] = turnover_sum / (len(self.strategy_positions[strategy]) - 1)
            else:
                turnover[strategy] = 0
        
        # Create turnover dataframe
        turnover_df = pd.DataFrame({
            'Average Monthly Turnover': turnover
        })
        
        # Plot average position by firm size
        plt.figure(figsize=(15, 10))
        for i, strategy in enumerate(['Strategy_1', 'Strategy_2', 'Strategy_3', 'Strategy_4']):
            plt.subplot(2, 2, i+1)
            
            # Get firm sizes
            sizes = self.firm_characteristics['log_market_cap']
            
            # Sort firms by size
            sorted_idx = sizes.argsort()
            sorted_sizes = sizes.iloc[sorted_idx]
            sorted_positions = avg_positions[strategy].iloc[sorted_idx]
            
            # Plot
            plt.scatter(sorted_sizes, sorted_positions, alpha=0.5)
            plt.xlabel('Log Market Cap')
            plt.ylabel('Average Position')
            plt.title(f'{strategy} - Position vs Firm Size')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(sorted_sizes, sorted_positions, 1)
            p = np.poly1d(z)
            plt.plot(sorted_sizes, p(sorted_sizes), "r--")
        
        plt.tight_layout()
        plt.savefig('portfolio_composition.png', dpi=300)
        plt.show()
        
        return turnover_df

def main():
    # Initialize and simulate
    print("Initializing network-based trading strategies simulation...")
    trading = NetworkBasedTradingStrategies(n_firms=100, n_periods=120, training_periods=60)
    
    # Generate firm characteristics
    print("Generating firm characteristics...")
    trading.generate_firm_characteristics()
    
    # Simulate returns
    print("Simulating stock and bond returns...")
    trading.simulate_returns()
    
    # Backtest strategies
    print("Backtesting trading strategies...")
    strategy_returns = trading.backtest_strategies()
    
    # Print performance summary
    print("\nStrategy Performance Summary:")
    print(trading.performance_summary)
    
    # Calculate portfolio turnover
    turnover_df = trading.analyze_portfolios()
    print("\nPortfolio Turnover:")
    print(turnover_df)
    
    # Plot results
    print("\nPlotting strategy performance...")
    trading.plot_strategy_performance()
    
    return trading

if __name__ == "__main__":
    trading = main()