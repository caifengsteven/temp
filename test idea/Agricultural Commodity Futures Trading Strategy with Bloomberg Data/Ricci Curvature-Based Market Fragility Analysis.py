import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

class RicciCurvatureAnalysis:
    def __init__(self):
        self.networks = []
        self.curvatures = []
        self.entropies = []
        self.avg_path_lengths = []
        self.diameters = []
        self.dates = []
        
    def simulate_stock_data(self, n_stocks=50, days=1000, start_date=None, include_crisis=True):
        """
        Simulate stock price data with optional crisis periods
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        days : int
            Number of days to simulate
        start_date : datetime
            Starting date for the simulation
        include_crisis : bool
            Whether to include simulated crisis periods
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with simulated stock prices
        """
        if start_date is None:
            start_date = dt.datetime(2010, 1, 1)
            
        dates = [start_date + dt.timedelta(days=i) for i in range(days)]
        
        # Base market movement
        market = np.zeros(days)
        
        # Add trend
        market += np.linspace(0, 0.4, days)
        
        # Add market noise
        market += np.random.normal(0, 0.01, days).cumsum()
        
        # Add seasonality
        market += 0.1 * np.sin(np.linspace(0, 8 * np.pi, days))
        
        # Define crisis periods
        crisis_periods = []
        if include_crisis:
            # First crisis: sharp drop followed by recovery
            crisis_start1 = int(days * 0.2)
            crisis_end1 = crisis_start1 + 60
            crisis_periods.append((crisis_start1, crisis_end1))
            
            # Second crisis: longer and more severe
            crisis_start2 = int(days * 0.6)
            crisis_end2 = crisis_start2 + 120
            crisis_periods.append((crisis_start2, crisis_end2))
            
            # Apply crisis effects to market
            for start, end in crisis_periods:
                # Decrease during crisis
                market[start:end] -= np.linspace(0, 0.3, end-start)
                # Increase correlations during crisis (handled later)
        
        # Generate stock prices
        stocks = {}
        correlations = np.zeros((days, n_stocks, n_stocks))
        
        # Normal correlation matrix (base state)
        base_corr = np.eye(n_stocks)
        
        # Add some realistic correlation structure - stocks are grouped into sectors
        n_sectors = 5
        stocks_per_sector = n_stocks // n_sectors
        
        for i in range(n_sectors):
            sector_start = i * stocks_per_sector
            sector_end = (i + 1) * stocks_per_sector if i < n_sectors - 1 else n_stocks
            
            # Stocks within the same sector have higher correlation
            for j in range(sector_start, sector_end):
                for k in range(j+1, sector_end):
                    base_corr[j, k] = 0.3 + 0.3 * np.random.random()
                    base_corr[k, j] = base_corr[j, k]
        
        # Fill in remaining correlations with low values
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                if base_corr[i, j] == 0:
                    base_corr[i, j] = 0.1 * np.random.random()
                    base_corr[j, i] = base_corr[i, j]
        
        # Generate stock specific movements and apply correlation structure
        specific_movements = np.random.normal(0, 0.01, (days, n_stocks))
        
        # Apply correlation structure
        for t in range(days):
            # During crisis, increase correlations
            in_crisis = False
            crisis_intensity = 0
            
            for start, end in crisis_periods:
                if start <= t < end:
                    in_crisis = True
                    # Intensity peaks in the middle of the crisis
                    progress = (t - start) / (end - start)
                    if progress < 0.5:
                        crisis_intensity = progress * 2
                    else:
                        crisis_intensity = (1 - progress) * 2
                    break
            
            if in_crisis:
                # Increase correlations during crisis
                day_corr = base_corr.copy()
                
                # Correlations increase with crisis intensity
                for i in range(n_stocks):
                    for j in range(i+1, n_stocks):
                        # Increase correlation during crisis but keep it <= 1
                        day_corr[i, j] = min(day_corr[i, j] + 0.5 * crisis_intensity, 0.95)
                        day_corr[j, i] = day_corr[i, j]
            else:
                day_corr = base_corr.copy()
            
            correlations[t] = day_corr
        
        # Generate stock prices
        stock_prices = np.zeros((days, n_stocks))
        
        # Initial prices
        stock_prices[0] = 100 * np.ones(n_stocks) * (0.9 + 0.2 * np.random.random(n_stocks))
        
        # Generate correlated returns
        for t in range(1, days):
            day_corr = correlations[t]
            
            # Cholesky decomposition to generate correlated random variables
            try:
                L = np.linalg.cholesky(day_corr)
                correlated_noise = np.dot(L, np.random.normal(0, 0.01, n_stocks))
            except np.linalg.LinAlgError:
                # If correlation matrix is not positive definite, use nearest PD approximation
                day_corr = self.nearest_pd(day_corr)
                L = np.linalg.cholesky(day_corr)
                correlated_noise = np.dot(L, np.random.normal(0, 0.01, n_stocks))
            
            # Market component + specific component
            returns = 0.0003 + 0.6 * (market[t] - market[t-1]) + 0.4 * correlated_noise
            
            # During crisis, add some extreme negative returns
            for start, end in crisis_periods:
                if start <= t < end:
                    # Some stocks have more extreme negative movements during crisis
                    crisis_effect = np.random.exponential(0.01, n_stocks)
                    returns -= crisis_effect
                    break
            
            # Update prices
            stock_prices[t] = stock_prices[t-1] * (1 + returns)
        
        # Convert to DataFrame
        stock_names = [f'Stock_{i+1}' for i in range(n_stocks)]
        df = pd.DataFrame(stock_prices, columns=stock_names, index=dates)
        
        return df, crisis_periods, dates
    
    def nearest_pd(self, A):
        """Find the nearest positive-definite matrix to input A"""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self.is_pd(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.is_pd(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
    def is_pd(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def create_correlation_network(self, returns, threshold=0.8, use_mst=True):
        """
        Create a correlation network from stock returns
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with stock returns
        threshold : float
            Correlation threshold for including edges
        use_mst : bool
            Whether to include the minimum spanning tree
        
        Returns:
        --------
        G : networkx.Graph
            Correlation network
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create empty graph
        G = nx.Graph()
        
        # Add nodes
        for stock in returns.columns:
            G.add_node(stock)
        
        # Create distance matrix from correlation
        distance_matrix = np.sqrt(2 * (1 - corr_matrix.values))
        
        # Add MST edges if requested
        if use_mst:
            # Create a graph from the distance matrix
            temp_G = nx.Graph()
            n = len(returns.columns)
            for i in range(n):
                for j in range(i+1, n):
                    temp_G.add_edge(i, j, weight=distance_matrix[i, j])
            
            # Compute MST
            mst = nx.minimum_spanning_tree(temp_G)
            
            # Map MST indices to stock names
            stocks = list(returns.columns)
            for u, v, data in mst.edges(data=True):
                G.add_edge(stocks[u], stocks[v], weight=corr_matrix.iloc[u, v])
        
        # Add high correlation edges
        for i, stock1 in enumerate(returns.columns):
            for j, stock2 in enumerate(returns.columns):
                if i < j:  # Avoid duplicates
                    correlation = corr_matrix.loc[stock1, stock2]
                    if correlation >= threshold:
                        G.add_edge(stock1, stock2, weight=correlation)
        
        return G
    
    def compute_ollivier_ricci_curvature(self, G):
        """
        Compute Ollivier-Ricci curvature for a network
        
        Parameters:
        -----------
        G : networkx.Graph
            Network for which to compute curvature
        
        Returns:
        --------
        edge_curvatures : dict
            Dictionary with edge curvatures
        avg_curvature : float
            Average curvature of the graph
        """
        edge_curvatures = {}
        
        for u, v in G.edges():
            # Get neighbors of u and v
            u_neighbors = list(G.neighbors(u))
            v_neighbors = list(G.neighbors(v))
            
            # Skip if either node has no neighbors
            if not u_neighbors or not v_neighbors:
                edge_curvatures[(u, v)] = 0
                edge_curvatures[(v, u)] = 0
                continue
            
            # Compute measures
            u_measure = {neighbor: 1/len(u_neighbors) for neighbor in u_neighbors}
            v_measure = {neighbor: 1/len(v_neighbors) for neighbor in v_neighbors}
            
            # Compute Wasserstein distance
            wasserstein_dist = self.compute_wasserstein_distance(G, u_measure, v_measure)
            
            # Compute curvature: κ(x,y) = 1 - W_1(μ_x, μ_y)/d(x,y)
            # In a graph, d(x,y) = 1 for adjacent vertices
            curvature = 1 - wasserstein_dist
            
            edge_curvatures[(u, v)] = curvature
            edge_curvatures[(v, u)] = curvature  # Symmetric
        
        # Compute average curvature
        avg_curvature = np.mean(list(edge_curvatures.values())) if edge_curvatures else 0
        
        return edge_curvatures, avg_curvature
    
    def compute_wasserstein_distance(self, G, source_measure, target_measure):
        """
        Compute the 1-Wasserstein distance between two probability measures
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph
        source_measure : dict
            Source probability measure
        target_measure : dict
            Target probability measure
        
        Returns:
        --------
        distance : float
            Wasserstein distance
        """
        # Setup the transportation problem
        source_nodes = list(source_measure.keys())
        target_nodes = list(target_measure.keys())
        
        # Create cost matrix
        n_source = len(source_nodes)
        n_target = len(target_nodes)
        cost_matrix = np.zeros((n_source, n_target))
        
        for i, s in enumerate(source_nodes):
            for j, t in enumerate(target_nodes):
                try:
                    # Use shortest path distance as the cost
                    cost_matrix[i, j] = nx.shortest_path_length(G, source=s, target=t)
                except nx.NetworkXNoPath:
                    # If no path exists, use a large value
                    cost_matrix[i, j] = len(G)
        
        # Ensure that we have probability measures (sum to 1)
        source_weights = np.array([source_measure[node] for node in source_nodes])
        target_weights = np.array([target_measure[node] for node in target_nodes])
        
        # Normalize if not already normalized
        if abs(sum(source_weights) - 1.0) > 1e-10:
            source_weights = source_weights / sum(source_weights)
        if abs(sum(target_weights) - 1.0) > 1e-10:
            target_weights = target_weights / sum(target_weights)
        
        # Solve optimal transport problem using a simplified approach for speed
        # This is an approximation that works well enough for our purposes
        flow = np.zeros((n_source, n_target))
        
        # Greedy algorithm for small problems
        if n_source * n_target < 1000:
            # Sort coordinates by cost
            coords = [(i, j, cost_matrix[i, j]) for i in range(n_source) for j in range(n_target)]
            coords.sort(key=lambda x: x[2])
            
            remaining_source = source_weights.copy()
            remaining_target = target_weights.copy()
            
            for i, j, _ in coords:
                # Flow amount is min of remaining weights
                amount = min(remaining_source[i], remaining_target[j])
                flow[i, j] = amount
                
                # Update remaining weights
                remaining_source[i] -= amount
                remaining_target[j] -= amount
                
                # Stop if all weights are assigned
                if np.sum(remaining_source) < 1e-10 or np.sum(remaining_target) < 1e-10:
                    break
        else:
            # For larger problems, use a simpler approximation
            # Normalize the cost matrix
            norm_cost = cost_matrix / np.sum(cost_matrix)
            
            # Compute an approximation of optimal transport
            for i in range(n_source):
                for j in range(n_target):
                    flow[i, j] = source_weights[i] * target_weights[j] * (1 - norm_cost[i, j])
            
            # Normalize the flow
            flow = flow / np.sum(flow)
        
        # Compute the Wasserstein distance
        distance = np.sum(flow * cost_matrix)
        return distance
    
    def compute_network_entropy(self, G):
        """
        Compute network entropy as defined in the paper
        
        Parameters:
        -----------
        G : networkx.Graph
            Network for which to compute entropy
        
        Returns:
        --------
        entropy : float
            Network entropy
        """
        # Create transition matrix
        nodes = list(G.nodes())
        n = len(nodes)
        transition_matrix = np.zeros((n, n))
        
        for i, u in enumerate(nodes):
            neighbors = list(G.neighbors(u))
            if neighbors:
                weights = np.array([G[u][v].get('weight', 1.0) for v in neighbors])
                weights = weights / np.sum(weights)  # Normalize
                
                for j, v in enumerate(neighbors):
                    v_idx = nodes.index(v)
                    transition_matrix[i, v_idx] = weights[j]
        
        # Calculate invariant distribution (approximation)
        pi = np.ones(n) / n
        for _ in range(50):  # Iterate to converge to invariant distribution
            pi = pi @ transition_matrix
        
        # Calculate entropy
        entropy = 0
        for i, u in enumerate(nodes):
            neighbors = list(G.neighbors(u))
            if neighbors:
                node_entropy = 0
                for j, v in enumerate(neighbors):
                    v_idx = nodes.index(v)
                    p = transition_matrix[i, v_idx]
                    if p > 0:
                        node_entropy -= p * np.log(p)
                entropy += pi[i] * node_entropy
        
        return entropy
    
    def analyze_stock_data(self, stock_data, window_size=22, step_size=1, correlation_threshold=0.8, use_mst=True):
        """
        Analyze stock data using sliding windows
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock prices
        window_size : int
            Size of sliding window in days
        step_size : int
            Step size for sliding window
        correlation_threshold : float
            Threshold for including edges in correlation network
        use_mst : bool
            Whether to include the minimum spanning tree
        """
        # Calculate returns
        returns = stock_data.pct_change().dropna()
        
        # Initialize lists to store results
        self.networks = []
        self.curvatures = []
        self.entropies = []
        self.avg_path_lengths = []
        self.diameters = []
        self.dates = []
        
        # Slide window through returns data
        for i in tqdm(range(0, len(returns) - window_size, step_size),
                     desc="Analyzing windows"):
            window_returns = returns.iloc[i:i+window_size]
            window_date = returns.index[i+window_size-1]
            
            # Create correlation network
            G = self.create_correlation_network(window_returns, threshold=correlation_threshold, use_mst=use_mst)
            
            # Store the network
            self.networks.append(G)
            self.dates.append(window_date)
            
            # Compute and store curvature
            _, avg_curvature = self.compute_ollivier_ricci_curvature(G)
            self.curvatures.append(avg_curvature)
            
            # Compute and store entropy
            entropy = self.compute_network_entropy(G)
            self.entropies.append(entropy)
            
            # Compute and store average path length
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
            else:
                # Handle disconnected graphs
                largest_cc = max(nx.connected_components(G), key=len)
                largest_cc_subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(largest_cc_subgraph)
                diameter = nx.diameter(largest_cc_subgraph)
            
            self.avg_path_lengths.append(avg_path_length)
            self.diameters.append(diameter)
    
    def compute_minimum_risk_portfolio(self, returns, window_size=22):
        """
        Compute Markowitz minimum risk portfolio and project onto curvature
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with stock returns
        window_size : int
            Size of lookback window for portfolio computation
        
        Returns:
        --------
        risk_profile : numpy.ndarray
            Minimum risk profile over time
        curvature_weight : numpy.ndarray
            Projection of portfolio weights onto curvature
        """
        n_days = len(returns) - window_size
        n_stocks = len(returns.columns)
        
        risk_profile = np.zeros(n_days)
        curvature_weight = np.zeros(n_days)
        
        for i in tqdm(range(n_days), desc="Computing portfolios"):
            # Get returns for this window
            window_returns = returns.iloc[i:i+window_size]
            
            # Calculate covariance matrix
            cov_matrix = window_returns.cov()
            
            # Mean returns
            mean_returns = window_returns.mean()
            
            # Compute minimum risk portfolio
            try:
                # Simple implementation for minimum variance portfolio
                # Solve: min w^T Σ w subject to sum(w) = 1, w >= 0
                
                def objective(w):
                    return w @ cov_matrix.values @ w
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
                ]
                bounds = [(0, None) for _ in range(n_stocks)]  # Non-negative weights
                
                # Initial guess: equal weights
                initial_weights = np.ones(n_stocks) / n_stocks
                
                result = opt.minimize(objective, initial_weights, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                else:
                    weights = np.ones(n_stocks) / n_stocks  # Equal weight fallback
                
                # Calculate portfolio risk
                risk = np.sqrt(weights @ cov_matrix.values @ weights)
                risk_profile[i] = risk
                
                # Project onto curvature if we have enough networks computed
                if i < len(self.networks):
                    G = self.networks[i]
                    edge_curvatures, _ = self.compute_ollivier_ricci_curvature(G)
                    
                    # Create curvature matrix
                    stocks = list(window_returns.columns)
                    curvature_matrix = np.zeros((n_stocks, n_stocks))
                    
                    for (u, v), curv in edge_curvatures.items():
                        if u in stocks and v in stocks:
                            u_idx = stocks.index(u)
                            v_idx = stocks.index(v)
                            curvature_matrix[u_idx, v_idx] = curv
                            curvature_matrix[v_idx, u_idx] = curv
                    
                    # Project weights onto curvature
                    curvature_weight[i] = weights @ curvature_matrix @ weights
                
            except Exception as e:
                print(f"Error in portfolio calculation, window {i}: {e}")
                # Fallback to equal weights
                weights = np.ones(n_stocks) / n_stocks
                risk_profile[i] = np.sqrt(weights @ cov_matrix.values @ weights)
                curvature_weight[i] = 0
        
        return risk_profile, curvature_weight
    
    def plot_results(self):
        """Plot the results of the analysis"""
        # Convert lists to numpy arrays
        dates = np.array(self.dates)
        curvatures = np.array(self.curvatures)
        entropies = np.array(self.entropies)
        avg_path_lengths = np.array(self.avg_path_lengths)
        diameters = np.array(self.diameters)
        
        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot curvature
        axes[0].plot(dates, curvatures, 'b-', linewidth=1.5)
        axes[0].set_title('Ollivier-Ricci Curvature')
        axes[0].set_ylabel('Curvature')
        axes[0].grid(True)
        
        # Plot entropy
        axes[1].plot(dates, entropies, 'g-', linewidth=1.5)
        axes[1].set_title('Network Entropy')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True)
        
        # Plot average path length
        axes[2].plot(dates, avg_path_lengths, 'r-', linewidth=1.5)
        axes[2].set_title('Average Shortest Path Length')
        axes[2].set_ylabel('Path Length')
        axes[2].grid(True)
        
        # Plot diameter
        axes[3].plot(dates, diameters, 'm-', linewidth=1.5)
        axes[3].set_title('Graph Diameter')
        axes[3].set_ylabel('Diameter')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_results(self, risk_profile, curvature_weight, dates):
        """
        Plot portfolio results
        
        Parameters:
        -----------
        risk_profile : numpy.ndarray
            Minimum risk profile over time
        curvature_weight : numpy.ndarray
            Projection of portfolio weights onto curvature
        dates : list
            List of dates
        """
        # Ensure matching lengths
        min_len = min(len(risk_profile), len(curvature_weight), len(dates))
        risk_profile = risk_profile[:min_len]
        curvature_weight = curvature_weight[:min_len]
        dates = dates[:min_len]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot curvature weight
        ax1.plot(dates, curvature_weight, 'b-', linewidth=1.5)
        ax1.set_title('Ricci Curvature Portfolio Weight')
        ax1.set_ylabel('Wκ_port')
        ax1.grid(True)
        
        # Plot minimum risk
        ax2.plot(dates, risk_profile, 'r-', linewidth=1.5)
        ax2.set_title('Minimum Risk Profile')
        ax2.set_ylabel('Risk')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_with_crisis(self, crisis_periods, all_dates):
        """
        Plot results with crisis periods highlighted
        
        Parameters:
        -----------
        crisis_periods : list
            List of (start, end) tuples for crisis periods
        all_dates : list
            Complete list of dates for the simulation
        """
        dates = np.array(self.dates)
        curvatures = np.array(self.curvatures)
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, curvatures, 'b-', linewidth=1.5)
        
        # Highlight crisis periods
        for start, end in crisis_periods:
            if start < len(all_dates) and end < len(all_dates):
                # Find closest dates in our analysis dates
                start_date = all_dates[start]
                end_date = all_dates[end]
                
                # Find indices in our analysis dates
                start_idx = np.where(dates >= start_date)[0]
                end_idx = np.where(dates <= end_date)[0]
                
                if len(start_idx) > 0 and len(end_idx) > 0:
                    plt.axvspan(dates[start_idx[0]], dates[end_idx[-1]], 
                               alpha=0.3, color='red', label='Crisis Period')
        
        plt.title('Ollivier-Ricci Curvature with Crisis Periods Highlighted')
        plt.ylabel('Curvature')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class RicciCurvatureStrategy:
    def __init__(self, lookback_window=22, curvature_threshold=None, 
                 moving_avg_window=10, correlation_threshold=0.7):
        """
        Initialize the Ricci Curvature trading strategy
        
        Parameters:
        -----------
        lookback_window : int
            Window size for calculating correlation networks
        curvature_threshold : float or None
            Threshold for curvature to detect market regime changes
            If None, will use dynamic thresholds based on historical values
        moving_avg_window : int
            Window size for smoothing curvature values
        correlation_threshold : float
            Threshold for including edges in correlation networks
        """
        self.analyzer = RicciCurvatureAnalysis()
        self.lookback_window = lookback_window
        self.curvature_threshold = curvature_threshold
        self.moving_avg_window = moving_avg_window
        self.correlation_threshold = correlation_threshold
        
        # State variables
        self.curvature_history = []
        self.positions = {}
        self.cash = 0
        self.portfolio_value_history = []
    
    def calculate_curvature(self, prices):
        """
        Calculate the current market curvature
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            Current market prices
        
        Returns:
        --------
        curvature : float
            Current market curvature
        """
        returns = prices.pct_change().dropna()
        
        # Use the most recent lookback_window days
        if len(returns) >= self.lookback_window:
            window_returns = returns.iloc[-self.lookback_window:]
            
            # Create correlation network
            G = self.analyzer.create_correlation_network(
                window_returns, 
                threshold=self.correlation_threshold,
                use_mst=True
            )
            
            # Compute curvature
            _, curvature = self.analyzer.compute_ollivier_ricci_curvature(G)
            return curvature
        
        return 0  # Not enough data
    
    def detect_market_regime(self, curvature):
        """
        Detect market regime based on curvature
        
        Parameters:
        -----------
        curvature : float
            Current market curvature
        
        Returns:
        --------
        regime : str
            'normal', 'crash', or 'recovery'
        """
        # Add curvature to history
        self.curvature_history.append(curvature)
        
        # Not enough history to make reliable detection
        if len(self.curvature_history) < self.moving_avg_window:
            return 'normal'
        
        # Calculate moving average
        curvature_ma = np.mean(self.curvature_history[-self.moving_avg_window:])
        
        # If threshold is not provided, use dynamic threshold
        if self.curvature_threshold is None:
            # Use 1.5 standard deviations above the mean as threshold
            mean_curvature = np.mean(self.curvature_history)
            std_curvature = np.std(self.curvature_history) if len(self.curvature_history) > 1 else 0.01
            threshold = mean_curvature + 1.5 * std_curvature
        else:
            threshold = self.curvature_threshold
        
        # Z-score of current curvature
        if len(self.curvature_history) > 30:  # Enough history for z-score
            z_score = (curvature - np.mean(self.curvature_history)) / (np.std(self.curvature_history) + 1e-10)
        else:
            z_score = 0
        
        # Detect regime
        if curvature_ma > threshold:
            # High curvature indicates crash or recovery
            
            # Check if curvature is increasing (potential crash beginning)
            curvature_trend = self.curvature_history[-1] - self.curvature_history[-self.moving_avg_window]
            
            if curvature_trend > 0 and z_score > 1.5:
                return 'crash'
            else:
                return 'recovery'
        else:
            return 'normal'
    
    def allocate_portfolio(self, prices, regime):
        """
        Allocate portfolio based on market regime
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            Current market prices
        regime : str
            Current market regime
        
        Returns:
        --------
        allocations : dict
            Target allocations for each asset
        """
        n_assets = len(prices.columns)
        
        if regime == 'normal':
            # In normal markets, use minimum variance portfolio
            returns = prices.pct_change().dropna().iloc[-self.lookback_window:]
            cov_matrix = returns.cov()
            
            try:
                # Optimize for minimum risk
                def objective(w):
                    return w @ cov_matrix.values @ w
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
                ]
                bounds = [(0, None) for _ in range(n_assets)]  # Non-negative weights
                
                # Initial guess: equal weights
                initial_weights = np.ones(n_assets) / n_assets
                
                result = opt.minimize(objective, initial_weights, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    weights = result.x
                else:
                    weights = np.ones(n_assets) / n_assets
                
                # Create allocation dictionary
                allocations = {asset: weight for asset, weight in zip(prices.columns, weights)}
                
            except Exception as e:
                print(f"Portfolio optimization error: {e}")
                # Equal weight fallback
                allocations = {asset: 1.0/n_assets for asset in prices.columns}
            
        elif regime == 'crash':
            # In crash markets, allocate to anti-fragile assets
            
            # Get edge curvatures to identify the most robust assets
            returns = prices.pct_change().dropna().iloc[-self.lookback_window:]
            G = self.analyzer.create_correlation_network(
                returns, 
                threshold=self.correlation_threshold,
                use_mst=True
            )
            
            edge_curvatures, _ = self.analyzer.compute_ollivier_ricci_curvature(G)
            
            # Calculate node curvatures (average curvature of edges connected to each node)
            node_curvatures = {}
            for asset in prices.columns:
                connected_edges = [(u, v) for (u, v) in edge_curvatures.keys() if u == asset or v == asset]
                if connected_edges:
                    node_curvatures[asset] = np.mean([edge_curvatures[edge] for edge in connected_edges])
                else:
                    node_curvatures[asset] = -np.inf
            
            # Sort assets by curvature (higher curvature = more robust)
            sorted_assets = sorted(node_curvatures.keys(), key=lambda k: node_curvatures[k], reverse=True)
            
            # Allocate more to high-curvature assets (top 30%)
            num_top_assets = max(1, int(n_assets * 0.3))
            top_assets = sorted_assets[:num_top_assets]
            
            # Equal weight to top assets
            allocations = {asset: 1.0/num_top_assets if asset in top_assets else 0 for asset in prices.columns}
            
        elif regime == 'recovery':
            # In recovery, allocate to high Sharpe ratio assets
            returns = prices.pct_change().dropna().iloc[-self.lookback_window:]
            mean_returns = returns.mean()
            std_returns = returns.std()
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratios = mean_returns / (std_returns + 1e-10)  # Avoid division by zero
            
            # Sort assets by Sharpe ratio
            sorted_assets = sharpe_ratios.sort_values(ascending=False).index
            
            # Allocate more to high Sharpe ratio assets (top 50%)
            num_top_assets = max(1, int(n_assets * 0.5))
            top_assets = sorted_assets[:num_top_assets]
            
            # Equal weight to top assets
            allocations = {asset: 1.0/num_top_assets if asset in top_assets else 0 for asset in prices.columns}
        
        return allocations
    
    def rebalance(self, prices, target_allocations, current_allocations, portfolio_value):
        """
        Calculate trades needed to rebalance to target allocations
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            Current market prices
        target_allocations : dict
            Target allocations for each asset
        current_allocations : dict
            Current number of shares for each asset
        portfolio_value : float
            Current portfolio value
        
        Returns:
        --------
        trades : dict
            Number of shares to buy/sell for each asset
        """
        trades = {}
        
        for asset in prices.columns:
            current_price = prices[asset].iloc[-1]
            current_shares = current_allocations.get(asset, 0)
            current_value = current_shares * current_price
            
            target_value = portfolio_value * target_allocations.get(asset, 0)
            target_shares = target_value / current_price if current_price > 0 else 0
            
            # Calculate trade
            trade_shares = target_shares - current_shares
            trades[asset] = trade_shares
        
        return trades
    
    def backtest(self, prices, initial_capital=100000, transaction_cost=0.001):
        """
        Backtest the strategy
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            Historical price data
        initial_capital : float
            Initial capital
        transaction_cost : float
            Transaction cost as a percentage of trade value
        
        Returns:
        --------
        results : dict
            Backtest results
        """
        # Initialize
        self.cash = initial_capital
        self.positions = {asset: 0 for asset in prices.columns}
        self.portfolio_value_history = []
        self.curvature_history = []
        
        # Store results
        dates = []
        portfolio_values = []
        curvatures = []
        regimes = []
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Need at least lookback_window days to start
        start_idx = self.lookback_window
        
        # Run backtest
        for i in tqdm(range(start_idx, len(prices)), desc="Backtesting"):
            # Current prices
            current_prices = prices.iloc[:i+1]
            current_date = prices.index[i]
            current_prices_row = prices.iloc[i]
            
            # Calculate curvature
            curvature = self.calculate_curvature(current_prices)
            curvatures.append(curvature)
            
            # Detect market regime
            regime = self.detect_market_regime(curvature)
            regimes.append(regime)
            
            # Calculate current portfolio value
            portfolio_value = self.cash
            
            for asset, shares in self.positions.items():
                portfolio_value += shares * current_prices_row[asset]
            
            # Store results
            dates.append(current_date)
            portfolio_values.append(portfolio_value)
            
            # Determine if we should rebalance:
            # 1. Every 5 days OR
            # 2. When regime changes (but only if we have at least 2 regimes to compare)
            should_rebalance = i % 5 == 0
            if len(regimes) >= 2:
                regime_changed = regimes[-1] != regimes[-2]
                should_rebalance = should_rebalance or regime_changed
            
            # Rebalance portfolio if needed
            if should_rebalance:
                # Calculate target allocations
                target_allocations = self.allocate_portfolio(current_prices, regime)
                
                # Calculate trades
                trades = self.rebalance(
                    current_prices, 
                    target_allocations, 
                    self.positions, 
                    portfolio_value
                )
                
                # Execute trades
                for asset, shares in trades.items():
                    price = current_prices_row[asset]
                    
                    if shares > 0:  # Buy
                        cost = shares * price * (1 + transaction_cost)
                        if cost <= self.cash:  # Check if enough cash
                            self.positions[asset] += shares
                            self.cash -= cost
                        else:  # Adjust trade if not enough cash
                            affordable_shares = self.cash / (price * (1 + transaction_cost))
                            self.positions[asset] += affordable_shares
                            self.cash = 0
                    
                    elif shares < 0:  # Sell
                        current_shares = self.positions[asset]
                        sell_shares = min(abs(shares), current_shares)  # Can't sell more than we have
                        revenue = sell_shares * price * (1 - transaction_cost)
                        self.positions[asset] -= sell_shares
                        self.cash += revenue
    
        # Compile results
        results = {
            'dates': dates,
            'portfolio_values': portfolio_values,
            'curvatures': curvatures,
            'regimes': regimes,
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'returns': np.array(portfolio_values) / initial_capital - 1
        }
        
        return results
    
    def plot_results(self, results, benchmark_prices=None):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : dict
            Backtest results
        benchmark_prices : pandas.Series or None
            Benchmark prices for comparison
        """
        dates = results['dates']
        portfolio_values = results['portfolio_values']
        curvatures = results['curvatures']
        regimes = results['regimes']
        
        # Normalize portfolio values for returns calculation
        initial_value = portfolio_values[0]
        portfolio_returns = np.array(portfolio_values) / initial_value - 1
        
        # Create benchmark returns if provided
        if benchmark_prices is not None:
            # Align dates
            benchmark_aligned = benchmark_prices.loc[dates]
            benchmark_returns = benchmark_aligned / benchmark_aligned.iloc[0] - 1
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        
        # Plot portfolio returns
        axes[0].plot(dates, portfolio_returns, 'b-', linewidth=2, label='Strategy')
        
        if benchmark_prices is not None:
            axes[0].plot(dates, benchmark_returns, 'g--', linewidth=1.5, label='Benchmark')
        
        axes[0].set_title('Cumulative Returns')
        axes[0].set_ylabel('Return')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot curvature
        axes[1].plot(dates, curvatures, 'r-', linewidth=1.5)
        axes[1].set_title('Ollivier-Ricci Curvature')
        axes[1].set_ylabel('Curvature')
        axes[1].grid(True)
        
        # Plot regimes as background colors
        unique_regimes = set(regimes)
        colors = {'normal': 'green', 'crash': 'red', 'recovery': 'orange'}
        
        prev_regime = regimes[0]
        regime_start = dates[0]
        
        for i, (date, regime) in enumerate(zip(dates, regimes)):
            if regime != prev_regime or i == len(dates) - 1:
                # End of a regime period
                axes[0].axvspan(regime_start, date, alpha=0.2, color=colors.get(prev_regime, 'gray'))
                axes[1].axvspan(regime_start, date, alpha=0.2, color=colors.get(prev_regime, 'gray'))
                
                # Start a new regime period
                regime_start = date
                prev_regime = regime
        
        # Plot regime labels
        regime_data = pd.Series(regimes, index=dates)
        regime_changes = regime_data.drop_duplicates()
        
        # Create dummy plot for regime legend
        for regime, color in colors.items():
            axes[1].plot([], [], color=color, linewidth=10, label=regime.capitalize())
        
        axes[1].legend(loc='upper right')
        
        # Calculate and plot drawdowns
        max_values = np.maximum.accumulate(portfolio_values)
        drawdowns = 1 - np.array(portfolio_values) / max_values
        
        axes[2].fill_between(dates, 0, drawdowns, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_ylim(0, max(0.05, max(drawdowns) * 1.1))  # Add some padding
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        self.print_performance_metrics(results, benchmark_returns if benchmark_prices is not None else None)
    
    def print_performance_metrics(self, results, benchmark_returns=None):
        """
        Print performance metrics
        
        Parameters:
        -----------
        results : dict
            Backtest results
        benchmark_returns : numpy.ndarray or None
            Benchmark returns for comparison
        """
        portfolio_values = results['portfolio_values']
        dates = results['dates']
        
        # Calculate daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Performance metrics
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate max drawdown
        max_values = np.maximum.accumulate(portfolio_values)
        drawdowns = 1 - np.array(portfolio_values) / max_values
        max_drawdown = np.max(drawdowns)
        
        # Calculate regime statistics
        regimes = results['regimes']
        regime_count = {regime: regimes.count(regime) for regime in set(regimes)}
        regime_percent = {regime: count / len(regimes) * 100 for regime, count in regime_count.items()}
        
        # Print results
        print("===== Strategy Performance =====")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print()
        
        print("===== Market Regime Analysis =====")
        for regime, count in regime_count.items():
            print(f"{regime.capitalize()}: {count} days ({regime_percent[regime]:.1f}%)")
        
        # Compare to benchmark if provided
        if benchmark_returns is not None:
            # Calculate benchmark metrics
            benchmark_return = benchmark_returns[-1]
            benchmark_daily_returns = np.diff(benchmark_returns + 1) / (benchmark_returns[:-1] + 1)
            benchmark_volatility = np.std(benchmark_daily_returns) * np.sqrt(252)
            benchmark_annualized_return = (1 + benchmark_return) ** (252 / len(benchmark_returns)) - 1
            benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility > 0 else 0
            
            # Calculate information ratio
            return_diff = daily_returns - benchmark_daily_returns
            tracking_error = np.std(return_diff) * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error > 0 else 0
            
            print("\n===== Benchmark Comparison =====")
            print(f"Strategy vs Benchmark Return: {total_return:.2%} vs {benchmark_return:.2%}")
            print(f"Annualized Alpha: {annualized_return - benchmark_annualized_return:.2%}")
            print(f"Information Ratio: {information_ratio:.2f}")

def compute_node_curvatures(G, edge_curvatures):
    """
    Compute node curvatures based on edge curvatures
    
    Parameters:
    -----------
    G : networkx.Graph
        Network graph
    edge_curvatures : dict
        Dictionary with edge curvatures
    
    Returns:
    --------
    node_curvatures : dict
        Dictionary with node curvatures
    """
    node_curvatures = {}
    
    for node in G.nodes():
        # Get all edges connected to this node
        connected_edges = [(u, v) for (u, v) in edge_curvatures.keys() 
                         if u == node or v == node]
        
        if connected_edges:
            # Average curvature of connected edges
            node_curvatures[node] = np.mean([edge_curvatures[edge] for edge in connected_edges])
        else:
            node_curvatures[node] = 0
    
    return node_curvatures

def long_short_ricci_strategy(prices, lookback_window=22, correlation_threshold=0.7):
    """
    Implement a long-short strategy based on node-level Ricci curvature
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    lookback_window : int
        Window size for calculating correlation networks
    correlation_threshold : float
        Threshold for including edges in correlation networks
    
    Returns:
    --------
    results : dict
        Strategy results
    """
    analyzer = RicciCurvatureAnalysis()
    
    # Initialize variables
    portfolio_values = []
    positions = {}
    
    for i in tqdm(range(lookback_window, len(prices)), desc="Running long-short strategy"):
        # Get window data
        window = prices.iloc[i-lookback_window:i]
        returns = window.pct_change().dropna()
        
        # Create correlation network
        G = analyzer.create_correlation_network(
            returns, 
            threshold=correlation_threshold,
            use_mst=True
        )
        
        # Compute edge curvatures
        edge_curvatures, _ = analyzer.compute_ollivier_ricci_curvature(G)
        
        # Compute node curvatures
        node_curvatures = compute_node_curvatures(G, edge_curvatures)
        
        # Rank assets by curvature
        ranked_assets = sorted(node_curvatures.keys(), 
                             key=lambda k: node_curvatures[k])
        
        # Long the top 20% assets with highest curvature (most robust)
        num_assets = len(ranked_assets)
        num_long = max(1, int(num_assets * 0.2))
        num_short = max(1, int(num_assets * 0.2))
        
        long_assets = ranked_assets[-num_long:]
        short_assets = ranked_assets[:num_short]
        
        # Set positions (equal weights)
        new_positions = {}
        
        for asset in prices.columns:
            if asset in long_assets:
                new_positions[asset] = 1.0 / num_long
            elif asset in short_assets:
                new_positions[asset] = -1.0 / num_short
            else:
                new_positions[asset] = 0
        
        positions = new_positions
        
        # Calculate portfolio return for this day
        next_day_return = 0
        if i < len(prices) - 1:
            current_prices = prices.iloc[i]
            next_prices = prices.iloc[i+1]
            
            for asset, weight in positions.items():
                asset_return = next_prices[asset] / current_prices[asset] - 1
                next_day_return += weight * asset_return
        
        # Update portfolio value
        if not portfolio_values:
            portfolio_values.append(100000)
        else:
            portfolio_values.append(portfolio_values[-1] * (1 + next_day_return))
    
    return {
        'portfolio_values': portfolio_values,
        'final_value': portfolio_values[-1] if portfolio_values else 100000,
        'return': portfolio_values[-1] / 100000 - 1 if portfolio_values else 0
    }

def sector_rotation_strategy(prices, lookback_window=22, correlation_threshold=0.7):
    """
    Implement a sector rotation strategy based on Ricci curvature
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    lookback_window : int
        Window size for calculating correlation networks
    correlation_threshold : float
        Threshold for including edges in correlation networks
    
    Returns:
    --------
    results : dict
        Strategy results
    """
    # Assume columns of prices DataFrame represent different sectors
    analyzer = RicciCurvatureAnalysis()
    
    # Initialize variables
    cash = 100000
    positions = {sector: 0 for sector in prices.columns}
    portfolio_values = []
    sector_curvatures = {}
    
    # Run strategy
    for i in tqdm(range(lookback_window, len(prices)), desc="Running sector rotation"):
        # Get window data
        window = prices.iloc[i-lookback_window:i]
        returns = window.pct_change().dropna()
        
        # Calculate Ricci curvature for each sector
        for sector in prices.columns:
            # Create a subset with this sector and others
            sector_returns = returns[[sector]].join(returns.drop(sector, axis=1))
            
            # Create correlation network
            G = analyzer.create_correlation_network(
                sector_returns, 
                threshold=correlation_threshold,
                use_mst=True
            )
            
            # Compute curvature
            _, curvature = analyzer.compute_ollivier_ricci_curvature(G)
            sector_curvatures[sector] = curvature
        
        # Rank sectors by curvature (higher curvature = more robust)
        ranked_sectors = sorted(sector_curvatures.keys(), 
                              key=lambda k: sector_curvatures[k], 
                              reverse=True)
        
        # Allocate to top 3 sectors with highest curvature
        top_sectors = ranked_sectors[:3]
        
        # Rebalance portfolio
        current_prices = prices.iloc[i]
        
        # Calculate current portfolio value
        portfolio_value = cash
        for sector, shares in positions.items():
            portfolio_value += shares * current_prices[sector]
        
        portfolio_values.append(portfolio_value)
        
        # Sell positions in sectors not in top_sectors
        for sector in prices.columns:
            if sector not in top_sectors and positions[sector] > 0:
                # Sell all shares
                cash += positions[sector] * current_prices[sector] * 0.999  # 0.1% transaction cost
                positions[sector] = 0
        
        # Buy positions in top sectors
        target_value_per_sector = portfolio_value / len(top_sectors)
        
        for sector in top_sectors:
            current_value = positions[sector] * current_prices[sector]
            target_value = target_value_per_sector
            
            if target_value > current_value:
                # Buy more
                buy_value = target_value - current_value
                if buy_value <= cash:
                    shares_to_buy = buy_value / current_prices[sector]
                    positions[sector] += shares_to_buy
                    cash -= buy_value * 1.001  # 0.1% transaction cost
    
    return {
        'portfolio_values': portfolio_values,
        'final_value': portfolio_values[-1] if portfolio_values else cash,
        'return': portfolio_values[-1] / 100000 - 1 if portfolio_values else 0
    }

def analyze_ricci_strategy_performance(results, benchmark_returns=None):
    """
    Analyze the performance of a Ricci curvature-based strategy
    
    Parameters:
    -----------
    results : dict
        Strategy results
    benchmark_returns : pandas.Series or None
        Benchmark returns for comparison
    
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    # Extract data
    portfolio_values = results['portfolio_values']
    dates = results['dates']
    regimes = results['regimes']
    
    # Calculate returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cum_returns = np.array(portfolio_values) / portfolio_values[0] - 1
    
    # Basic metrics
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    drawdowns = 1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)
    max_drawdown = np.max(drawdowns)
    
    # Calculate drawdown duration
    underwater = np.zeros_like(drawdowns)
    for i in range(len(drawdowns)):
        if drawdowns[i] == 0:
            underwater[i] = 0
        else:
            underwater[i] = underwater[i-1] + 1 if i > 0 else 1
    
    max_underwater = np.max(underwater)
    
    # Calculate metrics by regime
    regime_metrics = {}
    
    for regime in set(regimes):
        regime_indices = [i for i, r in enumerate(regimes) if r == regime]
        
        if len(regime_indices) > 1:
            regime_returns = [daily_returns[i-1] for i in regime_indices if i > 0]
            
            if regime_returns:
                regime_metrics[regime] = {
                    'avg_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns) * np.sqrt(252),
                    'sharpe': np.mean(regime_returns) / np.std(regime_returns) * np.sqrt(252) if np.std(regime_returns) > 0 else 0,
                    'win_rate': np.mean([r > 0 for r in regime_returns]),
                    'days': len(regime_indices)
                }
    
    # Add benchmark comparison
    if benchmark_returns is not None:
        benchmark_daily_returns = np.diff(benchmark_returns) / benchmark_returns[:-1]
        
        # Calculate beta
        if len(benchmark_daily_returns) == len(daily_returns):
            covariance = np.cov(daily_returns, benchmark_daily_returns)[0, 1]
            benchmark_variance = np.var(benchmark_daily_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            # Calculate alpha
            risk_free_rate = 0  # Assume zero risk-free rate for simplicity
            expected_return = risk_free_rate + beta * (np.mean(benchmark_daily_returns) - risk_free_rate)
            alpha = np.mean(daily_returns) - expected_return
            
            # Calculate tracking error
            tracking_error = np.std(daily_returns - benchmark_daily_returns) * np.sqrt(252)
            
            # Information ratio
            information_ratio = (annualized_return - np.mean(benchmark_daily_returns) * 252) / tracking_error if tracking_error > 0 else 0
            
            # Add to metrics
            benchmark_metrics = {
                'beta': beta,
                'alpha': alpha * 252,  # Annualized
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
        else:
            benchmark_metrics = {}
    else:
        benchmark_metrics = {}
    
    # Compile all metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_underwater,
        'regime_metrics': regime_metrics,
        'benchmark_metrics': benchmark_metrics
    }
    
    return metrics

def market_regime_detector(prices, lookback_window=22, correlation_threshold=0.7):
    """
    Market regime detection system based on Ricci curvature
    
    Parameters:
    -----------
    prices : pandas.DataFrame
        Historical price data
    lookback_window : int
        Window size for calculating correlation networks
    correlation_threshold : float
        Threshold for including edges in correlation networks
    
    Returns:
    --------
    regimes : list
        Detected market regimes
    """
    analyzer = RicciCurvatureAnalysis()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Initialize results
    curvatures = []
    regimes = []
    
    # Calculate curvature for each window
    for i in tqdm(range(lookback_window, len(returns)), desc="Detecting regimes"):
        window_returns = returns.iloc[i-lookback_window:i]
        
        # Create correlation network
        G = analyzer.create_correlation_network(
            window_returns, 
            threshold=correlation_threshold,
            use_mst=True
        )
        
        # Compute curvature
        _, curvature = analyzer.compute_ollivier_ricci_curvature(G)
        curvatures.append(curvature)
        
        # Detect regime
        if len(curvatures) < 30:
            # Not enough history, assume normal
            regimes.append('normal')
        else:
            # Calculate statistics
            mean_curvature = np.mean(curvatures[:-1])
            std_curvature = np.std(curvatures[:-1])
            z_score = (curvature - mean_curvature) / (std_curvature + 1e-10)
            
            # Detect regime based on z-score
            if z_score > 1.5:
                # High curvature - crisis or recovery
                
                # Check if curvature is rising or falling
                curvature_change = curvature - curvatures[-2]
                
                if curvature_change > 0:
                    regimes.append('crisis')
                else:
                    regimes.append('recovery')
            else:
                regimes.append('normal')
    
    return regimes

def optimize_curvature_calculation(strategy):
    """
    Optimize the curvature calculation for faster execution
    
    Parameters:
    -----------
    strategy : RicciCurvatureStrategy
        The strategy object
    
    Returns:
    --------
    optimized_strategy : RicciCurvatureStrategy
        Optimized strategy
    """
    # Use a smaller subset of assets for correlation network
    original_create_network = strategy.analyzer.create_correlation_network
    
    def optimized_create_network(returns, threshold=0.7, use_mst=True):
        # If there are too many assets, sample a subset
        if returns.shape[1] > 50:
            # Sample 50 assets that represent the market well
            # (could use clustering or other approaches)
            sampled_returns = returns.sample(n=50, axis=1)
            return original_create_network(sampled_returns, threshold, use_mst)
        else:
            return original_create_network(returns, threshold, use_mst)
    
    # Replace the method
    strategy.analyzer.create_correlation_network = optimized_create_network
    
    return strategy

def add_risk_management(strategy):
    """
    Add risk management rules to the strategy
    
    Parameters:
    -----------
    strategy : RicciCurvatureStrategy
        The strategy object
    
    Returns:
    --------
    strategy_with_risk_mgmt : RicciCurvatureStrategy
        Strategy with risk management
    """
    original_allocate = strategy.allocate_portfolio
    
    def allocate_with_risk_mgmt(prices, regime):
        # Get original allocations
        allocations = original_allocate(prices, regime)
        
        # Add volatility targeting
        returns = prices.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # If market volatility is too high, reduce exposure
        market_vol = returns.mean(axis=1).std() * np.sqrt(252)
        if market_vol > 0.25:  # 25% annualized volatility threshold
            # Scale down allocations and increase cash
            reduction_factor = 0.25 / market_vol
            allocations = {asset: weight * reduction_factor 
                         for asset, weight in allocations.items()}
        
        # Add stop-loss
        if len(strategy.portfolio_value_history) > 0:
            initial_value = strategy.portfolio_value_history[0]
            current_value = strategy.portfolio_value_history[-1]
            drawdown = 1 - current_value / max(strategy.portfolio_value_history)
            
            # If drawdown exceeds threshold, reduce exposure
            if drawdown > 0.15:  # 15% drawdown threshold
                reduction_factor = max(0, 1 - (drawdown - 0.15) / 0.15)
                allocations = {asset: weight * reduction_factor 
                             for asset, weight in allocations.items()}
        
        return allocations
    
    # Replace the method
    strategy.allocate_portfolio = allocate_with_risk_mgmt
    
    return strategy

def run_ricci_strategy():
    """Run the Ricci curvature trading strategy on simulated data"""
    # Create the analyzer
    analyzer = RicciCurvatureAnalysis()
    
    # Simulate stock data with crisis periods
    print("Simulating stock data...")
    stock_data, crisis_periods, dates = analyzer.simulate_stock_data(
        n_stocks=20,
        days=500,
        include_crisis=True
    )
    
    # Create an equal-weighted portfolio as a benchmark
    benchmark = stock_data.mean(axis=1)
    
    # Create and run the strategy
    strategy = RicciCurvatureStrategy(
        lookback_window=22,
        curvature_threshold=None,  # Use dynamic threshold
        moving_avg_window=10,
        correlation_threshold=0.7
    )
    
    # Backtest the strategy
    print("Backtesting strategy...")
    results = strategy.backtest(
        stock_data,
        initial_capital=100000,
        transaction_cost=0.001
    )
    
    # Plot results
    print("Plotting results...")
    strategy.plot_results(results, benchmark_prices=benchmark)
    
    return strategy, results, stock_data, benchmark

def compare_strategies():
    """Compare different Ricci curvature-based strategies"""
    # Create the analyzer
    analyzer = RicciCurvatureAnalysis()
    
    # Simulate stock data with crisis periods
    print("Simulating stock data...")
    stock_data, crisis_periods, dates = analyzer.simulate_stock_data(
        n_stocks=20,
        days=500,
        include_crisis=True
    )
    
    # Create an equal-weighted portfolio as a benchmark
    benchmark = stock_data.mean(axis=1)
    benchmark_returns = benchmark.pct_change().dropna()
    
    # Strategy 1: Regular Ricci strategy
    print("\nRunning Ricci Curvature Strategy...")
    strategy1 = RicciCurvatureStrategy(
        lookback_window=22,
        curvature_threshold=None,
        moving_avg_window=10,
        correlation_threshold=0.7
    )
    results1 = strategy1.backtest(stock_data)
    
    # Strategy 2: Long-Short Ricci strategy
    print("\nRunning Long-Short Ricci Strategy...")
    results2_raw = long_short_ricci_strategy(stock_data)
    results2 = {
        'dates': stock_data.index[22:22+len(results2_raw['portfolio_values'])],
        'portfolio_values': results2_raw['portfolio_values'],
        'final_value': results2_raw['final_value'],
        'returns': results2_raw['return']
    }
    
    # Strategy 3: Sector Rotation Ricci strategy
    print("\nRunning Sector Rotation Strategy...")
    results3_raw = sector_rotation_strategy(stock_data)
    results3 = {
        'dates': stock_data.index[22:22+len(results3_raw['portfolio_values'])],
        'portfolio_values': results3_raw['portfolio_values'],
        'final_value': results3_raw['final_value'],
        'returns': results3_raw['return']
    }
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Plot strategy returns
    plt.plot(results1['dates'], np.array(results1['portfolio_values']) / results1['portfolio_values'][0] - 1, 
             'b-', linewidth=2, label='Regime-Based Strategy')
    
    plt.plot(results2['dates'], np.array(results2['portfolio_values']) / results2['portfolio_values'][0] - 1, 
             'g-', linewidth=2, label='Long-Short Strategy')
    
    plt.plot(results3['dates'], np.array(results3['portfolio_values']) / results3['portfolio_values'][0] - 1, 
             'r-', linewidth=2, label='Sector Rotation Strategy')
    
    # Plot benchmark
    benchmark_aligned = benchmark.loc[results1['dates']]
    plt.plot(results1['dates'], benchmark_aligned / benchmark_aligned.iloc[0] - 1, 
             'k--', linewidth=1.5, label='Benchmark')
    
    plt.title('Strategy Comparison')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\n===== Strategy 1: Regime-Based =====")
    metrics1 = analyze_ricci_strategy_performance(results1, benchmark_aligned.values / benchmark_aligned.iloc[0] - 1)
    print(f"Total Return: {metrics1['total_return']:.2%}")
    print(f"Annualized Return: {metrics1['annualized_return']:.2%}")
    print(f"Volatility: {metrics1['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics1['max_drawdown']:.2%}")
    
    print("\n===== Strategy 2: Long-Short =====")
    print(f"Total Return: {results2['returns']:.2%}")
    
    print("\n===== Strategy 3: Sector Rotation =====")
    print(f"Total Return: {results3['returns']:.2%}")
    
    print("\n===== Benchmark =====")
    benchmark_return = benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1
    print(f"Total Return: {benchmark_return:.2%}")
    
    return {
        'strategy1': {'strategy': strategy1, 'results': results1},
        'strategy2': {'results': results2},
        'strategy3': {'results': results3},
        'benchmark': benchmark,
        'stock_data': stock_data
    }

if __name__ == "__main__":
    # Run the Ricci curvature trading strategy
    strategy, results, stock_data, benchmark = run_ricci_strategy()
    
    # Uncomment to run strategy comparison
    # strategy_comparison = compare_strategies()