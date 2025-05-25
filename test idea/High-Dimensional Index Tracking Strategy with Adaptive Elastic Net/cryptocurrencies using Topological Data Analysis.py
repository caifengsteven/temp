import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tadasets
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def compute_persistence_landscape(diagram, num_landscapes=5, resolution=100):
    """
    Compute persistence landscapes from a persistence diagram
    
    Parameters:
    -----------
    diagram : ndarray
        Persistence diagram
    num_landscapes : int
        Number of landscapes to compute
    resolution : int
        Resolution of the landscapes
        
    Returns:
    --------
    landscapes : ndarray
        Persistence landscapes
    """
    if len(diagram) == 0:
        return np.zeros((num_landscapes, resolution))
    
    # Extract birth and death times
    birth = diagram[:, 0]
    death = diagram[:, 1]
    
    # Define the range for the landscape
    min_birth = min(birth) if len(birth) > 0 else 0
    max_death = max(death) if len(death) > 0 else 1
    
    # Add some padding to the range
    padding = 0.1 * (max_death - min_birth)
    min_x = min_birth - padding
    max_x = max_death + padding
    
    # Generate x values for the landscape
    x_vals = np.linspace(min_x, max_x, resolution)
    
    # Compute the landscape functions
    landscapes = np.zeros((num_landscapes, resolution))
    
    for i, x in enumerate(x_vals):
        # Compute all critical points for this x value
        critical_points = []
        
        for b, d in zip(birth, death):
            if b <= x <= d:
                # Distance to diagonal is (d - b) / 2
                midpoint = (b + d) / 2
                if x <= midpoint:
                    # Increasing part of the triangle
                    critical_points.append(x - b)
                else:
                    # Decreasing part of the triangle
                    critical_points.append(d - x)
            else:
                critical_points.append(0)  # Outside the persistence interval
        
        # Sort critical points in descending order
        critical_points.sort(reverse=True)
        
        # Assign to landscape layers
        for k in range(min(num_landscapes, len(critical_points))):
            landscapes[k, i] = critical_points[k]
    
    return landscapes

class TDAPortfolioManager:
    """
    A class to implement the improved TDA-based portfolio management for cryptocurrencies
    as described in the paper "Topological Data Analysis for Portfolio Management of Cryptocurrencies"
    with additional enhancements for better performance.
    """
    
    def __init__(self, window_size=30, max_dimension=1, embedding_dimension=3,
                 num_landscapes=5, landscape_resolution=100, dim_reduction=None):
        """
        Initialize the TDA Portfolio Manager
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for time delay embeddings
        max_dimension : int
            Maximum homology dimension to compute
        embedding_dimension : int
            Dimension for time delay embedding
        num_landscapes : int
            Number of landscapes to compute
        landscape_resolution : int
            Resolution of the landscapes
        dim_reduction : function or None
            Dimensionality reduction function (not used in this implementation)
        """
        self.window_size = window_size
        self.max_dimension = max_dimension
        self.embedding_dimension = embedding_dimension
        self.num_landscapes = num_landscapes
        self.landscape_resolution = landscape_resolution
        self.dim_reduction = dim_reduction
        self.scaler = MinMaxScaler()
        
    def _time_delay_embedding(self, time_series, delay=1):
        """
        Create time delay embeddings from a time series
        
        Parameters:
        -----------
        time_series : array-like
            Time series data
        delay : int
            Delay between points
            
        Returns:
        --------
        point_cloud : ndarray
            The resulting point cloud
        """
        N = len(time_series)
        if N < self.embedding_dimension * delay:
            raise ValueError("Time series is too short for the requested embedding")
            
        point_cloud = np.zeros((N - (self.embedding_dimension-1) * delay, self.embedding_dimension))
        
        for i in range(self.embedding_dimension):
            point_cloud[:, i] = time_series[i*delay:N-(self.embedding_dimension-1-i)*delay]
            
        return point_cloud
    
    def _sliding_window(self, time_series):
        """
        Generate sliding windows for the time series
        
        Parameters:
        -----------
        time_series : array-like
            Time series data
            
        Returns:
        --------
        windows : list
            List of sliding windows
        """
        windows = []
        for i in range(len(time_series) - self.window_size + 1):
            window = time_series[i:i+self.window_size]
            windows.append(window)
        return windows
    
    def _compute_persistence_diagrams(self, point_cloud):
        """
        Compute persistence diagrams for a point cloud
        
        Parameters:
        -----------
        point_cloud : ndarray
            Point cloud data
            
        Returns:
        --------
        diagrams : list
            List of persistence diagrams for different dimensions
        """
        result = ripser(point_cloud, maxdim=self.max_dimension)
        diagrams = result['dgms']
        return diagrams
    
    def _compute_landscapes(self, diagrams, dimension=1):
        """
        Compute persistence landscapes from persistence diagrams
        
        Parameters:
        -----------
        diagrams : list
            List of persistence diagrams
        dimension : int
            Homology dimension to use
            
        Returns:
        --------
        landscapes : ndarray
            Persistence landscapes
        """
        if len(diagrams) <= dimension:
            # Return zero landscape if the dimension is not available
            return np.zeros((self.num_landscapes, self.landscape_resolution))
        
        diagram = diagrams[dimension]
        landscapes = compute_persistence_landscape(
            diagram, 
            num_landscapes=self.num_landscapes, 
            resolution=self.landscape_resolution
        )
        return landscapes
    
    def _compute_l2_norm(self, landscapes):
        """
        Compute L2 norm of persistence landscapes
        
        Parameters:
        -----------
        landscapes : ndarray
            Persistence landscapes
            
        Returns:
        --------
        l2_norm : float
            L2 norm of the landscapes
        """
        # Sum squares across all values in all landscapes and take square root
        return np.sqrt(np.sum(landscapes**2))
    
    def _compute_enhanced_rfm_features(self, diff_norms, prices, lookback_period=30):
        """
        Compute enhanced RFM features for a cryptocurrency
        
        Parameters:
        -----------
        diff_norms : array-like
            Array of differences between consecutive L2 norms
        prices : array-like
            Array of cryptocurrency prices
        lookback_period : int
            Period to look back for calculations
            
        Returns:
        --------
        rfm : tuple
            Tuple containing enhanced (recency, frequency, monetary) values
        """
        # Only consider the last part of the data for RFM calculation
        recent_diff_norms = diff_norms[-lookback_period:]
        recent_prices = prices[-lookback_period:]
        
        # Calculate returns
        returns = np.diff(recent_prices, prepend=recent_prices[0]) / recent_prices
        
        # Recency: Number of days since diff_norm was positive
        positive_indices = np.where(recent_diff_norms > 0)[0]
        recency = lookback_period - positive_indices[-1] if len(positive_indices) > 0 else lookback_period
        
        # Frequency: Number of times diff_norm has been positive in the period
        frequency = np.sum(recent_diff_norms > 0)
        
        # Monetary: Cumulative value of the positive diff_norms, weighted by returns
        monetary_base = np.sum(recent_diff_norms[recent_diff_norms > 0])
        
        # Enhanced Monetary: Weight by returns when diff_norm is positive
        positive_mask = recent_diff_norms > 0
        if np.any(positive_mask):
            return_weighted = recent_diff_norms[positive_mask] * (1 + returns[positive_mask])
            monetary = np.sum(return_weighted)
        else:
            monetary = monetary_base
            
        # Volatility: Measure of risk
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Trend: Recent price momentum
        trend = (recent_prices[-1] / recent_prices[0]) - 1
        
        # Add volatility as a penalty factor (lower is better)
        volatility_penalty = 1 / (1 + volatility) if volatility > 0 else 1
        
        # Adjust Monetary by trend and volatility
        monetary_adjusted = monetary * (1 + trend) * volatility_penalty
        
        return (recency, frequency, monetary_adjusted)
    
    def process_cryptocurrency(self, time_series, normalize=True):
        """
        Process a cryptocurrency time series using TDA
        
        Parameters:
        -----------
        time_series : array-like
            Time series of cryptocurrency prices
        normalize : bool
            Whether to normalize the time series before processing
            
        Returns:
        --------
        norms : array
            Array of L2 norms for the persistence landscapes
        diff_norms : array
            Array of differences between consecutive L2 norms
        """
        # Make a copy to avoid modifying the original
        ts = np.array(time_series)
        
        # Normalize if requested (better for comparing different price scales)
        if normalize:
            # Log returns for windows
            log_prices = np.log(ts)
            ts = np.diff(log_prices, prepend=log_prices[0])
        
        # Generate sliding windows
        windows = self._sliding_window(ts)
        
        # Compute persistence diagrams and landscapes for each window
        norms = []
        for window in windows:
            # Create point cloud using time delay embedding
            point_cloud = self._time_delay_embedding(window, delay=1)
            
            # Apply dimensionality reduction if specified
            if self.dim_reduction is not None:
                point_cloud = self.dim_reduction(point_cloud)
            
            # Compute persistence diagrams
            diagrams = self._compute_persistence_diagrams(point_cloud)
            
            # Compute persistence landscapes for dimension 1 (loops)
            landscapes = self._compute_landscapes(diagrams, dimension=1)
            
            # Compute L2 norm of landscapes
            norm = self._compute_l2_norm(landscapes)
            norms.append(norm)
        
        norms = np.array(norms)
        
        # Compute differences between consecutive norms
        diff_norms = np.diff(norms, prepend=norms[0])
        
        return norms, diff_norms
    
    def create_portfolio(self, crypto_data, lookback_period=60, risk_threshold=0.5, 
                         min_allocation_pct=0.01, max_allocation_pct=0.30):
        """
        Create a portfolio allocation based on enhanced TDA features with risk management
        
        Parameters:
        -----------
        crypto_data : dict
            Dictionary mapping cryptocurrency names to their price time series
        lookback_period : int
            Period to look back for RFM calculations
        risk_threshold : float
            Threshold for filtering high-risk assets
        min_allocation_pct : float
            Minimum allocation percentage for any included cryptocurrency
        max_allocation_pct : float
            Maximum allocation percentage for any cryptocurrency
            
        Returns:
        --------
        portfolio : dict
            Dictionary mapping cryptocurrency names to their allocation percentages
        metrics : dict
            Dictionary containing computed metrics for each cryptocurrency
        """
        rfm_features = {}
        norms_data = {}
        diff_norms_data = {}
        risk_data = {}
        
        # Process each cryptocurrency
        for name, time_series in crypto_data.items():
            print(f"Processing {name}...")
            
            # Calculate risk metrics before TDA processing
            if len(time_series) > lookback_period:
                recent_prices = time_series[-lookback_period:]
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                # Downside deviation (risk of losses)
                negative_returns = returns[returns < 0]
                downside_risk = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
                
                # Combined risk score
                risk_score = 0.7 * volatility + 0.3 * downside_risk
                risk_data[name] = risk_score
            else:
                risk_data[name] = float('inf')  # Very high risk if not enough data
            
            # TDA processing
            norms, diff_norms = self.process_cryptocurrency(time_series)
            norms_data[name] = norms
            diff_norms_data[name] = diff_norms
            
            # Compute enhanced RFM features
            rfm = self._compute_enhanced_rfm_features(diff_norms, time_series, lookback_period)
            rfm_features[name] = rfm
        
        # Convert RFM features to DataFrame for easier normalization
        rfm_df = pd.DataFrame.from_dict(rfm_features, orient='index', 
                                       columns=['Recency', 'Frequency', 'Monetary'])
        
        # Add risk as a feature (lower is better)
        risk_df = pd.DataFrame.from_dict(risk_data, orient='index', columns=['Risk'])
        
        # Combine RFM and risk
        feature_df = pd.concat([rfm_df, risk_df], axis=1)
        
        # Invert Recency so that lower values (more recent) get higher scores
        feature_df['Recency'] = lookback_period - feature_df['Recency']
        
        # Invert Risk so that lower risk gets higher scores
        if feature_df['Risk'].max() > feature_df['Risk'].min():
            feature_df['Risk'] = (feature_df['Risk'].max() - feature_df['Risk']) / (feature_df['Risk'].max() - feature_df['Risk'].min())
        else:
            feature_df['Risk'] = 1.0  # If all risks are the same
        
        # Normalize features to [0, 1]
        normalized_features = self.scaler.fit_transform(feature_df)
        normalized_df = pd.DataFrame(normalized_features, index=feature_df.index, 
                                    columns=feature_df.columns)
        
        # Calculate weighted scores with higher weight on Risk
        normalized_df['Score'] = (
            0.25 * normalized_df['Recency'] + 
            0.25 * normalized_df['Frequency'] + 
            0.25 * normalized_df['Monetary'] + 
            0.25 * normalized_df['Risk']
        )
        
        # Filter cryptocurrencies based on risk threshold
        low_risk_assets = normalized_df[normalized_df['Risk'] >= (1 - risk_threshold)]
        
        # Filter out assets with non-positive scores
        positive_scores = low_risk_assets[low_risk_assets['Score'] > 0]
        
        # Calculate allocation percentages
        if len(positive_scores) > 0:
            total_score = positive_scores['Score'].sum()
            positive_scores['RawAllocation'] = positive_scores['Score'] / total_score
            
            # Apply min and max allocation constraints
            allocations = {}
            for name, row in positive_scores.iterrows():
                allocation = row['RawAllocation']
                
                # Apply min/max constraints
                allocation = max(min_allocation_pct, min(allocation, max_allocation_pct))
                allocations[name] = allocation
            
            # Normalize to ensure allocations sum to 1
            total_allocation = sum(allocations.values())
            portfolio = {name: alloc/total_allocation for name, alloc in allocations.items()}
        else:
            # Fallback to less risky assets with equal allocation
            if len(normalized_df) > 0:
                # Get top 3 least risky assets
                top_by_risk = normalized_df.sort_values('Risk', ascending=False).head(3)
                portfolio = {name: 1.0/len(top_by_risk) for name in top_by_risk.index}
            else:
                # If no data, use equal allocation for all
                portfolio = {name: 1.0/len(crypto_data) for name in crypto_data.keys()}
        
        # Check if portfolio is too concentrated (more than 30% in one asset)
        max_alloc = max(portfolio.values()) if portfolio else 0
        if max_alloc > 0.3:
            print("Warning: Portfolio is highly concentrated. Applying diversification...")
            # Redistribute excess allocation
            excess = {name: (alloc - 0.3) for name, alloc in portfolio.items() if alloc > 0.3}
            total_excess = sum(excess.values())
            
            # Remove excess from assets above threshold
            for name in excess:
                portfolio[name] = 0.3
            
            # Redistribute excess proportionally to assets below threshold
            below_threshold = {name: alloc for name, alloc in portfolio.items() if alloc < 0.3}
            if below_threshold:
                total_below = sum(below_threshold.values())
                for name, alloc in below_threshold.items():
                    portfolio[name] += total_excess * (alloc / total_below)
            
        # Prepare metrics for return
        metrics = {
            'RFM': rfm_features,
            'Norms': norms_data,
            'DiffNorms': diff_norms_data,
            'NormalizedFeatures': normalized_df,
            'Risk': risk_data
        }
        
        return portfolio, metrics

def generate_realistic_crypto_data(num_cryptos=10, data_points=1000, 
                                  include_bear_market=True, bear_market_start=700):
    """
    Generate more realistic simulated cryptocurrency price data
    
    Parameters:
    -----------
    num_cryptos : int
        Number of cryptocurrencies to simulate
    data_points : int
        Number of daily price points to generate
    include_bear_market : bool
        Whether to include a bear market period
    bear_market_start : int
        Index where bear market begins
        
    Returns:
    --------
    crypto_data : dict
        Dictionary mapping cryptocurrency names to their price time series
    dates : array
        Array of dates for the time series
    """
    # Generate dates
    start_date = datetime(2013, 5, 1)
    dates = [start_date + timedelta(days=i) for i in range(data_points)]
    
    # Base market factor (common to all cryptocurrencies)
    market_returns = np.random.normal(0.001, 0.02, data_points)
    
    # Add cyclical component to simulate crypto cycles
    cycle_period = 365  # One year cycle
    t = np.arange(data_points)
    cycle = 0.1 * np.sin(2 * np.pi * t / cycle_period)
    market_returns += cycle
    
    # Add bear market if requested
    if include_bear_market and bear_market_start < data_points:
        bear_duration = min(250, data_points - bear_market_start)
        bear_severity = np.linspace(0, -0.01, bear_duration)  # Gradual decline
        market_returns[bear_market_start:bear_market_start+bear_duration] += bear_severity
    
    crypto_data = {}
    
    # Create different types of cryptocurrencies
    crypto_types = [
        {"name": "Large Cap", "count": max(1, num_cryptos // 5), "volatility": (0.01, 0.03), "start_price": (50, 500)},
        {"name": "Mid Cap", "count": max(2, num_cryptos // 3), "volatility": (0.03, 0.06), "start_price": (5, 50)},
        {"name": "Small Cap", "count": num_cryptos - max(1, num_cryptos // 5) - max(2, num_cryptos // 3), 
         "volatility": (0.06, 0.12), "start_price": (0.1, 5)}
    ]
    
    crypto_counter = 1
    
    for crypto_type in crypto_types:
        for i in range(crypto_type["count"]):
            # Generate unique parameters for this cryptocurrency
            volatility_range = crypto_type["volatility"]
            start_price_range = crypto_type["start_price"]
            
            volatility = np.random.uniform(*volatility_range)
            trend = np.random.uniform(-0.0002, 0.001)
            
            # Generate returns with correlation to market
            correlation = np.random.uniform(0.3, 0.7)  # Different correlations
            specific_returns = np.random.normal(trend, volatility, data_points)
            correlated_returns = correlation * market_returns + (1 - correlation) * specific_returns
            
            # Convert returns to prices with a random starting price
            start_price = np.random.uniform(*start_price_range)
            prices = start_price * np.cumprod(1 + correlated_returns)
            
            # Add some bubble-like behavior for realism
            if np.random.rand() < 0.7:  # 70% chance of having a bubble
                bubble_center = np.random.randint(data_points // 3, 2 * data_points // 3)
                bubble_width = np.random.randint(30, 90)
                bubble_height = np.random.uniform(0.5, 2.0) * (1.0 / start_price)  # Smaller bubbles for higher prices
                
                # Create bubble effect
                x = np.arange(data_points)
                bubble = bubble_height * np.exp(-0.5 * ((x - bubble_center) / bubble_width) ** 2)
                prices = prices * (1 + bubble)
            
            # Add periodic fluctuations
            if np.random.rand() < 0.5:  # 50% chance
                period = np.random.randint(30, 180)
                amplitude = np.random.uniform(0.03, 0.15)
                fluctuation = amplitude * np.sin(2 * np.pi * t / period)
                prices = prices * (1 + fluctuation)
            
            crypto_name = f"Crypto_{crypto_counter}"
            crypto_data[crypto_name] = prices
            crypto_counter += 1
    
    # Make Crypto_1 behave like Bitcoin if it exists
    if 'Crypto_1' in crypto_data:
        # Bitcoin-like parameters
        start_price = 100
        trend = 0.0015  # Higher trend
        volatility = 0.015  # Lower volatility
        specific_returns = np.random.normal(trend, volatility, data_points)
        market_correlation = 0.6  # Strong market correlation
        
        # Combine specific and market returns
        bitcoin_returns = market_correlation * market_returns + (1 - market_correlation) * specific_returns
        
        # Add periodic market cycles (about 4 years)
        bitcoin_cycle = 0.2 * np.sin(2 * np.pi * t / (4 * 365))
        bitcoin_returns += bitcoin_cycle
        
        prices = start_price * np.cumprod(1 + bitcoin_returns)
        
        # Add the 2017-2018 bubble
        bubble_center = int(data_points * 0.6)  # Around 60% through the time series
        bubble_width = 120
        bubble_height = 5  # Big bubble for Bitcoin
        
        x = np.arange(data_points)
        bubble = bubble_height * np.exp(-0.5 * ((x - bubble_center) / bubble_width) ** 2)
        prices = prices * (1 + bubble)
        
        crypto_data['Crypto_1'] = prices  # Bitcoin analog
    
    return crypto_data, dates

def evaluate_portfolio(portfolio, crypto_data, evaluation_period=30):
    """
    Evaluate the performance of a portfolio over a period
    
    Parameters:
    -----------
    portfolio : dict
        Dictionary mapping cryptocurrency names to their allocation percentages
    crypto_data : dict
        Dictionary mapping cryptocurrency names to their price time series
    evaluation_period : int
        Number of days to evaluate the portfolio
        
    Returns:
    --------
    portfolio_returns : array
        Daily returns of the portfolio
    """
    # Get the last evaluation_period days of data
    crypto_returns = {}
    for name, prices in crypto_data.items():
        # Calculate daily returns
        returns = np.diff(prices[-evaluation_period-1:]) / prices[-evaluation_period-1:-1]
        crypto_returns[name] = returns
    
    # Calculate portfolio returns
    portfolio_returns = np.zeros(evaluation_period)
    
    for name, allocation in portfolio.items():
        if name in crypto_returns:
            portfolio_returns += allocation * crypto_returns[name]
    
    return portfolio_returns

def naive_portfolio(crypto_data):
    """
    Create a naive 1/N portfolio allocation
    
    Parameters:
    -----------
    crypto_data : dict
        Dictionary mapping cryptocurrency names to their price time series
        
    Returns:
    --------
    portfolio : dict
        Dictionary mapping cryptocurrency names to their allocation percentages
    """
    allocation = 1.0 / len(crypto_data)
    return {name: allocation for name in crypto_data}

def plot_results(dates, crypto_data, tda_portfolio, naive_portfolio_allocation, metrics, 
                 evaluation_period=180, lookback_window=60):
    """
    Plot various visualizations of the results
    
    Parameters:
    -----------
    dates : array
        Array of dates for the time series
    crypto_data : dict
        Dictionary mapping cryptocurrency names to their price time series
    tda_portfolio : dict
        TDA-based portfolio allocation
    naive_portfolio_allocation : dict
        Naive portfolio allocation
    metrics : dict
        Dictionary containing computed metrics for each cryptocurrency
    evaluation_period : int
        Number of days to evaluate and display portfolio performance
    lookback_window : int
        Window used for RFM calculations
    """
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Plot prices of all cryptocurrencies
    ax1 = fig.add_subplot(3, 2, 1)
    for name, prices in crypto_data.items():
        if name == 'Crypto_1':  # Bitcoin analog
            ax1.plot(dates, prices, 'b-', linewidth=2, label=name)
        else:
            ax1.plot(dates, prices, 'k-', alpha=0.2)
    
    ax1.set_title('Cryptocurrency Prices (Crypto_1 in blue)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    
    # Highlight the evaluation period
    start_eval = dates[-evaluation_period]
    ax1.axvline(x=start_eval, color='r', linestyle='--', label='Evaluation Start')
    
    # 2. Plot L2 norms for selected cryptocurrencies
    ax2 = fig.add_subplot(3, 2, 2)
    portfolio_cryptos = list(tda_portfolio.keys())[:3]  # Top 3 allocated cryptos
    selected_cryptos = portfolio_cryptos + ['Crypto_1'] if 'Crypto_1' not in portfolio_cryptos else portfolio_cryptos
    selected_cryptos = selected_cryptos[:3]  # Limit to 3
    
    for name in selected_cryptos:
        if name in metrics['Norms']:
            norms = metrics['Norms'][name]
            norm_dates = dates[len(dates)-len(norms):]
            ax2.plot(norm_dates, norms, label=name)
    
    ax2.set_title('L2 Norms of Persistence Landscapes')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('L2 Norm')
    ax2.legend()
    
    # 3. Plot differences of L2 norms for selected cryptocurrencies
    ax3 = fig.add_subplot(3, 2, 3)
    for name in selected_cryptos:
        if name in metrics['DiffNorms']:
            diff_norms = metrics['DiffNorms'][name]
            diff_dates = dates[len(dates)-len(diff_norms):]
            ax3.plot(diff_dates, diff_norms, label=name)
    
    ax3.set_title('Differences of L2 Norms')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Diff L2 Norm')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.legend()
    
    # 4. Plot TDA portfolio allocation vs Naive
    ax4 = fig.add_subplot(3, 2, 4)
    
    # Combine allocations for comparison
    all_assets = set(tda_portfolio.keys()) | set(naive_portfolio_allocation.keys())
    combined_alloc = pd.DataFrame(index=all_assets, columns=['TDA', 'Naive'])
    
    for asset in all_assets:
        combined_alloc.loc[asset, 'TDA'] = tda_portfolio.get(asset, 0)
        combined_alloc.loc[asset, 'Naive'] = naive_portfolio_allocation.get(asset, 0)
    
    # Sort by TDA allocation
    combined_alloc = combined_alloc.sort_values('TDA', ascending=False)
    
    # Plot top 10 assets
    top_assets = combined_alloc.head(10)
    top_assets.plot(kind='bar', ax=ax4)
    
    ax4.set_title('Portfolio Allocation Comparison (Top 10 by TDA weight)')
    ax4.set_xlabel('Cryptocurrency')
    ax4.set_ylabel('Allocation')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Evaluate and plot portfolio performance
    ax5 = fig.add_subplot(3, 2, 5)
    
    # Calculate cumulative returns for both portfolios
    tda_returns = evaluate_portfolio(tda_portfolio, crypto_data, evaluation_period)
    naive_returns = evaluate_portfolio(naive_portfolio_allocation, crypto_data, evaluation_period)
    
    tda_cumulative = np.cumprod(1 + tda_returns) - 1
    naive_cumulative = np.cumprod(1 + naive_returns) - 1
    
    evaluation_dates = dates[-evaluation_period:]
    ax5.plot(evaluation_dates, tda_cumulative, 'g-', label='TDA Portfolio')
    ax5.plot(evaluation_dates, naive_cumulative, 'r-', label='Naive Portfolio')
    
    ax5.set_title('Portfolio Cumulative Returns')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Cumulative Return')
    ax5.legend()
    
    # 6. Show Risk vs. Score metrics
    ax6 = fig.add_subplot(3, 2, 6)
    
    if 'NormalizedFeatures' in metrics:
        features_df = metrics['NormalizedFeatures']
        
        # Create scatter plot of Risk vs Score for all cryptos
        ax6.scatter(features_df['Risk'], features_df['Score'], alpha=0.7)
        
        # Highlight selected assets in the portfolio
        portfolio_assets = list(tda_portfolio.keys())
        selected_data = features_df.loc[portfolio_assets]
        ax6.scatter(selected_data['Risk'], selected_data['Score'], color='g', s=100, label='In Portfolio')
        
        # Add labels for portfolio assets
        for idx, row in selected_data.iterrows():
            ax6.annotate(idx, (row['Risk'], row['Score']), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax6.set_title('Risk vs. Score for Cryptocurrencies')
        ax6.set_xlabel('Risk Score (higher is better)')
        ax6.set_ylabel('Total Score')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Display additional performance metrics
    print("\nPortfolio Performance Metrics:")
    print("-" * 30)
    print(f"TDA Portfolio Final Return: {tda_cumulative[-1]*100:.2f}%")
    print(f"Naive Portfolio Final Return: {naive_cumulative[-1]*100:.2f}%")
    
    # Calculate annualized Sharpe Ratio (assuming 0 risk-free rate)
    tda_sharpe = np.mean(tda_returns) / np.std(tda_returns) * np.sqrt(252)  # 252 trading days in a year
    naive_sharpe = np.mean(naive_returns) / np.std(naive_returns) * np.sqrt(252)
    
    print(f"TDA Portfolio Sharpe Ratio: {tda_sharpe:.2f}")
    print(f"Naive Portfolio Sharpe Ratio: {naive_sharpe:.2f}")
    
    # Calculate drawdowns
    tda_drawdown = (np.maximum.accumulate(1 + np.cumsum(tda_returns)) - (1 + np.cumsum(tda_returns))) / np.maximum.accumulate(1 + np.cumsum(tda_returns))
    naive_drawdown = (np.maximum.accumulate(1 + np.cumsum(naive_returns)) - (1 + np.cumsum(naive_returns))) / np.maximum.accumulate(1 + np.cumsum(naive_returns))
    
    print(f"TDA Portfolio Max Drawdown: {max(tda_drawdown)*100:.2f}%")
    print(f"Naive Portfolio Max Drawdown: {max(naive_drawdown)*100:.2f}%")
    
    # Calculate outperformance
    outperformance = tda_cumulative[-1] - naive_cumulative[-1]
    print(f"TDA Outperformance: {outperformance*100:.2f}%")
    
    # Calculate win rate (days TDA outperformed naive)
    win_rate = np.mean(tda_returns > naive_returns)
    print(f"TDA Win Rate: {win_rate*100:.2f}%")
    
    # Portfolio composition
    print("\nTDA Portfolio Allocation:")
    print("-" * 30)
    for name, allocation in sorted(tda_portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {allocation*100:.2f}%")

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate more realistic simulated cryptocurrency data
    num_cryptos = 20  # More cryptos for better diversification
    data_points = 1000  # ~2.7 years of daily data
    
    print(f"Generating simulated data for {num_cryptos} cryptocurrencies over {data_points} days...")
    crypto_data, dates = generate_realistic_crypto_data(
        num_cryptos=num_cryptos, 
        data_points=data_points, 
        include_bear_market=True,
        bear_market_start=750  # Start bear market later in the data
    )
    
    # Initialize TDA Portfolio Manager with improved parameters
    print("Initializing enhanced TDA Portfolio Manager...")
    tda_manager = TDAPortfolioManager(
        window_size=30,
        max_dimension=1,
        embedding_dimension=3,
        num_landscapes=8,  # More landscapes for better pattern detection
        landscape_resolution=150  # Higher resolution
    )
    
    # Create TDA-based portfolio with risk management
    print("Creating TDA-based portfolio with risk management...")
    tda_portfolio, metrics = tda_manager.create_portfolio(
        crypto_data, 
        lookback_period=60,
        risk_threshold=0.5,  # Filter out high-risk assets
        min_allocation_pct=0.02,  # Minimum 2% allocation
        max_allocation_pct=0.20   # Maximum 20% allocation
    )
    
    # Create naive portfolio for comparison
    print("Creating naive 1/N portfolio for comparison...")
    naive_portfolio_allocation = naive_portfolio(crypto_data)
    
    # Plot results with longer evaluation period
    print("Plotting results...")
    plot_results(
        dates, 
        crypto_data, 
        tda_portfolio, 
        naive_portfolio_allocation, 
        metrics, 
        evaluation_period=200,  # Evaluate over a longer period
        lookback_window=60
    )