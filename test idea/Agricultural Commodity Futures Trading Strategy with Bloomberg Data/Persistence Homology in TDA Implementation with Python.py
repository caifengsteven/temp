import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import gudhi as gd
from sklearn.manifold import MDS
from scipy.spatial.distance import directed_hausdorff
from itertools import product
from tqdm import tqdm
import seaborn as sns
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

##############################################################################
# TDA Core Functions
##############################################################################

def compute_persistent_homology(point_cloud, max_dimension=2, max_radius=2.0):
    """
    Compute persistent homology using Ripser
    
    Parameters:
    -----------
    point_cloud : numpy.ndarray
        Shape (n_points, d) array of points
    max_dimension : int
        Maximum homology dimension to compute
    max_radius : float
        Maximum radius for the Rips filtration
        
    Returns:
    --------
    diagrams : list of numpy.ndarray
        Persistence diagrams for each dimension
    """
    # Handle potential issues with the input data
    if point_cloud.size == 0 or np.isnan(point_cloud).any():
        # Return empty diagrams
        diagrams = [np.empty((0, 2)) for _ in range(max_dimension + 1)]
        return diagrams
    
    # Compute persistent homology
    try:
        results = ripser(point_cloud, maxdim=max_dimension, thresh=max_radius)
        diagrams = results['dgms']
    except Exception as e:
        print(f"Error in ripser: {e}")
        # Return empty diagrams
        diagrams = [np.empty((0, 2)) for _ in range(max_dimension + 1)]
    
    return diagrams

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualize a point cloud
    
    Parameters:
    -----------
    point_cloud : numpy.ndarray
        Shape (n_points, d) array of points
    title : str
        Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    
    if point_cloud.shape[1] == 2:
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1], alpha=0.7)
        plt.axis('equal')
        plt.title(title)
        plt.grid(True)
        
    elif point_cloud.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
    
    plt.show()

def compute_persistence_landscape(diagram, num_landscapes=5, resolution=1000):
    """
    Compute persistence landscape manually from a persistence diagram
    
    Parameters:
    -----------
    diagram : numpy.ndarray
        Persistence diagram with birth-death pairs
    num_landscapes : int
        Number of landscape functions to compute
    resolution : int
        Number of points to sample for each landscape function
        
    Returns:
    --------
    landscapes : numpy.ndarray
        Array of shape (num_landscapes, resolution, 2) containing landscape functions
    """
    # Filter out points on the diagonal
    if len(diagram) > 0:
        diagram = diagram[~np.isclose(diagram[:, 0], diagram[:, 1])]
    
    if len(diagram) == 0:
        # Return zero landscape if no off-diagonal points
        x_vals = np.linspace(0, 1, resolution)
        landscapes = np.zeros((num_landscapes, resolution, 2))
        landscapes[:, :, 0] = x_vals
        return landscapes
    
    # Define the range for the landscape functions
    min_birth = diagram[:, 0].min()
    max_death = diagram[:, 1].max()
    x_vals = np.linspace(min_birth, max_death, resolution)
    
    # For each birth-death pair, create a piecewise linear function
    y_vals = np.zeros((len(diagram), resolution))
    
    for i, (birth, death) in enumerate(diagram):
        mid = (birth + death) / 2
        y_vals[i] = np.maximum(0, 
                               np.minimum(x_vals - birth, death - x_vals))
    
    # Sort the values for each x
    landscapes = np.zeros((num_landscapes, resolution, 2))
    landscapes[:, :, 0] = x_vals
    
    for i in range(resolution):
        sorted_vals = np.sort(y_vals[:, i])[::-1]  # Sort in descending order
        for k in range(num_landscapes):
            if k < len(sorted_vals):
                landscapes[k, i, 1] = sorted_vals[k]
    
    return landscapes

##############################################################################
# Time Series Analysis Functions
##############################################################################

def create_time_delay_embedding(time_series, embedding_dimension=3, delay=1):
    """
    Create a time delay embedding of a time series
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        1D array containing the time series
    embedding_dimension : int
        Dimension of the embedding
    delay : int
        Delay between consecutive dimensions
        
    Returns:
    --------
    embedding : numpy.ndarray
        Shape (n_points, embedding_dimension) array containing the embedding
    """
    if len(time_series) < (embedding_dimension - 1) * delay + 1:
        # Not enough data points for this embedding
        return np.empty((0, embedding_dimension))
    
    n = len(time_series) - (embedding_dimension - 1) * delay
    embedding = np.zeros((n, embedding_dimension))
    
    for i in range(n):
        for j in range(embedding_dimension):
            embedding[i, j] = time_series[i + j * delay]
    
    return embedding

def calculate_sliding_window_homology(time_series, window_size=50, step_size=1, embedding_dimension=3, delay=1):
    """
    Calculate persistent homology for sliding windows of a time series
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        1D array containing the time series
    window_size : int
        Size of the sliding window
    step_size : int
        Step size between consecutive windows
    embedding_dimension : int
        Dimension of the time delay embedding
    delay : int
        Delay between consecutive dimensions in the embedding
        
    Returns:
    --------
    diagrams : list
        List of persistence diagrams for each window
    window_indices : list
        List of indices corresponding to the end of each window
    """
    if len(time_series) < window_size:
        print(f"Warning: time_series length {len(time_series)} is less than window_size {window_size}")
        return [], []
    
    diagrams = []
    window_indices = []
    
    for i in range(0, len(time_series) - window_size + 1, step_size):
        # Extract window
        window = time_series[i:i+window_size]
        
        if np.isnan(window).any():
            continue
        
        # Create time delay embedding
        embedding = create_time_delay_embedding(window, embedding_dimension, delay)
        
        if embedding.size == 0:
            continue
        
        # Normalize the embedding to reduce numerical issues
        embedding_std = np.std(embedding, axis=0)
        
        # Check for zero standard deviations
        if np.any(embedding_std < 1e-10):
            # Skip this window if any dimension has zero variance
            continue
            
        embedding = (embedding - np.mean(embedding, axis=0)) / embedding_std
        
        # Compute persistent homology
        diagram = compute_persistent_homology(embedding, max_dimension=1)
        
        diagrams.append(diagram)
        window_indices.append(i + window_size - 1)  # Index at the end of the window
    
    return diagrams, window_indices

def calculate_topological_features(diagrams):
    """
    Calculate topological features from persistence diagrams
    
    Parameters:
    -----------
    diagrams : list
        List of persistence diagrams
        
    Returns:
    --------
    features : pandas.DataFrame
        DataFrame containing topological features
    """
    features = []
    
    for diagram in diagrams:
        # Extract features from diagram
        feature = {}
        
        for dim in range(len(diagram)):
            # Calculate persistence (lifetime)
            persistence = diagram[dim][:, 1] - diagram[dim][:, 0]
            
            if len(persistence) > 0 and np.sum(persistence) > 0:
                # Total persistence
                feature[f'TotalPersistence_dim{dim}'] = np.sum(persistence)
                
                # Maximum persistence
                feature[f'MaxPersistence_dim{dim}'] = np.max(persistence)
                
                # Number of features
                feature[f'NumFeatures_dim{dim}'] = len(persistence)
                
                # Persistence entropy
                p = persistence / np.sum(persistence)
                feature[f'Entropy_dim{dim}'] = -np.sum(p * np.log2(p + 1e-10))
                
                # Average persistence
                feature[f'AvgPersistence_dim{dim}'] = np.mean(persistence)
                
                # Standard deviation of persistence
                if len(persistence) > 1:
                    feature[f'StdPersistence_dim{dim}'] = np.std(persistence)
                else:
                    feature[f'StdPersistence_dim{dim}'] = 0
                
                # Persistence ratio (max/total)
                feature[f'PersistenceRatio_dim{dim}'] = np.max(persistence) / np.sum(persistence)
            else:
                feature[f'TotalPersistence_dim{dim}'] = 0
                feature[f'MaxPersistence_dim{dim}'] = 0
                feature[f'NumFeatures_dim{dim}'] = 0
                feature[f'Entropy_dim{dim}'] = 0
                feature[f'AvgPersistence_dim{dim}'] = 0
                feature[f'StdPersistence_dim{dim}'] = 0
                feature[f'PersistenceRatio_dim{dim}'] = 0
        
        features.append(feature)
    
    return pd.DataFrame(features)

##############################################################################
# Trading Strategy Functions
##############################################################################

def generate_trading_signals(topological_features, window_indices, price_data_dates):
    """
    Generate trading signals based on topological features
    
    Parameters:
    -----------
    topological_features : pandas.DataFrame
        DataFrame containing topological features
    window_indices : list
        List of indices corresponding to the end of each window
    price_data_dates : pandas.DatetimeIndex
        Dates corresponding to price data
        
    Returns:
    --------
    signals : pandas.Series
        Series with trading signals (1=buy, -1=sell, 0=hold)
    """
    if len(topological_features) == 0:
        return pd.Series(index=price_data_dates, data=0)
    
    # Map window indices to dates
    dates = [price_data_dates[idx] if idx < len(price_data_dates) else price_data_dates[-1] for idx in window_indices]
    
    # Initialize signals
    signals = pd.Series(index=dates, data=0)
    
    # Parameter tuning based on analysis of topological features
    # Handle potential issues with empty or constant features
    if ('MaxPersistence_dim1' in topological_features.columns and
        topological_features['MaxPersistence_dim1'].nunique() > 1):
        max_persistence_threshold = np.percentile(
            topological_features['MaxPersistence_dim1'].replace([np.inf, -np.inf], np.nan).dropna(), 
            75
        )
    else:
        max_persistence_threshold = 0
    
    if ('Entropy_dim1' in topological_features.columns and
        topological_features['Entropy_dim1'].nunique() > 1):
        entropy_threshold = np.percentile(
            topological_features['Entropy_dim1'].replace([np.inf, -np.inf], np.nan).dropna(), 
            50
        )
    else:
        entropy_threshold = 0
    
    # Generate signals only if we have at least 2 data points
    if len(topological_features) > 1:
        for i in range(1, len(topological_features)):
            idx = dates[i]
            
            # Check for buy signals - strong topological structure formation
            if ('MaxPersistence_dim1' in topological_features.columns and
                topological_features.iloc[i]['MaxPersistence_dim1'] > max_persistence_threshold and
                topological_features.iloc[i-1]['MaxPersistence_dim1'] <= max_persistence_threshold):
                signals.loc[idx] = 1  # Buy signal
                
            # Check for sell signals - breakdown of topological structure
            elif ('MaxPersistence_dim1' in topological_features.columns and
                  topological_features.iloc[i]['MaxPersistence_dim1'] < max_persistence_threshold and
                  topological_features.iloc[i-1]['MaxPersistence_dim1'] >= max_persistence_threshold):
                signals.loc[idx] = -1  # Sell signal
                
            # Alternative signal based on entropy change
            elif ('Entropy_dim1' in topological_features.columns and
                  topological_features.iloc[i]['Entropy_dim1'] > entropy_threshold and
                  topological_features.iloc[i-1]['Entropy_dim1'] <= entropy_threshold):
                signals.loc[idx] = -1  # Sell signal on increase in entropy (more randomness)
                
            elif ('Entropy_dim1' in topological_features.columns and
                  topological_features.iloc[i]['Entropy_dim1'] < entropy_threshold and
                  topological_features.iloc[i-1]['Entropy_dim1'] >= entropy_threshold):
                signals.loc[idx] = 1  # Buy signal on decrease in entropy (more structure)
    
    return signals

def backtest_strategy(price_data, signals, initial_capital=10000, transaction_cost=0.001):
    """
    Backtest a trading strategy
    
    Parameters:
    -----------
    price_data : pandas.Series
        Series containing price data
    signals : pandas.Series
        Series with trading signals
    initial_capital : float
        Initial capital
    transaction_cost : float
        Transaction cost as a fraction of trade value
        
    Returns:
    --------
    portfolio : pandas.DataFrame
        DataFrame containing portfolio value, positions, and returns
    """
    # Make sure signals dates exist in price_data
    common_dates = signals.index.intersection(price_data.index)
    
    if len(common_dates) == 0:
        print("Warning: No common dates between signals and price data")
        # Return empty portfolio
        return pd.DataFrame(index=signals.index)
    
    # Filter signals to only include dates that exist in price_data
    signals = signals.loc[common_dates]
    
    # Initialize portfolio
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['Price'] = price_data.loc[signals.index]
    portfolio['Signal'] = signals
    portfolio['Position'] = 0
    portfolio['Cash'] = initial_capital
    portfolio['Holdings'] = 0
    portfolio['Portfolio'] = initial_capital
    portfolio['Returns'] = 0
    
    # Current position and cash
    position = 0
    cash = initial_capital
    
    # Process signals
    for i in range(len(signals)):
        idx = signals.index[i]
        
        # Update position based on signal
        if signals.iloc[i] == 1 and position <= 0:  # Buy signal
            # Close short position if any
            if position < 0:
                cash += position * portfolio.loc[idx, 'Price'] * (1 - transaction_cost)
                position = 0
            
            # Buy with all available cash
            position = cash / portfolio.loc[idx, 'Price'] * (1 - transaction_cost)
            cash = 0
            
        elif signals.iloc[i] == -1 and position >= 0:  # Sell signal
            # Close long position if any
            if position > 0:
                cash += position * portfolio.loc[idx, 'Price'] * (1 - transaction_cost)
                position = 0
            
            # Short with all available cash
            position = -cash / portfolio.loc[idx, 'Price'] * (1 - transaction_cost)
            cash = 2 * initial_capital  # Reserve cash for covering short
            
        # Update portfolio
        portfolio.loc[idx, 'Position'] = position
        portfolio.loc[idx, 'Cash'] = cash
        portfolio.loc[idx, 'Holdings'] = position * portfolio.loc[idx, 'Price']
        portfolio.loc[idx, 'Portfolio'] = cash + position * portfolio.loc[idx, 'Price']
        
        # Calculate returns
        if i > 0:
            prev_idx = signals.index[i-1]
            portfolio.loc[idx, 'Returns'] = (portfolio.loc[idx, 'Portfolio'] / portfolio.loc[prev_idx, 'Portfolio']) - 1
    
    return portfolio

def calculate_performance_metrics(portfolio):
    """
    Calculate performance metrics for a portfolio
    
    Parameters:
    -----------
    portfolio : pandas.DataFrame
        DataFrame containing portfolio value, positions, and returns
        
    Returns:
    --------
    metrics : dict
        Dictionary containing performance metrics
    """
    # Check if portfolio is empty or doesn't have required columns
    if len(portfolio) == 0 or 'Portfolio' not in portfolio.columns or 'Returns' not in portfolio.columns:
        return {
            'Total Return': 0,
            'Annualized Return': 0,
            'Volatility': 0,
            'Annualized Volatility': 0,
            'Sharpe Ratio': 0,
            'Maximum Drawdown': 0,
            'Calmar Ratio': 0,
            'Win Rate': 0,
            'Average Trade Return': 0,
            'Number of Trades': 0
        }
    
    # Extract portfolio value and returns
    value = portfolio['Portfolio']
    returns = portfolio['Returns']
    
    # Calculate metrics
    metrics = {}
    
    # Total return
    metrics['Total Return'] = (value.iloc[-1] / value.iloc[0]) - 1
    
    # Annualized return (assuming 252 trading days per year)
    metrics['Annualized Return'] = ((value.iloc[-1] / value.iloc[0]) ** (252 / len(value))) - 1
    
    # Volatility
    metrics['Volatility'] = returns.std()
    
    # Annualized volatility
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    if metrics['Annualized Volatility'] > 0:
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility']
    else:
        metrics['Sharpe Ratio'] = 0
    
    # Maximum drawdown
    drawdowns = 1 - value / value.cummax()
    metrics['Maximum Drawdown'] = drawdowns.max()
    
    # Calmar ratio
    if metrics['Maximum Drawdown'] > 0:
        metrics['Calmar Ratio'] = metrics['Annualized Return'] / metrics['Maximum Drawdown']
    else:
        metrics['Calmar Ratio'] = 0
    
    # Win rate
    if 'Signal' in portfolio.columns:
        trades = portfolio['Signal'].replace(0, np.nan).dropna()
        if len(trades) > 0:
            trade_returns = []
            entry_price = None
            entry_signal = None
            
            for i in range(len(trades)):
                if entry_price is None:
                    entry_price = portfolio['Price'].iloc[i]
                    entry_signal = trades.iloc[i]
                else:
                    exit_price = portfolio['Price'].iloc[i]
                    if entry_signal == 1:  # Long position
                        trade_return = (exit_price / entry_price) - 1
                    else:  # Short position
                        trade_return = 1 - (exit_price / entry_price)
                    
                    trade_returns.append(trade_return)
                    entry_price = exit_price
                    entry_signal = trades.iloc[i]
            
            metrics['Win Rate'] = sum(r > 0 for r in trade_returns) / len(trade_returns) if trade_returns else 0
            metrics['Average Trade Return'] = np.mean(trade_returns) if trade_returns else 0
            metrics['Number of Trades'] = len(trades)
        else:
            metrics['Win Rate'] = 0
            metrics['Average Trade Return'] = 0
            metrics['Number of Trades'] = 0
    else:
        metrics['Win Rate'] = 0
        metrics['Average Trade Return'] = 0
        metrics['Number of Trades'] = 0
    
    return metrics

##############################################################################
# Visualization Functions
##############################################################################

def visualize_topological_features(price_data, topological_features, signals):
    """
    Visualize topological features and trading signals
    
    Parameters:
    -----------
    price_data : pandas.Series
        Series containing price data
    topological_features : pandas.DataFrame
        DataFrame containing topological features
    signals : pandas.Series
        Series with trading signals
    """
    # Check if we have data to visualize
    if len(topological_features) == 0 or len(signals) == 0:
        print("Warning: Not enough data to visualize")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Get common dates between all datasets
    common_dates = signals.index.intersection(price_data.index)
    common_dates = common_dates.intersection(topological_features.index)
    
    if len(common_dates) == 0:
        print("Warning: No common dates to visualize")
        return
    
    # Plot price with signals
    axes[0].plot(price_data, label='Price', alpha=0.7, color='blue')
    
    # Add buy/sell markers
    buy_indices = signals[signals == 1].index
    buy_indices = buy_indices.intersection(price_data.index)
    
    sell_indices = signals[signals == -1].index
    sell_indices = sell_indices.intersection(price_data.index)
    
    if len(buy_indices) > 0:
        axes[0].scatter(buy_indices, price_data.loc[buy_indices], marker='^', color='green', s=100, label='Buy Signal')
        
    if len(sell_indices) > 0:
        axes[0].scatter(sell_indices, price_data.loc[sell_indices], marker='v', color='red', s=100, label='Sell Signal')
    
    axes[0].set_title('Price with Trading Signals')
    axes[0].legend()
    axes[0].grid(True)
    
    # Check which features are available
    available_features = topological_features.columns
    
    # Plot topological features - Max persistence
    if 'MaxPersistence_dim0' in available_features:
        axes[1].plot(topological_features.index, topological_features['MaxPersistence_dim0'], 
                    label='Max Persistence (Dim 0)', color='blue')
        
    if 'MaxPersistence_dim1' in available_features:
        axes[1].plot(topological_features.index, topological_features['MaxPersistence_dim1'], 
                    label='Max Persistence (Dim 1)', color='red')
        
    axes[1].set_title('Maximum Persistence')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot topological features - Entropy
    if 'Entropy_dim0' in available_features:
        axes[2].plot(topological_features.index, topological_features['Entropy_dim0'], 
                    label='Entropy (Dim 0)', color='blue')
        
    if 'Entropy_dim1' in available_features:
        axes[2].plot(topological_features.index, topological_features['Entropy_dim1'], 
                    label='Entropy (Dim 1)', color='red')
        
    axes[2].set_title('Persistence Entropy')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot topological features - Number of features
    if 'NumFeatures_dim0' in available_features:
        axes[3].plot(topological_features.index, topological_features['NumFeatures_dim0'], 
                    label='Num Features (Dim 0)', color='blue')
        
    if 'NumFeatures_dim1' in available_features:
        axes[3].plot(topological_features.index, topological_features['NumFeatures_dim1'], 
                    label='Num Features (Dim 1)', color='red')
        
    axes[3].set_title('Number of Topological Features')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_portfolio_performance(portfolio, price_data):
    """
    Visualize portfolio performance
    
    Parameters:
    -----------
    portfolio : pandas.DataFrame
        DataFrame containing portfolio value, positions, and returns
    price_data : pandas.Series
        Series containing price data
    """
    # Check if portfolio is empty
    if len(portfolio) == 0 or 'Portfolio' not in portfolio.columns:
        print("Warning: Not enough portfolio data to visualize")
        return
    
    # Get common dates
    common_dates = portfolio.index.intersection(price_data.index)
    
    if len(common_dates) == 0:
        print("Warning: No common dates between portfolio and price data")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot price
    axes[0].plot(price_data, label='Price', alpha=0.7, color='blue')
    axes[0].set_title('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot portfolio value
    axes[1].plot(portfolio['Portfolio'], label='Portfolio Value', color='green')
    axes[1].set_title('Portfolio Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot position
    if 'Position' in portfolio.columns:
        axes[2].fill_between(portfolio.index, portfolio['Position'], 0, 
                            where=portfolio['Position'] > 0, color='green', alpha=0.3, label='Long')
        axes[2].fill_between(portfolio.index, portfolio['Position'], 0, 
                            where=portfolio['Position'] < 0, color='red', alpha=0.3, label='Short')
    
    axes[2].set_title('Position')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown if we have portfolio values
    if 'Portfolio' in portfolio.columns and len(portfolio) > 1:
        plt.figure(figsize=(14, 6))
        drawdown = 1 - portfolio['Portfolio'] / portfolio['Portfolio'].cummax()
        plt.fill_between(portfolio.index, drawdown, 0, color='red', alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

##############################################################################
# Simulation Functions
##############################################################################

def simulate_agricultural_futures(n_days=1000, n_commodities=5, start_date='2010-01-01'):
    """
    Simulate agricultural futures data
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    n_commodities : int
        Number of commodities to simulate
    start_date : str
        Start date for the simulation
        
    Returns:
    --------
    futures_data : dict
        Dictionary containing simulated futures data
    """
    # Make sure n_days and n_commodities are valid
    n_days = max(100, n_days)
    n_commodities = min(5, max(1, n_commodities))
    
    # Commodities
    commodities = ['Wheat', 'Corn', 'Soybeans', 'Cotton', 'Sugar'][:n_commodities]
    
    # Create a base price series with seasonality
    base_dates = pd.date_range(start=start_date, periods=n_days, freq='B')  # Business days
    base_price = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_days))  # Seasonal component
    
    # Add trend
    trend = np.linspace(0, 40, n_days)
    base_price += trend
    
    # Add noise
    base_price += np.random.normal(0, 5, n_days)
    
    futures_data = {}
    
    for i, commodity in enumerate(commodities):
        # Create prices with some correlation to base price but different characteristics
        price = base_price.copy()
        
        # Add commodity-specific trend
        commodity_trend = np.linspace(0, np.random.uniform(-20, 20), n_days)
        price += commodity_trend
        
        # Add commodity-specific seasonality
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(5, 15)
        price += amplitude * np.sin(np.linspace(phase, phase + 6*np.pi, n_days))
        
        # Add commodity-specific noise
        price += np.random.normal(0, np.random.uniform(2, 8), n_days)
        
        # Ensure prices are positive
        price = np.maximum(price, 1)
        
        # Create DataFrame with OHLC data
        df = pd.DataFrame(index=base_dates)
        df['Open'] = price
        
        # Add some intraday variation
        high_factor = np.random.uniform(1.01, 1.03, n_days)
        low_factor = np.random.uniform(0.97, 0.99, n_days)
        
        df['High'] = df['Open'] * high_factor
        df['Low'] = df['Open'] * low_factor
        df['Close'] = df['Open'] + np.random.normal(0, 1, n_days)
        
        # Ensure High >= Open >= Close >= Low
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        df['Close'] = np.maximum(np.minimum(df['Close'], df['High']), df['Low'])
        
        # Add volume
        df['Volume'] = np.random.lognormal(10, 1, n_days)
        
        futures_data[commodity] = df
    
    return futures_data

def add_market_regimes(futures_data, n_regimes=3, regime_length=60):
    """
    Add market regimes to simulated futures data
    
    Parameters:
    -----------
    futures_data : dict
        Dictionary containing simulated futures data
    n_regimes : int
        Number of different regimes to simulate
    regime_length : int
        Average length of each regime in days
        
    Returns:
    --------
    futures_data : dict
        Dictionary containing futures data with regime information
    regime_info : pandas.DataFrame
        DataFrame containing regime information
    """
    # Get dates
    dates = list(futures_data.values())[0].index
    
    # Ensure valid parameters
    n_regimes = min(3, max(1, n_regimes))
    regime_length = max(10, min(regime_length, len(dates) // 3))
    
    # Initialize regime information
    regime_info = pd.DataFrame(index=dates)
    regime_info['Regime'] = 0
    
    # Generate regime transitions
    current_regime = 0
    current_length = 0
    
    for i in range(len(dates)):
        # Check if it's time to change regime
        if current_length >= regime_length:
            # Switch to a new regime
            current_regime = (current_regime + 1) % n_regimes
            current_length = 0
        
        # Set regime
        regime_info.loc[dates[i], 'Regime'] = current_regime
        current_length += 1
    
    # Define factors for high and low prices in each regime
    high_factors = {
        0: np.random.uniform(1.01, 1.03, len(dates)),  # Trending up
        1: np.random.uniform(1.005, 1.015, len(dates)),  # Trending down
        2: np.random.uniform(1.02, 1.05, len(dates))   # Volatile
    }
    
    low_factors = {
        0: np.random.uniform(0.97, 0.99, len(dates)),  # Trending up
        1: np.random.uniform(0.95, 0.98, len(dates)),  # Trending down
        2: np.random.uniform(0.92, 0.97, len(dates))   # Volatile
    }
    
    # Modify price behavior based on regime
    for commodity, data in futures_data.items():
        # Add regime-specific behavior
        for regime in range(n_regimes):
            regime_dates = regime_info[regime_info['Regime'] == regime].index
            
            if regime == 0:  # Trending up
                data.loc[regime_dates, 'Close'] *= 1.0005  # Slight uptrend
                
            elif regime == 1:  # Trending down
                data.loc[regime_dates, 'Close'] *= 0.9995  # Slight downtrend
                
            elif regime == 2:  # Volatile
                data.loc[regime_dates, 'Close'] *= (1 + np.random.normal(0, 0.01, len(regime_dates)))
                
            # Recalculate High and Low to maintain consistency
            data.loc[regime_dates, 'High'] = data.loc[regime_dates, 'Close'] * high_factors[regime][:len(regime_dates)]
            data.loc[regime_dates, 'Low'] = data.loc[regime_dates, 'Close'] * low_factors[regime][:len(regime_dates)]
    
    # Add regime description
    regime_descriptions = {0: 'Trending Up', 1: 'Trending Down', 2: 'Volatile'}
    regime_info['Description'] = regime_info['Regime'].map(regime_descriptions)
    
    return futures_data, regime_info

##############################################################################
# Main Functions
##############################################################################

def analyze_commodity(price_data, window_size=50, embedding_dimension=3, delay=1):
    """
    Analyze a commodity using TDA
    
    Parameters:
    -----------
    price_data : pandas.Series
        Series containing price data
    window_size : int
        Size of the sliding window
    embedding_dimension : int
        Dimension of the time delay embedding
    delay : int
        Delay between consecutive dimensions in the embedding
        
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    print(f"  Window size: {window_size}, Embedding dimension: {embedding_dimension}, Delay: {delay}")
    
    # Handle empty or invalid price data
    if len(price_data) < window_size + embedding_dimension * delay:
        print(f"  Warning: Not enough data points for analysis. Need at least {window_size + embedding_dimension * delay}, got {len(price_data)}")
        return {
            'diagrams': [],
            'window_indices': [],
            'features': pd.DataFrame(),
            'signals': pd.Series(index=price_data.index, data=0),
            'portfolio': pd.DataFrame(),
            'metrics': {}
        }
    
    # Calculate sliding window homology
    print("  Calculating sliding window homology...")
    diagrams, window_indices = calculate_sliding_window_homology(
        price_data.values, window_size, step_size=1, 
        embedding_dimension=embedding_dimension, delay=delay
    )
    
    if len(diagrams) == 0 or len(window_indices) == 0:
        print("  Warning: No valid persistence diagrams generated")
        return {
            'diagrams': [],
            'window_indices': [],
            'features': pd.DataFrame(),
            'signals': pd.Series(index=price_data.index, data=0),
            'portfolio': pd.DataFrame(),
            'metrics': {}
        }
    
    # Calculate topological features
    print("  Calculating topological features...")
    features = calculate_topological_features(diagrams)
    
    # Map window indices to dates and set as index for features
    dates = [price_data.index[idx] if idx < len(price_data) else price_data.index[-1] for idx in window_indices]
    features.index = dates
    
    # Generate trading signals
    print("  Generating trading signals...")
    signals = generate_trading_signals(features, window_indices, price_data.index)
    
    # Backtest strategy
    print("  Backtesting strategy...")
    portfolio = backtest_strategy(price_data, signals)
    
    # Calculate performance metrics
    print("  Calculating performance metrics...")
    metrics = calculate_performance_metrics(portfolio)
    
    return {
        'diagrams': diagrams,
        'window_indices': window_indices,
        'features': features,
        'signals': signals,
        'portfolio': portfolio,
        'metrics': metrics
    }

def run_tda_trading_simulation(n_days=1000, n_commodities=3, window_size=50):
    """
    Run a TDA-based trading simulation on agricultural futures data
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    n_commodities : int
        Number of commodities to simulate
    window_size : int
        Size of the sliding window for TDA
    """
    # Make sure parameters are valid
    n_days = max(100, n_days)
    n_commodities = min(5, max(1, n_commodities))
    window_size = min(n_days // 10, max(10, window_size))
    
    print(f"Simulating agricultural futures data with {n_days} days and {n_commodities} commodities...")
    futures_data = simulate_agricultural_futures(n_days, n_commodities)
    
    # Add market regimes
    print("Adding market regimes...")
    futures_data, regime_info = add_market_regimes(futures_data)
    
    results = {}
    all_metrics = {}
    
    print("Analyzing commodities...")
    for commodity, data in futures_data.items():
        print(f"Analyzing {commodity}...")
        
        # Extract close prices
        close_prices = data['Close']
        
        # Analyze commodity
        results[commodity] = analyze_commodity(close_prices, window_size)
        
        # Store metrics
        all_metrics[commodity] = results[commodity]['metrics']
        
        # Visualize results only if we have valid data
        if len(results[commodity]['features']) > 0 and len(results[commodity]['signals']) > 0:
            visualize_topological_features(
                close_prices, 
                results[commodity]['features'],
                results[commodity]['signals']
            )
            
            visualize_portfolio_performance(
                results[commodity]['portfolio'],
                close_prices
            )
    
    # Compare performance across commodities
    print("\nPerformance Comparison:")
    performance_df = pd.DataFrame(all_metrics).T
    
    # Handle potentially empty DataFrames
    if not performance_df.empty:
        print(performance_df)
        
        # Plot comparison of portfolio values
        plt.figure(figsize=(12, 6))
        
        for commodity in futures_data.keys():
            if (len(results[commodity]['portfolio']) > 0 and 
                'Portfolio' in results[commodity]['portfolio'].columns):
                portfolio = results[commodity]['portfolio']
                plt.plot(
                    portfolio.index,
                    portfolio['Portfolio'] / portfolio['Portfolio'].iloc[0],
                    label=commodity
                )
        
        plt.title('Portfolio Performance Comparison')
        plt.ylabel('Normalized Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Compare with regime information
        plt.figure(figsize=(14, 8))
        
        # Plot regime information
        ax1 = plt.subplot(211)
        for regime in sorted(regime_info['Regime'].unique()):
            regime_dates = regime_info[regime_info['Regime'] == regime].index
            if len(regime_dates) > 0:
                description = regime_info.loc[regime_dates[0], 'Description']
                ax1.fill_between(
                    regime_dates,
                    0, 1,
                    alpha=0.3,
                    label=f'Regime {regime} ({description})'
                )
        
        ax1.set_title('Market Regimes')
        ax1.legend()
        ax1.get_yaxis().set_visible(False)
        
        # Plot portfolio performance
        ax2 = plt.subplot(212, sharex=ax1)
        
        for commodity in futures_data.keys():
            if (len(results[commodity]['portfolio']) > 0 and 
                'Portfolio' in results[commodity]['portfolio'].columns):
                portfolio = results[commodity]['portfolio']
                ax2.plot(
                    portfolio.index,
                    portfolio['Portfolio'] / portfolio['Portfolio'].iloc[0],
                    label=commodity
                )
        
        ax2.set_title('Portfolio Performance by Regime')
        ax2.set_ylabel('Normalized Portfolio Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze performance by regime
        regime_performance = {}
        
        for regime in sorted(regime_info['Regime'].unique()):
            regime_dates = regime_info[regime_info['Regime'] == regime].index
            if len(regime_dates) > 0:
                description = regime_info.loc[regime_dates[0], 'Description']
                
                regime_performance[description] = {}
                
                for commodity in futures_data.keys():
                    portfolio = results[commodity]['portfolio']
                    
                    if len(portfolio) > 0 and 'Portfolio' in portfolio.columns:
                        # Filter portfolio by regime dates
                        common_dates = portfolio.index.intersection(regime_dates)
                        
                        if len(common_dates) > 1:
                            regime_portfolio = portfolio.loc[common_dates]
                            
                            # Calculate regime-specific metrics
                            regime_return = (regime_portfolio['Portfolio'].iloc[-1] / regime_portfolio['Portfolio'].iloc[0]) - 1
                            if 'Returns' in regime_portfolio.columns:
                                regime_volatility = regime_portfolio['Returns'].std()
                            else:
                                regime_volatility = 0
                            
                            regime_performance[description][commodity] = {
                                'Return': regime_return,
                                'Volatility': regime_volatility,
                                'Sharpe': regime_return / regime_volatility if regime_volatility > 0 else 0
                            }
        
        # Print regime-specific performance
        print("\nPerformance by Regime:")
        for regime, commodities in regime_performance.items():
            if commodities:  # Only print if we have data
                print(f"\nRegime: {regime}")
                regime_df = pd.DataFrame(commodities).T
                print(regime_df)
    
    return results, futures_data, regime_info

##############################################################################
# Main Execution
##############################################################################

if __name__ == "__main__":
    try:
        # Run the TDA trading simulation with smaller values to ensure it runs quickly
        results, futures_data, regime_info = run_tda_trading_simulation(
            n_days=200,         # Number of days to simulate
            n_commodities=2,    # Number of commodities to simulate
            window_size=20      # Size of sliding window for TDA
        )
        
        print("Simulation complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()