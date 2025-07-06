import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Set the TF_CPP_MIN_LOG_LEVEL environment variable to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FractionalDifferentiator:
    """
    Class for performing fractional differentiation on time series data.
    Based on the methodology described in the paper.
    """
    def __init__(self, d=0.4, window=10):
        """
        Initialize with the differentiation order and window size.
        
        Parameters:
        -----------
        d : float
            Order of differentiation (between 0 and 1)
        window : int
            Size of the window for calculating weights
        """
        self.d = d
        self.window = window
        self.weights = self._compute_weights()
    
    def _compute_weights(self):
        """Compute weights for the given differentiation order."""
        weights = [1.0]
        for k in range(1, self.window):
            weight = weights[-1] * (self.d - k + 1) / k
            weights.append(-weight)
        return np.array(weights)
    
    def transform(self, series):
        """
        Apply fractional differentiation to a time series.
        
        Parameters:
        -----------
        series : array-like
            The time series to differentiate
            
        Returns:
        --------
        diff_series : array-like
            The differentiated time series
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Initialize the output array with NaNs
        diff_series = np.full_like(series, np.nan, dtype=float)
        
        # For each point in the series
        for i in range(self.window - 1, len(series)):
            # Get the window of data
            window_data = series[i-(self.window-1):i+1]
            
            # Apply the weights
            diff_series[i] = np.sum(self.weights * window_data)
        
        return diff_series

class TripleBarrierLabeler:
    """
    Class for implementing triple barrier labeling.
    Based on the methodology described in the paper.
    """
    def __init__(self, window_size=20, barrier_size=0.01):
        """
        Initialize with window size and barrier size.
        
        Parameters:
        -----------
        window_size : int
            Number of periods to look ahead
        barrier_size : float
            Size of the barrier as a percentage
        """
        self.window_size = window_size
        self.barrier_size = barrier_size
    
    def label(self, prices):
        """
        Apply triple barrier labeling to a price series.
        
        Parameters:
        -----------
        prices : array-like
            The price series to label
            
        Returns:
        --------
        labels : array-like
            The labels (1 for long, -1 for short, 0 for neutral)
        """
        # Convert to numpy array if it's a pandas Series
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        # Initialize the output array with zeros (neutral)
        labels = np.zeros(len(prices))
        
        # For each point in the series (except the last window_size points)
        for i in range(len(prices) - self.window_size):
            entry_price = prices[i]
            # Upper and lower barriers
            upper_barrier = entry_price * (1 + self.barrier_size)
            lower_barrier = entry_price * (1 - self.barrier_size)
            
            # Get the window of future prices
            future_prices = prices[i+1:i+self.window_size+1]
            
            # Check if price hits upper barrier first
            if np.any(future_prices >= upper_barrier):
                upper_idx = np.where(future_prices >= upper_barrier)[0][0]
                # Check if price hits lower barrier before upper barrier
                if np.any(future_prices[:upper_idx] <= lower_barrier):
                    lower_idx = np.where(future_prices[:upper_idx] <= lower_barrier)[0][0]
                    labels[i] = -1  # Short position
                else:
                    labels[i] = 1  # Long position
            # Check if price hits lower barrier
            elif np.any(future_prices <= lower_barrier):
                labels[i] = -1  # Short position
            # If neither barrier is hit, leave as neutral
        
        return labels

class SupervisedAutoencoderMLP:
    """
    Supervised Autoencoder MLP for time series forecasting.
    Based on the methodology described in the paper.
    """
    def __init__(self, input_dim, encoding_dim=5, noise_level=0.05, hidden_layers=1, 
                 dropout_rate=0.2, activation='swish', learning_rate=0.01):
        """
        Initialize the supervised autoencoder MLP.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        encoding_dim : int
            Dimension of the bottleneck layer
        noise_level : float
            Level of noise to add for data augmentation (as a fraction of volatility)
        hidden_layers : int
            Number of hidden layers in encoder and decoder
        dropout_rate : float
            Dropout rate for regularization
        activation : str
            Activation function to use
        learning_rate : float
            Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.noise_level = noise_level
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Build the model
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.classifier = None
        self.build_model()
    
    def build_model(self):
        """Build the encoder, decoder, autoencoder, and classifier."""
        # Encoder
        encoder_input = Input(shape=(self.input_dim,), name='encoder_input')
        
        # Add noise for data augmentation
        encoder_noise = keras.layers.GaussianNoise(self.noise_level)(encoder_input)
        
        # First layer of encoder
        x = Dense(self.input_dim, activation=self.activation)(encoder_noise)
        x = Dropout(self.dropout_rate)(x)
        
        # Additional hidden layers for encoder
        for i in range(self.hidden_layers - 1):
            x = Dense(self.input_dim // (2**(i+1)), activation=self.activation)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Bottleneck layer
        encoded = Dense(self.encoding_dim, activation=self.activation, name='bottleneck')(x)
        
        # Decoder
        # First layer of decoder
        x = Dense(self.input_dim // (2**(self.hidden_layers-1)), activation=self.activation)(encoded)
        x = Dropout(self.dropout_rate)(x)
        
        # Additional hidden layers for decoder
        for i in range(self.hidden_layers - 1):
            x = Dense(self.input_dim // (2**(self.hidden_layers-2-i)), activation=self.activation)(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer of decoder
        decoded = Dense(self.input_dim, name='decoder_output')(x)
        
        # Classifier
        # Concatenate original input and encoded features
        classifier_input = keras.layers.concatenate([encoder_input, encoded], axis=1)
        
        # Hidden layers for classifier
        x = Dense(32, activation=self.activation)(classifier_input)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(16, activation=self.activation)(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer for classifier (3 classes: long, neutral, short)
        classifier_output = Dense(3, activation='softmax', name='classifier_output')(x)
        
        # Create models
        self.encoder = Model(encoder_input, encoded, name='encoder')
        self.decoder = Model(encoded, decoded, name='decoder')
        self.autoencoder = Model(encoder_input, decoded, name='autoencoder')
        self.classifier = Model(encoder_input, classifier_output, name='classifier')
        
        # Compile autoencoder
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Compile classifier
        self.classifier.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
            
        Returns:
        --------
        history : History
            Training history
        """
        # Pre-train the autoencoder
        autoencoder_history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs // 2,
            batch_size=batch_size,
            validation_data=(X_val, X_val) if X_val is not None else None,
            verbose=verbose
        )
        
        # Train the classifier
        classifier_history = self.classifier.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            verbose=verbose
        )
        
        return classifier_history
    
    def predict(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        probabilities : array-like
            Class probabilities
        """
        return self.classifier.predict(X)
    
    def predict_classes(self, X):
        """
        Predict classes.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        classes : array-like
            Predicted classes (0, 1, 2 for short, neutral, long)
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1) - 1  # Convert to -1, 0, 1

def generate_simulated_price_data(n_samples=5000, trend=0.0001, volatility=0.01, mean_reversion=0.001):
    """
    Generate simulated price data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    trend : float
        Trend parameter
    volatility : float
        Volatility parameter
    mean_reversion : float
        Mean reversion parameter
    
    Returns:
    --------
    prices : DataFrame
        Simulated price data with OHLC and volume
    """
    # Initialize prices
    prices = np.zeros(n_samples)
    prices[0] = 100.0
    
    # Generate returns with trend, volatility and mean reversion
    for i in range(1, n_samples):
        # Mean reversion component
        mean_reversion_component = mean_reversion * (prices[0] - prices[i-1])
        # Trend component
        trend_component = trend
        # Random component
        random_component = np.random.normal(0, volatility)
        
        # Combine components
        return_i = trend_component + mean_reversion_component + random_component
        
        # Calculate new price
        prices[i] = prices[i-1] * (1 + return_i)
    
    # Generate OHLC and volume
    df = pd.DataFrame()
    df['close'] = prices
    
    # Generate open, high, low based on close
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = df['close'].iloc[0] * (1 - np.random.uniform(0, 0.01))
    
    df['high'] = df['close'] * (1 + np.random.uniform(0, 0.01, n_samples))
    df['low'] = df['close'] * (1 - np.random.uniform(0, 0.01, n_samples))
    
    # Ensure high is the highest and low is the lowest
    df['high'] = np.maximum(np.maximum(df['high'], df['open']), df['close'])
    df['low'] = np.minimum(np.minimum(df['low'], df['open']), df['close'])
    
    # Generate volume
    df['volume'] = np.random.uniform(1000, 10000, n_samples)
    
    # Create date index
    df.index = pd.date_range(start='2020-01-01', periods=n_samples, freq='5min')
    
    return df

def generate_feature_data(price_data, additional_features=6):
    """
    Generate feature data for the model.
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data with OHLC and volume
    additional_features : int
        Number of additional features to generate (simulating external data like in the paper)
    
    Returns:
    --------
    features : DataFrame
        Feature data
    """
    # Calculate basic features from price data
    features = pd.DataFrame(index=price_data.index)
    
    # Price-based features
    features['log_return'] = np.log(price_data['close'] / price_data['close'].shift(1))
    features['log_return_5'] = np.log(price_data['close'] / price_data['close'].shift(5))
    features['log_return_10'] = np.log(price_data['close'] / price_data['close'].shift(10))
    
    # Volatility features
    features['volatility_5'] = features['log_return'].rolling(5).std()
    features['volatility_10'] = features['log_return'].rolling(10).std()
    features['volatility_20'] = features['log_return'].rolling(20).std()
    
    # Volume features
    features['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(5).mean()
    features['volume_ratio_10'] = price_data['volume'] / price_data['volume'].rolling(10).mean()
    
    # Price range features
    features['price_range'] = (price_data['high'] - price_data['low']) / price_data['low']
    features['price_range_5'] = features['price_range'].rolling(5).mean()
    
    # Simple moving averages
    features['sma_5'] = price_data['close'].rolling(5).mean() / price_data['close']
    features['sma_10'] = price_data['close'].rolling(10).mean() / price_data['close']
    features['sma_20'] = price_data['close'].rolling(20).mean() / price_data['close']
    
    # Generate additional features (simulating the external data mentioned in the paper)
    for i in range(additional_features):
        # Generate a correlated feature with some random noise
        base = np.log(price_data['close'])
        noise = np.random.normal(0, 0.1, len(price_data))
        feature = base + noise
        
        # Apply some mean reversion and trend to make it look like different economic indicators
        feature = feature * (0.9 + 0.2 * np.random.random())  # Random scaling
        feature = feature + np.random.normal(0, 0.5)  # Random shift
        
        # Add trend
        trend = np.linspace(0, np.random.uniform(-1, 1), len(price_data))
        feature = feature + trend
        
        # Normalize and add to features
        feature = (feature - np.mean(feature)) / np.std(feature)
        features[f'external_feature_{i+1}'] = feature
    
    # Drop NaN values (from rolling calculations)
    features = features.dropna()
    
    return features

def apply_fractional_differentiation(features, d=0.4, window=10):
    """
    Apply fractional differentiation to all features.
    
    Parameters:
    -----------
    features : DataFrame
        Feature data
    d : float
        Order of differentiation
    window : int
        Window size for differentiation
    
    Returns:
    --------
    diff_features : DataFrame
        Differentiated features
    """
    diff_features = pd.DataFrame(index=features.index)
    
    # Initialize the differentiator
    differentiator = FractionalDifferentiator(d=d, window=window)
    
    # Apply to each feature
    for col in features.columns:
        diff_series = differentiator.transform(features[col])
        diff_features[col] = diff_series
    
    # Drop NaN values (from the differentiator window)
    diff_features = diff_features.dropna()
    
    return diff_features

def create_input_target_data(features, prices, window_size=10, target_type='triple_barrier'):
    """
    Create input and target data for the model.
    
    Parameters:
    -----------
    features : DataFrame
        Feature data
    prices : DataFrame
        Price data
    window_size : int
        Size of the input window
    target_type : str
        Type of target ('direction' or 'triple_barrier')
    
    Returns:
    --------
    X : array-like
        Input data
    y : array-like
        Target data
    """
    # Align features and prices
    common_index = features.index.intersection(prices.index)
    features = features.loc[common_index]
    prices = prices.loc[common_index]
    
    # Initialize lists for X and y
    X = []
    y = []
    
    # Create sliding windows for input features
    for i in range(len(features) - window_size):
        # Get window of features
        window_features = features.iloc[i:i+window_size].values.flatten()
        X.append(window_features)
        
        # Get target based on target_type
        if target_type == 'direction':
            # Simple direction classification
            future_return = np.log(prices['close'].iloc[i+window_size] / prices['close'].iloc[i+window_size-1])
            if future_return > 0:
                y.append(1)  # Up
            elif future_return < 0:
                y.append(-1)  # Down
            else:
                y.append(0)  # No change
        else:  # triple_barrier
            # Use triple barrier labeling
            labeler = TripleBarrierLabeler(window_size=20, barrier_size=0.01)
            labels = labeler.label(prices['close'].iloc[i:i+window_size+20])
            y.append(labels[0])  # Get the label for the current position
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to 0, 1, 2 for classifier
    y = y + 1
    
    return X, y

def backtest_strategy(model, X_test, prices_test, window_size=10, transaction_cost=0.0005):
    """
    Backtest the trading strategy.
    
    Parameters:
    -----------
    model : SupervisedAutoencoderMLP
        Trained model
    X_test : array-like
        Test input data
    prices_test : DataFrame
        Test price data
    window_size : int
        Size of the input window
    transaction_cost : float
        Transaction cost as a fraction
    
    Returns:
    --------
    results : DataFrame
        Backtest results
    """
    # Get predictions
    predictions = model.predict_classes(X_test)
    
    # Initialize results
    results = pd.DataFrame(index=prices_test.index[window_size:window_size+len(predictions)])
    results['close'] = prices_test['close'].iloc[window_size:window_size+len(predictions)].values
    results['signal'] = predictions
    
    # Calculate position (previous signal)
    results['position'] = results['signal'].shift(1)
    results['position'].iloc[0] = 0  # No position at start
    
    # Calculate returns
    results['return'] = np.log(results['close'] / results['close'].shift(1))
    results['strategy_return'] = results['position'] * results['return']
    
    # Apply transaction costs
    results['trade'] = results['position'].diff().abs()
    results['cost'] = results['trade'] * transaction_cost
    results['strategy_return'] = results['strategy_return'] - results['cost']
    
    # Calculate cumulative returns
    results['cumulative_return'] = np.exp(results['return'].cumsum()) - 1
    results['strategy_cumulative_return'] = np.exp(results['strategy_return'].cumsum()) - 1
    
    return results

def calculate_performance_metrics(results):
    """
    Calculate performance metrics for the strategy.
    
    Parameters:
    -----------
    results : DataFrame
        Backtest results
    
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    # Calculate returns
    strategy_returns = results['strategy_return'].dropna()
    benchmark_returns = results['return'].dropna()
    
    # Calculate annualized return (assuming 252 trading days)
    annualized_factor = 252 * 24 * 60 / 5  # For 5-minute data
    
    strategy_annual_return = np.exp(strategy_returns.mean() * annualized_factor) - 1
    benchmark_annual_return = np.exp(benchmark_returns.mean() * annualized_factor) - 1
    
    # Calculate annualized volatility
    strategy_annual_vol = strategy_returns.std() * np.sqrt(annualized_factor)
    benchmark_annual_vol = benchmark_returns.std() * np.sqrt(annualized_factor)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    strategy_sharpe = strategy_annual_return / strategy_annual_vol
    benchmark_sharpe = benchmark_annual_return / benchmark_annual_vol
    
    # Calculate maximum drawdown
    strategy_cum_returns = (1 + results['strategy_return']).cumprod()
    benchmark_cum_returns = (1 + results['return']).cumprod()
    
    strategy_running_max = strategy_cum_returns.cummax()
    benchmark_running_max = benchmark_cum_returns.cummax()
    
    strategy_drawdown = (strategy_cum_returns / strategy_running_max) - 1
    benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
    
    strategy_max_drawdown = strategy_drawdown.min()
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Calculate information ratio
    tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(annualized_factor)
    information_ratio = (strategy_annual_return - benchmark_annual_return) / tracking_error if tracking_error != 0 else 0
    
    # Calculate IR**
    ir_star_star = (strategy_annual_return**2 * np.sign(strategy_annual_return)) / (strategy_annual_vol * abs(strategy_max_drawdown)) if strategy_max_drawdown != 0 else 0
    
    return {
        'Cumulative Return': results['strategy_cumulative_return'].iloc[-1],
        'Annual Return': strategy_annual_return,
        'Annual Volatility': strategy_annual_vol,
        'Sharpe Ratio': strategy_sharpe,
        'Information Ratio': information_ratio,
        'Max Drawdown': strategy_max_drawdown,
        'IR**': ir_star_star,
        'Benchmark Cumulative Return': results['cumulative_return'].iloc[-1],
        'Benchmark Annual Return': benchmark_annual_return,
        'Benchmark Annual Volatility': benchmark_annual_vol,
        'Benchmark Sharpe Ratio': benchmark_sharpe,
        'Benchmark Max Drawdown': benchmark_max_drawdown
    }

def plot_results(results, metrics, title="Strategy Backtest Results"):
    """
    Plot backtest results.
    
    Parameters:
    -----------
    results : DataFrame
        Backtest results
    metrics : dict
        Performance metrics
    title : str
        Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    plt.plot(results['strategy_cumulative_return'], label='Strategy', color='blue')
    plt.plot(results['cumulative_return'], label='Benchmark', color='gray', alpha=0.7)
    plt.title(f"{title}\nCumulative Returns")
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    # Plot positions and price
    plt.subplot(2, 1, 2)
    
    # Plot price on the right axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(results['close'], color='gray', alpha=0.5, label='Price')
    
    # Plot positions (colored by long/short)
    buy_signals = results[results['position'] == 1].index
    sell_signals = results[results['position'] == -1].index
    
    ax2.scatter(buy_signals, [1.1] * len(buy_signals), color='green', marker='^', label='Long')
    ax2.scatter(sell_signals, [-1.1] * len(sell_signals), color='red', marker='v', label='Short')
    
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Position')
    ax2.set_ylim([-1.5, 1.5])
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Positions and Price')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create a separate figure for metrics
    plt.figure(figsize=(12, 6))
    
    # Display metrics as a table
    cell_text = []
    for key, value in metrics.items():
        if isinstance(value, float):
            cell_text.append([key, f"{value:.4f}"])
        else:
            cell_text.append([key, value])
    
    plt.axis('off')
    plt.table(cellText=cell_text, loc='center', cellLoc='left', colWidths=[0.5, 0.3])
    plt.title("Performance Metrics")
    
    plt.tight_layout()
    plt.show()

def compare_approaches(price_data, feature_data, window_size=10):
    """
    Compare different approaches as described in the paper.
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data
    feature_data : DataFrame
        Feature data
    window_size : int
        Size of the input window
    
    Returns:
    --------
    results : dict
        Results for each approach
    """
    # Split data into train and test
    train_size = int(len(feature_data) * 0.8)
    
    # Apply fractional differentiation to all features
    diff_features = apply_fractional_differentiation(feature_data, d=0.4, window=10)
    
    # Create input and target data for different approaches
    # Approach 1: Regression (not implemented here for simplicity)
    # Approach 2: Simple Direction Classification
    X_direction, y_direction = create_input_target_data(
        diff_features, price_data, window_size=window_size, target_type='direction'
    )
    
    # Approach 3: SAE-MLP with Direction Classification
    X_sae_direction, y_sae_direction = create_input_target_data(
        diff_features, price_data, window_size=window_size, target_type='direction'
    )
    
    # Approach 4: SAE-MLP with Triple Barrier Labeling
    X_sae_tbl, y_sae_tbl = create_input_target_data(
        diff_features, price_data, window_size=window_size, target_type='triple_barrier'
    )
    
    # Split data into train and test
    X_direction_train, X_direction_test = X_direction[:train_size], X_direction[train_size:]
    y_direction_train, y_direction_test = y_direction[:train_size], y_direction[train_size:]
    
    X_sae_direction_train, X_sae_direction_test = X_sae_direction[:train_size], X_sae_direction[train_size:]
    y_sae_direction_train, y_sae_direction_test = y_sae_direction[:train_size], y_sae_direction[train_size:]
    
    X_sae_tbl_train, X_sae_tbl_test = X_sae_tbl[:train_size], X_sae_tbl[train_size:]
    y_sae_tbl_train, y_sae_tbl_test = y_sae_tbl[:train_size], y_sae_tbl[train_size:]
    
    # Normalize data
    scaler = StandardScaler()
    X_direction_train = scaler.fit_transform(X_direction_train)
    X_direction_test = scaler.transform(X_direction_test)
    
    X_sae_direction_train = scaler.fit_transform(X_sae_direction_train)
    X_sae_direction_test = scaler.transform(X_sae_direction_test)
    
    X_sae_tbl_train = scaler.fit_transform(X_sae_tbl_train)
    X_sae_tbl_test = scaler.transform(X_sae_tbl_test)
    
    # Initialize and train models for each approach
    # Approach 2: Simple Direction Classification
    model_direction = SupervisedAutoencoderMLP(
        input_dim=X_direction_train.shape[1],
        encoding_dim=int(X_direction_train.shape[1] * 0.3),  # 30% of input_dim
        noise_level=0.0,  # No noise for Approach 2
        hidden_layers=1,
        activation='tanh',
        learning_rate=0.01
    )
    
    # Approach 3: SAE-MLP with Direction Classification
    model_sae_direction = SupervisedAutoencoderMLP(
        input_dim=X_sae_direction_train.shape[1],
        encoding_dim=int(X_sae_direction_train.shape[1] * 0.3),
        noise_level=0.05,  # 5% noise for Approach 3
        hidden_layers=1,
        activation='swish',
        learning_rate=0.01
    )
    
    # Approach 4: SAE-MLP with Triple Barrier Labeling
    model_sae_tbl = SupervisedAutoencoderMLP(
        input_dim=X_sae_tbl_train.shape[1],
        encoding_dim=int(X_sae_tbl_train.shape[1] * 0.3),
        noise_level=0.05,  # 5% noise for Approach 4
        hidden_layers=1,
        activation='swish',
        learning_rate=0.01
    )
    
    # Train models
    print("Training Approach 2: Simple Direction Classification...")
    model_direction.fit(
        X_direction_train, y_direction_train,
        epochs=20, batch_size=32, verbose=0
    )
    
    print("Training Approach 3: SAE-MLP with Direction Classification...")
    model_sae_direction.fit(
        X_sae_direction_train, y_sae_direction_train,
        epochs=20, batch_size=32, verbose=0
    )
    
    print("Training Approach 4: SAE-MLP with Triple Barrier Labeling...")
    model_sae_tbl.fit(
        X_sae_tbl_train, y_sae_tbl_train,
        epochs=20, batch_size=32, verbose=0
    )
    
    # Get test prices for backtesting
    test_prices = price_data.iloc[train_size+window_size:]
    
    # Backtest each approach
    print("Backtesting Approach 2: Simple Direction Classification...")
    results_direction = backtest_strategy(
        model_direction, X_direction_test, test_prices, window_size=window_size
    )
    
    print("Backtesting Approach 3: SAE-MLP with Direction Classification...")
    results_sae_direction = backtest_strategy(
        model_sae_direction, X_sae_direction_test, test_prices, window_size=window_size
    )
    
    print("Backtesting Approach 4: SAE-MLP with Triple Barrier Labeling...")
    results_sae_tbl = backtest_strategy(
        model_sae_tbl, X_sae_tbl_test, test_prices, window_size=window_size
    )
    
    # Calculate performance metrics for each approach
    metrics_direction = calculate_performance_metrics(results_direction)
    metrics_sae_direction = calculate_performance_metrics(results_sae_direction)
    metrics_sae_tbl = calculate_performance_metrics(results_sae_tbl)
    
    # Plot results
    plot_results(results_direction, metrics_direction, "Approach 2: Simple Direction Classification")
    plot_results(results_sae_direction, metrics_sae_direction, "Approach 3: SAE-MLP with Direction Classification")
    plot_results(results_sae_tbl, metrics_sae_tbl, "Approach 4: SAE-MLP with Triple Barrier Labeling")
    
    # Create and plot a comparison of cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(results_direction['strategy_cumulative_return'], label='Approach 2: Simple Direction')
    plt.plot(results_sae_direction['strategy_cumulative_return'], label='Approach 3: SAE-MLP + Direction')
    plt.plot(results_sae_tbl['strategy_cumulative_return'], label='Approach 4: SAE-MLP + TBL')
    plt.plot(results_direction['cumulative_return'], label='Benchmark', color='gray', alpha=0.7)
    plt.title('Comparison of Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'Approach 2': {
            'results': results_direction,
            'metrics': metrics_direction
        },
        'Approach 3': {
            'results': results_sae_direction,
            'metrics': metrics_sae_direction
        },
        'Approach 4': {
            'results': results_sae_tbl,
            'metrics': metrics_sae_tbl
        }
    }

def create_metrics_comparison_table(results):
    """
    Create a comparison table of metrics for different approaches.
    
    Parameters:
    -----------
    results : dict
        Results for each approach
    
    Returns:
    --------
    comparison_df : DataFrame
        Comparison table
    """
    # Extract metrics for each approach
    metrics_dict = {
        'Approach 2': results['Approach 2']['metrics'],
        'Approach 3': results['Approach 3']['metrics'],
        'Approach 4': results['Approach 4']['metrics']
    }
    
    # Create DataFrame
    comparison_df = pd.DataFrame(metrics_dict)
    
    # Display the table
    print("Metrics Comparison:")
    print(comparison_df.round(4))
    
    # Create a bar chart for key metrics
    key_metrics = ['Annual Return', 'Sharpe Ratio', 'Information Ratio', 'IR**']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(key_metrics):
        plt.subplot(2, 2, i+1)
        values = [metrics_dict[approach][metric] for approach in metrics_dict.keys()]
        plt.bar(metrics_dict.keys(), values, color=['blue', 'green', 'red'])
        plt.title(metric)
        plt.grid(True, axis='y')
        
        # Add value labels
        for j, value in enumerate(values):
            plt.text(j, value, f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def sensitivity_analysis_noise_bottleneck(price_data, feature_data, window_size=10, epochs=20):
    """
    Perform sensitivity analysis on noise level and bottleneck size.
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data
    feature_data : DataFrame
        Feature data
    window_size : int
        Size of the input window
    epochs : int
        Number of training epochs
    """
    # Apply fractional differentiation to all features
    diff_features = apply_fractional_differentiation(feature_data, d=0.4, window=10)
    
    # Create input and target data for triple barrier labeling
    X, y = create_input_target_data(
        diff_features, price_data, window_size=window_size, target_type='triple_barrier'
    )
    
    # Split data into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define parameter ranges
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    bottleneck_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # As a fraction of input_dim
    
    # Initialize results matrix
    results_matrix = np.zeros((len(noise_levels), len(bottleneck_sizes)))
    
    # Test each combination
    for i, noise_level in enumerate(noise_levels):
        for j, bottleneck_size in enumerate(bottleneck_sizes):
            print(f"Testing noise_level={noise_level}, bottleneck_size={bottleneck_size}...")
            
            # Initialize and train model
            model = SupervisedAutoencoderMLP(
                input_dim=X_train.shape[1],
                encoding_dim=int(X_train.shape[1] * bottleneck_size),
                noise_level=noise_level,
                hidden_layers=1,
                activation='swish',
                learning_rate=0.01
            )
            
            model.fit(
                X_train, y_train,
                epochs=epochs, batch_size=32, verbose=0
            )
            
            # Backtest the model
            test_prices = price_data.iloc[train_size+window_size:]
            results = backtest_strategy(
                model, X_test, test_prices, window_size=window_size
            )
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(results)
            
            # Store the Information Ratio
            results_matrix[i, j] = metrics['Information Ratio']
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_matrix, annot=True, fmt=".2f", 
                xticklabels=bottleneck_sizes, yticklabels=noise_levels,
                cmap="YlGnBu")
    plt.title('Information Ratio by Noise Level and Bottleneck Size')
    plt.xlabel('Bottleneck Size (fraction of input_dim)')
    plt.ylabel('Noise Level')
    plt.tight_layout()
    plt.show()

def sensitivity_analysis_tbl_params(price_data, feature_data, window_size=10, epochs=20):
    """
    Perform sensitivity analysis on triple barrier labeling parameters.
    
    Parameters:
    -----------
    price_data : DataFrame
        Price data
    feature_data : DataFrame
        Feature data
    window_size : int
        Size of the input window
    epochs : int
        Number of training epochs
    """
    # Apply fractional differentiation to all features
    diff_features = apply_fractional_differentiation(feature_data, d=0.4, window=10)
    
    # Define parameter ranges
    barrier_sizes = [0.005, 0.01, 0.015, 0.02, 0.025]
    barrier_windows = [10, 15, 20, 25, 30]
    
    # Initialize results matrix
    results_matrix = np.zeros((len(barrier_sizes), len(barrier_windows)))
    
    # Test each combination
    for i, barrier_size in enumerate(barrier_sizes):
        for j, barrier_window in enumerate(barrier_windows):
            print(f"Testing barrier_size={barrier_size}, barrier_window={barrier_window}...")
            
            # Create labeler with current parameters
            labeler = TripleBarrierLabeler(window_size=barrier_window, barrier_size=barrier_size)
            
            # Create input and target data
            X, y = create_input_target_data(
                diff_features, price_data, window_size=window_size, target_type='triple_barrier'
            )
            
            # Split data into train and test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Normalize data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Initialize and train model
            model = SupervisedAutoencoderMLP(
                input_dim=X_train.shape[1],
                encoding_dim=int(X_train.shape[1] * 0.3),
                noise_level=0.05,
                hidden_layers=1,
                activation='swish',
                learning_rate=0.01
            )
            
            model.fit(
                X_train, y_train,
                epochs=epochs, batch_size=32, verbose=0
            )
            
            # Backtest the model
            test_prices = price_data.iloc[train_size+window_size:]
            results = backtest_strategy(
                model, X_test, test_prices, window_size=window_size
            )
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(results)
            
            # Store the Information Ratio
            results_matrix[i, j] = metrics['Information Ratio']
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_matrix, annot=True, fmt=".2f", 
                xticklabels=barrier_windows, yticklabels=barrier_sizes,
                cmap="YlGnBu")
    plt.title('Information Ratio by Barrier Size and Window')
    plt.xlabel('Barrier Window (periods)')
    plt.ylabel('Barrier Size (fraction)')
    plt.tight_layout()
    plt.show()

def run_simulation():
    """Run the complete simulation and analysis."""
    print("Generating simulated price data...")
    price_data = generate_simulated_price_data(n_samples=5000, trend=0.0001, volatility=0.01)
    
    print("Generating feature data...")
    feature_data = generate_feature_data(price_data, additional_features=6)
    
    print("Comparing approaches...")
    results = compare_approaches(price_data, feature_data, window_size=10)
    
    print("Creating metrics comparison table...")
    metrics_df = create_metrics_comparison_table(results)
    
    print("Performing sensitivity analysis on noise level and bottleneck size...")
    sensitivity_analysis_noise_bottleneck(price_data, feature_data, window_size=10, epochs=10)
    
    print("Performing sensitivity analysis on triple barrier labeling parameters...")
    sensitivity_analysis_tbl_params(price_data, feature_data, window_size=10, epochs=10)
    
    return results, metrics_df

if __name__ == "__main__":
    results, metrics_df = run_simulation()