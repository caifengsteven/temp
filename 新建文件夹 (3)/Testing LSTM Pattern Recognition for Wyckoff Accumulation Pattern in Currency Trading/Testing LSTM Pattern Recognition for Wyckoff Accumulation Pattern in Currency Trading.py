import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# PART 1: DATA GENERATION

def generate_trading_range_pattern(validity=True, noise_level=0.05):
    """
    Generate a synthetic trading range pattern based on Wyckoff principles.
    
    Parameters:
    - validity: Whether to generate a valid (True) or invalid (False) pattern
    - noise_level: Amount of noise to add to the pattern
    
    Returns:
    - Pattern array with label as first element
    """
    if validity:
        # Create a valid trading range pattern
        # p1: high point, p2: low point, p3: lower high, p4: higher low
        p1 = random.uniform(80, 100)
        p2 = random.uniform(20, 40)
        p3 = random.uniform(p2 + (p1-p2)*0.4, p2 + (p1-p2)*0.7)  # Lower high
        p4 = random.uniform(p2 + (p1-p2)*0.2, p2 + (p1-p2)*0.4)  # Higher low
        
        # Create basic pattern
        pattern = [p1, p2, p3, p4]
        
        # Add filler points to create a more realistic pattern
        full_pattern = add_filler_points(pattern, num_between=3, noise_level=noise_level)
        
        # Add label
        full_pattern.insert(0, 1)  # 1 for valid pattern
        
    else:
        # Generate random points for invalid pattern
        pattern = [
            random.uniform(20, 100),
            random.uniform(20, 100),
            random.uniform(20, 100),
            random.uniform(20, 100)
        ]
        
        # If by chance we created a valid pattern, make sure it's invalid
        if (pattern[0] > pattern[1] and 
            pattern[1] < pattern[2] and 
            pattern[2] < pattern[0] and 
            pattern[3] < pattern[2] and 
            pattern[3] > pattern[1]):
            # Swap values to make it invalid
            pattern[2], pattern[3] = pattern[3], pattern[2]
        
        # Add filler points
        full_pattern = add_filler_points(pattern, num_between=3, noise_level=noise_level)
        
        # Add label
        full_pattern.insert(0, 0)  # 0 for invalid pattern
    
    return full_pattern

def generate_secondary_test_pattern(validity=True, noise_level=0.05):
    """
    Generate a synthetic secondary test pattern based on Wyckoff principles.
    
    Parameters:
    - validity: Whether to generate a valid (True) or invalid (False) pattern
    - noise_level: Amount of noise to add to the pattern
    
    Returns:
    - Pattern array with label as first element
    """
    if validity:
        # Create a valid secondary test pattern
        # p1: high point, p2: low point, p3 & p4: equal/similar lows (secondary test)
        p1 = random.uniform(80, 100)
        p2 = random.uniform(20, 40)
        
        # Secondary test points - close to the original low (p2)
        p3 = random.gauss(p2, 2)
        p3 = min(max(p3, p2-3), p2+3)  # Keep close to p2
        
        p4 = random.gauss(p2, 2)
        p4 = min(max(p4, p2-3), p2+3)  # Keep close to p2
        
        # p5: higher low (sign of strength)
        p5 = random.uniform(p2 + (p1-p2)*0.2, p2 + (p1-p2)*0.4)
        
        # Create basic pattern
        pattern = [p1, p2, p3, p4, p5]
        
        # Add filler points to create a more realistic pattern
        full_pattern = add_filler_points(pattern, num_between=2, noise_level=noise_level)
        
        # Add label
        full_pattern.insert(0, 1)  # 1 for valid pattern
        
    else:
        # Generate random points for invalid pattern
        pattern = [
            random.uniform(20, 100),
            random.uniform(20, 100),
            random.uniform(20, 100),
            random.uniform(20, 100),
            random.uniform(20, 100)
        ]
        
        # If by chance we created a valid-looking pattern, make it invalid
        # (For secondary test, the key is having p3 and p4 close to p2)
        if abs(pattern[2] - pattern[1]) < 5 and abs(pattern[3] - pattern[1]) < 5:
            # Make the "retest" points clearly different from the low
            pattern[2] = pattern[1] + 15
            pattern[3] = pattern[1] + 20
        
        # Add filler points
        full_pattern = add_filler_points(pattern, num_between=2, noise_level=noise_level)
        
        # Add label
        full_pattern.insert(0, 0)  # 0 for invalid pattern
    
    return full_pattern

def add_filler_points(pattern, num_between=2, noise_level=0.05):
    """
    Add filler points between the main swing points to create a more realistic pattern.
    
    Parameters:
    - pattern: List of main swing points
    - num_between: Number of points to add between each pair of swing points
    - noise_level: Level of noise to add to the filler points
    
    Returns:
    - List with original points and filler points
    """
    new_pattern = []
    
    for i in range(len(pattern) - 1):
        # Add the current swing point
        new_pattern.append(pattern[i])
        
        # Calculate the step between current and next point
        step = (pattern[i+1] - pattern[i]) / (num_between + 1)
        
        # Add filler points
        for j in range(1, num_between + 1):
            # Base value for filler point
            filler_val = pattern[i] + step * j
            
            # Add some noise
            noise = random.uniform(-noise_level * abs(step), noise_level * abs(step))
            filler_val += noise
            
            new_pattern.append(filler_val)
    
    # Add the last swing point
    new_pattern.append(pattern[-1])
    
    return new_pattern

def generate_dataset(num_samples, pattern_type='trading_range', validity_ratio=0.5, normalize=True):
    """
    Generate a dataset of patterns for training the model.
    
    Parameters:
    - num_samples: Number of samples to generate
    - pattern_type: 'trading_range' or 'secondary_test'
    - validity_ratio: Ratio of valid patterns in the dataset
    - normalize: Whether to normalize the patterns
    
    Returns:
    - X: Features (patterns)
    - y: Labels (valid/invalid)
    """
    # Determine how many valid and invalid patterns to generate
    num_valid = int(num_samples * validity_ratio)
    num_invalid = num_samples - num_valid
    
    # Generate patterns
    patterns = []
    
    if pattern_type == 'trading_range':
        for _ in range(num_valid):
            patterns.append(generate_trading_range_pattern(validity=True))
        for _ in range(num_invalid):
            patterns.append(generate_trading_range_pattern(validity=False))
    else:  # secondary_test
        for _ in range(num_valid):
            patterns.append(generate_secondary_test_pattern(validity=True))
        for _ in range(num_invalid):
            patterns.append(generate_secondary_test_pattern(validity=False))
    
    # Convert to numpy arrays
    patterns = np.array(patterns)
    
    # Extract labels and features
    y = patterns[:, 0]
    X = patterns[:, 1:]
    
    # Normalize features if requested
    if normalize:
        for i in range(len(X)):
            # Min-max normalization for each pattern
            min_val = np.min(X[i])
            max_val = np.max(X[i])
            if max_val > min_val:  # Avoid division by zero
                X[i] = (X[i] - min_val) / (max_val - min_val)
    
    # Reshape for LSTM: (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y

def generate_price_series_from_pattern(pattern, n_before=50, n_after=50, price_level=100, trend_strength=0.1):
    """
    Generate a complete price series with the pattern embedded in it.
    
    Parameters:
    - pattern: The pattern to embed (without the label)
    - n_before: Number of candles before the pattern
    - n_after: Number of candles after the pattern
    - price_level: Initial price level
    - trend_strength: Strength of the trend after the pattern
    
    Returns:
    - prices: A series of prices
    """
    # Generate prices before the pattern (with some noise and slight downtrend)
    prices_before = []
    current_price = price_level
    
    for _ in range(n_before):
        # Add slight downtrend with noise
        change = random.gauss(-0.05, 0.2)  # Slight downtrend with noise
        current_price = max(current_price + change, 20)  # Don't go below 20
        prices_before.append(current_price)
    
    # Scale the pattern to match the current price level
    pattern_min = min(pattern)
    pattern_max = max(pattern)
    pattern_range = pattern_max - pattern_min
    
    price_range = current_price * 0.2  # Use 20% of current price as range
    
    # Scale and shift the pattern
    scaled_pattern = []
    for p in pattern:
        # Normalize the pattern point
        normalized = (p - pattern_min) / pattern_range
        
        # Scale to the desired range and shift to current price level
        scaled = current_price - price_range/2 + normalized * price_range
        scaled_pattern.append(scaled)
    
    # Generate prices after the pattern (with uptrend for valid patterns)
    prices_after = []
    current_price = scaled_pattern[-1]
    
    is_valid = 1  # Assume valid pattern
    
    for _ in range(n_after):
        # Add uptrend with noise for valid patterns
        if is_valid:
            change = random.gauss(trend_strength, 0.3)  # Uptrend with noise
        else:
            change = random.gauss(-0.05, 0.3)  # No clear trend for invalid patterns
        
        current_price = max(current_price + change, 20)  # Don't go below 20
        prices_after.append(current_price)
    
    # Combine all prices
    all_prices = prices_before + scaled_pattern + prices_after
    
    return all_prices

def visualize_pattern(pattern, title="Wyckoff Pattern"):
    """
    Visualize a pattern.
    
    Parameters:
    - pattern: The pattern to visualize (without the label)
    - title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(pattern, marker='o')
    plt.title(title)
    plt.xlabel("Points")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def visualize_price_series_with_pattern(prices, pattern_start, pattern_end, title="Price Series with Wyckoff Pattern"):
    """
    Visualize a price series with the pattern highlighted.
    
    Parameters:
    - prices: Complete price series
    - pattern_start: Index where pattern starts
    - pattern_end: Index where pattern ends
    - title: Title for the plot
    """
    plt.figure(figsize=(15, 7))
    
    # Plot the full price series
    plt.plot(prices, label="Price")
    
    # Highlight the pattern
    plt.plot(range(pattern_start, pattern_end+1), 
             prices[pattern_start:pattern_end+1], 
             'r-', linewidth=3, label="Pattern")
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# PART 2: MODEL BUILDING

def build_lstm_model(input_shape, dropout_rate=0.2):
    """
    Build an LSTM model for pattern recognition.
    
    Parameters:
    - input_shape: Shape of input data (time steps, features)
    - dropout_rate: Dropout rate for regularization
    
    Returns:
    - model: Compiled Keras model
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(32),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, batch_size=32, epochs=50):
    """
    Train and evaluate the LSTM model.
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - batch_size: Batch size for training
    - epochs: Number of training epochs
    
    Returns:
    - model: Trained model
    - history: Training history
    - test_loss, test_acc: Test loss and accuracy
    """
    # Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history, test_loss, test_acc

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for the model.
    
    Parameters:
    - model: Trained model
    - X_test, y_test: Test data
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    return roc_auc

def evaluate_model_performance(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model performance with detailed metrics.
    
    Parameters:
    - model: Trained model
    - X_test, y_test: Test data
    - threshold: Classification threshold
    
    Returns:
    - report: Classification report
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Invalid', 'Valid'],
                yticklabels=['Invalid', 'Valid'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    return report

# PART 3: TRADING STRATEGY

def create_trading_strategy(price_data, model, window_size, threshold=0.5, entry_threshold=0.7):
    """
    Create a trading strategy based on LSTM pattern recognition.
    
    Parameters:
    - price_data: Series of price data
    - model: Trained LSTM model
    - window_size: Size of the window for pattern detection
    - threshold: Threshold for pattern classification
    - entry_threshold: Higher threshold for trade entry to reduce false positives
    
    Returns:
    - signals: DataFrame with signals and position
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(price_data, pd.Series):
        prices = price_data.values
    else:
        prices = price_data
    
    # Initialize signals DataFrame
    signals = pd.DataFrame({
        'price': prices,
        'pattern_prob': np.zeros(len(prices)),
        'signal': np.zeros(len(prices)),
        'position': np.zeros(len(prices))
    })
    
    # Scan through the price data with a rolling window
    for i in range(len(prices) - window_size + 1):
        # Extract the window
        window = prices[i:i+window_size]
        
        # Normalize the window
        window_norm = (window - np.min(window)) / (np.max(window) - np.min(window))
        
        # Reshape for LSTM input
        window_input = window_norm.reshape(1, window_size, 1)
        
        # Get model prediction
        pattern_prob = model.predict(window_input, verbose=0)[0][0]
        
        # Store prediction probability
        if i + window_size - 1 < len(signals):
            signals.loc[i + window_size - 1, 'pattern_prob'] = pattern_prob
    
    # Generate trading signals
    for i in range(len(signals)):
        # Entry signal: Pattern detected with high probability
        if signals.loc[i, 'pattern_prob'] > entry_threshold:
            signals.loc[i, 'signal'] = 1  # Buy signal
    
    # Calculate positions
    position = 0
    holding_period = 0
    max_holding_period = 20  # Maximum holding period after pattern detection
    
    for i in range(len(signals)):
        if signals.loc[i, 'signal'] == 1 and position == 0:
            # Enter long position
            position = 1
            holding_period = 0
        elif position == 1:
            # Increment holding period
            holding_period += 1
            
            # Exit after max holding period
            if holding_period >= max_holding_period:
                position = 0
                holding_period = 0
        
        signals.loc[i, 'position'] = position
    
    return signals

def backtest_strategy(signals, initial_capital=10000, position_size=0.5):
    """
    Backtest the trading strategy.
    
    Parameters:
    - signals: DataFrame with signals and position
    - initial_capital: Initial capital for the strategy
    - position_size: Portion of capital to invest in each trade
    
    Returns:
    - results: DataFrame with backtest results
    """
    # Create a copy of signals DataFrame
    results = signals.copy()
    
    # Initialize columns
    results['returns'] = results['price'].pct_change()
    results['strategy_returns'] = results['position'].shift(1) * results['returns']
    results['equity_curve'] = (1 + results['strategy_returns']).cumprod()
    results['capital'] = initial_capital * results['equity_curve']
    
    # Calculate drawdown
    results['peak'] = results['capital'].cummax()
    results['drawdown'] = (results['capital'] - results['peak']) / results['peak']
    
    return results

def plot_backtest_results(results):
    """
    Plot backtest results.
    
    Parameters:
    - results: DataFrame with backtest results
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price and Entry Points
    plt.subplot(3, 1, 1)
    plt.plot(results['price'], label='Price')
    
    # Mark entry points
    entries = results[results['signal'] == 1].index
    plt.scatter(entries, results.loc[entries, 'price'], marker='^', color='g', s=100, label='Entry')
    
    # Mark position periods
    for i in range(len(results)):
        if results.iloc[i]['position'] == 1:
            plt.axvspan(i, i+1, alpha=0.2, color='g')
    
    plt.title('Price and Entry Points')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Pattern Probability
    plt.subplot(3, 1, 2)
    plt.plot(results['pattern_prob'], label='Pattern Probability')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Entry Threshold')
    plt.title('Pattern Detection Probability')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Equity Curve
    plt.subplot(3, 1, 3)
    plt.plot(results['capital'], label='Strategy Capital')
    plt.title('Equity Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown separately
    plt.figure(figsize=(12, 5))
    plt.plot(results['drawdown'], color='r')
    plt.title('Drawdown')
    plt.xlabel('Time')
    plt.ylabel('Drawdown %')
    plt.grid(True)
    plt.show()

def print_strategy_metrics(results):
    """
    Print key metrics for the trading strategy.
    
    Parameters:
    - results: DataFrame with backtest results
    """
    # Calculate metrics
    total_return = results['capital'].iloc[-1] / results['capital'].iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(results)) - 1
    
    # Calculate Sharpe ratio (assuming 252 trading days per year)
    risk_free_rate = 0.02  # 2% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = results['strategy_returns'] - daily_risk_free
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Calculate maximum drawdown
    max_drawdown = results['drawdown'].min()
    
    # Calculate win rate
    results['trade'] = results['position'].diff()
    trade_ends = results[results['trade'] == -1].index
    trade_starts = results[results['position'].diff() == 1].index
    
    if len(trade_starts) > 0 and len(trade_ends) > 0:
        trades = []
        for start in trade_starts:
            # Find the next end after this start
            ends_after_start = trade_ends[trade_ends > start]
            if len(ends_after_start) > 0:
                end = ends_after_start[0]
                # Calculate trade return
                entry_price = results.loc[start, 'price']
                exit_price = results.loc[end, 'price']
                trade_return = (exit_price / entry_price) - 1
                trades.append(trade_return)
        
        if trades:
            win_rate = sum(1 for t in trades if t > 0) / len(trades)
            avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
            avg_loss = np.mean([t for t in trades if t <= 0]) if any(t <= 0 for t in trades) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    
    # Print metrics
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Number of Trades: {len(trades) if 'trades' in locals() else 0}")

# PART 4: REAL-WORLD TESTING

def download_forex_data(pair='EURUSD=X', start_date='2020-01-01', end_date='2023-12-31'):
    """
    Download forex data from Yahoo Finance.
    
    Parameters:
    - pair: Forex pair
    - start_date: Start date
    - end_date: End date
    
    Returns:
    - data: DataFrame with forex data
    """
    try:
        data = yf.download(pair, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def preprocess_real_data(data, window_size):
    """
    Preprocess real forex data for pattern detection.
    
    Parameters:
    - data: DataFrame with forex data
    - window_size: Size of the window for pattern detection
    
    Returns:
    - processed_data: DataFrame with processed data
    """
    # Keep only Close prices
    close_prices = data['Close']
    
    # Calculate additional features
    processed_data = pd.DataFrame({
        'close': close_prices,
        'returns': close_prices.pct_change(),
        'volatility': close_prices.pct_change().rolling(20).std()
    })
    
    # Fill NaN values
    processed_data = processed_data.fillna(0)
    
    return processed_data

def apply_strategy_to_real_data(data, model, window_size):
    """
    Apply the trading strategy to real forex data.
    
    Parameters:
    - data: DataFrame with forex data
    - model: Trained LSTM model
    - window_size: Size of the window for pattern detection
    
    Returns:
    - results: DataFrame with strategy results
    """
    # Preprocess data
    processed_data = preprocess_real_data(data, window_size)
    
    # Apply trading strategy
    signals = create_trading_strategy(processed_data['close'], model, window_size)
    
    # Backtest strategy
    results = backtest_strategy(signals)
    
    return results

# PART 5: MAIN FUNCTION

def main():
    """
    Main function to run the entire pipeline.
    """
    print("Starting Wyckoff Pattern Recognition with LSTM")
    print("-" * 50)
    
    # Set parameters
    num_samples = 10000
    pattern_type = 'trading_range'  # 'trading_range' or 'secondary_test'
    
    # 1. Data Generation
    print(f"\nGenerating {num_samples} {pattern_type} patterns...")
    X, y = generate_dataset(num_samples, pattern_type=pattern_type)
    
    # Print shapes
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Visualize some patterns
    print("\nVisualizing sample patterns...")
    
    # Generate and visualize a valid pattern
    valid_pattern = generate_trading_range_pattern(validity=True)[1:]
    visualize_pattern(valid_pattern, "Valid Trading Range Pattern")
    
    # Generate a price series with the pattern embedded
    valid_price_series = generate_price_series_from_pattern(valid_pattern)
    visualize_price_series_with_pattern(
        valid_price_series, 
        pattern_start=50, 
        pattern_end=50+len(valid_pattern)-1,
        title="Price Series with Valid Wyckoff Trading Range Pattern"
    )
    
    # 3. Model Training
    print("\nTraining LSTM model...")
    model, history, test_loss, test_acc = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, epochs=30
    )
    
    # 4. Model Evaluation
    print("\nEvaluating model performance...")
    roc_auc = plot_roc_curve(model, X_test, y_test)
    report = evaluate_model_performance(model, X_test, y_test)
    
    # 5. Trading Strategy on Simulated Data
    print("\nTesting trading strategy on simulated data...")
    
    # Generate a longer price series for backtesting
    window_size = X.shape[1]
    
    # Create 500 days of price data with multiple patterns
    simulated_prices = []
    is_valid_patterns = []
    pattern_locations = []
    
    current_price = 100
    day_count = 0
    
    while day_count < 500:
        # Decide whether to insert a pattern
        if random.random() < 0.1 and day_count > 50:  # 10% chance to insert a pattern
            # Decide whether it's a valid or invalid pattern
            is_valid = random.random() < 0.7  # 70% valid patterns
            
            if pattern_type == 'trading_range':
                pattern = generate_trading_range_pattern(validity=is_valid)[1:]
            else:
                pattern = generate_secondary_test_pattern(validity=is_valid)[1:]
            
            # Generate price series with pattern
            pattern_prices = generate_price_series_from_pattern(
                pattern, 
                n_before=0, 
                n_after=30, 
                price_level=current_price,
                trend_strength=0.2 if is_valid else -0.05
            )
            
            # Record pattern information
            pattern_start = day_count
            pattern_end = day_count + len(pattern) - 1
            pattern_locations.append((pattern_start, pattern_end, is_valid))
            is_valid_patterns.append(is_valid)
            
            # Add to simulated prices
            simulated_prices.extend(pattern_prices)
            
            # Update day count and current price
            day_count += len(pattern_prices)
            current_price = pattern_prices[-1]
        else:
            # Generate a single day with random walk
            change = random.gauss(0, 0.5)
            current_price = max(current_price + change, 20)
            simulated_prices.append(current_price)
            day_count += 1
    
    # Create signals using the model
    signals = create_trading_strategy(
        simulated_prices, 
        model, 
        window_size, 
        entry_threshold=0.7
    )
    
    # Backtest the strategy
    results = backtest_strategy(signals)
    
    # Plot results
    plot_backtest_results(results)
    
    # Print metrics
    print("\nTrading Strategy Metrics:")
    print_strategy_metrics(results)
    
    # 6. Optional: Testing on Real Forex Data
    print("\nTesting on real forex data...")
    
    # Download forex data
    forex_data = download_forex_data(pair='EURUSD=X', start_date='2020-01-01', end_date='2023-12-31')
    
    if forex_data is not None:
        # Apply strategy to real data
        real_results = apply_strategy_to_real_data(forex_data, model, window_size)
        
        # Plot results
        plot_backtest_results(real_results)
        
        # Print metrics
        print("\nReal Forex Data Strategy Metrics:")
        print_strategy_metrics(real_results)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()