import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import random
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============== DATA GENERATION =================

def generate_stock_data(n_stocks=500, n_days=5000, seed=42):
    """
    Generate synthetic stock price data similar to S&P 500 stocks.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to generate
    n_days : int
        Number of trading days
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict: Dictionary with stock data frames containing open, close prices
    """
    np.random.seed(seed)
    
    # Create dictionary to store stock data
    stock_data = {}
    
    # Generate dates
    dates = pd.date_range(start='1990-01-01', periods=n_days)
    
    # Market factor (affects all stocks)
    market_return = 0.0001  # Small positive drift
    market_vol = 0.01  # Market volatility
    market_factor = np.random.normal(market_return, market_vol, n_days)
    market_factor = np.cumsum(market_factor)
    
    # Create different sectors/factors (5 sectors)
    n_sectors = 5
    sector_factors = np.zeros((n_sectors, n_days))
    for s in range(n_sectors):
        sector_vol = 0.008
        sector_drift = np.random.normal(-0.0001, 0.0002)  # Different drift for each sector
        sector_returns = np.random.normal(sector_drift, sector_vol, n_days)
        sector_factors[s] = np.cumsum(sector_returns)
    
    # Assign each stock to a sector
    stock_sectors = np.random.randint(0, n_sectors, n_stocks)
    
    for i in range(n_stocks):
        # Base price for this stock
        base_price = np.random.uniform(20, 200)
        
        # Stock-specific parameters
        stock_vol = np.random.uniform(0.01, 0.03)  # Individual stock volatility
        stock_beta = np.random.uniform(0.5, 1.5)    # Market sensitivity
        sector_beta = np.random.uniform(0.5, 1.5)   # Sector sensitivity
        
        # Generate stock-specific noise
        stock_noise = np.random.normal(0, stock_vol, n_days)
        
        # Combine market, sector and stock-specific factors to create log returns
        sector = stock_sectors[i]
        log_returns = (stock_beta * market_factor + 
                      sector_beta * sector_factors[sector] + 
                      np.cumsum(stock_noise)) / 100
        
        # Convert to price series
        prices = base_price * np.exp(log_returns)
        
        # Generate Open, High, Low prices around the Close prices
        opens = prices * np.exp(np.random.normal(0, 0.005, n_days))
        highs = np.maximum(prices, opens) * np.exp(np.random.uniform(0.001, 0.01, n_days))
        lows = np.minimum(prices, opens) * np.exp(-np.random.uniform(0.001, 0.01, n_days))
        
        # Create DataFrame for this stock
        stock_df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': np.random.randint(100000, 10000000, n_days)
        })
        
        # Store in dictionary
        stock_data[f'Stock_{i+1}'] = stock_df
    
    return stock_data

# =============== DATA PREPROCESSING =================

def prepare_study_periods(stock_data, train_years=3, trade_years=1):
    """
    Split data into study periods with non-overlapping testing periods.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of stock DataFrames
    train_years : int
        Number of years for training
    trade_years : int
        Number of years for trading
        
    Returns:
    --------
    list: List of study periods, each containing training and trading data
    """
    # Extract dates from the first stock
    first_stock = list(stock_data.keys())[0]
    dates = stock_data[first_stock]['Date']
    
    # Calculate days per year (approximately 252 trading days)
    days_per_year = 252
    train_days = train_years * days_per_year
    trade_days = trade_years * days_per_year
    study_period_days = train_days + trade_days
    
    # Calculate number of complete study periods
    total_days = len(dates)
    n_periods = (total_days - 240) // (trade_days)  # Subtract 240 for feature creation
    
    study_periods = []
    
    for i in range(n_periods):
        start_idx = i * trade_days
        train_end_idx = start_idx + train_days
        trade_end_idx = train_end_idx + trade_days
        
        if trade_end_idx > total_days:
            break
            
        # Create a study period
        study_period = {
            'train_start': start_idx,
            'train_end': train_end_idx,
            'trade_start': train_end_idx,
            'trade_end': trade_end_idx
        }
        
        study_periods.append(study_period)
    
    return study_periods

def create_features_targets(stock_data, period):
    """
    Create features and targets for a specific study period.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of stock DataFrames
    period : dict
        Study period indices
        
    Returns:
    --------
    tuple: X_train, y_train, X_trade, metadata
    """
    train_start = period['train_start']
    train_end = period['train_end']
    trade_start = period['trade_start']
    trade_end = period['trade_end']
    
    # Lists to store features and targets
    X_train_rf = []
    X_train_lstm = []
    y_train = []
    
    X_trade_rf = []
    X_trade_lstm = []
    
    # Store metadata for trading
    trade_dates = []
    trade_stocks = []
    trade_intraday_returns = []
    
    # Process each day in the training period
    for t in range(train_start + 240, train_end):  # Start from 240 to have enough history
        # Store all stocks' intraday returns for this day
        day_intraday_returns = []
        
        # Process each stock
        for stock_name, stock_df in stock_data.items():
            # Extract features for this stock on this day
            features_rf = extract_rf_features(stock_df, t)
            features_lstm = extract_lstm_features(stock_df, t)
            
            # Calculate intraday return (target)
            intraday_return = stock_df.iloc[t]['Close'] / stock_df.iloc[t]['Open'] - 1
            day_intraday_returns.append((stock_name, intraday_return))
            
            # Store features and targets
            X_train_rf.append(features_rf)
            X_train_lstm.append(features_lstm)
        
        # Sort stocks by intraday return and assign classes
        day_intraday_returns.sort(key=lambda x: x[1])
        median_idx = len(day_intraday_returns) // 2
        
        for i, (stock_name, _) in enumerate(day_intraday_returns):
            # Class 1 if above median, Class 0 if below median
            target = 1 if i >= median_idx else 0
            y_train.append(target)
    
    # Process each day in the trading period
    for t in range(trade_start + 240, trade_end):
        # Store this trading day's date
        date = stock_data[list(stock_data.keys())[0]].iloc[t]['Date']
        trade_dates.append(date)
        
        # Store all stocks' intraday returns for this day
        day_intraday_returns = []
        day_stocks = []
        
        # Process each stock
        for stock_name, stock_df in stock_data.items():
            # Extract features for this stock on this day
            features_rf = extract_rf_features(stock_df, t)
            features_lstm = extract_lstm_features(stock_df, t)
            
            # Calculate intraday return (for evaluation)
            intraday_return = stock_df.iloc[t]['Close'] / stock_df.iloc[t]['Open'] - 1
            day_intraday_returns.append(intraday_return)
            day_stocks.append(stock_name)
            
            # Store features
            X_trade_rf.append(features_rf)
            X_trade_lstm.append(features_lstm)
        
        trade_stocks.append(day_stocks)
        trade_intraday_returns.append(day_intraday_returns)
    
    # Convert to numpy arrays
    X_train_rf = np.array(X_train_rf)
    X_train_lstm = np.array(X_train_lstm)
    y_train = np.array(y_train)
    
    X_trade_rf = np.array(X_trade_rf)
    X_trade_lstm = np.array(X_trade_lstm)
    
    # Create metadata dictionary
    metadata = {
        'dates': trade_dates,
        'stocks': trade_stocks,
        'intraday_returns': trade_intraday_returns
    }
    
    return X_train_rf, X_train_lstm, y_train, X_trade_rf, X_trade_lstm, metadata

def extract_rf_features(stock_df, t):
    """
    Extract features for Random Forest model.
    
    Parameters:
    -----------
    stock_df : DataFrame
        DataFrame containing stock data
    t : int
        Current time index
        
    Returns:
    --------
    numpy.array: Array of features
    """
    features = []
    
    # Feature set 1: Intraday returns
    for m in list(range(1, 21)) + list(range(40, 241, 20)):
        if t - m >= 0:
            intraday_return = stock_df.iloc[t-m]['Close'] / stock_df.iloc[t-m]['Open'] - 1
            features.append(intraday_return)
        else:
            features.append(0)
    
    # Feature set 2: Returns with respect to last closing price
    for m in list(range(1, 21)) + list(range(40, 241, 20)):
        if t - m - 1 >= 0:
            close_return = stock_df.iloc[t-1]['Close'] / stock_df.iloc[t-m-1]['Close'] - 1
            features.append(close_return)
        else:
            features.append(0)
    
    # Feature set 3: Returns with respect to opening price
    for m in list(range(1, 21)) + list(range(40, 241, 20)):
        if t - m >= 0:
            open_return = stock_df.iloc[t]['Open'] / stock_df.iloc[t-m]['Close'] - 1
            features.append(open_return)
        else:
            features.append(0)
    
    return np.array(features)

def extract_lstm_features(stock_df, t):
    """
    Extract features for LSTM model.
    
    Parameters:
    -----------
    stock_df : DataFrame
        DataFrame containing stock data
    t : int
        Current time index
        
    Returns:
    --------
    numpy.array: Array of features for the past 240 days
    """
    features = np.zeros((240, 3))
    
    for i in range(240):
        idx = t - 240 + i
        if idx >= 0:
            # Intraday return
            features[i, 0] = stock_df.iloc[idx]['Close'] / stock_df.iloc[idx]['Open'] - 1
            
            # Return with respect to previous close
            if idx > 0:
                features[i, 1] = stock_df.iloc[idx]['Close'] / stock_df.iloc[idx-1]['Close'] - 1
            
            # Return with respect to previous close for open
            if idx > 0:
                features[i, 2] = stock_df.iloc[idx]['Open'] / stock_df.iloc[idx-1]['Close'] - 1
    
    return features

# =============== MODELS =================

class LSTMClassifier(nn.Module):
    """
    LSTM model for binary classification.
    """
    def __init__(self, input_size, hidden_size=25, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_out)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class StockDataset(Dataset):
    """
    Dataset for PyTorch DataLoader
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

def train_models(X_train_rf, X_train_lstm, y_train, batch_size=512, epochs=100, patience=10):
    """
    Train Random Forest and LSTM models.
    
    Parameters:
    -----------
    X_train_rf : numpy.array
        Training features for Random Forest
    X_train_lstm : numpy.array
        Training features for LSTM
    y_train : numpy.array
        Target values
    batch_size : int
        Batch size for LSTM training
    epochs : int
        Maximum number of epochs for LSTM training
    patience : int
        Early stopping patience
        
    Returns:
    --------
    tuple: Trained Random Forest and LSTM models, and scaler
    """
    # Train Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_rf, y_train)
    rf_training_time = time.time() - start_time
    print(f"Random Forest training completed in {rf_training_time:.2f} seconds")
    
    # Scale LSTM features
    print("Scaling LSTM features...")
    scaler = RobustScaler()
    n_samples = X_train_lstm.shape[0]
    n_timesteps = X_train_lstm.shape[1]
    n_features = X_train_lstm.shape[2]
    
    # Reshape for scaling
    X_train_lstm_reshaped = X_train_lstm.reshape(n_samples * n_timesteps, n_features)
    X_train_lstm_scaled = scaler.fit_transform(X_train_lstm_reshaped)
    X_train_lstm = X_train_lstm_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Train LSTM model
    print("Training LSTM...")
    start_time = time.time()
    
    # Create dataset and dataloader
    train_dataset = StockDataset(X_train_lstm, y_train)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    # Initialize model
    lstm_model = LSTMClassifier(input_size=n_features).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(lstm_model.parameters(), lr=0.001)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        lstm_model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = lstm_model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        lstm_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = lstm_model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Check early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = lstm_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    lstm_model.load_state_dict(best_model_state)
    
    lstm_training_time = time.time() - start_time
    print(f"LSTM training completed in {lstm_training_time:.2f} seconds")
    
    return rf_model, lstm_model, scaler

def predict_and_trade(rf_model, lstm_model, scaler, X_trade_rf, X_trade_lstm, metadata, k=10):
    """
    Make predictions and simulate trading.
    
    Parameters:
    -----------
    rf_model : RandomForestClassifier
        Trained Random Forest model
    lstm_model : pytorch.Module
        Trained LSTM model
    scaler : RobustScaler
        Scaler for LSTM features
    X_trade_rf : numpy.array
        Trading features for Random Forest
    X_trade_lstm : numpy.array
        Trading features for LSTM
    metadata : dict
        Metadata for trading
    k : int
        Number of stocks to trade (top k and bottom k)
        
    Returns:
    --------
    dict: Trading results
    """
    # Scale LSTM input
    n_samples = X_trade_lstm.shape[0]
    n_timesteps = X_trade_lstm.shape[1]
    n_features = X_trade_lstm.shape[2]
    
    # Reshape for scaling
    X_trade_lstm_reshaped = X_trade_lstm.reshape(n_samples * n_timesteps, n_features)
    X_trade_lstm_scaled = scaler.transform(X_trade_lstm_reshaped)
    X_trade_lstm = X_trade_lstm_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Make Random Forest predictions
    rf_probs = rf_model.predict_proba(X_trade_rf)[:, 1]  # Probabilities of class 1
    
    # Make LSTM predictions
    lstm_model.eval()
    trade_dataset = StockDataset(X_trade_lstm)
    trade_loader = DataLoader(trade_dataset, batch_size=256)
    
    lstm_probs = []
    with torch.no_grad():
        for inputs in trade_loader:
            inputs = inputs.to(device)
            outputs = lstm_model(inputs)
            probs = outputs[:, 1].cpu().numpy()  # Probabilities of class 1
            lstm_probs.extend(probs)
    
    lstm_probs = np.array(lstm_probs)
    
    # Initialize trading results
    rf_daily_returns = []
    lstm_daily_returns = []
    
    # Get trading days and stocks
    dates = metadata['dates']
    stocks_by_day = metadata['stocks']
    intraday_returns_by_day = metadata['intraday_returns']
    
    # Simulate trading for each day
    n_days = len(dates)
    n_stocks = len(stocks_by_day[0])
    
    rf_prob_idx = 0
    lstm_prob_idx = 0
    
    for day in range(n_days):
        # Get probabilities for this day
        day_rf_probs = rf_probs[rf_prob_idx:rf_prob_idx + n_stocks]
        day_lstm_probs = lstm_probs[lstm_prob_idx:lstm_prob_idx + n_stocks]
        rf_prob_idx += n_stocks
        lstm_prob_idx += n_stocks
        
        # Get intraday returns for this day
        day_returns = intraday_returns_by_day[day]
        
        # Sort stocks by probability
        rf_sorted_indices = np.argsort(day_rf_probs)
        lstm_sorted_indices = np.argsort(day_lstm_probs)
        
        # Random Forest trading
        rf_long_indices = rf_sorted_indices[-k:]  # Top k stocks
        rf_short_indices = rf_sorted_indices[:k]  # Bottom k stocks
        
        rf_long_return = np.mean([day_returns[i] for i in rf_long_indices])
        rf_short_return = -np.mean([day_returns[i] for i in rf_short_indices])  # Negative for short position
        rf_daily_return = (rf_long_return + rf_short_return) / 2  # Equal weighting
        
        # LSTM trading
        lstm_long_indices = lstm_sorted_indices[-k:]  # Top k stocks
        lstm_short_indices = lstm_sorted_indices[:k]  # Bottom k stocks
        
        lstm_long_return = np.mean([day_returns[i] for i in lstm_long_indices])
        lstm_short_return = -np.mean([day_returns[i] for i in lstm_short_indices])  # Negative for short position
        lstm_daily_return = (lstm_long_return + lstm_short_return) / 2  # Equal weighting
        
        # Apply transaction costs (0.05% per half-turn, so 0.2% total daily)
        transaction_cost = 0.002  # 0.2%
        rf_daily_return -= transaction_cost
        lstm_daily_return -= transaction_cost
        
        # Store daily returns
        rf_daily_returns.append(rf_daily_return)
        lstm_daily_returns.append(lstm_daily_return)
    
    # Calculate cumulative returns
    rf_cum_returns = np.cumprod(1 + np.array(rf_daily_returns))
    lstm_cum_returns = np.cumprod(1 + np.array(lstm_daily_returns))
    
    # Calculate performance metrics
    rf_total_return = rf_cum_returns[-1] - 1
    lstm_total_return = lstm_cum_returns[-1] - 1
    
    rf_annual_return = (1 + rf_total_return) ** (252 / len(rf_daily_returns)) - 1
    lstm_annual_return = (1 + lstm_total_return) ** (252 / len(lstm_daily_returns)) - 1
    
    rf_sharpe = np.mean(rf_daily_returns) / np.std(rf_daily_returns) * np.sqrt(252)
    lstm_sharpe = np.mean(lstm_daily_returns) / np.std(lstm_daily_returns) * np.sqrt(252)
    
    # Return results
    results = {
        'dates': dates,
        'rf_daily_returns': rf_daily_returns,
        'lstm_daily_returns': lstm_daily_returns,
        'rf_cum_returns': rf_cum_returns,
        'lstm_cum_returns': lstm_cum_returns,
        'rf_total_return': rf_total_return,
        'lstm_total_return': lstm_total_return,
        'rf_annual_return': rf_annual_return,
        'lstm_annual_return': lstm_annual_return,
        'rf_sharpe': rf_sharpe,
        'lstm_sharpe': lstm_sharpe
    }
    
    return results

def plot_results(results):
    """
    Plot the trading results.
    
    Parameters:
    -----------
    results : dict
        Trading results
    """
    dates = results['dates']
    rf_cum_returns = results['rf_cum_returns']
    lstm_cum_returns = results['lstm_cum_returns']
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates, rf_cum_returns, label=f'Random Forest (Return: {results["rf_total_return"]:.2%}, Sharpe: {results["rf_sharpe"]:.2f})')
    plt.plot(dates, lstm_cum_returns, label=f'LSTM (Return: {results["lstm_total_return"]:.2%}, Sharpe: {results["lstm_sharpe"]:.2f})')
    
    # Create a S&P 500-like benchmark (1% per year)
    benchmark_returns = np.ones(len(dates)) * (1.01 ** (1/252)) - 1
    benchmark_cum_returns = np.cumprod(1 + benchmark_returns)
    plt.plot(dates, benchmark_cum_returns, label='Benchmark', linestyle='--')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Plot daily returns distribution
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    sns.histplot(results['rf_daily_returns'], kde=True)
    plt.title(f'Random Forest Daily Returns (Mean: {np.mean(results["rf_daily_returns"]):.4f})')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(results['lstm_daily_returns'], kde=True)
    plt.title(f'LSTM Daily Returns (Mean: {np.mean(results["lstm_daily_returns"]):.4f})')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n======= PERFORMANCE SUMMARY =======")
    print(f"Random Forest Total Return: {results['rf_total_return']:.2%}")
    print(f"LSTM Total Return: {results['lstm_total_return']:.2%}")
    print(f"Random Forest Annual Return: {results['rf_annual_return']:.2%}")
    print(f"LSTM Annual Return: {results['lstm_annual_return']:.2%}")
    print(f"Random Forest Sharpe Ratio: {results['rf_sharpe']:.2f}")
    print(f"LSTM Sharpe Ratio: {results['lstm_sharpe']:.2f}")
    print(f"Random Forest Avg Daily Return: {np.mean(results['rf_daily_returns']):.4f}")
    print(f"LSTM Avg Daily Return: {np.mean(results['lstm_daily_returns']):.4f}")
    print(f"Random Forest Std Dev Daily Return: {np.std(results['rf_daily_returns']):.4f}")
    print(f"LSTM Std Dev Daily Return: {np.std(results['lstm_daily_returns']):.4f}")
    print(f"% Positive Days (RF): {np.mean(np.array(results['rf_daily_returns']) > 0):.2%}")
    print(f"% Positive Days (LSTM): {np.mean(np.array(results['lstm_daily_returns']) > 0):.2%}")
    
    # Calculate drawdown
    rf_drawdown = 1 - rf_cum_returns / np.maximum.accumulate(rf_cum_returns)
    lstm_drawdown = 1 - lstm_cum_returns / np.maximum.accumulate(lstm_cum_returns)
    
    print(f"Random Forest Max Drawdown: {np.max(rf_drawdown):.2%}")
    print(f"LSTM Max Drawdown: {np.max(lstm_drawdown):.2%}")
    
    # Plot drawdown
    plt.figure(figsize=(14, 7))
    plt.plot(dates, rf_drawdown, label='Random Forest')
    plt.plot(dates, lstm_drawdown, label='LSTM')
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to run the experiment."""
    print("Generating synthetic stock data...")
    # Generate fewer stocks and days for demonstration
    stock_data = generate_stock_data(n_stocks=50, n_days=2000)
    
    print("Creating study periods...")
    study_periods = prepare_study_periods(stock_data)
    
    print(f"Number of study periods: {len(study_periods)}")
    
    # Run for one study period (for demonstration)
    period_idx = 0
    period = study_periods[period_idx]
    
    print(f"Processing study period {period_idx+1}/{len(study_periods)}")
    print(f"Training period: {period['train_start']} to {period['train_end']}")
    print(f"Trading period: {period['trade_start']} to {period['trade_end']}")
    
    # Create features and targets
    X_train_rf, X_train_lstm, y_train, X_trade_rf, X_trade_lstm, metadata = create_features_targets(stock_data, period)
    
    print(f"Training data shapes:")
    print(f"X_train_rf: {X_train_rf.shape}")
    print(f"X_train_lstm: {X_train_lstm.shape}")
    print(f"y_train: {y_train.shape}")
    
    print(f"Trading data shapes:")
    print(f"X_trade_rf: {X_trade_rf.shape}")
    print(f"X_trade_lstm: {X_trade_lstm.shape}")
    
    # Train models
    rf_model, lstm_model, scaler = train_models(X_train_rf, X_train_lstm, y_train)
    
    # Predict and trade
    results = predict_and_trade(rf_model, lstm_model, scaler, X_trade_rf, X_trade_lstm, metadata)
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()