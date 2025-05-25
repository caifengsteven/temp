import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate synthetic stock market data
def generate_stock_data(n_days=1500, starting_price=100, volatility=0.01,
                        trend=0.0001, cyclical_period=252, cyclical_amplitude=0.05):
    """
    Generate synthetic stock market data with trend, cyclical patterns, and noise.
    
    Parameters:
    - n_days: Number of trading days
    - starting_price: Initial price
    - volatility: Daily volatility
    - trend: Daily upward/downward trend
    - cyclical_period: Period of cyclical patterns in days
    - cyclical_amplitude: Amplitude of cyclical patterns
    
    Returns:
    - dates: Array of dates
    - prices: Dictionary with OHLCV data
    """
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    # Generate random returns with normal distribution
    returns = np.random.normal(trend, volatility, n_days)
    
    # Add cyclical component
    t = np.arange(n_days)
    cyclical = cyclical_amplitude * np.sin(2 * np.pi * t / cyclical_period)
    returns = returns + cyclical
    
    # Calculate prices from returns
    log_prices = np.zeros(n_days)
    log_prices[0] = np.log(starting_price)
    for i in range(1, n_days):
        log_prices[i] = log_prices[i-1] + returns[i]
    
    close_prices = np.exp(log_prices)
    
    # Generate Open, High, Low prices based on Close
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)
    
    # First day
    open_prices[0] = close_prices[0] * (1 + np.random.normal(0, volatility/2))
    intraday_range = np.random.uniform(0.01, 0.03) * close_prices[0]
    if open_prices[0] > close_prices[0]:
        high_prices[0] = open_prices[0] * (1 + np.random.uniform(0, 0.01))
        low_prices[0] = close_prices[0] * (1 - np.random.uniform(0, 0.01))
    else:
        high_prices[0] = close_prices[0] * (1 + np.random.uniform(0, 0.01))
        low_prices[0] = open_prices[0] * (1 - np.random.uniform(0, 0.01))
    
    # Rest of the days
    for i in range(1, n_days):
        open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, volatility/2))
        intraday_range = np.random.uniform(0.01, 0.03) * close_prices[i]
        if open_prices[i] > close_prices[i]:
            high_prices[i] = open_prices[i] * (1 + np.random.uniform(0, 0.01))
            low_prices[i] = close_prices[i] * (1 - np.random.uniform(0, 0.01))
        else:
            high_prices[i] = close_prices[i] * (1 + np.random.uniform(0, 0.01))
            low_prices[i] = open_prices[i] * (1 - np.random.uniform(0, 0.01))
    
    # Generate volume with some correlation to price changes
    volume = np.abs(np.diff(np.concatenate([[0], close_prices]))) * 1000000 + np.random.normal(500000, 200000, n_days)
    volume = np.clip(volume, 50000, 5000000).astype(int)
    
    prices = {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    
    return dates, prices

# Generate technical indicators
def calculate_technical_indicators(data):
    """
    Calculate technical indicators for the given price data.
    
    Parameters:
    - data: DataFrame with OHLCV data
    
    Returns:
    - DataFrame with technical indicators
    """
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['STD20'] = df['close'].rolling(window=20).std()
    df['BOLL_upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BOLL_lower'] = df['MA20'] - (df['STD20'] * 2)
    
    # ATR (Average True Range)
    df['TR'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1))
        ),
        np.abs(df['low'] - df['close'].shift(1))
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # CCI (Commodity Channel Index)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['TP_MA20'] = df['TP'].rolling(window=20).mean()
    df['TP_STD20'] = df['TP'].rolling(window=20).std()
    df['CCI'] = (df['TP'] - df['TP_MA20']) / (0.015 * df['TP_STD20'])
    
    # ROC (Rate of Change)
    df['ROC'] = (df['close'] / df['close'].shift(10) - 1) * 100
    
    # Momentum
    df['MTM6'] = df['close'] - df['close'].shift(6)
    df['MTM12'] = df['close'] - df['close'].shift(12)
    
    # Stochastic Momentum Index (simplified)
    df['highest_high'] = df['high'].rolling(window=14).max()
    df['lowest_low'] = df['low'].rolling(window=14).min()
    df['SMI'] = ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])) * 100
    
    # Williams Variable Accumulation/Distribution (simplified)
    df['WVAD'] = ((df['close'] - df['open']) / (df['high'] - df['low'])) * df['volume']
    
    return df

# Generate macroeconomic data
def generate_macro_data(dates):
    """
    Generate synthetic macroeconomic data for the given dates.
    
    Parameters:
    - dates: Array of dates
    
    Returns:
    - DataFrame with macroeconomic data
    """
    n_days = len(dates)
    
    # USD Index
    usd_index_trend = 0.0001
    usd_index_vol = 0.003
    usd_index = np.zeros(n_days)
    usd_index[0] = 96.0  # Starting value
    
    for i in range(1, n_days):
        usd_index[i] = usd_index[i-1] * (1 + np.random.normal(usd_index_trend, usd_index_vol))
    
    # Interest Rate (e.g., LIBOR)
    interest_rate_trend = 0.00001
    interest_rate_vol = 0.001
    interest_rate = np.zeros(n_days)
    interest_rate[0] = 0.01  # Starting value (1%)
    
    for i in range(1, n_days):
        interest_rate[i] = interest_rate[i-1] + np.random.normal(interest_rate_trend, interest_rate_vol)
        interest_rate[i] = max(0.001, min(0.1, interest_rate[i]))  # Keep in reasonable range
    
    macro_data = pd.DataFrame({
        'usd_index': usd_index,
        'interest_rate': interest_rate
    }, index=dates)
    
    return macro_data

# Wavelet Transform for denoising
def wavelet_denoising(data, wavelet='haar', level=2):
    """
    Apply wavelet transform to denoise time series data.
    
    Parameters:
    - data: Input data series
    - wavelet: Wavelet function to use
    - level: Decomposition level
    
    Returns:
    - Denoised data series
    """
    # Get coefficients
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Apply thresholding
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    new_coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        new_coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # Reconstruct signal
    denoised_data = pywt.waverec(new_coeffs, wavelet)
    
    # Make sure the length is the same as the input
    return denoised_data[:len(data)]

# PyTorch Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.Sigmoid())
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1]))  # Don't include the bottleneck layer twice
        prev_dim = hidden_dims[-1]
        for dim in reversed_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.Sigmoid())
            prev_dim = dim
        
        # Final layer to reconstruct input
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        
        # Get the last time step output
        last_time_step = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_time_step)
        
        # Linear layer for prediction
        out = self.fc(out)
        
        return out

# Simple RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        
        self.rnn = nn.RNN(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        rnn_out, _ = self.rnn(x)
        # rnn_out shape: (batch_size, sequence_length, hidden_dim)
        
        # Get the last time step output
        last_time_step = rnn_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_time_step)
        
        # Linear layer for prediction
        out = self.fc(out)
        
        return out

# Prepare sequences
def prepare_sequences(data, sequence_length):
    """
    Prepare sequences for time series prediction.
    
    Parameters:
    - data: Input data
    - sequence_length: Length of the sequence
    
    Returns:
    - X: Sequences (features)
    - y: Targets (next value)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, -1])  # Last column is the target
    
    return np.array(X), np.array(y).reshape(-1, 1)

# Train model
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    """
    Train the model.
    
    Parameters:
    - model: PyTorch model
    - train_loader: DataLoader with training data
    - criterion: Loss function
    - optimizer: Optimizer
    - num_epochs: Number of training epochs
    
    Returns:
    - Trained model
    - List of training losses
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    
    return model, losses

# Evaluation metrics
def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics.
    
    Parameters:
    - actual: Actual values
    - predicted: Predicted values
    
    Returns:
    - Dict with metrics
    """
    mape = mean_absolute_percentage_error(actual, predicted)
    r = pearsonr(actual, predicted)[0]
    
    # Theil's U
    actual_change = np.diff(actual)
    predicted_change = np.diff(predicted)
    rmse_prediction = np.sqrt(np.mean((actual_change - predicted_change) ** 2))
    rmse_naive = np.sqrt(np.mean(actual_change ** 2))
    theil_u = rmse_prediction / rmse_naive
    
    return {
        'MAPE': mape,
        'R': r,
        'Theil_U': theil_u
    }

# Trading strategy
def implement_trading_strategy(actual, predicted, transaction_cost=0.0001):
    """
    Implement the buy-and-sell trading strategy.
    
    Parameters:
    - actual: Actual prices
    - predicted: Predicted prices
    - transaction_cost: Transaction cost percentage
    
    Returns:
    - Dict with trading results
    """
    positions = np.zeros(len(actual))
    returns = np.zeros(len(actual))
    
    # Start from second day
    for t in range(1, len(actual)):
        # Buy signal: predicted price > current price
        if predicted[t] > actual[t-1]:
            positions[t] = 1
        # Sell signal: predicted price < current price
        else:
            positions[t] = -1
        
        # Calculate returns
        if positions[t] == 1:
            # Long position: gain if price goes up
            returns[t] = (actual[t] / actual[t-1] - 1) - transaction_cost
        else:
            # Short position: gain if price goes down
            returns[t] = (1 - actual[t] / actual[t-1]) - transaction_cost
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    # Calculate buy-and-hold returns
    buy_hold_returns = (actual[-1] / actual[0]) - 1
    
    # Calculate average annual return (assuming 252 trading days per year)
    annual_return = (1 + cumulative_returns[-1]) ** (252 / len(actual)) - 1
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
    
    return {
        'positions': positions,
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'buy_hold_returns': buy_hold_returns,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio
    }

# Run WSAEs-LSTM model
def run_wsaes_lstm(data_df, sequence_length=10, batch_size=32, num_epochs=50):
    """
    Run the WSAEs-LSTM model.
    
    Parameters:
    - data_df: DataFrame with input features
    - sequence_length: Length of sequences
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Dict with results
    """
    # 1. Normalize data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale features
    features = data_df.drop(columns=['close'])
    targets = data_df[['close']]
    
    features_scaled = scaler_x.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # 2. Apply Wavelet Transform for denoising
    features_denoised = np.zeros_like(features_scaled)
    for i in range(features_scaled.shape[1]):
        features_denoised[:, i] = wavelet_denoising(features_scaled[:, i])
    
    targets_denoised = wavelet_denoising(targets_scaled.flatten()).reshape(-1, 1)
    
    # 3. Build and train Stacked Autoencoder
    input_dim = features_denoised.shape[1]
    hidden_dims = [
        int(input_dim * 0.8), 
        int(input_dim * 0.6), 
        int(input_dim * 0.4), 
        int(input_dim * 0.2)
    ]
    
    autoencoder = Autoencoder(input_dim, hidden_dims).to(device)
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features_denoised, dtype=torch.float32).to(device)
    
    # Create DataLoader
    ae_dataset = torch.utils.data.TensorDataset(features_tensor, features_tensor)
    ae_loader = torch.utils.data.DataLoader(ae_dataset, batch_size=batch_size, shuffle=True)
    
    # Train autoencoder
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    print("Training autoencoder...")
    autoencoder, _ = train_model(autoencoder, ae_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # Extract deep features
    autoencoder.eval()
    with torch.no_grad():
        deep_features = autoencoder.encode(features_tensor).cpu().numpy()
    
    # Combine deep features with denoised target
    combined_data = np.hstack((deep_features, targets_denoised))
    
    # 4. Prepare sequences for LSTM
    X, y = prepare_sequences(combined_data, sequence_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Build and train LSTM model
    input_dim = X_train.shape[2]
    lstm_model = LSTMModel(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    print("Training LSTM model...")
    lstm_model, _ = train_model(lstm_model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # 6. Evaluate model
    lstm_model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert predictions to original scale
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Get original scale prices
    y_actual = scaler_y.inverse_transform(actuals)
    y_pred = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_actual.flatten(), y_pred.flatten())
    
    # Calculate trading strategy results
    trading_results = implement_trading_strategy(y_actual.flatten(), y_pred.flatten())
    
    return {
        'actual': y_actual.flatten(),
        'predicted': y_pred.flatten(),
        'metrics': metrics,
        'trading_results': trading_results
    }

# Run WLSTM model
def run_wlstm(data_df, sequence_length=10, batch_size=32, num_epochs=50):
    """
    Run the WLSTM model (Wavelet + LSTM).
    
    Parameters:
    - data_df: DataFrame with input features
    - sequence_length: Length of sequences
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Dict with results
    """
    # 1. Normalize data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale features
    features = data_df.drop(columns=['close'])
    targets = data_df[['close']]
    
    features_scaled = scaler_x.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # 2. Apply Wavelet Transform for denoising
    features_denoised = np.zeros_like(features_scaled)
    for i in range(features_scaled.shape[1]):
        features_denoised[:, i] = wavelet_denoising(features_scaled[:, i])
    
    targets_denoised = wavelet_denoising(targets_scaled.flatten()).reshape(-1, 1)
    
    # Combine features
    combined_data = np.hstack((features_denoised, targets_denoised))
    
    # 3. Prepare sequences for LSTM
    X, y = prepare_sequences(combined_data, sequence_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Build and train LSTM model
    input_dim = X_train.shape[2]
    lstm_model = LSTMModel(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    print("Training WLSTM model...")
    lstm_model, _ = train_model(lstm_model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # 5. Evaluate model
    lstm_model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert predictions to original scale
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Get original scale prices
    y_actual = scaler_y.inverse_transform(actuals)
    y_pred = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_actual.flatten(), y_pred.flatten())
    
    # Calculate trading strategy results
    trading_results = implement_trading_strategy(y_actual.flatten(), y_pred.flatten())
    
    return {
        'actual': y_actual.flatten(),
        'predicted': y_pred.flatten(),
        'metrics': metrics,
        'trading_results': trading_results
    }

# Run LSTM model
def run_lstm(data_df, sequence_length=10, batch_size=32, num_epochs=50):
    """
    Run the standard LSTM model.
    
    Parameters:
    - data_df: DataFrame with input features
    - sequence_length: Length of sequences
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Dict with results
    """
    # 1. Normalize data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale features
    features = data_df.drop(columns=['close'])
    targets = data_df[['close']]
    
    features_scaled = scaler_x.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # Combine features without denoising
    combined_data = np.hstack((features_scaled, targets_scaled))
    
    # 2. Prepare sequences for LSTM
    X, y = prepare_sequences(combined_data, sequence_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Build and train LSTM model
    input_dim = X_train.shape[2]
    lstm_model = LSTMModel(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    print("Training LSTM model...")
    lstm_model, _ = train_model(lstm_model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # 4. Evaluate model
    lstm_model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert predictions to original scale
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Get original scale prices
    y_actual = scaler_y.inverse_transform(actuals)
    y_pred = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_actual.flatten(), y_pred.flatten())
    
    # Calculate trading strategy results
    trading_results = implement_trading_strategy(y_actual.flatten(), y_pred.flatten())
    
    return {
        'actual': y_actual.flatten(),
        'predicted': y_pred.flatten(),
        'metrics': metrics,
        'trading_results': trading_results
    }

# Run RNN model
def run_rnn(data_df, sequence_length=10, batch_size=32, num_epochs=50):
    """
    Run the standard RNN model.
    
    Parameters:
    - data_df: DataFrame with input features
    - sequence_length: Length of sequences
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Dict with results
    """
    # 1. Normalize data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale features
    features = data_df.drop(columns=['close'])
    targets = data_df[['close']]
    
    features_scaled = scaler_x.fit_transform(features)
    targets_scaled = scaler_y.fit_transform(targets)
    
    # Combine features without denoising
    combined_data = np.hstack((features_scaled, targets_scaled))
    
    # 2. Prepare sequences for RNN
    X, y = prepare_sequences(combined_data, sequence_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Build and train RNN model
    input_dim = X_train.shape[2]
    rnn_model = RNNModel(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    
    print("Training RNN model...")
    rnn_model, _ = train_model(rnn_model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # 4. Evaluate model
    rnn_model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = rnn_model(X_batch)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Convert predictions to original scale
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Get original scale prices
    y_actual = scaler_y.inverse_transform(actuals)
    y_pred = scaler_y.inverse_transform(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_actual.flatten(), y_pred.flatten())
    
    # Calculate trading strategy results
    trading_results = implement_trading_strategy(y_actual.flatten(), y_pred.flatten())
    
    return {
        'actual': y_actual.flatten(),
        'predicted': y_pred.flatten(),
        'metrics': metrics,
        'trading_results': trading_results
    }

# Compare models
def compare_models(data_df, sequence_length=10, batch_size=32, num_epochs=50):
    """
    Compare all models.
    
    Parameters:
    - data_df: DataFrame with input features
    - sequence_length: Length of sequences
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Dict with results for all models
    """
    results = {}
    
    # Run WSAEs-LSTM
    print("\nRunning WSAEs-LSTM model...")
    results['WSAEs-LSTM'] = run_wsaes_lstm(data_df, sequence_length, batch_size, num_epochs)
    
    # Run WLSTM
    print("\nRunning WLSTM model...")
    results['WLSTM'] = run_wlstm(data_df, sequence_length, batch_size, num_epochs)
    
    # Run LSTM
    print("\nRunning LSTM model...")
    results['LSTM'] = run_lstm(data_df, sequence_length, batch_size, num_epochs)
    
    # Run RNN
    print("\nRunning RNN model...")
    results['RNN'] = run_rnn(data_df, sequence_length, batch_size, num_epochs)
    
    return results

# Function to display results
def display_results(results):
    """
    Display the results of the model comparison.
    
    Parameters:
    - results: Dict with results for all models
    """
    # 1. Display metrics
    metrics_df = pd.DataFrame({
        model: {
            'MAPE': results[model]['metrics']['MAPE'],
            'R': results[model]['metrics']['R'],
            'Theil_U': results[model]['metrics']['Theil_U']
        }
        for model in results
    }).T
    
    print("\nPredictive Accuracy Metrics:")
    print(metrics_df)
    print()
    
    # 2. Display trading results
    trading_df = pd.DataFrame({
        model: {
            'Annual Return (%)': results[model]['trading_results']['annual_return'] * 100,
            'Sharpe Ratio': results[model]['trading_results']['sharpe_ratio'],
            'Buy & Hold Return (%)': results[model]['trading_results']['buy_hold_returns'] * 100
        }
        for model in results
    }).T
    
    print("Trading Performance:")
    print(trading_df)
    
    # 3. Plot predictions vs actual
    plt.figure(figsize=(15, 10))
    
    # Plot for each model
    for i, model in enumerate(results.keys()):
        plt.subplot(2, 2, i+1)
        plt.plot(results[model]['actual'], label='Actual')
        plt.plot(results[model]['predicted'], label='Predicted')
        plt.title(f'{model} Predictions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    
    # 4. Plot cumulative returns
    plt.figure(figsize=(10, 6))
    
    for model in results.keys():
        plt.plot(results[model]['trading_results']['cumulative_returns'], label=model)
    
    # Also plot buy & hold returns
    actual_returns = np.diff(results['WSAEs-LSTM']['actual']) / results['WSAEs-LSTM']['actual'][:-1]
    buy_hold_cumulative = np.cumprod(1 + np.concatenate([[0], actual_returns])) - 1
    plt.plot(buy_hold_cumulative, label='Buy & Hold', linestyle='--')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    dates, prices = generate_stock_data(n_days=1000)
    
    # Create DataFrame with stock data
    stock_df = pd.DataFrame({
        'open': prices['open'],
        'high': prices['high'],
        'low': prices['low'],
        'close': prices['close'],
        'volume': prices['volume']
    }, index=dates)
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    data_with_indicators = calculate_technical_indicators(stock_df)
    
    # Generate macroeconomic data
    print("Generating macroeconomic data...")
    macro_df = generate_macro_data(dates)
    
    # Combine all data
    combined_df = pd.concat([data_with_indicators, macro_df], axis=1)
    
    # Drop NaN values (due to indicators calculation)
    combined_df = combined_df.dropna()
    
    # Select features to use
    features = [
        'open', 'high', 'low', 'close', 'volume',  # OHLCV
        'MA5', 'MA10', 'EMA20',                    # Moving Averages
        'MACD', 'CCI', 'ATR', 'BOLL_upper', 'BOLL_lower',  # Technical Indicators
        'ROC', 'MTM6', 'MTM12', 'SMI', 'WVAD',     # More Technical Indicators
        'usd_index', 'interest_rate'               # Macroeconomic variables
    ]
    
    feature_df = combined_df[features].copy()
    
    # Using fewer epochs and a smaller batch size for demonstration
    print("\nComparing models...")
    all_results = compare_models(
        feature_df, 
        sequence_length=10, 
        batch_size=32, 
        num_epochs=10  # Reduced for demonstration
    )
    
    # Display results
    display_results(all_results)