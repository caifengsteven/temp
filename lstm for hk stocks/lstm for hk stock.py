import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import talib
import random
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for stock data
class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Calculate attention weights
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # Apply attention weights to input
        context = attention_weights * x
        return context

# LSTM Model - Corrected for PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x, _ = self.lstm1(x)  # PyTorch LSTM returns output and hidden state
        x = self.dropout1(x)
        # We take the output of the last timestep for the second LSTM
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # Take the last timestep output
        x = self.fc(x)
        x = self.bn(x)
        return nn.functional.softmax(x, dim=1)

# Attention LSTM Model - Corrected for PyTorch
class AttentionLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=2):
        super(AttentionLSTMModel, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.attention(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # Take the last timestep output
        x = self.fc(x)
        x = self.bn(x)
        return nn.functional.softmax(x, dim=1)

# Function to generate simulated stock data with realistic features
def generate_simulated_stock_data(num_days=3000, seed=42):
    np.random.seed(seed)
    
    # Base parameters
    price = 100.0
    mu = 0.0001  # Drift (daily)
    sigma = 0.01  # Volatility (daily)
    
    # Generate basic price series
    returns = np.random.normal(mu, sigma, num_days)
    
    # Add regime shifts (bull/bear markets)
    num_regimes = 6  # Number of regime shifts
    regime_length = num_days // num_regimes
    regimes = []
    
    for i in range(num_regimes):
        if i % 2 == 0:  # Bull market
            adjustment = 0.001  # Higher drift
            vol_adjustment = 0.9  # Lower volatility
        else:  # Bear market
            adjustment = -0.001  # Lower drift
            vol_adjustment = 1.3  # Higher volatility
            
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, num_days)
        returns[start_idx:end_idx] += adjustment
        returns[start_idx:end_idx] *= vol_adjustment
        
        # Store regime labels (0 for bull, 1 for bear)
        regimes.extend([i % 2] * (end_idx - start_idx))
    
    # Add momentum effects
    momentum = np.zeros(num_days)
    momentum_strength = 0.2
    momentum_length = 10
    
    for i in range(momentum_length, num_days):
        momentum[i] = np.mean(returns[i-momentum_length:i]) * momentum_strength
        returns[i] += momentum[i]
    
    # Add some mean reversion
    for i in range(5, num_days):
        mean_reversion = -0.1 * np.mean(returns[i-5:i])
        returns[i] += mean_reversion
    
    # Add seasonal patterns
    for i in range(num_days):
        # Weekly pattern (higher returns mid-week)
        weekday = i % 5
        if weekday == 2:  # Wednesday
            returns[i] += 0.0005
        elif weekday == 4:  # Friday
            returns[i] -= 0.0003
        
        # Monthly pattern (higher returns at month start)
        day_of_month = i % 22
        if day_of_month < 3:
            returns[i] += 0.0007
    
    # Add occasional jumps (market shocks)
    jump_days = np.random.choice(range(num_days), size=int(num_days * 0.01), replace=False)
    jump_sizes = np.random.choice([-1, 1], size=len(jump_days)) * np.random.uniform(0.02, 0.05, size=len(jump_days))
    
    for i, day in enumerate(jump_days):
        returns[day] += jump_sizes[i]
    
    # Generate price series from returns
    prices = price * np.cumprod(1 + returns)
    
    # Generate volume data
    avg_volume = 1000000
    volume = np.random.lognormal(np.log(avg_volume), 0.3, num_days)
    
    # Make volume correlate with absolute returns (higher volume on big moves)
    volume = volume * (1 + 5 * np.abs(returns))
    
    # Generate OHLC data
    high = prices * (1 + np.random.uniform(0, 0.01, num_days))
    low = prices * (1 - np.random.uniform(0, 0.01, num_days))
    
    # Ensure high is always >= price and low is always <= price
    high = np.maximum(high, prices)
    low = np.minimum(low, prices)
    
    # Generate open price (yesterday's close with some adjustment)
    open_price = np.zeros(num_days)
    open_price[0] = price * (1 - 0.001)  # First day
    open_price[1:] = prices[:-1] * (1 + np.random.normal(0, 0.003, num_days-1))
    
    # Create date range
    start_date = dt.datetime(2010, 1, 1)
    dates = [start_date + dt.timedelta(days=i) for i in range(num_days)]
    
    # Keep only weekdays
    weekday_indices = [i for i, date in enumerate(dates) if date.weekday() < 5]
    dates = [dates[i] for i in weekday_indices]
    prices = prices[weekday_indices]
    returns = returns[weekday_indices]
    open_price = open_price[weekday_indices]
    high = high[weekday_indices]
    low = low[weekday_indices]
    volume = volume[weekday_indices]
    regimes = [regimes[i] for i in weekday_indices]
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates[:len(prices)],  # Truncate if needed
        'Close': prices,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Volume': volume,
        'Return': returns,
        'Regime': regimes[:len(prices)]  # 0 for bull, 1 for bear
    })
    
    data.set_index('Date', inplace=True)
    return data

# Calculate technical indicators
def calculate_technical_indicators(df):
    data = df.copy()
    
    # Extract price and volume data
    close = data['Close'].values
    open_price = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    volume = data['Volume'].values
    
    # Calculate ROCP (Rate of Change Percentage)
    data['ROCP'] = talib.ROC(close, timeperiod=1) / 100
    data['OROCP'] = talib.ROC(open_price, timeperiod=1) / 100
    data['HROCP'] = talib.ROC(high, timeperiod=1) / 100
    data['LROCP'] = talib.ROC(low, timeperiod=1) / 100
    
    # Calculate Moving Averages
    data['MA5'] = talib.MA(close, timeperiod=5)
    data['MA10'] = talib.MA(close, timeperiod=10)
    data['MA20'] = talib.MA(close, timeperiod=20)
    data['MA30'] = talib.MA(close, timeperiod=30)
    data['MA60'] = talib.MA(close, timeperiod=60)
    
    # Calculate MACP (Moving Average Change Percentage)
    data['MACP5'] = (data['MA5'] - close) / close
    data['MACP10'] = (data['MA10'] - close) / close
    data['MACP20'] = (data['MA20'] - close) / close
    data['MACP30'] = (data['MA30'] - close) / close
    data['MACP60'] = (data['MA60'] - close) / close
    
    # Calculate MACD
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACDsignal'] = macdsignal
    data['MACDhist'] = macdhist
    
    # Calculate DIF, DEA for MACD
    data['DIF'] = macd
    data['DEA'] = macdsignal
    
    # Calculate DIFROCP, DEAROCP, MACDROCP
    data['DIFROCP'] = talib.ROC(macd, timeperiod=1) / 100
    data['DEAROCP'] = talib.ROC(macdsignal, timeperiod=1) / 100
    data['MACDROCP'] = talib.ROC(macdhist, timeperiod=1) / 100
    
    # Calculate RSI
    data['RSI6'] = talib.RSI(close, timeperiod=6)
    data['RSI12'] = talib.RSI(close, timeperiod=12)
    data['RSI24'] = talib.RSI(close, timeperiod=24)
    
    # Calculate RSIROCP
    data['RSIROCP6'] = talib.ROC(data['RSI6'], timeperiod=1) / 100
    data['RSIROCP12'] = talib.ROC(data['RSI12'], timeperiod=1) / 100
    data['RSIROCP24'] = talib.ROC(data['RSI24'], timeperiod=1) / 100
    
    # Calculate VROCP (Volume Rate of Change Percentage)
    data['VROCP'] = np.arctan(talib.ROC(volume, timeperiod=1) / 100)
    
    # Calculate Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['BOLLupper'] = upperband
    data['BOLLmiddle'] = middleband
    data['BOLLlower'] = lowerband
    
    # Calculate VMAROCP (Volume Moving Average ROCP)
    volume_ma5 = talib.MA(volume, timeperiod=5)
    volume_ma10 = talib.MA(volume, timeperiod=10)
    data['VMA5ROCP'] = talib.ROC(volume_ma5, timeperiod=1) / 100
    data['VMA10ROCP'] = talib.ROC(volume_ma10, timeperiod=1) / 100
    
    # Calculate VMACP (Volume Moving Average Change Percentage)
    data['VMACP5'] = (volume_ma5 - volume) / volume
    data['VMACP10'] = (volume_ma10 - volume) / volume
    
    # Calculate PRICE_VOLUME = ROCP * VROCP
    data['PRICE_VOLUME'] = data['ROCP'] * data['VROCP']
    
    # Drop NaN values created by indicators requiring historical data
    data = data.dropna()
    
    return data

# Create sequences for LSTM input
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

# Train function
def train_model(model, train_loader, val_loader, epochs=1000, early_stopping_patience=10, verbose=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # For early stopping
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Print progress
        if verbose > 0 and (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch+1}, Best Val Acc: {best_val_acc:.4f}")
                model.load_state_dict(best_model_state)
                break
                
    return model, best_val_acc

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(targets.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds)
    return test_acc, test_preds, test_targets

# Function to train and evaluate models
def train_evaluate_models(data, stock_name, seq_length=10, test_size=700, batch_size=512, verbose=1):
    if verbose > 0:
        print(f"Processing {stock_name}")
    
    # Calculate technical indicators
    if verbose > 0:
        print(f"Calculating technical indicators for {stock_name}")
    data = calculate_technical_indicators(data)
    
    # Create target variable - direction of next day's price movement
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop the last row as it doesn't have a target
    data = data.iloc[:-1]
    
    # Split data
    test_data = data.iloc[-test_size:]
    train_data = data.iloc[:-test_size]
    
    # Further split train data into train and validation (70/30)
    train_size = int(0.7 * len(train_data))
    train_data, val_data = train_data.iloc[:train_size], train_data.iloc[train_size:]
    
    # Extract features and targets
    feature_columns = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Regime']]
    
    # Scale the features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_data[feature_columns])
    val_features = scaler.transform(val_data[feature_columns])
    test_features = scaler.transform(test_data[feature_columns])
    
    # Create sequences
    X_train, y_train = create_sequences(train_features, train_data['Target'].values, seq_length)
    X_val, y_val = create_sequences(val_features, val_data['Target'].values, seq_length)
    X_test, y_test = create_sequences(test_features, test_data['Target'].values, seq_length)
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create and train LSTM model
    if verbose > 0:
        print(f"Training LSTM model for {stock_name}")
    input_dim = X_train.shape[2]
    lstm_model = LSTMModel(input_dim).to(device)
    lstm_model, _ = train_model(lstm_model, train_loader, val_loader, verbose=verbose)
    
    # Create and train AttLSTM model
    if verbose > 0:
        print(f"Training AttLSTM model for {stock_name}")
    attlstm_model = AttentionLSTMModel(input_dim).to(device)
    attlstm_model, _ = train_model(attlstm_model, train_loader, val_loader, verbose=verbose)
    
    # Evaluate models
    lstm_acc, lstm_preds, _ = evaluate_model(lstm_model, test_loader)
    attlstm_acc, attlstm_preds, _ = evaluate_model(attlstm_model, test_loader)
    
    if verbose > 0:
        print(f"LSTM Accuracy: {lstm_acc:.4f}")
        print(f"AttLSTM Accuracy: {attlstm_acc:.4f}")
    
    return {
        'lstm_acc': lstm_acc,
        'attlstm_acc': attlstm_acc,
        'lstm_pred': lstm_preds,
        'attlstm_pred': attlstm_preds,
        'actual': y_test,
        'test_data': test_data
    }

# Tune model parameters
def tune_model_parameters(data, stock_name, input_sizes=[10, 20], batch_sizes=[512, 256, 128], learning_rates=[0.001, 0.1]):
    print(f"Tuning parameters for {stock_name}")
    
    best_params = {
        'lstm': {'input_size': None, 'batch_size': None, 'learning_rate': None, 'accuracy': 0},
        'attlstm': {'input_size': None, 'batch_size': None, 'learning_rate': None, 'accuracy': 0}
    }
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Create target variable - direction of next day's price movement
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop the last row as it doesn't have a target
    data = data.iloc[:-1]
    
    # Split data
    test_data = data.iloc[-700:]  # 700 days for test as in the paper
    train_data = data.iloc[:-700]
    
    for input_size in input_sizes:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                print(f"Testing parameters: input_size={input_size}, batch_size={batch_size}, learning_rate={lr}")
                
                try:
                    # Further split train data into train and validation (70/30)
                    train_size = int(0.7 * len(train_data))
                    train_subset, val_data = train_data.iloc[:train_size], train_data.iloc[train_size:]
                    
                    # Extract features and targets
                    feature_columns = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Regime']]
                    
                    # Scale the features
                    scaler = StandardScaler()
                    train_features = scaler.fit_transform(train_subset[feature_columns])
                    val_features = scaler.transform(val_data[feature_columns])
                    
                    # Create sequences
                    X_train, y_train = create_sequences(train_features, train_subset['Target'].values, input_size)
                    X_val, y_val = create_sequences(val_features, val_data['Target'].values, input_size)
                    
                    # Create datasets and dataloaders
                    train_dataset = StockDataset(X_train, y_train)
                    val_dataset = StockDataset(X_val, y_val)
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    
                    # Create LSTM model
                    input_dim = X_train.shape[2]
                    lstm_model = LSTMModel(input_dim).to(device)
                    
                    # Custom training for parameter tuning with specific learning rate
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
                    
                    # Train LSTM model for a few epochs
                    for epoch in range(30):  # Just 30 epochs for parameter tuning
                        lstm_model.train()
                        for features, targets in train_loader:
                            features, targets = features.to(device), targets.to(device)
                            
                            optimizer.zero_grad()
                            outputs = lstm_model(features)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                    
                    # Evaluate LSTM model
                    lstm_acc, _, _ = evaluate_model(lstm_model, val_loader)
                    
                    # Create AttLSTM model
                    attlstm_model = AttentionLSTMModel(input_dim).to(device)
                    optimizer = optim.Adam(attlstm_model.parameters(), lr=lr)
                    
                    # Train AttLSTM model for a few epochs
                    for epoch in range(30):  # Just 30 epochs for parameter tuning
                        attlstm_model.train()
                        for features, targets in train_loader:
                            features, targets = features.to(device), targets.to(device)
                            
                            optimizer.zero_grad()
                            outputs = attlstm_model(features)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                    
                    # Evaluate AttLSTM model
                    attlstm_acc, _, _ = evaluate_model(attlstm_model, val_loader)
                    
                    print(f"LSTM Accuracy: {lstm_acc:.4f}, AttLSTM Accuracy: {attlstm_acc:.4f}")
                    
                    # Update best parameters
                    if lstm_acc > best_params['lstm']['accuracy']:
                        best_params['lstm'] = {
                            'input_size': input_size,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'accuracy': lstm_acc
                        }
                    
                    if attlstm_acc > best_params['attlstm']['accuracy']:
                        best_params['attlstm'] = {
                            'input_size': input_size,
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'accuracy': attlstm_acc
                        }
                
                except Exception as e:
                    print(f"Error during parameter tuning: {e}")
    
    print("\nBest Parameters:")
    print(f"LSTM: {best_params['lstm']}")
    print(f"AttLSTM: {best_params['attlstm']}")
    
    return best_params

# Implement Long-Only trading strategy
def long_only_strategy(predictions_dict, market_data):
    print("\nImplementing Long-Only trading strategy...")
    
    # Initialize portfolios
    portfolio_lstm = 1.0
    portfolio_attlstm = 1.0
    portfolio_hold = 1.0
    
    # Create market index from average of all stocks
    market_index = pd.DataFrame()
    for stock_name, stock_data in market_data.items():
        if market_index.empty:
            market_index['Close'] = stock_data['Close'] / stock_data['Close'].iloc[0]
        else:
            market_index['Close'] += stock_data['Close'] / stock_data['Close'].iloc[0]
    
    market_index['Close'] /= len(market_data)
    
    # Get test data dates from the first stock
    first_stock = next(iter(predictions_dict.values()))
    test_data = first_stock['test_data']
    dates = test_data.index
    
    # Initialize dictionary for daily returns
    returns_data = pd.DataFrame(index=dates[:len(first_stock['lstm_pred'])])
    returns_data['LSTM'] = 1.0
    returns_data['AttLSTM'] = 1.0
    returns_data['HoldOnly'] = 1.0
    returns_data['Market'] = 1.0
    
    # For each day in the test period
    for day in range(len(first_stock['lstm_pred'])):
        # Lists to store stocks predicted to increase by each model
        lstm_increase_stocks = []
        attlstm_increase_stocks = []
        
        # Check each stock's prediction for the current day
        for stock_name, pred_data in predictions_dict.items():
            # LSTM prediction
            if day < len(pred_data['lstm_pred']) and pred_data['lstm_pred'][day] == 1:
                lstm_increase_stocks.append(stock_name)
            
            # AttLSTM prediction
            if day < len(pred_data['attlstm_pred']) and pred_data['attlstm_pred'][day] == 1:
                attlstm_increase_stocks.append(stock_name)
        
        # Calculate daily returns for each strategy
        lstm_daily_return = 0
        attlstm_daily_return = 0
        hold_daily_return = 0
        
        # LSTM strategy - invest equally in stocks predicted to increase
        if lstm_increase_stocks:
            for stock_name in lstm_increase_stocks:
                # Get the stock's actual return for the day
                test_data = predictions_dict[stock_name]['test_data']
                if day < len(test_data) - 1:
                    today_close = test_data.iloc[day]['Close']
                    tomorrow_close = test_data.iloc[day+1]['Close']
                    stock_return = tomorrow_close / today_close - 1
                    lstm_daily_return += stock_return / len(lstm_increase_stocks)
        
        # AttLSTM strategy - invest equally in stocks predicted to increase
        if attlstm_increase_stocks:
            for stock_name in attlstm_increase_stocks:
                # Get the stock's actual return for the day
                test_data = predictions_dict[stock_name]['test_data']
                if day < len(test_data) - 1:
                    today_close = test_data.iloc[day]['Close']
                    tomorrow_close = test_data.iloc[day+1]['Close']
                    stock_return = tomorrow_close / today_close - 1
                    attlstm_daily_return += stock_return / len(attlstm_increase_stocks)
        
        # Hold-Only strategy - invest equally in all stocks
        for stock_name in predictions_dict.keys():
            test_data = predictions_dict[stock_name]['test_data']
            if day < len(test_data) - 1:
                today_close = test_data.iloc[day]['Close']
                tomorrow_close = test_data.iloc[day+1]['Close']
                stock_return = tomorrow_close / today_close - 1
                hold_daily_return += stock_return / len(predictions_dict)
        
        # Get market return
        if day < len(market_index) - 1:
            market_return = market_index['Close'].iloc[day+1] / market_index['Close'].iloc[day] - 1
        else:
            market_return = 0
        
        # Update portfolio values
        portfolio_lstm *= (1 + lstm_daily_return)
        portfolio_attlstm *= (1 + attlstm_daily_return)
        portfolio_hold *= (1 + hold_daily_return)
        
        # Store daily values
        returns_data.iloc[day, 0] = portfolio_lstm
        returns_data.iloc[day, 1] = portfolio_attlstm
        returns_data.iloc[day, 2] = portfolio_hold
        returns_data.iloc[day, 3] = market_index['Close'].iloc[day] / market_index['Close'].iloc[0]
    
    # Plot performance
    plt.figure(figsize=(12, 6))
    plt.plot(returns_data.index, returns_data['LSTM'], label='LSTM Strategy')
    plt.plot(returns_data.index, returns_data['AttLSTM'], label='AttLSTM Strategy')
    plt.plot(returns_data.index, returns_data['HoldOnly'], label='Hold-Only Strategy')
    plt.plot(returns_data.index, returns_data['Market'], label='Market Index')
    
    plt.title('Cumulative Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Starting at 1.0)')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy_performance.png')
    plt.show()
    
    # Print final performance
    print("\nFinal Performance:")
    print(f"LSTM Strategy: {portfolio_lstm:.4f}")
    print(f"AttLSTM Strategy: {portfolio_attlstm:.4f}")
    print(f"Hold-Only Strategy: {portfolio_hold:.4f}")
    print(f"Market Index: {market_index['Close'].iloc[-1] / market_index['Close'].iloc[0]:.4f}")
    
    return returns_data

# Main function
def main():
    # Number of simulated stocks to generate
    num_stocks = 30
    
    # Generate simulated stocks
    print(f"Generating {num_stocks} simulated stocks...")
    stocks_data = {}
    for i in range(num_stocks):
        stock_name = f"Stock_{i:03d}"
        stocks_data[stock_name] = generate_simulated_stock_data(seed=i)
    
    # Plot a few sample stocks
    plt.figure(figsize=(15, 10))
    for i, (stock_name, stock_data) in enumerate(list(stocks_data.items())[:4]):
        plt.subplot(2, 2, i+1)
        plt.plot(stock_data.index, stock_data['Close'])
        plt.title(f"{stock_name} Price")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('sample_stocks.png')
    plt.show()
    
    # Plot regimes for a sample stock
    sample_stock = list(stocks_data.items())[0][1]
    plt.figure(figsize=(15, 6))
    plt.plot(sample_stock.index, sample_stock['Close'])
    
    # Color background based on regime
    regime_changes = []
    last_regime = sample_stock['Regime'].iloc[0]
    
    for i, regime in enumerate(sample_stock['Regime']):
        if i > 0 and regime != last_regime:
            regime_changes.append(i)
            last_regime = regime
    
    for i in regime_changes:
        plt.axvline(sample_stock.index[i], color='k', linestyle='--', alpha=0.5)
    
    # Shade regions based on regime
    last_change = 0
    last_regime = sample_stock['Regime'].iloc[0]
    
    for i in regime_changes:
        plt.axvspan(sample_stock.index[last_change], sample_stock.index[i],
                   color='green' if last_regime == 0 else 'red', alpha=0.2)
        last_change = i
        last_regime = sample_stock['Regime'].iloc[i]
    
    # Shade the final region
    plt.axvspan(sample_stock.index[last_change], sample_stock.index[-1],
               color='green' if last_regime == 0 else 'red', alpha=0.2)
    
    plt.title('Simulated Stock with Bull/Bear Regimes (Green=Bull, Red=Bear)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.savefig('stock_regimes.png')
    plt.show()
    
    # Dictionary to store predictions
    predictions_dict = {}
    
    # Process each stock
    for stock_name, stock_data in tqdm(stocks_data.items(), desc="Processing stocks"):
        result = train_evaluate_models(stock_data, stock_name, verbose=0)
        predictions_dict[stock_name] = result
    
    # Compare LSTM and AttLSTM accuracy across all stocks
    lstm_accs = [pred['lstm_acc'] for pred in predictions_dict.values()]
    attlstm_accs = [pred['attlstm_acc'] for pred in predictions_dict.values()]
    
    print("\nAccuracy Comparison:")
    print(f"Average LSTM Accuracy: {np.mean(lstm_accs):.4f}")
    print(f"Average AttLSTM Accuracy: {np.mean(attlstm_accs):.4f}")
    
    # Count number of stocks where AttLSTM outperforms LSTM
    attlstm_wins = sum(1 for lstm_acc, attlstm_acc in zip(lstm_accs, attlstm_accs) if attlstm_acc >= lstm_acc)
    print(f"AttLSTM outperforms LSTM in {attlstm_wins}/{num_stocks} stocks ({attlstm_wins/num_stocks*100:.2f}%)")
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.scatter(range(num_stocks), lstm_accs, label='LSTM', alpha=0.7)
    plt.scatter(range(num_stocks), attlstm_accs, label='AttLSTM', alpha=0.7)
    plt.plot(range(num_stocks), [np.mean(lstm_accs)]*num_stocks, 'b--', label=f'LSTM Avg: {np.mean(lstm_accs):.4f}')
    plt.plot(range(num_stocks), [np.mean(attlstm_accs)]*num_stocks, 'r--', label=f'AttLSTM Avg: {np.mean(attlstm_accs):.4f}')
    plt.xlabel('Stock Index')
    plt.ylabel('Prediction Accuracy')
    plt.title('LSTM vs AttLSTM Prediction Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    plt.show()
    
    # Implement trading strategy
    performance = long_only_strategy(predictions_dict, stocks_data)
    
    # Parameter tuning for a specific stock
    print("\nTuning parameters for a sample stock...")
    sample_stock_name = list(stocks_data.keys())[0]
    best_params = tune_model_parameters(stocks_data[sample_stock_name], sample_stock_name)
    
    # Train models with best parameters
    print("\nTraining with best parameters...")
    sample_stock_data = stocks_data[sample_stock_name]
    
    # For LSTM
    best_input_size = best_params['lstm']['input_size']
    best_batch_size = best_params['lstm']['batch_size']
    best_lr = best_params['lstm']['learning_rate']
    
    print(f"Best LSTM parameters: input_size={best_input_size}, batch_size={best_batch_size}, learning_rate={best_lr}")
    print(f"Best AttLSTM parameters: input_size={best_params['attlstm']['input_size']}, batch_size={best_params['attlstm']['batch_size']}, learning_rate={best_params['attlstm']['learning_rate']}")
    
    # Train with best parameters and evaluate
    result_best = train_evaluate_models(sample_stock_data, sample_stock_name, 
                                       seq_length=best_input_size, 
                                       batch_size=best_batch_size)
    
    print(f"\nAccuracy with best parameters for {sample_stock_name}:")
    print(f"LSTM Accuracy: {result_best['lstm_acc']:.4f}")
    print(f"AttLSTM Accuracy: {result_best['attlstm_acc']:.4f}")

if __name__ == "__main__":
    main()