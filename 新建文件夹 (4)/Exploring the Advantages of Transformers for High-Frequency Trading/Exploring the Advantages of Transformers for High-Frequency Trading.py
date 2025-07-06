import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SpikingActivation(nn.Module):
    """
    Spiking activation function as described in the paper.
    """
    def __init__(self):
        super(SpikingActivation, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        return torch.where(x >= self.threshold, x, torch.zeros_like(x))

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LSTM(nn.Module):
    """
    LSTM model for time series forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.PReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out[:, -1, :])
        out = self.fc(out)
        
        return out

class HFformer(nn.Module):
    """
    HFformer model as described in the paper, combining Transformer encoder with linear decoder.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, use_pos_encoding=False):
        super(HFformer, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding (optional)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Replace ReLU with PReLU in the encoder
        for layer in self.transformer_encoder.layers:
            layer.linear1 = nn.Sequential(
                layer.linear1,
                SpikingActivation(),
                nn.PReLU()
            )
        
        # Linear decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.PReLU(),
            nn.Linear(dim_feedforward, output_dim)
        )
        
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the last sequence element for prediction
        x = x[:, -1, :]
        
        # Linear decoder
        x = self.decoder(x)
        
        return x

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.
    """
    def __init__(self, data, look_back, forecast_horizon):
        self.data = data
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.look_back - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        # Get features (input sequence)
        features = self.data[idx:idx+self.look_back]
        
        # Get target (log return after forecast_horizon steps)
        current_price = self.data[idx+self.look_back-1, -1]  # Assuming last column is price
        future_price = self.data[idx+self.look_back+self.forecast_horizon-1, -1]
        log_return = np.log(future_price / current_price)
        
        return torch.FloatTensor(features), torch.FloatTensor([log_return])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train a model and return the training history.
    """
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate R2 score
        r2 = r2_score(all_targets, all_outputs)
        val_r2_scores.append(r2)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val R2: {r2:.6f}')
    
    return train_losses, val_losses, val_r2_scores

def generate_simulated_data(num_samples=100000, num_levels=10):
    """
    Generate simulated cryptocurrency price and order book data.
    """
    # Generate initial price
    initial_price = 30000.0
    
    # Generate price time series with random walk
    prices = [initial_price]
    volatility = 0.001  # Daily volatility
    
    for _ in range(num_samples-1):
        # Add some autocorrelation and mean reversion to price changes
        prev_price = prices[-1]
        random_component = np.random.normal(0, volatility * prev_price)
        mean_reversion = 0.001 * (initial_price - prev_price)
        new_price = prev_price * (1 + random_component + mean_reversion)
        prices.append(new_price)
    
    # Generate order book data
    data = []
    
    for i in range(num_samples):
        # Current price
        current_price = prices[i]
        
        # Generate bid and ask levels
        spread = current_price * 0.0005  # 0.05% spread
        
        bid_prices = [current_price - spread/2]
        ask_prices = [current_price + spread/2]
        
        # Add more levels with increasing spreads
        for j in range(1, num_levels):
            # Add random variation to spreads
            bid_variation = np.random.uniform(0.8, 1.2) * j * spread
            ask_variation = np.random.uniform(0.8, 1.2) * j * spread
            
            bid_prices.append(current_price - spread/2 - bid_variation)
            ask_prices.append(current_price + spread/2 + ask_variation)
        
        # Generate quantities for each level
        bid_quantities = [np.random.exponential(2) for _ in range(num_levels)]
        ask_quantities = [np.random.exponential(2) for _ in range(num_levels)]
        
        # Add autocorrelation to quantities
        if i > 0:
            prev_bid_quantities = data[i-1][num_levels:2*num_levels]
            prev_ask_quantities = data[i-1][2*num_levels:3*num_levels]
            
            for j in range(num_levels):
                bid_quantities[j] = 0.7 * bid_quantities[j] + 0.3 * prev_bid_quantities[j]
                ask_quantities[j] = 0.7 * ask_quantities[j] + 0.3 * prev_ask_quantities[j]
        
        # Create row with all data
        row = bid_prices + bid_quantities + ask_prices + ask_quantities + [current_price]
        data.append(row)
    
    # Create feature names
    feature_names = []
    for i in range(num_levels):
        feature_names.append(f'bid_price_{i+1}')
    for i in range(num_levels):
        feature_names.append(f'bid_qty_{i+1}')
    for i in range(num_levels):
        feature_names.append(f'ask_price_{i+1}')
    for i in range(num_levels):
        feature_names.append(f'ask_qty_{i+1}')
    feature_names.append('price')
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    # Calculate weighted midprice
    df['weighted_midprice'] = (df['bid_qty_1'] * df['bid_price_1'] + df['ask_qty_1'] * df['ask_price_1']) / (df['bid_qty_1'] + df['ask_qty_1'])
    
    # Calculate historical log returns for different horizons
    for horizon in range(1, 31):
        df[f'log_return_{horizon}'] = np.log(df['price'].shift(-horizon) / df['price']).shift(1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data_for_training(df, look_back=100, forecast_horizon=1, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare data for training, validation, and testing.
    """
    # Select features (bid prices, bid quantities, ask prices, ask quantities, weighted midprice)
    features = df.drop(columns=[f'log_return_{i}' for i in range(1, 31)]).values
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Split data
    n_samples = len(normalized_features)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_data = normalized_features[:train_size]
    val_data = normalized_features[train_size:train_size+val_size]
    test_data = normalized_features[train_size+val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, look_back, forecast_horizon)
    val_dataset = TimeSeriesDataset(val_data, look_back, forecast_horizon)
    test_dataset = TimeSeriesDataset(test_data, look_back, forecast_horizon)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    return train_loader, val_loader, test_loader, scaler

def evaluate_model(model, test_loader):
    """
    Evaluate a model on the test set.
    """
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    # Calculate R2 score
    r2 = r2_score(all_targets, all_outputs)
    
    # Calculate classification metrics (buy/sell signals)
    true_signs = np.sign(all_targets)
    pred_signs = np.sign(all_outputs)
    
    correct_buys = np.sum((true_signs > 0) & (pred_signs > 0))
    total_buys = np.sum(true_signs > 0)
    
    correct_sells = np.sum((true_signs < 0) & (pred_signs < 0))
    total_sells = np.sum(true_signs < 0)
    
    buy_accuracy = correct_buys / total_buys if total_buys > 0 else 0
    sell_accuracy = correct_sells / total_sells if total_sells > 0 else 0
    overall_accuracy = (correct_buys + correct_sells) / len(true_signs)
    
    # Calculate weighted accuracy
    weighted_correct_buys = np.sum(np.abs(all_targets) * ((true_signs > 0) & (pred_signs > 0)))
    weighted_total_buys = np.sum(np.abs(all_targets) * (true_signs > 0))
    
    weighted_correct_sells = np.sum(np.abs(all_targets) * ((true_signs < 0) & (pred_signs < 0)))
    weighted_total_sells = np.sum(np.abs(all_targets) * (true_signs < 0))
    
    weighted_buy_accuracy = weighted_correct_buys / weighted_total_buys if weighted_total_buys > 0 else 0
    weighted_sell_accuracy = weighted_correct_sells / weighted_total_sells if weighted_total_sells > 0 else 0
    weighted_overall_accuracy = (weighted_correct_buys + weighted_correct_sells) / np.sum(np.abs(all_targets))
    
    metrics = {
        'r2': r2,
        'buy_accuracy': buy_accuracy,
        'sell_accuracy': sell_accuracy,
        'overall_accuracy': overall_accuracy,
        'weighted_buy_accuracy': weighted_buy_accuracy,
        'weighted_sell_accuracy': weighted_sell_accuracy,
        'weighted_overall_accuracy': weighted_overall_accuracy
    }
    
    return metrics, all_targets, all_outputs

def backtest_trading_strategy(model, test_loader, strategy='single', num_signals=1, trade_sizing=False, min_threshold=None):
    """
    Backtest a trading strategy using the model predictions.
    
    Parameters:
    -----------
    model: PyTorch model
        The trained model for prediction
    test_loader: DataLoader
        DataLoader for test data
    strategy: str
        'single' for single signal, 'multiple' for multiple signals
    num_signals: int
        Number of signals to use (if strategy is 'multiple')
    trade_sizing: bool
        Whether to use trade sizing based on signal strength
    min_threshold: float
        Minimum threshold for a trade to occur
    
    Returns:
    --------
    trades: list
        List of trade details
    cumulative_pnl: list
        Cumulative PnL over time
    """
    model.eval()
    trades = []
    cumulative_pnl = [0]
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Generate prediction
            outputs = model(features)
            
            # Get signals
            signals = outputs.cpu().numpy()
            true_values = targets.cpu().numpy()
            
            # Process each sample
            for i in range(len(signals)):
                signal = signals[i][0]
                true_value = true_values[i][0]
                
                # Determine whether to trade based on strategy
                trade_decision = False
                trade_size = 0.1  # Default trade size
                
                if strategy == 'single':
                    # Single signal strategy
                    trade_decision = True
                    trade_direction = 1 if signal > 0 else -1
                
                elif strategy == 'multiple' and num_signals > 1:
                    # Multiple signals strategy (simulated)
                    # In a real implementation, we would use predictions from multiple models
                    # Here we'll simulate by adding some noise to the original signal
                    multi_signals = []
                    for _ in range(num_signals):
                        # Add some noise to the original signal
                        noisy_signal = signal + np.random.normal(0, 0.02)
                        multi_signals.append(noisy_signal)
                    
                    # Only trade if all signals have the same sign
                    all_positive = all(s > 0 for s in multi_signals)
                    all_negative = all(s < 0 for s in multi_signals)
                    
                    if all_positive:
                        trade_decision = True
                        trade_direction = 1
                    elif all_negative:
                        trade_decision = True
                        trade_direction = -1
                    else:
                        trade_decision = False
                
                # Apply minimum threshold if specified
                if min_threshold is not None:
                    if abs(signal) < min_threshold:
                        trade_decision = False
                
                # Apply trade sizing if enabled
                if trade_sizing and trade_decision:
                    # Scale trade size based on signal strength
                    # This is a simple linear scaling; in practice, more sophisticated methods could be used
                    signal_strength = abs(signal)
                    if signal_strength > 0.2:
                        trade_size = 0.15
                    elif signal_strength > 0.1:
                        trade_size = 0.1
                    else:
                        trade_size = 0.05
                
                # Record trade and PnL
                if trade_decision:
                    pnl = trade_direction * true_value * trade_size
                    
                    # Subtract a small amount for slippage (0.0002%)
                    pnl -= 0.000002 * trade_size
                    
                    trades.append({
                        'signal': signal,
                        'direction': trade_direction,
                        'size': trade_size,
                        'true_value': true_value,
                        'pnl': pnl
                    })
                    
                    cumulative_pnl.append(cumulative_pnl[-1] + pnl)
                else:
                    # No trade
                    cumulative_pnl.append(cumulative_pnl[-1])
    
    return trades, cumulative_pnl

def plot_cumulative_pnl(pnl_results, title="Cumulative PnL"):
    """
    Plot cumulative PnL for different trading strategies.
    
    Parameters:
    -----------
    pnl_results: dict
        Dictionary with strategy names as keys and cumulative PnL lists as values
    title: str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, pnl in pnl_results.items():
        plt.plot(pnl, label=strategy_name)
    
    plt.title(title)
    plt.xlabel('Trades')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_signal_vs_pnl(trades, title="Signal vs PnL"):
    """
    Plot the relationship between signal strength and PnL.
    
    Parameters:
    -----------
    trades: list
        List of trade details
    title: str
        Plot title
    """
    signals = [trade['signal'] for trade in trades]
    pnls = [trade['pnl'] for trade in trades]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(signals, pnls, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('Signal')
    plt.ylabel('PnL')
    plt.grid(True)
    plt.show()

def main():
    # Generate simulated data
    print("Generating simulated data...")
    df = generate_simulated_data(num_samples=100000, num_levels=10)
    
    # Explore the data
    print("\nData overview:")
    print(df.head())
    
    # Prepare data for training
    print("\nPreparing data for training...")
    look_back = 100
    forecast_horizon = 28  # As used in the paper
    train_loader, val_loader, test_loader, scaler = prepare_data_for_training(
        df, look_back=look_back, forecast_horizon=forecast_horizon
    )
    
    # Define models
    input_dim = df.shape[1] - 30  # Excluding log returns
    
    # LSTM model
    lstm_model = LSTM(
        input_dim=input_dim,
        hidden_dim=16,
        num_layers=5,
        output_dim=1
    ).to(device)
    
    # HFformer model
    hfformer_model = HFformer(
        input_dim=input_dim,
        d_model=64,
        nhead=6,
        num_encoder_layers=1,
        dim_feedforward=64,
        output_dim=1,
        use_pos_encoding=False  # As per the paper's findings
    ).to(device)
    
    # Define loss function and optimizers
    criterion = nn.MSELoss()
    lstm_optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=0.001)
    hfformer_optimizer = torch.optim.AdamW(hfformer_model.parameters(), lr=0.04)
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_train_losses, lstm_val_losses, lstm_val_r2_scores = train_model(
        lstm_model, train_loader, val_loader, criterion, lstm_optimizer, num_epochs=50
    )
    
    # Train HFformer model
    print("\nTraining HFformer model...")
    hfformer_train_losses, hfformer_val_losses, hfformer_val_r2_scores = train_model(
        hfformer_model, train_loader, val_loader, criterion, hfformer_optimizer, num_epochs=50
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM Train')
    plt.plot(lstm_val_losses, label='LSTM Val')
    plt.plot(hfformer_train_losses, label='HFformer Train')
    plt.plot(hfformer_val_losses, label='HFformer Val')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_val_r2_scores, label='LSTM')
    plt.plot(hfformer_val_r2_scores, label='HFformer')
    plt.title('R² Score During Training')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate models on test set
    print("\nEvaluating models on test set...")
    lstm_metrics, lstm_targets, lstm_outputs = evaluate_model(lstm_model, test_loader)
    hfformer_metrics, hfformer_targets, hfformer_outputs = evaluate_model(hfformer_model, test_loader)
    
    print("\nLSTM Metrics:")
    for key, value in lstm_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nHFformer Metrics:")
    for key, value in hfformer_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot model predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.scatter(lstm_targets, lstm_outputs, alpha=0.3, label='LSTM')
    plt.scatter(hfformer_targets, hfformer_outputs, alpha=0.3, label='HFformer')
    plt.plot([-0.05, 0.05], [-0.05, 0.05], 'r--')
    plt.xlabel('Actual Log Return')
    plt.ylabel('Predicted Log Return')
    plt.title('Model Predictions vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Backtest trading strategies
    print("\nBacktesting trading strategies...")
    
    # Strategy 1: Single signal
    print("\nStrategy 1: Single signal")
    lstm_trades_1, lstm_pnl_1 = backtest_trading_strategy(lstm_model, test_loader, strategy='single')
    hfformer_trades_1, hfformer_pnl_1 = backtest_trading_strategy(hfformer_model, test_loader, strategy='single')
    
    # Strategy 2: Multiple signals (3)
    print("\nStrategy 2: Multiple signals (3)")
    lstm_trades_2, lstm_pnl_2 = backtest_trading_strategy(lstm_model, test_loader, strategy='multiple', num_signals=3)
    hfformer_trades_2, hfformer_pnl_2 = backtest_trading_strategy(hfformer_model, test_loader, strategy='multiple', num_signals=3)
    
    # Strategy 3: Multiple signals (5)
    print("\nStrategy 3: Multiple signals (5)")
    lstm_trades_3, lstm_pnl_3 = backtest_trading_strategy(lstm_model, test_loader, strategy='multiple', num_signals=5)
    hfformer_trades_3, hfformer_pnl_3 = backtest_trading_strategy(hfformer_model, test_loader, strategy='multiple', num_signals=5)
    
    # Strategy 4: Multiple signals (7) with trade sizing and minimum threshold
    print("\nStrategy 4: Multiple signals (7) with trade sizing and minimum threshold")
    lstm_trades_4, lstm_pnl_4 = backtest_trading_strategy(
        lstm_model, test_loader, strategy='multiple', num_signals=7, 
        trade_sizing=True, min_threshold=0.01
    )
    hfformer_trades_4, hfformer_pnl_4 = backtest_trading_strategy(
        hfformer_model, test_loader, strategy='multiple', num_signals=7, 
        trade_sizing=True, min_threshold=0.01
    )
    
    # Plot cumulative PnL for all strategies
    pnl_results = {
        'LSTM - Strategy 1': lstm_pnl_1,
        'HFformer - Strategy 1': hfformer_pnl_1,
        'LSTM - Strategy 2': lstm_pnl_2,
        'HFformer - Strategy 2': hfformer_pnl_2,
        'LSTM - Strategy 3': lstm_pnl_3,
        'HFformer - Strategy 3': hfformer_pnl_3,
        'LSTM - Strategy 4': lstm_pnl_4,
        'HFformer - Strategy 4': hfformer_pnl_4
    }
    
    plot_cumulative_pnl(pnl_results, "Cumulative PnL for Different Trading Strategies")
    
    # Print trading statistics
    print("\nTrading Statistics:")
    for strategy_name, trades_list, pnl_list in [
        ("LSTM - Strategy 1", lstm_trades_1, lstm_pnl_1),
        ("HFformer - Strategy 1", hfformer_trades_1, hfformer_pnl_1),
        ("LSTM - Strategy 2", lstm_trades_2, lstm_pnl_2),
        ("HFformer - Strategy 2", hfformer_trades_2, hfformer_pnl_2),
        ("LSTM - Strategy 3", lstm_trades_3, lstm_pnl_3),
        ("HFformer - Strategy 3", hfformer_trades_3, hfformer_pnl_3),
        ("LSTM - Strategy 4", lstm_trades_4, lstm_pnl_4),
        ("HFformer - Strategy 4", hfformer_trades_4, hfformer_pnl_4)
    ]:
        num_trades = len(trades_list)
        final_pnl = pnl_list[-1] if pnl_list else 0
        
        profitable_trades = sum(1 for trade in trades_list if trade['pnl'] > 0)
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        
        avg_profit = sum(trade['pnl'] for trade in trades_list if trade['pnl'] > 0) / profitable_trades if profitable_trades > 0 else 0
        avg_loss = sum(trade['pnl'] for trade in trades_list if trade['pnl'] <= 0) / (num_trades - profitable_trades) if (num_trades - profitable_trades) > 0 else 0
        
        print(f"\n{strategy_name}:")
        print(f"Number of trades: {num_trades}")
        print(f"Final PnL: {final_pnl:.4f}")
        print(f"Win rate: {win_rate:.4f}")
        print(f"Average profit: {avg_profit:.4f}")
        print(f"Average loss: {avg_loss:.4f}")
    
    # Plot relationship between signal and PnL for the best strategy
    plot_signal_vs_pnl(hfformer_trades_4, "HFformer Strategy 4: Signal vs PnL")

if __name__ == "__main__":
    main()