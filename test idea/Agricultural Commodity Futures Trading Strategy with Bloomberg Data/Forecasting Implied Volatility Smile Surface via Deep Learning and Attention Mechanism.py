import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from tqdm import tqdm
from datetime import datetime, timedelta
import math

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic implied volatility surface data
def generate_volatility_surface(n_days=2520, n_maturities=5, n_moneyness=9, 
                                base_vol=0.2, vol_of_vol=0.05, smile_factor=0.03, 
                                mean_reversion=0.85, noise_level=0.01):
    """
    Generate synthetic implied volatility surface data
    
    Parameters:
    n_days: Number of days to simulate
    n_maturities: Number of option maturities
    n_moneyness: Number of moneyness levels
    base_vol: Base volatility level
    vol_of_vol: Volatility of volatility
    smile_factor: How pronounced the smile is
    mean_reversion: Mean reversion factor for base volatility
    noise_level: Random noise level
    
    Returns:
    vol_surfaces: A time series of volatility surfaces (days, maturities, moneyness)
    index_prices: Series of underlying index prices
    """
    # Define maturities (in days) and moneyness levels
    maturities = np.array([30, 60, 90, 180, 360])  # Days to expiration
    moneyness = np.array([0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.20])  # K/S
    
    # Initialize surfaces and index price
    vol_surfaces = np.zeros((n_days, n_maturities, n_moneyness))
    index_prices = np.zeros(n_days)
    
    # Generate a base volatility process (mean-reverting)
    base_volatility = np.zeros(n_days)
    base_volatility[0] = base_vol
    
    # Index price starts at 100
    index_prices[0] = 100
    
    # Simulate base volatility and index price
    for t in range(1, n_days):
        # Mean-reverting volatility process
        dv = mean_reversion * (base_vol - base_volatility[t-1]) + vol_of_vol * np.random.randn()
        base_volatility[t] = max(0.05, base_volatility[t-1] + dv)
        
        # Index price follows a geometric Brownian motion
        daily_return = np.random.normal(0, base_volatility[t-1]/np.sqrt(252))
        index_prices[t] = index_prices[t-1] * np.exp(daily_return)
    
    # Generate volatility surfaces
    for t in range(n_days):
        for i, maturity in enumerate(maturities):
            for j, money in enumerate(moneyness):
                # Create smile effect (higher vol for OTM options)
                smile = smile_factor * (money - 1.0)**2
                
                # Time effect (longer maturities have less pronounced smile)
                time_effect = np.sqrt(maturity/360)
                
                # Combine effects
                vol = base_volatility[t] + smile/time_effect + noise_level * np.random.randn()
                vol_surfaces[t, i, j] = max(0.05, vol)  # Ensure positive volatility
    
    return vol_surfaces, index_prices, maturities, moneyness

# Black-Scholes Option Pricing Functions
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return S * norm_cdf(d1_val) - K * np.exp(-r * T) * norm_cdf(d2_val)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(0, K - S)
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm_cdf(-d2_val) - S * norm_cdf(-d1_val)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Create a PyTorch dataset for volatility surface prediction
class VolatilitySurfaceDataset(Dataset):
    def __init__(self, vol_surfaces, lookback=3):
        """
        Dataset for volatility surface prediction
        
        Parameters:
        vol_surfaces: Time series of volatility surfaces (days, maturities, moneyness)
        lookback: Number of days to look back for prediction
        """
        self.vol_surfaces = torch.tensor(vol_surfaces, dtype=torch.float32)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.vol_surfaces) - self.lookback
    
    def __getitem__(self, idx):
        # Get daily, weekly, and monthly features as described in the paper
        daily_x = self.vol_surfaces[idx + self.lookback - 1]
        
        # Weekly average (last 5 days)
        weekly_start = max(0, idx + self.lookback - 5)
        weekly_x = torch.mean(self.vol_surfaces[weekly_start:idx + self.lookback], dim=0)
        
        # Monthly average (last 22 days)
        monthly_start = max(0, idx + self.lookback - 22)
        monthly_x = torch.mean(self.vol_surfaces[monthly_start:idx + self.lookback], dim=0)
        
        # Stack features
        features = torch.stack([daily_x, weekly_x, monthly_x], dim=0)
        
        # Target is the next day's volatility surface
        target = self.vol_surfaces[idx + self.lookback]
        
        return features, target

# LSTM with Attention mechanism
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # First LSTM layer processes the input sequences
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Second LSTM layer processes the attended output
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Dropout for regularization
        self.dropout2 = nn.Dropout(dropout)
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_maturities, n_moneyness)
        batch_size, seq_len, n_maturities, n_moneyness = x.size()
        
        # Reshape for LSTM
        x_reshaped = x.view(batch_size, seq_len, n_maturities * n_moneyness)
        
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x_reshaped)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Attention mechanism
        attention_weights = self.attention(lstm1_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm1_out, dim=1).unsqueeze(1)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(context_vector)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Final fully connected layer
        output = self.fc(lstm2_out.squeeze(1))
        
        # Reshape back to original form
        output = output.view(batch_size, n_maturities, n_moneyness)
        
        return output

# Basic LSTM model (without attention) for comparison
class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(BasicLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_maturities, n_moneyness)
        batch_size, seq_len, n_maturities, n_moneyness = x.size()
        
        # Reshape for LSTM
        x_reshaped = x.view(batch_size, seq_len, n_maturities * n_moneyness)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x_reshaped)
        
        # We only need the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer
        output = self.fc(lstm_out)
        
        # Reshape back to original form
        output = output.view(batch_size, n_maturities, n_moneyness)
        
        return output

# Multi-Layer Perceptron model for comparison
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(MLP, self).__init__()
        
        # Calculate input size based on sequence length * feature dimensions
        self.input_size = 3 * input_dim  # 3 time steps (daily, weekly, monthly)
        
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_maturities, n_moneyness)
        batch_size, seq_len, n_maturities, n_moneyness = x.size()
        
        # Flatten the input: combine sequence and features
        x_reshaped = x.view(batch_size, seq_len * n_maturities * n_moneyness)
        
        # Forward pass
        x = self.relu(self.fc1(x_reshaped))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Reshape back to original form (without sequence dimension)
        output = x.view(batch_size, n_maturities, n_moneyness)
        
        return output

# Function to train models
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, 
                patience=10, scheduler=None, device=None):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

# Function to evaluate models
def evaluate_model(model, test_loader, device):
    model.eval()
    mse = 0.0
    mae = 0.0
    qlike = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Store for later
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            
            # Calculate metrics
            mse += ((outputs - targets) ** 2).mean().item()
            mae += torch.abs(outputs - targets).mean().item()
            # QLIKE = 𝑙𝑜𝑔(𝜎̂) + 𝜎/𝜎̂ where 𝜎̂ is the prediction and 𝜎 is the target
            epsilon = 1e-8  # Small epsilon to prevent division by zero
            qlike += (torch.log(outputs + epsilon) + targets / (outputs + epsilon)).mean().item()
    
    mse /= len(test_loader)
    mae /= len(test_loader)
    qlike /= len(test_loader)
    
    all_targets = np.concatenate([x.reshape(x.shape[0], -1) for x in all_targets])
    all_predictions = np.concatenate([x.reshape(x.shape[0], -1) for x in all_predictions])
    
    return {
        'MSE': mse,
        'MAE': mae,
        'QLIKE': qlike
    }, all_targets, all_predictions

# Calendar (Time) Spread Strategy
def calendar_spread_strategy(vol_surface_pred, vol_surface_actual, index_prices, maturities, moneyness, 
                            risk_free_rate=0.02, days_to_hold=30, take_profit=0.05, stop_loss=-0.03):
    """
    Implement a calendar spread strategy based on predicted volatility surface
    
    Parameters:
    vol_surface_pred: Predicted volatility surfaces
    vol_surface_actual: Actual volatility surfaces
    index_prices: Underlying index prices
    maturities: Option maturities
    moneyness: Moneyness levels
    risk_free_rate: Risk-free interest rate
    days_to_hold: Number of days to hold each spread
    take_profit: Take profit threshold
    stop_loss: Stop loss threshold
    
    Returns:
    pnl_series: Profit and loss series
    """
    n_days = len(index_prices) - 1  # Skip the first day as we need a prediction
    pnl_series = np.zeros(n_days)
    positions = []  # Track open positions
    
    for day in range(n_days):
        current_price = index_prices[day]
        
        # Close positions that have reached take profit, stop loss, or expired
        for pos in positions[:]:  # Use a copy to safely remove elements during iteration
            entry_day, entry_price, short_maturity, long_maturity, strike, entry_cost = pos
            days_in_position = day - entry_day
            
            # Calculate current value of the spread
            short_days_left = max(0, short_maturity - days_in_position)
            long_days_left = max(0, long_maturity - days_in_position)
            
            # Get volatility for pricing
            short_vol_idx = np.argmin(np.abs(maturities - short_days_left))
            long_vol_idx = np.argmin(np.abs(maturities - long_days_left))
            money_idx = np.argmin(np.abs(moneyness - (strike / current_price)))
            
            # Use the actual volatility surface
            short_vol = vol_surface_actual[day, short_vol_idx, money_idx]
            long_vol = vol_surface_actual[day, long_vol_idx, money_idx]
            
            # Price the options
            short_call_price = bs_call(current_price, strike, short_days_left/365, risk_free_rate, short_vol)
            long_call_price = bs_call(current_price, strike, long_days_left/365, risk_free_rate, long_vol)
            
            # Current value of the spread
            current_value = long_call_price - short_call_price
            
            # Calculate P&L
            pnl = current_value - entry_cost
            
            # Check if we should close
            if (days_in_position >= days_to_hold or 
                pnl / abs(entry_cost) >= take_profit or 
                pnl / abs(entry_cost) <= stop_loss or
                short_days_left == 0):
                
                positions.remove(pos)
                pnl_series[day] += pnl
        
        # Open new positions if our prediction shows opportunity
        if day % 5 == 0:  # Only check for new positions every 5 days to reduce turnover
            # Check different strike prices for potential spreads
            for money_idx, money in enumerate(moneyness):
                strike = current_price * money
                
                # Select option maturities
                short_mat_idx = 0  # 30-day option
                long_mat_idx = 2   # 90-day option
                
                # Make sure we have enough time difference
                if maturities[long_mat_idx] - maturities[short_mat_idx] < 30:
                    continue
                
                # Get volatilities for pricing
                short_vol_pred = vol_surface_pred[day, short_mat_idx, money_idx]
                long_vol_pred = vol_surface_pred[day, long_mat_idx, money_idx]
                
                # Use actual volatility for real pricing
                short_vol_actual = vol_surface_actual[day, short_mat_idx, money_idx]
                long_vol_actual = vol_surface_actual[day, long_mat_idx, money_idx]
                
                # Price the options with predicted volatility
                short_call_pred = bs_call(current_price, strike, maturities[short_mat_idx]/365, 
                                         risk_free_rate, short_vol_pred)
                long_call_pred = bs_call(current_price, strike, maturities[long_mat_idx]/365, 
                                        risk_free_rate, long_vol_pred)
                
                # Price the options with actual volatility
                short_call_actual = bs_call(current_price, strike, maturities[short_mat_idx]/365, 
                                          risk_free_rate, short_vol_actual)
                long_call_actual = bs_call(current_price, strike, maturities[long_mat_idx]/365, 
                                         risk_free_rate, long_vol_actual)
                
                # Calculate spread value
                spread_pred = long_call_pred - short_call_pred
                spread_actual = long_call_actual - short_call_actual
                
                # Check if there's a significant difference that indicates opportunity
                if spread_pred > spread_actual * 1.1:  # If predicted spread is 10% higher, it's cheap now
                    # Enter a long calendar spread position
                    entry_cost = spread_actual
                    positions.append((day, current_price, maturities[short_mat_idx], 
                                     maturities[long_mat_idx], strike, entry_cost))
    
    return pnl_series

# Butterfly Spread Strategy
def butterfly_spread_strategy(vol_surface_pred, vol_surface_actual, index_prices, maturities, moneyness,
                             risk_free_rate=0.02, days_to_hold=30, take_profit=0.05, stop_loss=-0.03):
    """
    Implement a butterfly spread strategy based on predicted volatility surface
    
    Parameters:
    vol_surface_pred: Predicted volatility surfaces
    vol_surface_actual: Actual volatility surfaces
    index_prices: Underlying index prices
    maturities: Option maturities
    moneyness: Moneyness levels
    risk_free_rate: Risk-free interest rate
    days_to_hold: Number of days to hold each spread
    take_profit: Take profit threshold
    stop_loss: Stop loss threshold
    
    Returns:
    pnl_series: Profit and loss series
    """
    n_days = len(index_prices) - 1  # Skip the first day as we need a prediction
    pnl_series = np.zeros(n_days)
    positions = []  # Track open positions
    
    for day in range(n_days):
        current_price = index_prices[day]
        
        # Close positions that have reached take profit, stop loss, or expired
        for pos in positions[:]:  # Use a copy to safely remove elements during iteration
            entry_day, entry_price, maturity, strikes, entry_cost, is_long = pos
            days_in_position = day - entry_day
            days_left = max(0, maturity - days_in_position)
            
            # Get volatilities for pricing
            mat_idx = np.argmin(np.abs(maturities - days_left))
            money_idxs = [np.argmin(np.abs(moneyness - (strike / current_price))) for strike in strikes]
            
            # Use the actual volatility surface
            vols = [vol_surface_actual[day, mat_idx, mid] for mid in money_idxs]
            
            # Price the options
            call_prices = [bs_call(current_price, strike, days_left/365, risk_free_rate, vol) 
                          for strike, vol in zip(strikes, vols)]
            
            # Calculate current spread value
            current_value = call_prices[0] - 2 * call_prices[1] + call_prices[2]
            if not is_long:
                current_value = -current_value
            
            # Calculate P&L
            pnl = current_value - entry_cost
            
            # Check if we should close
            if (days_in_position >= days_to_hold or 
                pnl / abs(entry_cost) >= take_profit or 
                pnl / abs(entry_cost) <= stop_loss or
                days_left == 0):
                
                positions.remove(pos)
                pnl_series[day] += pnl
        
        # Open new positions if our prediction shows opportunity
        if day % 5 == 0:  # Only check for new positions every 5 days
            # Check for butterfly spreads at different maturities
            for mat_idx, maturity in enumerate(maturities):
                # We need at least K2 = (K1 + K3)/2 for butterfly
                for i in range(len(moneyness) - 2):
                    # Define strikes for the butterfly
                    k1 = current_price * moneyness[i]
                    k3 = current_price * moneyness[i+2]
                    k2 = (k1 + k3) / 2
                    
                    # Find closest moneyness for k2
                    k2_money = k2 / current_price
                    j = np.argmin(np.abs(moneyness - k2_money))
                    
                    # Get volatilities for pricing
                    vol1_pred = vol_surface_pred[day, mat_idx, i]
                    vol2_pred = vol_surface_pred[day, mat_idx, j]
                    vol3_pred = vol_surface_pred[day, mat_idx, i+2]
                    
                    # Use actual volatility for real pricing
                    vol1_actual = vol_surface_actual[day, mat_idx, i]
                    vol2_actual = vol_surface_actual[day, mat_idx, j]
                    vol3_actual = vol_surface_actual[day, mat_idx, i+2]
                    
                    # Price the options with predicted volatility
                    call1_pred = bs_call(current_price, k1, maturity/365, risk_free_rate, vol1_pred)
                    call2_pred = bs_call(current_price, k2, maturity/365, risk_free_rate, vol2_pred)
                    call3_pred = bs_call(current_price, k3, maturity/365, risk_free_rate, vol3_pred)
                    
                    # Price the options with actual volatility
                    call1_actual = bs_call(current_price, k1, maturity/365, risk_free_rate, vol1_actual)
                    call2_actual = bs_call(current_price, k2, maturity/365, risk_free_rate, vol2_actual)
                    call3_actual = bs_call(current_price, k3, maturity/365, risk_free_rate, vol3_actual)
                    
                    # Calculate butterfly spread values
                    spread_pred = call1_pred - 2 * call2_pred + call3_pred
                    spread_actual = call1_actual - 2 * call2_actual + call3_actual
                    
                    # Check for opportunities
                    if spread_pred > spread_actual * 1.2:  # If predicted spread is 20% higher
                        # Long butterfly (buy wings, sell body)
                        is_long = True
                        entry_cost = spread_actual
                        positions.append((day, current_price, maturity, [k1, k2, k3], entry_cost, is_long))
                    
                    elif spread_pred < spread_actual * 0.8:  # If predicted spread is 20% lower
                        # Short butterfly (sell wings, buy body)
                        is_long = False
                        entry_cost = -spread_actual
                        positions.append((day, current_price, maturity, [k1, k2, k3], entry_cost, is_long))
    
    return pnl_series

# Backtesting function to compare strategies
def backtest_strategies(vol_surface_pred, vol_surface_actual, index_prices, maturities, moneyness):
    # Calendar spread strategy
    calendar_pnl = calendar_spread_strategy(vol_surface_pred, vol_surface_actual, 
                                           index_prices, maturities, moneyness)
    
    # Butterfly spread strategy
    butterfly_pnl = butterfly_spread_strategy(vol_surface_pred, vol_surface_actual, 
                                             index_prices, maturities, moneyness)
    
    # Calculate cumulative PnL and other metrics
    calendar_cumulative = np.cumsum(calendar_pnl)
    butterfly_cumulative = np.cumsum(butterfly_pnl)
    
    # Calculate returns assuming initial capital of 100
    initial_capital = 100
    calendar_returns = calendar_cumulative / initial_capital
    butterfly_returns = butterfly_cumulative / initial_capital
    
    # Calculate Sharpe ratio (annualized)
    calendar_sharpe = np.mean(calendar_pnl) / (np.std(calendar_pnl) + 1e-8) * np.sqrt(252)
    butterfly_sharpe = np.mean(butterfly_pnl) / (np.std(butterfly_pnl) + 1e-8) * np.sqrt(252)
    
    # Calculate drawdowns
    calendar_dd = np.zeros_like(calendar_cumulative)
    butterfly_dd = np.zeros_like(butterfly_cumulative)
    
    peak_calendar = 0
    peak_butterfly = 0
    
    for i in range(len(calendar_cumulative)):
        if calendar_cumulative[i] > peak_calendar:
            peak_calendar = calendar_cumulative[i]
        if butterfly_cumulative[i] > peak_butterfly:
            peak_butterfly = butterfly_cumulative[i]
        
        calendar_dd[i] = (calendar_cumulative[i] - peak_calendar) / (peak_calendar + 1e-8)
        butterfly_dd[i] = (butterfly_cumulative[i] - peak_butterfly) / (peak_butterfly + 1e-8)
    
    max_calendar_dd = np.min(calendar_dd)
    max_butterfly_dd = np.min(butterfly_dd)
    
    # Create a results dictionary
    results = {
        'calendar_cumulative': calendar_cumulative,
        'butterfly_cumulative': butterfly_cumulative,
        'calendar_returns': calendar_returns,
        'butterfly_returns': butterfly_returns,
        'calendar_sharpe': calendar_sharpe,
        'butterfly_sharpe': butterfly_sharpe,
        'calendar_max_dd': max_calendar_dd,
        'butterfly_max_dd': max_butterfly_dd
    }
    
    return results

# Compare models for volatility surface prediction
def compare_models(vol_surfaces, train_size=0.6, val_size=0.2):
    """
    Compare different models for volatility surface prediction
    
    Parameters:
    vol_surfaces: Volatility surfaces data
    train_size: Proportion of data for training
    val_size: Proportion of data for validation
    
    Returns:
    results: Dictionary with model comparison results
    """
    n_days, n_maturities, n_moneyness = vol_surfaces.shape
    
    # Split data into train, validation, and test sets
    train_end = int(n_days * train_size)
    val_end = train_end + int(n_days * val_size)
    
    train_data = vol_surfaces[:train_end]
    val_data = vol_surfaces[train_end:val_end]
    test_data = vol_surfaces[val_end:]
    
    # Create datasets
    train_dataset = VolatilitySurfaceDataset(train_data)
    val_dataset = VolatilitySurfaceDataset(val_data)
    test_dataset = VolatilitySurfaceDataset(test_data)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model dimensions
    input_dim = n_maturities * n_moneyness
    hidden_dim = 135  # As used in the paper
    
    # Initialize models
    att_lstm = AttentionLSTM(input_dim, hidden_dim).to(device)
    basic_lstm = BasicLSTM(input_dim, hidden_dim).to(device)
    mlp = MLP(input_dim, hidden_dim).to(device)
    
    # Define loss function and optimizers
    criterion = nn.MSELoss()
    att_lstm_optimizer = optim.RMSprop(att_lstm.parameters(), lr=0.001)
    basic_lstm_optimizer = optim.RMSprop(basic_lstm.parameters(), lr=0.001)
    mlp_optimizer = optim.RMSprop(mlp.parameters(), lr=0.001)
    
    # Define schedulers
    att_lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(att_lstm_optimizer, patience=5, factor=0.5)
    basic_lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(basic_lstm_optimizer, patience=5, factor=0.5)
    mlp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(mlp_optimizer, patience=5, factor=0.5)
    
    # Train models
    print("Training Attention LSTM model...")
    att_lstm, att_lstm_history = train_model(att_lstm, train_loader, val_loader, criterion, 
                                           att_lstm_optimizer, num_epochs=50, patience=10, 
                                           scheduler=att_lstm_scheduler, device=device)
    
    print("\nTraining Basic LSTM model...")
    basic_lstm, basic_lstm_history = train_model(basic_lstm, train_loader, val_loader, criterion, 
                                               basic_lstm_optimizer, num_epochs=50, patience=10, 
                                               scheduler=basic_lstm_scheduler, device=device)
    
    print("\nTraining MLP model...")
    mlp, mlp_history = train_model(mlp, train_loader, val_loader, criterion, 
                                 mlp_optimizer, num_epochs=50, patience=10, 
                                 scheduler=mlp_scheduler, device=device)
    
    # Evaluate models
    print("\nEvaluating models...")
    att_lstm_metrics, att_lstm_targets, att_lstm_preds = evaluate_model(att_lstm, test_loader, device)
    basic_lstm_metrics, basic_lstm_targets, basic_lstm_preds = evaluate_model(basic_lstm, test_loader, device)
    mlp_metrics, mlp_targets, mlp_preds = evaluate_model(mlp, test_loader, device)
    
    # Print results
    print("\nModel comparison:")
    print(f"Attention LSTM - MSE: {att_lstm_metrics['MSE']:.6f}, MAE: {att_lstm_metrics['MAE']:.6f}, QLIKE: {att_lstm_metrics['QLIKE']:.6f}")
    print(f"Basic LSTM - MSE: {basic_lstm_metrics['MSE']:.6f}, MAE: {basic_lstm_metrics['MAE']:.6f}, QLIKE: {basic_lstm_metrics['QLIKE']:.6f}")
    print(f"MLP - MSE: {mlp_metrics['MSE']:.6f}, MAE: {mlp_metrics['MAE']:.6f}, QLIKE: {mlp_metrics['QLIKE']:.6f}")
    
    # Plot training histories
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(att_lstm_history['train_losses'], label='Att-LSTM Train')
    plt.plot(att_lstm_history['val_losses'], label='Att-LSTM Val')
    plt.plot(basic_lstm_history['train_losses'], label='LSTM Train')
    plt.plot(basic_lstm_history['val_losses'], label='LSTM Val')
    plt.plot(mlp_history['train_losses'], label='MLP Train')
    plt.plot(mlp_history['val_losses'], label='MLP Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(['Att-LSTM', 'LSTM', 'MLP'], 
           [att_lstm_metrics['MSE'], basic_lstm_metrics['MSE'], mlp_metrics['MSE']])
    plt.ylabel('MSE')
    plt.title('Model Test MSE Comparison')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create result dictionary for return
    results = {
        'models': {
            'Att-LSTM': att_lstm,
            'LSTM': basic_lstm,
            'MLP': mlp
        },
        'metrics': {
            'Att-LSTM': att_lstm_metrics,
            'LSTM': basic_lstm_metrics,
            'MLP': mlp_metrics
        },
        'predictions': {
            'Att-LSTM': att_lstm_preds,
            'LSTM': basic_lstm_preds,
            'MLP': mlp_preds
        },
        'targets': att_lstm_targets,
        'history': {
            'Att-LSTM': att_lstm_history,
            'LSTM': basic_lstm_history,
            'MLP': mlp_history
        }
    }
    
    return results

# Main function to run the entire experiment
def main():
    # Generate synthetic data
    print("Generating synthetic volatility surface data...")
    vol_surfaces, index_prices, maturities, moneyness = generate_volatility_surface(
        n_days=1000,  # Use fewer days for faster runtime
        n_maturities=5, 
        n_moneyness=9
    )
    
    # Visualize a sample volatility surface
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(maturities, moneyness)
    ax.plot_surface(X, Y, vol_surfaces[500].T, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Maturity (days)')
    ax.set_ylabel('Moneyness (K/S)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Sample Implied Volatility Surface')
    plt.show()
    
    # Compare models for volatility surface prediction
    print("\nComparing models for volatility surface prediction...")
    model_results = compare_models(vol_surfaces)
    
    # Create rolling predictions for backtesting
    print("\nCreating rolling predictions for backtesting...")
    train_size = int(0.6 * len(vol_surfaces))
    val_size = int(0.2 * len(vol_surfaces))
    test_size = len(vol_surfaces) - train_size - val_size
    test_start = train_size + val_size
    
    # Generate predictions for the test period
    att_lstm_preds = np.zeros((test_size, vol_surfaces.shape[1], vol_surfaces.shape[2]))
    
    # Use the best model for predictions
    best_model = model_results['models']['Att-LSTM']
    
    # Create a dataset for the test period
    test_dataset = VolatilitySurfaceDataset(vol_surfaces[test_start-3:])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Generate predictions
    best_model.eval()
    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):
            if i >= test_size:
                break
            features = features.to(device)
            prediction = best_model(features).cpu().numpy()[0]
            att_lstm_preds[i] = prediction
    
    # Backtest trading strategies
    print("\nBacktesting trading strategies...")
    test_index_prices = index_prices[test_start:test_start+test_size]
    test_vol_surfaces = vol_surfaces[test_start:test_start+test_size]
    
    # Backtest with predicted volatility surface
    backtest_results_pred = backtest_strategies(
        att_lstm_preds, test_vol_surfaces, test_index_prices, maturities, moneyness
    )
    
    # Backtest with actual volatility surface (perfect prediction)
    backtest_results_actual = backtest_strategies(
        test_vol_surfaces, test_vol_surfaces, test_index_prices, maturities, moneyness
    )
    
    # Backtest with naive prediction (yesterday's volatility)
    naive_preds = np.zeros_like(test_vol_surfaces)
    for i in range(1, len(naive_preds)):
        naive_preds[i] = test_vol_surfaces[i-1]
    naive_preds[0] = test_vol_surfaces[0]
    
    backtest_results_naive = backtest_strategies(
        naive_preds, test_vol_surfaces, test_index_prices, maturities, moneyness
    )
    
    # Plot strategy performance
    plt.figure(figsize=(15, 10))
    
    # Calendar Spread Performance
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results_pred['calendar_cumulative'], label='AI Prediction')
    plt.plot(backtest_results_actual['calendar_cumulative'], label='Perfect Prediction')
    plt.plot(backtest_results_naive['calendar_cumulative'], label='Naive Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative P&L')
    plt.title('Calendar Spread Strategy Performance')
    plt.legend()
    plt.grid(True)
    
    # Butterfly Spread Performance
    plt.subplot(2, 1, 2)
    plt.plot(backtest_results_pred['butterfly_cumulative'], label='AI Prediction')
    plt.plot(backtest_results_actual['butterfly_cumulative'], label='Perfect Prediction')
    plt.plot(backtest_results_naive['butterfly_cumulative'], label='Naive Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative P&L')
    plt.title('Butterfly Spread Strategy Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print strategy performance metrics
    print("\nStrategy Performance Metrics:")
    print("\nCalendar Spread Strategy:")
    print(f"AI Prediction - Final Return: {backtest_results_pred['calendar_returns'][-1]:.4f}, Sharpe: {backtest_results_pred['calendar_sharpe']:.4f}, Max DD: {backtest_results_pred['calendar_max_dd']:.4f}")
    print(f"Perfect Prediction - Final Return: {backtest_results_actual['calendar_returns'][-1]:.4f}, Sharpe: {backtest_results_actual['calendar_sharpe']:.4f}, Max DD: {backtest_results_actual['calendar_max_dd']:.4f}")
    print(f"Naive Prediction - Final Return: {backtest_results_naive['calendar_returns'][-1]:.4f}, Sharpe: {backtest_results_naive['calendar_sharpe']:.4f}, Max DD: {backtest_results_naive['calendar_max_dd']:.4f}")
    
    print("\nButterfly Spread Strategy:")
    print(f"AI Prediction - Final Return: {backtest_results_pred['butterfly_returns'][-1]:.4f}, Sharpe: {backtest_results_pred['butterfly_sharpe']:.4f}, Max DD: {backtest_results_pred['butterfly_max_dd']:.4f}")
    print(f"Perfect Prediction - Final Return: {backtest_results_actual['butterfly_returns'][-1]:.4f}, Sharpe: {backtest_results_actual['butterfly_sharpe']:.4f}, Max DD: {backtest_results_actual['butterfly_max_dd']:.4f}")
    print(f"Naive Prediction - Final Return: {backtest_results_naive['butterfly_returns'][-1]:.4f}, Sharpe: {backtest_results_naive['butterfly_sharpe']:.4f}, Max DD: {backtest_results_naive['butterfly_max_dd']:.4f}")

    # Create a summary table
    summary_data = {
        'Strategy': ['Calendar Spread', 'Calendar Spread', 'Calendar Spread', 
                    'Butterfly Spread', 'Butterfly Spread', 'Butterfly Spread'],
        'Model': ['AI Prediction', 'Perfect Prediction', 'Naive Prediction', 
                 'AI Prediction', 'Perfect Prediction', 'Naive Prediction'],
        'Final Return': [backtest_results_pred['calendar_returns'][-1], 
                        backtest_results_actual['calendar_returns'][-1],
                        backtest_results_naive['calendar_returns'][-1],
                        backtest_results_pred['butterfly_returns'][-1],
                        backtest_results_actual['butterfly_returns'][-1],
                        backtest_results_naive['butterfly_returns'][-1]],
        'Sharpe Ratio': [backtest_results_pred['calendar_sharpe'], 
                         backtest_results_actual['calendar_sharpe'],
                         backtest_results_naive['calendar_sharpe'],
                         backtest_results_pred['butterfly_sharpe'],
                         backtest_results_actual['butterfly_sharpe'],
                         backtest_results_naive['butterfly_sharpe']],
        'Max Drawdown': [backtest_results_pred['calendar_max_dd'], 
                         backtest_results_actual['calendar_max_dd'],
                         backtest_results_naive['calendar_max_dd'],
                         backtest_results_pred['butterfly_max_dd'],
                         backtest_results_actual['butterfly_max_dd'],
                         backtest_results_naive['butterfly_max_dd']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\nStrategy Performance Summary:")
    print(summary_df)

if __name__ == "__main__":
    main()