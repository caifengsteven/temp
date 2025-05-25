import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define LSTM-based model architecture following the paper
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers as described in the paper
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final fully-connected layer with ReLU as described in the paper
        self.fc = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        
        # Output layer with softmax for predicting up/down probabilities
        self.output = nn.Linear(64, 2)  # 2 outputs: probability of up/down move
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state if not provided
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = lstm_out[:, -1, :]
        
        # Pass through fully connected layer with ReLU
        out = self.relu(self.fc(out))
        
        # Output layer
        out = self.output(out)
        out = self.softmax(out)
        
        return out, (hn, cn)

# Linear model for comparison (VAR model as described in the paper)
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # For simplicity, we'll just use the last time step from the sequence
        out = self.linear(x[:, -1, :])
        out = self.softmax(out)
        return out

# Order Book Simulator with more realistic market microstructure dynamics
class OrderBookSimulator:
    def __init__(self, 
                 num_stocks=100, 
                 days=60, 
                 levels=10, 
                 observations_per_day=390,
                 tick_size=0.01):
        """
        Create a realistic order book simulator
        
        Parameters:
        - num_stocks: Number of different stocks to simulate
        - days: Number of days to simulate
        - levels: Number of price levels in the order book
        - observations_per_day: Number of observations per day (e.g., 390 for 1-minute bars in 6.5 hour trading day)
        - tick_size: Minimum price movement
        """
        self.num_stocks = num_stocks
        self.days = days
        self.levels = levels
        self.observations_per_day = observations_per_day
        self.tick_size = tick_size
        
        # Stock characteristics
        self.stock_volatilities = np.random.uniform(0.1, 0.5, num_stocks) / np.sqrt(252)  # Annualized volatility
        self.stock_prices = np.random.uniform(20, 200, num_stocks)
        self.stock_volumes = np.random.uniform(100000, 10000000, num_stocks)  # Daily volume
        self.stock_tick_sizes = np.where(self.stock_prices < 50, 0.01, 
                                        np.where(self.stock_prices < 100, 0.05, 0.1))
        
        # Market regime parameters
        self.market_volatility = 1.0  # Volatility scalar
        self.market_regime = 'normal'  # Options: 'normal', 'high_vol', 'trending', 'mean_reverting'
        self.regime_change_prob = 0.05  # Probability of regime change each day
        
    def simulate(self):
        """Simulate order book data for all stocks"""
        all_data = {}
        
        for stock_idx in range(self.num_stocks):
            ticker = f"STOCK{stock_idx+1}"
            stock_price = self.stock_prices[stock_idx]
            volatility = self.stock_volatilities[stock_idx]
            tick_size = self.stock_tick_sizes[stock_idx]
            
            # Store all days for this stock
            stock_data = []
            
            # Initialize daily volume profile (U-shape)
            volume_profile = self._create_volume_profile()
            
            print(f"Simulating order book data for {ticker} (starting price: ${stock_price:.2f})")
            
            for day in range(self.days):
                # Check for regime change
                if np.random.rand() < self.regime_change_prob:
                    self._change_market_regime()
                
                date = (datetime.now() - timedelta(days=self.days-day)).strftime('%Y%m%d')
                
                # Create time index for the day
                times = pd.date_range(
                    start=f"{date} 09:30:00", 
                    end=f"{date} 16:00:00", 
                    periods=self.observations_per_day
                )
                
                # Initialize DataFrame
                data = pd.DataFrame(index=times)
                data['time'] = times
                
                # Add market regime specific behavior
                daily_returns, price_path = self._simulate_price_path(
                    stock_price, 
                    volatility, 
                    self.observations_per_day
                )
                
                # Simulate order book
                data = self._simulate_order_book(data, price_path, tick_size, volume_profile)
                
                # Calculate mid-price and price changes
                data['mid_price'] = (data['BEST_BID1'] + data['BEST_ASK1']) / 2
                data['price_change'] = data['mid_price'].diff()
                data['label'] = (data['price_change'] > 0).astype(int)
                
                # Update stock price for next day
                stock_price = price_path[-1]
                
                # Store the day's data
                stock_data.append((date, data))
            
            all_data[ticker] = stock_data
        
        return all_data
    
    def _create_volume_profile(self):
        """Create a realistic U-shaped intraday volume profile"""
        x = np.linspace(0, 1, self.observations_per_day)
        # U-shape for volume (high at open and close)
        y = 1.5 - np.sin((x - 0.5) * np.pi) * 0.8
        return y
    
    def _change_market_regime(self):
        """Randomly change the market regime"""
        regimes = ['normal', 'high_vol', 'trending', 'mean_reverting']
        weights = [0.4, 0.2, 0.2, 0.2]  # Normal regime is more likely
        
        # Don't select the current regime
        current_idx = regimes.index(self.market_regime)
        new_regimes = regimes.copy()
        new_weights = weights.copy()
        new_regimes.pop(current_idx)
        new_weights.pop(current_idx)
        
        # Normalize weights
        new_weights = [w/sum(new_weights) for w in new_weights]
        
        # Select new regime
        self.market_regime = np.random.choice(new_regimes, p=new_weights)
        
        if self.market_regime == 'high_vol':
            self.market_volatility = np.random.uniform(1.5, 3.0)
        elif self.market_regime == 'normal':
            self.market_volatility = 1.0
        else:
            self.market_volatility = np.random.uniform(0.7, 1.3)
            
        print(f"Market regime changed to: {self.market_regime} (volatility multiplier: {self.market_volatility:.2f})")
    
    def _simulate_price_path(self, start_price, volatility, steps):
        """Simulate price path based on market regime"""
        daily_vol = volatility * self.market_volatility
        
        if self.market_regime == 'normal':
            # Standard geometric Brownian motion
            daily_returns = np.random.normal(0, daily_vol, steps)
            
        elif self.market_regime == 'high_vol':
            # Higher volatility with occasional jumps
            daily_returns = np.random.normal(0, daily_vol, steps)
            
            # Add occasional jumps
            jump_points = np.random.rand(steps) < 0.02
            jump_sizes = np.random.normal(0, daily_vol * 5, steps)
            daily_returns += jump_points * jump_sizes
            
        elif self.market_regime == 'trending':
            # Add a trend component
            trend = np.random.choice([-1, 1]) * daily_vol * 0.5
            daily_returns = np.random.normal(trend, daily_vol, steps)
            
        elif self.market_regime == 'mean_reverting':
            # Mean reverting process
            mean_reversion_strength = 0.05
            previous_return = 0
            daily_returns = np.zeros(steps)
            
            for i in range(steps):
                # Mean-reverting component pulls back toward zero
                mean_reversion = -mean_reversion_strength * previous_return
                daily_returns[i] = mean_reversion + np.random.normal(0, daily_vol)
                previous_return = daily_returns[i]
        
        # Convert returns to price path
        price_path = start_price * np.cumprod(1 + daily_returns)
        
        # Ensure minimum tick size
        price_path = np.round(price_path / self.tick_size) * self.tick_size
        
        return daily_returns, price_path
    
    def _simulate_order_book(self, data, price_path, tick_size, volume_profile):
        """Simulate a realistic limit order book"""
        num_obs = len(price_path)
        
        # Base spread as a function of price and volatility
        avg_price = np.mean(price_path)
        base_spread = max(tick_size, avg_price * 0.0005 * self.market_volatility)
        
        # Order book imbalance (will affect price direction)
        imbalance = np.zeros(num_obs)
        
        # Add some autocorrelation to imbalance (order flow persistence)
        ar_param = 0.7  # Autocorrelation parameter
        for i in range(1, num_obs):
            imbalance[i] = ar_param * imbalance[i-1] + np.random.normal(0, 0.2)
        
        # Normalize imbalance to [-1, 1]
        imbalance = np.tanh(imbalance)
        
        # Calculate spread for each observation, affected by volatility and imbalance
        spreads = np.zeros(num_obs)
        for i in range(num_obs):
            # Higher imbalance tends to widen spreads
            spread_multiplier = 1 + 0.5 * abs(imbalance[i])
            # Volume profile affects spread (higher volume -> tighter spreads)
            vol_effect = 1 / np.sqrt(volume_profile[i])
            spreads[i] = max(tick_size, base_spread * spread_multiplier * vol_effect)
        
        # Populate order book levels
        for level in range(1, self.levels + 1):
            # Deeper levels have wider spreads
            level_multiplier = 1 + (level - 1) * 0.5
            
            # Add some noise to make it realistic
            level_noise = np.random.normal(0, 0.2 * level, num_obs)
            
            # Best bid and ask prices for this level
            if level == 1:
                # For level 1, use mid price and half spread
                bid_prices = price_path - spreads / 2
                ask_prices = price_path + spreads / 2
            else:
                # For deeper levels, increment from previous level
                prev_bid = data[f'BEST_BID{level-1}'].values
                prev_ask = data[f'BEST_ASK{level-1}'].values
                
                # Gaps between levels follow approximately exponential distribution
                bid_gaps = np.maximum(tick_size, np.random.exponential(tick_size * level_multiplier, num_obs))
                ask_gaps = np.maximum(tick_size, np.random.exponential(tick_size * level_multiplier, num_obs))
                
                bid_prices = prev_bid - bid_gaps
                ask_prices = prev_ask + ask_gaps
            
            # Round to tick size
            bid_prices = np.round(bid_prices / tick_size) * tick_size
            ask_prices = np.round(ask_prices / tick_size) * tick_size
            
            # Ensure ask > bid
            min_spread = tick_size
            invalid_idx = ask_prices - bid_prices < min_spread
            if np.any(invalid_idx):
                mid_points = (ask_prices[invalid_idx] + bid_prices[invalid_idx]) / 2
                bid_prices[invalid_idx] = mid_points - min_spread / 2
                ask_prices[invalid_idx] = mid_points + min_spread / 2
            
            # Store in DataFrame
            data[f'BEST_BID{level}'] = bid_prices
            data[f'BEST_ASK{level}'] = ask_prices
            
            # Generate order sizes for this level
            # Size decreases with depth in the order book
            base_size_multiplier = np.exp(-0.3 * (level - 1))
            
            # Order imbalance affects sizes (more on side where price is going)
            for i in range(num_obs):
                # Base size affected by volume profile
                base_size = np.random.exponential(100 * volume_profile[i])
                
                # Imbalance affects bid/ask sizes
                if imbalance[i] > 0:  # More buying pressure
                    bid_size = base_size * (1 + 0.5 * imbalance[i]) * base_size_multiplier
                    ask_size = base_size * (1 - 0.3 * imbalance[i]) * base_size_multiplier
                else:  # More selling pressure
                    bid_size = base_size * (1 + 0.3 * imbalance[i]) * base_size_multiplier
                    ask_size = base_size * (1 - 0.5 * imbalance[i]) * base_size_multiplier
                
                # Ensure positive sizes with some noise
                data.loc[data.index[i], f'BEST_BID{level}_SZ'] = max(1, bid_size * (1 + 0.1 * np.random.randn()))
                data.loc[data.index[i], f'BEST_ASK{level}_SZ'] = max(1, ask_size * (1 + 0.1 * np.random.randn()))
        
        return data
    
    def prepare_dataset(self, window_size=100, feature_type='raw'):
        """Prepare dataset suitable for deep learning model training"""
        print(f"Simulating order book data with {self.num_stocks} stocks over {self.days} days...")
        all_data = self.simulate()
        
        all_features = []
        all_labels = []
        stock_indices = []  # Track which stock each sample comes from
        
        print("Preparing training data...")
        for stock_idx, (ticker, stock_data) in enumerate(all_data.items()):
            stock_features = []
            stock_labels = []
            
            for date, data in stock_data:
                # Skip if not enough data
                if len(data) <= window_size:
                    continue
                
                # Extract features based on requested type
                if feature_type == 'raw':
                    features = self._extract_raw_features(data)
                elif feature_type == 'derived':
                    features = self._extract_derived_features(data)
                else:
                    features = self._extract_raw_features(data)
                
                # Create sequences for the LSTM
                X, y = self._create_sequences(features, data['label'], window_size)
                
                if len(X) > 0:
                    stock_features.append(X)
                    stock_labels.append(y)
                    # Add stock index for each sample
                    stock_indices.extend([stock_idx] * len(y))
            
            if stock_features:
                # Combine all days for this stock
                X_stock = np.vstack(stock_features)
                y_stock = np.concatenate(stock_labels)
                
                print(f"{ticker}: Generated {len(X_stock)} training examples with shape {X_stock.shape}")
                
                all_features.append(X_stock)
                all_labels.append(y_stock)
        
        # Combine all stocks
        if all_features:
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            stock_indices = np.array(stock_indices)
            
            print(f"Total dataset: {len(X)} samples with {X.shape[2]} features")
            
            return X, y, stock_indices
        else:
            return np.array([]), np.array([]), np.array([])
    
    def _extract_raw_features(self, data):
        """Extract raw features from order book data"""
        # Initialize features list
        features = []
        
        # Calculate mid-price for normalization
        mid_price = (data['BEST_BID1'] + data['BEST_ASK1']) / 2
        
        # Extract features for all levels available
        for i in range(1, self.levels + 1):
            bid_col = f'BEST_BID{i}'
            ask_col = f'BEST_ASK{i}'
            bid_size_col = f'BEST_BID{i}_SZ'
            ask_size_col = f'BEST_ASK{i}_SZ'
            
            if all(col in data.columns for col in [bid_col, ask_col, bid_size_col, ask_size_col]):
                # Normalize prices by mid-price
                rel_bid = (data[bid_col] / mid_price - 1)
                rel_ask = (data[ask_col] / mid_price - 1)
                
                # Add as features
                features.append(rel_bid)
                features.append(rel_ask)
                features.append(data[bid_size_col])
                features.append(data[ask_size_col])
        
        # Combine features into a DataFrame
        return pd.concat(features, axis=1)
    
    def _extract_derived_features(self, data):
        """Extract more complex derived features from order book data"""
        # Start with raw features
        raw_features = self._extract_raw_features(data)
        
        # Initialize derived features list
        derived_features = []
        
        # Calculate bid-ask spread
        spread = data['BEST_ASK1'] - data['BEST_BID1']
        derived_features.append(spread)
        
        # Calculate mid-price
        mid_price = (data['BEST_BID1'] + data['BEST_ASK1']) / 2
        
        # Calculate price volatility (rolling standard deviation)
        price_vol = mid_price.rolling(window=10).std()
        derived_features.append(price_vol)
        
        # Calculate order book imbalance at level 1
        imbalance1 = (data['BEST_BID1_SZ'] - data['BEST_ASK1_SZ']) / (data['BEST_BID1_SZ'] + data['BEST_ASK1_SZ'])
        derived_features.append(imbalance1)
        
        # Calculate total order book imbalance across all levels
        total_bid_size = sum(data[f'BEST_BID{i}_SZ'] for i in range(1, self.levels + 1) if f'BEST_BID{i}_SZ' in data.columns)
        total_ask_size = sum(data[f'BEST_ASK{i}_SZ'] for i in range(1, self.levels + 1) if f'BEST_ASK{i}_SZ' in data.columns)
        total_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        derived_features.append(total_imbalance)
        
        # Create DataFrame from derived features
        derived_df = pd.concat(derived_features, axis=1)
        
        # Combine raw and derived features
        combined = pd.concat([raw_features, derived_df], axis=1)
        
        # Fill NAs
        combined = combined.fillna(0)
        
        return combined
    
    def _create_sequences(self, features, labels, window_size):
        """Create sequences of data for LSTM input"""
        X, y = [], []
        
        for i in range(len(features) - window_size):
            # Only include rows where there's a price change
            if i + window_size < len(labels) and not np.isnan(labels.iloc[i + window_size]):
                X.append(features.iloc[i:i+window_size].values)
                y.append(labels.iloc[i+window_size])
        
        return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])

# Training and evaluation functions
def train_model(model, train_loader, val_loader, n_epochs=50, patience=10, learning_rate=0.001):
    """Train the model with early stopping"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, LSTMModel):
                output, _ = model(data)
            else:
                output = model(data)
                
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{n_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if isinstance(model, LSTMModel):
                    output, _ = model(data)
                else:
                    output = model(data)
                    
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        
        print(f'Epoch: {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate model on test data"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    prediction_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if isinstance(model, LSTMModel):
                output, _ = model(data)
            else:
                output = model(data)
                
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            prediction_probs.extend(output.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Print confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate class-wise accuracies
    print("\nClass-wise Performance:")
    print(classification_report(all_targets, all_preds, target_names=['Down', 'Up']))
    
    return accuracy, all_preds, all_targets, prediction_probs

# Trading strategy using the trained model
class OrderBookTradingStrategy:
    def __init__(self, model, window_size=100, threshold=0.55, 
                 initial_capital=100000, position_size=100,
                 stop_loss_pct=0.01, take_profit_pct=0.02):
        self.model = model
        self.window_size = window_size
        self.threshold = threshold  # Confidence threshold for making trades
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.positions = {}  # Current positions {ticker: {size, entry_price, stop_loss, take_profit}}
        self.trades = []  # Record of all trades
        self.daily_pnl = []  # Daily P&L
    
    def prepare_features(self, order_book_data):
        """Extract features from order book data"""
        if order_book_data.empty:
            return None
        
        # Calculate mid-price for normalization
        mid_price = (order_book_data['BEST_BID1'] + order_book_data['BEST_ASK1']) / 2
        
        # Extract features for all levels available
        features = []
        for i in range(1, 11):  # Assuming up to 10 levels
            bid_col = f'BEST_BID{i}'
            ask_col = f'BEST_ASK{i}'
            bid_size_col = f'BEST_BID{i}_SZ'
            ask_size_col = f'BEST_ASK{i}_SZ'
            
            if all(col in order_book_data.columns for col in [bid_col, ask_col, bid_size_col, ask_size_col]):
                # Normalize prices by mid-price
                rel_bid = (order_book_data[bid_col] / mid_price - 1)
                rel_ask = (order_book_data[ask_col] / mid_price - 1)
                
                # Add as features
                features.append(rel_bid)
                features.append(rel_ask)
                features.append(order_book_data[bid_size_col])
                features.append(order_book_data[ask_size_col])
        
        if not features:
            return None
            
        features_df = pd.concat(features, axis=1)
        
        # Create sequences for the LSTM
        if len(features_df) >= self.window_size:
            # Take the most recent window_size observations
            X = features_df.iloc[-self.window_size:].values
            X = X.reshape(1, self.window_size, -1)  # Add batch dimension
            return torch.FloatTensor(X).to(device)
        else:
            return None
    
    def generate_signal(self, order_book_data, ticker):
        """Generate a trading signal based on model prediction"""
        X = self.prepare_features(order_book_data)
        
        if X is None:
            return 0, None  # No signal if not enough data
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, LSTMModel):
                output, _ = self.model(X)
            else:
                output = self.model(X)
            
            # Get probabilities
            probs = output.cpu().numpy()[0]
            
            # Generate signal based on probability and threshold
            if probs[1] > self.threshold:  # Probability of up move
                return 1, probs  # Buy signal
            elif probs[0] > self.threshold:  # Probability of down move
                return -1, probs  # Sell signal
            else:
                return 0, probs  # No signal
    
    def execute_trade(self, ticker, signal, price, timestamp, confidence=None):
        """Execute a trade based on the signal"""
        # Skip if we don't have enough capital
        trade_value = self.position_size * price
        if trade_value > self.capital and signal > 0:
            print(f"Not enough capital to buy {self.position_size} shares of {ticker} at ${price:.2f}")
            return None
        
        # Calculate position size based on available capital
        if signal > 0:  # Buy
            # Currently no position or short position
            if ticker not in self.positions or self.positions[ticker]['size'] <= 0:
                # Close short position if any
                if ticker in self.positions and self.positions[ticker]['size'] < 0:
                    old_size = self.positions[ticker]['size']
                    old_price = self.positions[ticker]['entry_price']
                    # Close existing short position
                    pnl = old_size * (old_price - price)
                    self.capital += pnl
                    
                    # Record the trade
                    self.trades.append({
                        'ticker': ticker,
                        'timestamp': timestamp,
                        'action': 'close_short',
                        'size': abs(old_size),
                        'price': price,
                        'pnl': pnl,
                        'confidence': confidence[0] if confidence is not None else None
                    })
                
                # Open new long position
                self.positions[ticker] = {
                    'size': self.position_size,
                    'entry_price': price,
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'take_profit': price * (1 + self.take_profit_pct)
                }
                
                # Deduct from capital
                self.capital -= trade_value
                
                # Record the trade
                trade = {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'action': 'buy',
                    'size': self.position_size,
                    'price': price,
                    'confidence': confidence[1] if confidence is not None else None
                }
                self.trades.append(trade)
                
                return trade
        
        elif signal < 0:  # Sell
            # Currently no position or long position
            if ticker not in self.positions or self.positions[ticker]['size'] >= 0:
                # Close long position if any
                if ticker in self.positions and self.positions[ticker]['size'] > 0:
                    old_size = self.positions[ticker]['size']
                    old_price = self.positions[ticker]['entry_price']
                    # Close existing long position
                    pnl = old_size * (price - old_price)
                    self.capital += trade_value + pnl
                    
                    # Record the trade
                    self.trades.append({
                        'ticker': ticker,
                        'timestamp': timestamp,
                        'action': 'close_long',
                        'size': old_size,
                        'price': price,
                        'pnl': pnl,
                        'confidence': confidence[0] if confidence is not None else None
                    })
                
                # Open new short position
                self.positions[ticker] = {
                    'size': -self.position_size,
                    'entry_price': price,
                    'stop_loss': price * (1 + self.stop_loss_pct),
                    'take_profit': price * (1 - self.take_profit_pct)
                }
                
                # Add to capital (for short selling, we initially gain capital)
                self.capital += trade_value
                
                # Record the trade
                trade = {
                    'ticker': ticker,
                    'timestamp': timestamp,
                    'action': 'sell',
                    'size': self.position_size,
                    'price': price,
                    'confidence': confidence[0] if confidence is not None else None
                }
                self.trades.append(trade)
                
                return trade
        
        return None
    
    def check_stop_loss_take_profit(self, ticker, current_price, timestamp):
        """Check if stop loss or take profit has been hit"""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        # Check stop loss for long positions
        if position['size'] > 0 and current_price <= position['stop_loss']:
            # Close the position
            pnl = position['size'] * (current_price - position['entry_price'])
            self.capital += position['size'] * current_price
            
            # Record the trade
            self.trades.append({
                'ticker': ticker,
                'timestamp': timestamp,
                'action': 'stop_loss_long',
                'size': position['size'],
                'price': current_price,
                'pnl': pnl
            })
            
            # Remove the position
            del self.positions[ticker]
            return True
        
        # Check take profit for long positions
        elif position['size'] > 0 and current_price >= position['take_profit']:
            # Close the position
            pnl = position['size'] * (current_price - position['entry_price'])
            self.capital += position['size'] * current_price
            
            # Record the trade
            self.trades.append({
                'ticker': ticker,
                'timestamp': timestamp,
                'action': 'take_profit_long',
                'size': position['size'],
                'price': current_price,
                'pnl': pnl
            })
            
            # Remove the position
            del self.positions[ticker]
            return True
        
        # Check stop loss for short positions
        elif position['size'] < 0 and current_price >= position['stop_loss']:
            # Close the position
            pnl = position['size'] * (position['entry_price'] - current_price)
            self.capital += abs(position['size']) * current_price + pnl
            
            # Record the trade
            self.trades.append({
                'ticker': ticker,
                'timestamp': timestamp,
                'action': 'stop_loss_short',
                'size': abs(position['size']),
                'price': current_price,
                'pnl': pnl
            })
            
            # Remove the position
            del self.positions[ticker]
            return True
        
        # Check take profit for short positions
        elif position['size'] < 0 and current_price <= position['take_profit']:
            # Close the position
            pnl = position['size'] * (position['entry_price'] - current_price)
            self.capital += abs(position['size']) * current_price + pnl
            
            # Record the trade
            self.trades.append({
                'ticker': ticker,
                'timestamp': timestamp,
                'action': 'take_profit_short',
                'size': abs(position['size']),
                'price': current_price,
                'pnl': pnl
            })
            
            # Remove the position
            del self.positions[ticker]
            return True
        
        return False
    
    def backtest(self, simulated_data):
        """Backtest the trading strategy on simulated data"""
        results = {}
        daily_results = []
        
        # Reset strategy state
        self.positions = {}
        self.trades = []
        self.capital = self.initial_capital
        portfolio_value_history = []
        
        print("Starting backtest...")
        
        # Process each stock
        for ticker, stock_data in simulated_data.items():
            print(f"Backtesting {ticker}...")
            
            # Process each day for this stock
            for date, data in stock_data:
                day_date = datetime.strptime(date, '%Y%m%d')
                day_results = {
                    'date': day_date,
                    'ticker': ticker,
                    'trades': 0,
                    'pnl': 0
                }
                
                if data.empty:
                    daily_results.append(day_results)
                    continue
                
                # Process each timestep
                for idx in range(len(data)):
                    if idx < self.window_size:
                        continue
                    
                    # Current snapshot of the order book
                    current_data = data.iloc[:idx+1]
                    
                    # Current mid price
                    mid_price = (current_data['BEST_BID1'].iloc[-1] + current_data['BEST_ASK1'].iloc[-1]) / 2
                    
                    # Check stop loss/take profit for existing positions
                    if ticker in self.positions:
                        if self.check_stop_loss_take_profit(ticker, mid_price, current_data.index[-1]):
                            day_results['trades'] += 1
                    
                    # Generate signal
                    signal, probs = self.generate_signal(current_data, ticker)
                    
                    # Execute trade if signal is non-zero
                    if signal != 0:
                        trade = self.execute_trade(
                            ticker=ticker,
                            signal=signal,
                            price=mid_price,
                            timestamp=current_data.index[-1],
                            confidence=probs
                        )
                        
                        if trade is not None:
                            day_results['trades'] += 1
                
                # Calculate daily PnL
                day_pnl = 0
                
                # Calculate PnL for closed positions from this day's trades
                for trade in self.trades:
                    if trade['ticker'] == ticker and 'pnl' in trade:
                        if isinstance(trade['timestamp'], pd.Timestamp) and trade['timestamp'].date() == day_date.date():
                            day_pnl += trade['pnl']
                
                # Calculate mark-to-market PnL for open positions
                if ticker in self.positions:
                    position = self.positions[ticker]
                    last_price = data['mid_price'].iloc[-1]
                    
                    if position['size'] > 0:  # Long
                        day_pnl += position['size'] * (last_price - position['entry_price'])
                    else:  # Short
                        day_pnl += position['size'] * (position['entry_price'] - last_price)
                
                day_results['pnl'] = day_pnl
                daily_results.append(day_results)
                
                # Calculate portfolio value at end of day
                total_portfolio_value = self.capital
                for pos_ticker, position in self.positions.items():
                    # For the current ticker, use the last price from data
                    if pos_ticker == ticker:
                        last_price = data['mid_price'].iloc[-1]
                    else:
                        # This is a simplification - in a real backtest you'd need the price for all stocks
                        # For now, just use the entry price
                        last_price = position['entry_price']
                    
                    total_portfolio_value += position['size'] * last_price
                
                portfolio_value_history.append({
                    'date': day_date,
                    'portfolio_value': total_portfolio_value
                })
        
        # Process results
        results['daily'] = daily_results
        results['portfolio_history'] = portfolio_value_history
        results['final_capital'] = self.capital
        results['trades'] = self.trades
        results['positions'] = self.positions
        
        return results
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        trades = results['trades']
        portfolio_history = results['portfolio_history']
        
        # Group trades by day
        trade_df = pd.DataFrame(trades)
        if not trade_df.empty and 'timestamp' in trade_df.columns:
            if isinstance(trade_df['timestamp'].iloc[0], pd.Timestamp):
                trade_df['date'] = trade_df['timestamp'].dt.date
            else:
                # Handle case where timestamp might be a string or other format
                trade_df['date'] = trade_df['timestamp']
        
        # Create portfolio value series
        portfolio_df = pd.DataFrame(portfolio_history)
        
        # Calculate daily returns
        if not portfolio_df.empty:
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            # Calculate metrics
            total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
            daily_returns = portfolio_df['daily_return'].dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
            max_drawdown = 0
            
            # Calculate drawdown
            peak = portfolio_df['portfolio_value'].iloc[0]
            for value in portfolio_df['portfolio_value']:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                peak = max(peak, value)
            
            # Winning trades
            winning_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
            losing_trades = [t for t in trades if 'pnl' in t and t['pnl'] <= 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Average P&L per trade
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else float('inf')
            
            # Print metrics
            print("\n=== Strategy Performance ===")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Final Portfolio Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Number of Trades: {len(trades)}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Win: ${avg_win:,.2f}")
            print(f"Average Loss: ${avg_loss:,.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            
            # Plot portfolio value
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('portfolio_value.png')
            plt.show()
            
            # Plot daily returns
            plt.figure(figsize=(12, 6))
            plt.bar(portfolio_df['date'][1:], portfolio_df['daily_return'].iloc[1:])
            plt.title('Daily Returns')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('daily_returns.png')
            plt.show()
            
            # Plot drawdown
            drawdown_series = []
            peak = portfolio_df['portfolio_value'].iloc[0]
            for i, value in enumerate(portfolio_df['portfolio_value']):
                drawdown = (peak - value) / peak
                drawdown_series.append(drawdown)
                peak = max(peak, value)
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(portfolio_df['date'], 0, drawdown_series, color='red', alpha=0.3)
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('drawdown.png')
            plt.show()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
        else:
            print("No portfolio history data available for analysis.")
            return None

def train_universal_vs_stock_specific_models(sim_data, window_size=100, batch_size=64):
    """Train and compare universal model vs stock-specific models"""
    # Extract data for all stocks
    X, y, stock_indices = sim_data.prepare_dataset(window_size=window_size)
    
    if len(X) == 0:
        print("No data available for training. Please check the simulator parameters.")
        return None, None, None
    
    print(f"Total dataset: {len(X)} samples, X shape: {X.shape}, y shape: {y.shape}")
    
    # Split into train/validation/test sets
    X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(
        X, y, stock_indices, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
        X_train_val, y_train_val, indices_train_val, test_size=0.25, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create data loaders for universal model
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train universal LSTM model
    print("\n=== Training Universal LSTM Model ===")
    input_size = X_train.shape[2]  # Number of features
    universal_lstm = LSTMModel(input_size=input_size).to(device)
    
    universal_lstm, train_losses, val_losses = train_model(
        universal_lstm, 
        train_loader, 
        val_loader,
        n_epochs=30,
        patience=5
    )
    
    # Train universal linear model for comparison
    print("\n=== Training Universal Linear Model ===")
    universal_linear = LinearModel(input_size=input_size).to(device)
    
    universal_linear, linear_train_losses, linear_val_losses = train_model(
        universal_linear,
        train_loader,
        val_loader,
        n_epochs=20,
        patience=5
    )
    
    # Evaluate universal models
    print("\n=== Evaluating Universal LSTM Model ===")
    lstm_accuracy, lstm_preds, lstm_targets, lstm_probs = evaluate_model(universal_lstm, test_loader)
    
    print("\n=== Evaluating Universal Linear Model ===")
    linear_accuracy, linear_preds, linear_targets, linear_probs = evaluate_model(universal_linear, test_loader)
    
    # Train and evaluate stock-specific models
    unique_stocks = np.unique(indices_train)
    stock_specific_accuracies = []
    
    for stock_idx in unique_stocks:
        # Get data for this stock
        stock_train_idx = indices_train == stock_idx
        stock_val_idx = indices_val == stock_idx
        stock_test_idx = indices_test == stock_idx
        
        if np.sum(stock_train_idx) < 100 or np.sum(stock_val_idx) < 50 or np.sum(stock_test_idx) < 50:
            print(f"Skipping stock {stock_idx} due to insufficient data")
            continue
        
        X_stock_train = X_train[stock_train_idx]
        y_stock_train = y_train[stock_train_idx]
        X_stock_val = X_val[stock_val_idx]
        y_stock_val = y_val[stock_val_idx]
        X_stock_test = X_test[stock_test_idx]
        y_stock_test = y_test[stock_test_idx]
        
        print(f"\n=== Training Stock-Specific Model for Stock {stock_idx} ===")
        print(f"Stock-specific train set: {len(X_stock_train)} samples")
        
        # Create stock-specific data loaders
        stock_train_dataset = TensorDataset(torch.FloatTensor(X_stock_train), torch.LongTensor(y_stock_train))
        stock_val_dataset = TensorDataset(torch.FloatTensor(X_stock_val), torch.LongTensor(y_stock_val))
        stock_test_dataset = TensorDataset(torch.FloatTensor(X_stock_test), torch.LongTensor(y_stock_test))
        
        stock_train_loader = DataLoader(stock_train_dataset, batch_size=batch_size, shuffle=True)
        stock_val_loader = DataLoader(stock_val_dataset, batch_size=batch_size)
        stock_test_loader = DataLoader(stock_test_dataset, batch_size=batch_size)
        
        # Train stock-specific LSTM model
        stock_lstm = LSTMModel(input_size=input_size).to(device)
        
        stock_lstm, _, _ = train_model(
            stock_lstm, 
            stock_train_loader, 
            stock_val_loader,
            n_epochs=30,
            patience=5
        )
        
        # Evaluate stock-specific model
        print(f"Evaluating Stock-Specific Model for Stock {stock_idx}")
        stock_accuracy, _, _, _ = evaluate_model(stock_lstm, stock_test_loader)
        
        # Evaluate universal model on this stock's test data
        print(f"Evaluating Universal Model on Stock {stock_idx}")
        universal_on_stock_accuracy, _, _, _ = evaluate_model(universal_lstm, stock_test_loader)
        
        stock_specific_accuracies.append({
            'stock_idx': stock_idx,
            'stock_specific_accuracy': stock_accuracy,
            'universal_accuracy': universal_on_stock_accuracy,
            'difference': universal_on_stock_accuracy - stock_accuracy
        })
    
    # Plot comparison
    if stock_specific_accuracies:
        stock_indices = [item['stock_idx'] for item in stock_specific_accuracies]
        stock_specific_accs = [item['stock_specific_accuracy'] for item in stock_specific_accuracies]
        universal_accs = [item['universal_accuracy'] for item in stock_specific_accuracies]
        
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(stock_indices))
        plt.bar(x - width/2, stock_specific_accs, width, label='Stock-Specific Model')
        plt.bar(x + width/2, universal_accs, width, label='Universal Model')
        plt.xlabel('Stock Index')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Universal vs Stock-Specific Model Accuracy')
        plt.xticks(x, stock_indices)
        plt.legend()
        plt.tight_layout()
        plt.savefig('universal_vs_stock_specific.png')
        plt.show()
        
        # Show difference
        differences = [item['difference'] for item in stock_specific_accuracies]
        plt.figure(figsize=(12, 6))
        plt.bar(stock_indices, differences)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Stock Index')
        plt.ylabel('Accuracy Difference (%)')
        plt.title('Universal - Stock-Specific Model Accuracy')
        plt.tight_layout()
        plt.savefig('model_accuracy_difference.png')
        plt.show()
        
        # Plot histogram of differences
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=10)
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Accuracy Difference (%)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Universal - Stock-Specific Model Accuracy')
        plt.tight_layout()
        plt.savefig('accuracy_difference_histogram.png')
        plt.show()
    
    return universal_lstm, universal_linear, stock_specific_accuracies

def compare_window_sizes(sim_data, window_sizes=[50, 100, 200, 500], batch_size=64):
    """Compare model performance with different historical window sizes"""
    results = []
    
    for window_size in window_sizes:
        print(f"\n=== Training model with window size {window_size} ===")
        
        # Prepare dataset with this window size
        X, y, _ = sim_data.prepare_dataset(window_size=window_size)
        
        if len(X) == 0:
            print(f"No data available for window size {window_size}")
            continue
        
        # Split into train/validation/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Train LSTM model
        input_size = X_train.shape[2]
        model = LSTMModel(input_size=input_size).to(device)
        
        model, _, _ = train_model(
            model, 
            train_loader, 
            val_loader,
            n_epochs=20,
            patience=5
        )
        
        # Evaluate
        accuracy, _, _, _ = evaluate_model(model, test_loader)
        
        results.append({
            'window_size': window_size,
            'accuracy': accuracy
        })
    
    # Plot results
    if results:
        plt.figure(figsize=(10, 6))
        window_sizes = [r['window_size'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        plt.plot(window_sizes, accuracies, 'o-')
        plt.xlabel('Window Size (Number of Time Steps)')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Effect of Historical Window Size on Model Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('window_size_comparison.png')
        plt.show()
    
    return results

def main():
    """Main function to run the simulation, model training, and backtest"""
    # Define simulation parameters
    num_stocks = 20  # Number of stocks to simulate
    sim_days = 60    # Number of trading days to simulate
    levels = 10      # Number of order book levels
    
    # Create simulator and generate data
    print(f"Creating order book simulator with {num_stocks} stocks over {sim_days} days...")
    simulator = OrderBookSimulator(
        num_stocks=num_stocks,
        days=sim_days,
        levels=levels,
        observations_per_day=390
    )
    
    # Generate simulated order book data
    simulated_data = simulator.simulate()
    
    # Compare models with different window sizes
    print("\n=== Comparing models with different window sizes ===")
    window_size_results = compare_window_sizes(
        simulator, 
        window_sizes=[50, 100, 250, 500], 
        batch_size=64
    )
    
    # Train universal and stock-specific models for comparison
    print("\n=== Training universal vs stock-specific models ===")
    universal_lstm, universal_linear, stock_specific_results = train_universal_vs_stock_specific_models(
        simulator,
        window_size=100,
        batch_size=64
    )
    
    if universal_lstm is None:
        print("Training failed. Exiting.")
        return
    
    # Create a trading strategy with the universal LSTM model
    print("\n=== Running trading strategy backtest ===")
    strategy = OrderBookTradingStrategy(
        model=universal_lstm,
        window_size=100,
        threshold=0.55,
        initial_capital=100000,
        position_size=100
    )
    
    # Run backtest
    backtest_results = strategy.backtest(simulated_data)
    
    # Analyze results
    strategy.analyze_results(backtest_results)
    
    # Compare with linear model strategy
    print("\n=== Running trading strategy with linear model for comparison ===")
    linear_strategy = OrderBookTradingStrategy(
        model=universal_linear,
        window_size=100,
        threshold=0.55,
        initial_capital=100000,
        position_size=100
    )
    
    # Run backtest with linear model
    linear_backtest_results = linear_strategy.backtest(simulated_data)
    
    # Analyze linear model results
    linear_metrics = linear_strategy.analyze_results(linear_backtest_results)
    
    # Print final comparison
    print("\n=== Final Comparison ===")
    print("LSTM vs Linear Model Performance:")
    print(f"LSTM Total Return: {backtest_results['portfolio_history'][-1]['portfolio_value'] / 100000 - 1:.2%}")
    print(f"Linear Total Return: {linear_backtest_results['portfolio_history'][-1]['portfolio_value'] / 100000 - 1:.2%}")
    
    # Validate the paper's findings
    print("\n=== Key Findings from Paper Validation ===")
    
    # 1. Nonlinearity - LSTM outperforms linear models
    lstm_accuracy = stock_specific_results[0]['universal_accuracy'] if stock_specific_results else 0
    linear_accuracy = linear_metrics['total_return'] if linear_metrics else 0
    print(f"1. Nonlinearity: LSTM outperforms linear models - {'Validated' if lstm_accuracy > linear_accuracy else 'Not Validated'}")
    
    # 2. Universality - Universal model outperforms stock-specific models
    if stock_specific_results:
        avg_diff = np.mean([item['difference'] for item in stock_specific_results])
        print(f"2. Universality: Universal model outperforms stock-specific models - {'Validated' if avg_diff > 0 else 'Not Validated'} (Avg Diff: {avg_diff:.2f}%)")
    
    # 3. Path-dependence - Longer history improves performance
    if window_size_results:
        # Check if larger window sizes generally improve performance
        window_sizes = [r['window_size'] for r in window_size_results]
        accuracies = [r['accuracy'] for r in window_size_results]
        correlation = np.corrcoef(window_sizes, accuracies)[0, 1]
        print(f"3. Path-dependence: Longer history improves performance - {'Validated' if correlation > 0 else 'Not Validated'} (Correlation: {correlation:.2f})")

if __name__ == "__main__":
    main()