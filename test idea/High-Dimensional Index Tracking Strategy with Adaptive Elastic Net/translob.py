import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Helper function to add realistic market noise
def add_realistic_market_noise(data, noise_level=0.2):
    """
    Add realistic market noise to data:
    - Small percentage of random price spikes
    - Volume bursts
    - Non-uniform noise (more noise during volatile periods)
    """
    noisy_data = data.copy()
    time_steps, features = data.shape[1:]
    
    # Add occasional price spikes (flash crashes or sudden jumps)
    for i in range(len(data)):
        # Decide if this sample gets a price spike
        if random.random() < 0.05:  # 5% of samples get spikes
            spike_time = random.randint(0, time_steps-1)
            spike_direction = random.choice([-1, 1])
            spike_magnitude = random.uniform(0.005, 0.02)  # 0.5% to 2% price move
            
            # Apply spike to price columns (even indices 0,2,4... for ask, and later evens for bid)
            for j in range(0, features//2, 2):
                noisy_data[i, spike_time, j] += spike_direction * spike_magnitude * abs(noisy_data[i, spike_time, j])
                
            # Volume often spikes with price
            for j in range(1, features//2, 2):
                noisy_data[i, spike_time, j] *= random.uniform(1.5, 3.0)
    
    # Add non-uniform noise (more in volatile periods)
    for i in range(len(data)):
        # Calculate volatility for this sample
        price_diffs = np.diff(noisy_data[i, :, 0])
        volatility = np.std(price_diffs) if len(price_diffs) > 0 else 0.001
        rel_volatility = min(3.0, max(0.5, volatility / 0.001))  # Scale to reasonable range
        
        # Add more noise during volatile periods
        for t in range(time_steps):
            noise_scalar = noise_level * rel_volatility
            for f in range(features):
                # Add proportional noise
                if f % 2 == 0:  # Price columns
                    noisy_data[i, t, f] *= (1 + np.random.normal(0, noise_scalar * 0.2))
                else:  # Volume columns
                    noisy_data[i, t, f] *= (1 + np.random.normal(0, noise_scalar * 0.5))
    
    return noisy_data

# Realistic LOB data generator with imperfect signals
def generate_realistic_lob_data(num_samples=10000, num_levels=10, time_steps=100):
    """
    Generate more realistic synthetic limit order book data with:
    - Imperfect price signals (sometimes the signal doesn't lead to expected moves)
    - Realistic order flow patterns 
    - Signal delays (price changes sometimes lag the signals)
    - Noise and random fluctuations
    """
    features = num_levels * 4  # price and volume for each level on both sides
    
    # Initialize empty arrays
    X = np.zeros((num_samples, time_steps, features))
    y = np.zeros(num_samples, dtype=int)
    
    # Base parameters
    price_vol = 0.001  
    vol_vol = 0.15    
    mean_price = 100.0
    
    # More balanced class distribution
    direction_probs = [0.33, 0.34, 0.33]  # Down, Neutral, Up
    
    midprice = mean_price
    spread = 0.01 * mean_price
    
    # Variables to track market regime (trending or mean-reverting)
    regime_duration = random.randint(50, 150)  # How long a regime lasts
    regime_counter = 0
    is_trending = random.choice([True, False])  # Start with either regime
    
    for i in range(num_samples):
        # Check if we need to switch market regime
        regime_counter += 1
        if regime_counter >= regime_duration:
            is_trending = not is_trending
            regime_duration = random.randint(50, 150)
            regime_counter = 0
        
        # Generate actual price movement (ground truth)
        # In trending regime: trend has more influence
        # In mean-reverting regime: deviations tend to revert
        
        # Generate direction with appropriate probabilities
        if is_trending:
            # In trending regime, consecutive samples are more likely to move in same direction
            if i > 0 and y[i-1] != 1:  # If previous was not neutral
                # 60% chance to continue in same direction, 30% neutral, 10% reverse
                if y[i-1] == 0:  # Previous was down
                    direction_probs = [0.6, 0.3, 0.1]
                else:  # Previous was up
                    direction_probs = [0.1, 0.3, 0.6]
            else:
                direction_probs = [0.33, 0.34, 0.33]
        else:
            # In mean-reverting regime, moves are more likely to reverse
            if i > 0 and y[i-1] != 1:  # If previous was not neutral
                # 20% chance to continue in same direction, 30% neutral, 50% reverse
                if y[i-1] == 0:  # Previous was down
                    direction_probs = [0.2, 0.3, 0.5]
                else:  # Previous was up
                    direction_probs = [0.5, 0.3, 0.2]
            else:
                direction_probs = [0.33, 0.34, 0.33]
        
        price_direction = np.random.choice([-1, 0, 1], p=direction_probs)
        
        # Sometimes the signals are misleading (false signals)
        signal_reliability = 0.8  # 80% chance the order book signals match the price direction
        
        # Generate the apparent signal in the order book (what the model sees)
        if random.random() > signal_reliability:
            # False signal - book suggests different direction than actual
            apparent_direction = random.choice([-1, 0, 1])
            while apparent_direction == price_direction:
                apparent_direction = random.choice([-1, 0, 1])
        else:
            # True signal - book signals match price direction
            apparent_direction = price_direction
            
        # Generate a price series with realistic dynamics
        curr_midprice = midprice
        price_series = []
        
        # Base spread changes based on volatility and direction
        base_spread = spread * (1 + 0.3 * abs(price_direction))
        
        # Signal parameters (vary by direction)
        if apparent_direction == 1:  # Uptrend signal
            # Buying pressure: tighter spreads, more bid volume
            spread_factor_start = 0.9
            spread_factor_end = 0.7
            bid_vol_factor_start = 1.1
            bid_vol_factor_end = 1.5
            ask_vol_factor_start = 0.9
            ask_vol_factor_end = 0.7
        elif apparent_direction == -1:  # Downtrend signal
            # Selling pressure: wider spreads, more ask volume
            spread_factor_start = 1.1
            spread_factor_end = 1.3
            bid_vol_factor_start = 0.9
            bid_vol_factor_end = 0.7
            ask_vol_factor_start = 1.1
            ask_vol_factor_end = 1.5
        else:  # Neutral signal
            # Balanced: stable spreads, balanced volume
            spread_factor_start = 1.0
            spread_factor_end = 1.0
            bid_vol_factor_start = 1.0
            bid_vol_factor_end = 1.0
            ask_vol_factor_start = 1.0
            ask_vol_factor_end = 1.0
        
        # Add signal delay - signals don't always lead price immediately
        signal_delay = random.randint(0, 15)  # Signal may precede price by 0-15 steps
        
        # Generate price movement with realistic dynamics
        for t in range(time_steps):
            # Calculate trend component based on price direction
            # Add autocorrelation and momentum effects
            
            # Autocorrelation: recent price changes influence current change
            recent_trend = 0
            if t >= 3:
                recent_change = (price_series[-1] - price_series[-3]) / price_series[-3]
                recent_trend = recent_change * 0.4  # 40% influence from recent changes
            
            # Add trend component
            if t >= signal_delay:
                # After delay, price moves in the direction of the actual movement
                trend = price_direction * price_vol * curr_midprice * 0.6 + recent_trend
            else:
                # Before delay, price moves less predictably
                trend = random.uniform(-0.5, 0.5) * price_vol * curr_midprice + recent_trend
            
            # Add realistic noise (more volatile when prices are changing)
            noise_factor = 1.0 + 0.5 * abs(price_direction) + 0.3 * abs(recent_trend)
            noise = np.random.normal(0, price_vol * curr_midprice * 0.3 * noise_factor)
            
            # Update price 
            curr_midprice += trend + noise
            price_series.append(curr_midprice)
            
            # Determine current spread factor (interpolate from start to end)
            progress = t / (time_steps - 1)
            curr_spread_factor = spread_factor_start + (spread_factor_end - spread_factor_start) * progress
            
            # Calculate current volume factors
            curr_bid_vol_factor = bid_vol_factor_start + (bid_vol_factor_end - bid_vol_factor_start) * progress
            curr_ask_vol_factor = ask_vol_factor_start + (ask_vol_factor_end - ask_vol_factor_start) * progress
            
            # Add randomness to factors
            curr_spread_factor *= (1 + np.random.normal(0, 0.1))
            curr_bid_vol_factor *= (1 + np.random.normal(0, 0.15))
            curr_ask_vol_factor *= (1 + np.random.normal(0, 0.15))
            
            # Calculate current spread
            curr_spread = base_spread * curr_spread_factor
            
            # Best bid and ask prices
            best_ask = curr_midprice + curr_spread / 2
            best_bid = curr_midprice - curr_spread / 2
            
            # Generate LOB data for this time step
            for level in range(num_levels):
                # Price increases as level increases
                # Volume typically decreases as level increases
                
                # Ask side (higher prices)
                level_factor = 1 + level * 0.2  # Prices get higher as level increases
                ask_price = best_ask * level_factor * (1 + np.random.normal(0, 0.01))
                # Volume decreases with level, with more randomness at higher levels
                volume_decay = np.exp(-0.3 * level) * (1 + np.random.normal(0, 0.1 + 0.02 * level))
                ask_volume = max(10, 100 * volume_decay * curr_ask_vol_factor)
                
                # Bid side (lower prices)
                bid_price = best_bid / level_factor * (1 + np.random.normal(0, 0.01))
                bid_volume = max(10, 100 * volume_decay * curr_bid_vol_factor)
                
                # Store in feature matrix
                X[i, t, level*2] = ask_price
                X[i, t, level*2+1] = ask_volume
                X[i, t, num_levels*2 + level*2] = bid_price
                X[i, t, num_levels*2 + level*2+1] = bid_volume
        
        # Set the label based on the actual price direction
        if price_direction == 1:
            y[i] = 2  # Up (0-indexed)
        elif price_direction == 0:
            y[i] = 1  # Neutral
        else:
            y[i] = 0  # Down
        
        # Update midprice for the next sample
        midprice = curr_midprice
    
    # Add realistic market noise
    X = add_realistic_market_noise(X)
    
    # Normalize the data using z-score for each feature
    X_reshaped = X.reshape(-1, features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(num_samples, time_steps, features)
    
    # Print label distribution
    print("Label distribution:")
    for i, label_name in enumerate(["Down", "Neutral", "Up"]):
        count = np.sum(y == i)
        percentage = (count / num_samples) * 100
        print(f"{label_name}: {count} ({percentage:.2f}%)")
    
    return X, y

# Enhanced CNN-LSTM model - simpler but effective
class EnhancedCNNLSTM(nn.Module):
    def __init__(self, input_channels=40, num_classes=3, dropout_rate=0.3):
        super(EnhancedCNNLSTM, self).__init__()
        
        # Feature extraction with CNNs
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduce sequence length by half
            nn.Dropout(dropout_rate)
        )
        
        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layer with regularization
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    
    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
        batch_size = x.size(0)
        
        # Transpose for CNN: (batch_size, features, seq_length)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Transpose back for LSTM: (batch_size, seq_length/2, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length/2, hidden_size*2)
        
        # Apply attention
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_length/2, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Apply classifier
        out = self.classifier(context)
        
        return out  # Return logits (without softmax)

# Cross entropy loss with label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        
        # Create one-hot encoding
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

# Training function with more regularization
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=7):
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            
            # Track predictions
            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Print progress
            if (i+1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Calculate epoch training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_predictions)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track loss
                val_loss += loss.item()
                
                # Track predictions
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate epoch validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_predictions)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

# Evaluate model performance
def evaluate_model(model, test_loader):
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    # Class-specific metrics
    print("\nClass-specific metrics:")
    for cls in range(3):
        mask = np.array(all_targets) == cls
        if np.sum(mask) > 0:
            cls_accuracy = accuracy_score(np.array(all_targets)[mask], np.array(all_predictions)[mask])
        else:
            cls_accuracy = 0
            
        direction = "Down" if cls == 0 else "Neutral" if cls == 1 else "Up"
        print(f"{direction}: Accuracy={cls_accuracy:.4f}")
    
    print(f"\nOverall metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Down", "Neutral", "Up"],
                yticklabels=["Down", "Neutral", "Up"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Confidence distribution
    plt.figure(figsize=(10, 6))
    
    # Get max probability for each prediction
    max_probs = np.max(all_probabilities, axis=1)
    
    # Use matplotlib hist instead of seaborn histplot
    plt.hist(max_probs, bins=30, density=True, alpha=0.7)
    plt.axvline(x=0.33, color='r', linestyle='--', label='Random Guess (33%)')
    plt.axvline(x=0.5, color='orange', linestyle='--', label='50% Confidence')
    plt.axvline(x=0.75, color='g', linestyle='--', label='75% Confidence')
    
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'targets': all_targets
    }

# Practical trading strategy with risk management
def practical_trading_strategy(model, X_test, y_test):
    model.eval()
    
    # Process test data in batches
    batch_size = 64
    num_batches = (len(X_test) + batch_size - 1) // batch_size
    
    all_predictions = []
    all_probabilities = []
    
    # Get all predictions and probabilities
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(X_test))
        
        test_tensor = torch.FloatTensor(X_test[start_idx:end_idx]).to(device)
        
        with torch.no_grad():
            outputs = model(test_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Print prediction distribution
    print("\nModel prediction distribution:")
    for i, label_name in enumerate(["Down", "Neutral", "Up"]):
        count = np.sum(all_predictions == i)
        percentage = (count / len(all_predictions)) * 100
        print(f"{label_name}: {count} ({percentage:.2f}%)")
    
    # Convert predictions to trading signals (-1, 0, 1)
    prediction_signals = np.zeros_like(all_predictions, dtype=int)
    prediction_signals[all_predictions == 0] = -1  # Down
    prediction_signals[all_predictions == 1] = 0   # Neutral
    prediction_signals[all_predictions == 2] = 1   # Up
    
    # Convert true labels to signals for price generation
    true_signals = np.zeros_like(y_test, dtype=int)
    true_signals[y_test == 0] = -1  # Down
    true_signals[y_test == 1] = 0   # Neutral
    true_signals[y_test == 2] = 1   # Up
    
    # Trading parameters
    initial_balance = 10000
    balance = initial_balance
    position = 0  # 0=neutral, 1=long, -1=short
    
    # Risk parameters
    confidence_threshold = 0.6
    min_hold_period = 5  # Minimum hold time to avoid overtrading
    max_hold_period = 20  # Maximum hold time to limit exposure
    stop_loss_pct = 0.015  # 1.5% stop loss
    take_profit_pct = 0.025  # 2.5% take profit
    position_sizing = 0.2  # Max 20% of balance per trade
    max_trades_per_day = 2  # Limit number of trades
    transaction_cost = 0.001  # 0.1% per trade
    
    # Tracking
    current_hold_time = 0
    trades_today = 0
    day_counter = 0
    last_trade_idx = -min_hold_period  # Start with ability to trade
    balances = [initial_balance]
    positions = [0]
    trade_history = []
    entry_price = 0
    
    # Price simulation
    price = 100
    price_series = [price]
    
    # Daily resets
    day_change_frequency = 10  # Every 10 steps represents a new trading day
    
    for i in range(1, len(y_test)):
        # Check for new trading day
        if i % day_change_frequency == 0:
            day_counter += 1
            trades_today = 0
        
        # Update price with true movement direction and noise
        price_change_pct = true_signals[i] * 0.001 + np.random.normal(0, 0.002)
        price_change = price * price_change_pct
        price += price_change
        price_series.append(price)
        
        # Update balance based on current position
        if position != 0:
            # Update position profit/loss
            balance += position * price_change * position_sizing * balance / entry_price
            current_hold_time += 1
        
        # Calculate current profit percentage if in position
        profit_pct = 0
        if position != 0 and entry_price > 0:
            profit_pct = (price - entry_price) / entry_price * position
        
        # Check stop loss
        if position != 0 and profit_pct <= -stop_loss_pct:
            # Close position - stop loss hit
            balance -= abs(position) * price * transaction_cost * position_sizing
            
            trade_history.append({
                'index': i,
                'day': day_counter,
                'price': price,
                'old_position': position,
                'new_position': 0,
                'reason': 'Stop Loss',
                'confidence': np.max(all_probabilities[i]),
                'profit_pct': profit_pct * 100,
                'balance': balance
            })
            
            position = 0
            current_hold_time = 0
        
        # Check take profit
        elif position != 0 and profit_pct >= take_profit_pct:
            # Close position - take profit hit
            balance -= abs(position) * price * transaction_cost * position_sizing
            
            trade_history.append({
                'index': i,
                'day': day_counter,
                'price': price,
                'old_position': position,
                'new_position': 0,
                'reason': 'Take Profit',
                'confidence': np.max(all_probabilities[i]),
                'profit_pct': profit_pct * 100,
                'balance': balance
            })
            
            position = 0
            current_hold_time = 0
        
        # Check max hold time
        elif position != 0 and current_hold_time >= max_hold_period:
            # Close position - max hold time reached
            balance -= abs(position) * price * transaction_cost * position_sizing
            
            trade_history.append({
                'index': i,
                'day': day_counter,
                'price': price,
                'old_position': position,
                'new_position': 0,
                'reason': 'Max Hold Time',
                'confidence': np.max(all_probabilities[i]),
                'profit_pct': profit_pct * 100,
                'balance': balance
            })
            
            position = 0
            current_hold_time = 0
        
        # Trading logic
        if (trades_today < max_trades_per_day and 
            i - last_trade_idx > min_hold_period and 
            position == 0):  # Only enter if not already in position
            
            # Get prediction and confidence
            prediction = prediction_signals[i]
            confidence = all_probabilities[i][all_predictions[i]]
            
            # Only trade if confident and not neutral
            if confidence > confidence_threshold and prediction != 0:
                # Enter new position
                entry_price = price
                position = prediction
                last_trade_idx = i
                trades_today += 1
                current_hold_time = 0
                
                # Apply transaction cost
                balance -= abs(position) * price * transaction_cost * position_sizing
                
                trade_history.append({
                    'index': i,
                    'day': day_counter,
                    'price': price,
                    'old_position': 0,
                    'new_position': position,
                    'reason': 'New Signal',
                    'confidence': confidence,
                    'profit_pct': 0,
                    'balance': balance
                })
        
        balances.append(balance)
        positions.append(position)
    
    # Print recent trades
    if trade_history:
        print("\nRecent trades:")
        for trade in trade_history[-5:]:
            print(f"Day: {trade['day']}, Price: {trade['price']:.2f}, Old: {trade['old_position']}, "
                  f"New: {trade['new_position']}, Reason: {trade['reason']}, "
                  f"Confidence: {trade['confidence']:.4f}, Profit: {trade['profit_pct']:.2f}%")
    else:
        print("No trades were executed.")
    
    # Calculate trading performance metrics
    returns = np.diff(balances) / balances[:-1]
    daily_returns = []
    
    # Group returns by day
    for day in range(int(len(returns) / day_change_frequency) + 1):
        start_idx = day * day_change_frequency
        end_idx = min(start_idx + day_change_frequency, len(returns))
        if start_idx < end_idx:
            daily_return = (balances[end_idx] / balances[start_idx]) - 1
            daily_returns.append(daily_return)
    
    daily_returns = np.array(daily_returns)
    
    # Calculate metrics
    total_return = (balance / initial_balance - 1) * 100
    annual_trading_days = 252
    total_days = len(daily_returns)
    annualized_return = ((balance / initial_balance) ** (annual_trading_days / total_days) - 1) * 100 if total_days > 0 else 0
    
    # Sharpe ratio
    sharpe_ratio = 0
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe_ratio = np.sqrt(annual_trading_days) * np.mean(daily_returns) / np.std(daily_returns)
    
    # Drawdown analysis
    peak = np.maximum.accumulate(balances)
    drawdown = (peak - balances) / peak
    max_drawdown = drawdown.max() * 100
    
    # Win rate
    winning_trades = sum(1 for trade in trade_history 
                        if 'profit_pct' in trade and trade['profit_pct'] > 0)
    win_rate = winning_trades / len(trade_history) * 100 if trade_history else 0
    
    # Average profit per trade
    profits = [trade['profit_pct'] for trade in trade_history if 'profit_pct' in trade]
    avg_profit = np.mean(profits) if profits else 0
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Price with trade signals
    plt.subplot(3, 1, 1)
    plt.plot(price_series, label='Price', color='gray')
    
    # Add markers for different trade types
    buy_indices = [trade['index'] for trade in trade_history if trade['new_position'] == 1]
    sell_indices = [trade['index'] for trade in trade_history if trade['new_position'] == -1]
    tp_indices = [trade['index'] for trade in trade_history if trade['reason'] == 'Take Profit']
    sl_indices = [trade['index'] for trade in trade_history if trade['reason'] == 'Stop Loss']
    
    if buy_indices:
        buy_prices = [price_series[i] for i in buy_indices]
        plt.plot(buy_indices, buy_prices, 'g^', markersize=8, label='Buy')
    
    if sell_indices:
        sell_prices = [price_series[i] for i in sell_indices]
        plt.plot(sell_indices, sell_prices, 'rv', markersize=8, label='Sell')
    
    if tp_indices:
        tp_prices = [price_series[i] for i in tp_indices]
        plt.plot(tp_indices, tp_prices, 'ko', markersize=6, label='Take Profit')
    
    if sl_indices:
        sl_prices = [price_series[i] for i in sl_indices]
        plt.plot(sl_indices, sl_prices, 'ro', markersize=6, label='Stop Loss')
    
    plt.title('Price Series with Trade Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Account Balance
    plt.subplot(3, 1, 2)
    plt.plot(balances, label='Balance')
    
    # Add day markers
    for day in range(1, day_counter + 1):
        plt.axvline(x=day * day_change_frequency, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Trading Account Balance')
    plt.xlabel('Time')
    plt.ylabel('Balance ($)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    plt.subplot(3, 1, 3)
    plt.plot(drawdown * 100, color='red', label='Drawdown %')
    plt.axhline(y=max_drawdown, color='black', linestyle='--', label=f'Max Drawdown: {max_drawdown:.2f}%')
    
    # Add day markers
    for day in range(1, day_counter + 1):
        plt.axvline(x=day * day_change_frequency, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Drawdown Over Time')
    plt.xlabel('Time')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print(f"\nTrading Performance Summary:")
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Number of Trades: {len(trade_history)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Profit per Trade: {avg_profit:.2f}%")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Return results
    return {
        'balances': balances,
        'price_series': price_series,
        'trade_history': trade_history,
        'returns': returns,
        'daily_returns': daily_returns,
        'positions': positions,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_profit': avg_profit
    }

# Main function
def main():
    # Parameters
    num_samples = 10000  # Number of samples (reduced to speed up training)
    num_levels = 10      # Number of LOB levels
    time_steps = 100     # Time steps per sample
    batch_size = 64      # Batch size
    num_epochs = 20      # Training epochs
    
    print("Generating realistic LOB data with imperfect signals...")
    X, y = generate_realistic_lob_data(num_samples=num_samples, num_levels=num_levels, time_steps=time_steps)
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, "
          f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
          f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_channels = X.shape[2]  # Number of features
    model = EnhancedCNNLSTM(input_channels=input_channels, num_classes=3).to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Define loss function
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train model
    print("\nTraining the CNN-LSTM model...")
    start_time = time.time()
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train Acc')
    plt.plot(history['val_accuracies'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate model
    print("\nEvaluating model on test data:")
    eval_results = evaluate_model(model, test_loader)
    
    # Run trading strategy
    print("\nRunning practical trading strategy with risk management...")
    trading_results = practical_trading_strategy(model, X_test, y_test)
    
    return model, history, eval_results, trading_results

if __name__ == "__main__":
    model, history, eval_results, trading_results = main()