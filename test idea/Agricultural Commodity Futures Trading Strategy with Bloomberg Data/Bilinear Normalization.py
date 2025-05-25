import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#############################
### DATA GENERATION PHASE ###
#############################

def generate_simple_price_series(n_samples=10000, volatility=0.01, drift=0.0005):
    """Generate a simple price series with trends for testing trading strategies"""
    # Initialize price at 100
    prices = np.zeros(n_samples)
    prices[0] = 100.0
    
    # Generate random price changes with drift
    for i in range(1, n_samples):
        # Occasionally introduce stronger trends
        if random.random() < 0.2:  # 20% chance of stronger trend (increased from 10%)
            trend_factor = random.choice([-5, 5])  # Stronger trends (increased from 3)
        else:
            trend_factor = 1
        
        # Calculate price change
        change = np.random.normal(drift * trend_factor, volatility)
        prices[i] = prices[i-1] * (1 + change)
    
    return prices

def generate_features_and_labels(prices, window_size=20, horizon=5):
    """Generate features and labels from price series"""
    n_samples = len(prices)
    features = []
    labels = []
    
    # For each point, we'll use window_size previous prices as features
    for i in range(window_size, n_samples - horizon):
        # Features: window of prices, plus basic indicators
        price_window = prices[i-window_size:i]
        
        # Calculate some basic indicators:
        # - Price changes
        price_changes = np.diff(price_window) / price_window[:-1]
        
        # - Moving averages
        ma5 = np.mean(price_window[-5:])
        ma10 = np.mean(price_window[-10:])
        ma_ratio = ma5 / ma10 - 1  # Percentage difference
        
        # - Volatility (standard deviation of returns)
        volatility = np.std(price_changes) 
        
        # Combine features
        feature_vector = np.concatenate([
            price_window / price_window[-1] - 1,  # Normalized price changes relative to current price
            [ma_ratio, volatility]  # Technical indicators
        ])
        
        features.append(feature_vector)
        
        # Label: Direction of future price movement
        current_price = prices[i]
        future_price = prices[i + horizon]
        
        # SMALLER threshold to generate more signals (changed from 0.001)
        threshold = 0.0005  
        
        if future_price > current_price * (1 + threshold):
            labels.append(2)  # Price up
        elif future_price < current_price * (1 - threshold):
            labels.append(0)  # Price down
        else:
            labels.append(1)  # Price flat
    
    return np.array(features), np.array(labels)

def prepare_dataset(n_samples=10000, window_size=20, horizon=5, train_ratio=0.7, val_ratio=0.15):
    """Prepare dataset for training, validation and testing"""
    # Generate price series
    prices = generate_simple_price_series(n_samples=n_samples)
    
    # Create features and labels
    features, labels = generate_features_and_labels(
        prices=prices, window_size=window_size, horizon=horizon
    )
    
    # Split data
    train_size = int(len(features) * train_ratio)
    val_size = int(len(features) * val_ratio)
    
    X_train = features[:train_size]
    y_train = labels[:train_size]
    
    X_val = features[train_size:train_size+val_size]
    y_val = labels[train_size:train_size+val_size]
    
    X_test = features[train_size+val_size:]
    y_test = labels[train_size+val_size:]
    
    # Also return the prices for backtesting
    test_prices = prices[train_size+val_size+window_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, test_prices

###########################
### MODEL DEFINITION ###
###########################

class SimpleNetwork(nn.Module):
    """Simple feed-forward network for price prediction"""
    def __init__(self, input_size, n_classes=3):
        super(SimpleNetwork, self).__init__()
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, n_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Forward pass
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PriceDataset(Dataset):
    """Dataset for price prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """Train the model"""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate average metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: ")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

###########################
### TRADING STRATEGY ###
###########################

class TradingStrategy:
    """Simple trading strategy based on model predictions"""
    def __init__(self, model):
        self.model = model
        self.position = 0  # 0: no position, +: long, -: short
        self.entry_price = 0
        self.trades = []
        
    def predict(self, features):
        """Get prediction from model"""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(features).unsqueeze(0).to(device)
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
        return prediction, confidence
    
    def decide(self, features, current_price):
        """Make trading decision based on prediction"""
        prediction, confidence = self.predict(features)
        
        action = "HOLD"
        size = 0
        reason = ""
        
        # No position - potentially enter
        if self.position == 0:
            # REDUCED confidence threshold from 0.6 to 0.4
            if confidence > 0.4:  
                if prediction == 2:  # Bullish prediction
                    action = "BUY"
                    size = 1
                    reason = f"Bullish signal (conf: {confidence:.2f})"
                    self.position = size
                    self.entry_price = current_price
                elif prediction == 0:  # Bearish prediction
                    action = "SELL"
                    size = 1
                    reason = f"Bearish signal (conf: {confidence:.2f})"
                    self.position = -size
                    self.entry_price = current_price
        
        # Long position - potentially exit
        elif self.position > 0:
            # REDUCED confidence threshold from 0.7 to 0.5
            if prediction == 0 or (prediction == 1 and confidence > 0.5):
                action = "SELL"
                size = self.position
                reason = f"Exit long (conf: {confidence:.2f})"
                
                # Record trade
                pnl = (current_price - self.entry_price) * size
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl,
                    'holding_period': 1  # Placeholder
                })
                
                self.position = 0
                self.entry_price = 0
        
        # Short position - potentially exit
        elif self.position < 0:
            # REDUCED confidence threshold from 0.7 to 0.5
            if prediction == 2 or (prediction == 1 and confidence > 0.5):
                action = "BUY"
                size = abs(self.position)
                reason = f"Exit short (conf: {confidence:.2f})"
                
                # Record trade
                pnl = (self.entry_price - current_price) * size
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl,
                    'holding_period': 1  # Placeholder
                })
                
                self.position = 0
                self.entry_price = 0
        
        return {
            'action': action,
            'size': size,
            'reason': reason,
            'confidence': confidence,
            'prediction': prediction
        }
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0
            }
        
        # Calculate metrics
        total_trades = len(self.trades)
        profitable_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t['pnl'] for t in profitable_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate returns for Sharpe ratio
        returns = [t['pnl'] / t['entry_price'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }

# ADDED A SIMPLER STRATEGY FOR BACKTESTING
class SimpleMovingAverageStrategy:
    """Simple Moving Average Crossover Strategy for comparison"""
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window
        self.position = 0
        self.entry_price = 0
        self.trades = []
    
    def calculate_signal(self, prices):
        """Calculate trading signal based on MA crossover"""
        # Need at least long_window prices
        if len(prices) < self.long_window:
            return 0
        
        # Calculate moving averages
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Generate signal
        if short_ma > long_ma:
            return 1  # Bullish
        elif short_ma < long_ma:
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def decide(self, prices, current_price):
        """Make trading decision based on MA crossover"""
        signal = self.calculate_signal(prices)
        
        action = "HOLD"
        size = 0
        reason = ""
        
        # No position - potentially enter
        if self.position == 0:
            if signal == 1:  # Bullish
                action = "BUY"
                size = 1
                reason = "MA Crossover: Bullish"
                self.position = size
                self.entry_price = current_price
            elif signal == -1:  # Bearish
                action = "SELL"
                size = 1
                reason = "MA Crossover: Bearish"
                self.position = -size
                self.entry_price = current_price
        
        # Long position - potentially exit
        elif self.position > 0:
            if signal == -1:  # Bearish
                action = "SELL"
                size = self.position
                reason = "MA Crossover: Exit Long"
                
                # Record trade
                pnl = (current_price - self.entry_price) * size
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl,
                    'holding_period': 1  # Placeholder
                })
                
                self.position = 0
                self.entry_price = 0
        
        # Short position - potentially exit
        elif self.position < 0:
            if signal == 1:  # Bullish
                action = "BUY"
                size = abs(self.position)
                reason = "MA Crossover: Exit Short"
                
                # Record trade
                pnl = (self.entry_price - current_price) * size
                self.trades.append({
                    'entry_price': self.entry_price,
                    'exit_price': current_price,
                    'position': self.position,
                    'pnl': pnl,
                    'holding_period': 1  # Placeholder
                })
                
                self.position = 0
                self.entry_price = 0
        
        return {
            'action': action,
            'size': size,
            'reason': reason,
            'signal': signal
        }
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate metrics
        total_trades = len(self.trades)
        profitable_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t['pnl'] for t in profitable_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

def backtest_strategy(model, features, prices, initial_capital=100000):
    """Run backtest of trading strategy"""
    # Initialize strategy
    strategy = TradingStrategy(model)
    
    # Initialize account
    account_value = initial_capital
    holdings = 0
    trades = []
    equity_curve = [initial_capital]
    
    # Also initialize the MA strategy for comparison
    ma_strategy = SimpleMovingAverageStrategy()
    ma_account_value = initial_capital
    ma_holdings = 0
    ma_trades = []
    ma_equity_curve = [initial_capital]
    
    # Store price history
    price_history = []
    
    # Run through each time step
    for i in range(len(features)):
        try:
            # Get current feature vector and price
            feature_vector = features[i]
            current_price = prices[i]
            
            # Store price in history
            price_history.append(current_price)
            
            # Get ML strategy decision
            decision = strategy.decide(feature_vector, current_price)
            
            # Execute ML strategy decision
            if decision['action'] == 'BUY':
                # Calculate purchase
                size = decision['size']
                cost = current_price * size
                
                # Update account
                account_value -= cost
                holdings += size
                
                # Record trade
                trades.append({
                    'timestamp': i,
                    'action': 'BUY',
                    'price': current_price,
                    'size': size,
                    'cost': cost,
                    'reason': decision['reason']
                })
                
            elif decision['action'] == 'SELL':
                # Calculate sale
                size = decision['size']
                revenue = current_price * size
                
                # Update account
                account_value += revenue
                holdings -= size
                
                # Record trade
                trades.append({
                    'timestamp': i,
                    'action': 'SELL',
                    'price': current_price,
                    'size': size,
                    'revenue': revenue,
                    'reason': decision['reason']
                })
            
            # Get MA strategy decision
            ma_decision = ma_strategy.decide(price_history, current_price)
            
            # Execute MA strategy decision
            if ma_decision['action'] == 'BUY':
                # Calculate purchase
                size = ma_decision['size']
                cost = current_price * size
                
                # Update account
                ma_account_value -= cost
                ma_holdings += size
                
                # Record trade
                ma_trades.append({
                    'timestamp': i,
                    'action': 'BUY',
                    'price': current_price,
                    'size': size,
                    'cost': cost,
                    'reason': ma_decision['reason']
                })
                
            elif ma_decision['action'] == 'SELL':
                # Calculate sale
                size = ma_decision['size']
                revenue = current_price * size
                
                # Update account
                ma_account_value += revenue
                ma_holdings -= size
                
                # Record trade
                ma_trades.append({
                    'timestamp': i,
                    'action': 'SELL',
                    'price': current_price,
                    'size': size,
                    'revenue': revenue,
                    'reason': ma_decision['reason']
                })
            
            # Update equity curves
            ml_equity = account_value + holdings * current_price
            equity_curve.append(ml_equity)
            
            ma_equity = ma_account_value + ma_holdings * current_price
            ma_equity_curve.append(ma_equity)
            
        except Exception as e:
            print(f"Error at time step {i}: {str(e)}")
            if equity_curve:
                equity_curve.append(equity_curve[-1])
            if ma_equity_curve:
                ma_equity_curve.append(ma_equity_curve[-1])
            continue
    
    # Calculate performance metrics
    ml_performance = strategy.get_performance_metrics()
    ma_performance = ma_strategy.get_performance_metrics()
    
    return {
        'ml_trades': trades,
        'ml_equity_curve': equity_curve,
        'ml_final_equity': equity_curve[-1] if equity_curve else initial_capital,
        'ml_return_pct': ((equity_curve[-1] / initial_capital - 1) * 100) if equity_curve else 0.0,
        'ml_performance': ml_performance,
        'ma_trades': ma_trades,
        'ma_equity_curve': ma_equity_curve,
        'ma_final_equity': ma_equity_curve[-1] if ma_equity_curve else initial_capital,
        'ma_return_pct': ((ma_equity_curve[-1] / initial_capital - 1) * 100) if ma_equity_curve else 0.0,
        'ma_performance': ma_performance
    }

def analyze_backtest_results(backtest_results):
    """Analyze and visualize backtest results"""
    ml_trades = backtest_results.get('ml_trades', [])
    ml_equity_curve = backtest_results.get('ml_equity_curve', [])
    ml_performance = backtest_results.get('ml_performance', {})
    
    ma_trades = backtest_results.get('ma_trades', [])
    ma_equity_curve = backtest_results.get('ma_equity_curve', [])
    ma_performance = backtest_results.get('ma_performance', {})
    
    # Print ML strategy performance summary
    print("===== ML STRATEGY PERFORMANCE SUMMARY =====")
    print(f"Initial Capital: ${ml_equity_curve[0]:,.2f}" if ml_equity_curve else "No equity curve data")
    print(f"Final Equity: ${backtest_results.get('ml_final_equity', 0):,.2f}")
    print(f"Total Return: {backtest_results.get('ml_return_pct', 0):,.2f}%")
    print(f"Total Trades: {ml_performance.get('total_trades', 0)}")
    print(f"Win Rate: {ml_performance.get('win_rate', 0)*100:,.2f}%")
    print(f"Average Profit: ${ml_performance.get('avg_profit', 0):,.2f}")
    print(f"Average Loss: ${ml_performance.get('avg_loss', 0):,.2f}")
    print(f"Profit Factor: {ml_performance.get('profit_factor', 0):,.2f}")
    
    # Print MA strategy performance summary
    print("\n===== MA STRATEGY PERFORMANCE SUMMARY =====")
    print(f"Initial Capital: ${ma_equity_curve[0]:,.2f}" if ma_equity_curve else "No equity curve data")
    print(f"Final Equity: ${backtest_results.get('ma_final_equity', 0):,.2f}")
    print(f"Total Return: {backtest_results.get('ma_return_pct', 0):,.2f}%")
    print(f"Total Trades: {ma_performance.get('total_trades', 0)}")
    print(f"Win Rate: {ma_performance.get('win_rate', 0)*100:,.2f}%")
    print(f"Average Profit: ${ma_performance.get('avg_profit', 0):,.2f}")
    print(f"Average Loss: ${ma_performance.get('avg_loss', 0):,.2f}")
    print(f"Profit Factor: {ma_performance.get('profit_factor', 0):,.2f}")
    
    # Plot equity curves
    if ml_equity_curve and ma_equity_curve:
        plt.figure(figsize=(12, 6))
        plt.plot(ml_equity_curve, label='ML Strategy')
        plt.plot(ma_equity_curve, label='MA Strategy')
        plt.title('Strategy Equity Curves')
        plt.xlabel('Trading Events')
        plt.ylabel('Account Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Analyze ML trade distribution
    if ml_trades:
        ml_trades_df = pd.DataFrame(ml_trades)
        
        # Plot trade reasons
        if 'reason' in ml_trades_df.columns:
            plt.figure(figsize=(12, 6))
            reason_counts = ml_trades_df['reason'].apply(lambda x: x.split('(')[0].strip()).value_counts()
            reason_counts.plot(kind='bar')
            plt.title('ML Strategy Trade Reasons')
            plt.xlabel('Reason')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        
        # Plot actions
        if 'action' in ml_trades_df.columns:
            plt.figure(figsize=(8, 5))
            action_counts = ml_trades_df['action'].value_counts()
            action_counts.plot(kind='bar')
            plt.title('ML Strategy Trade Actions')
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.grid(True)
            plt.show()

###########################
### MAIN FUNCTION ###
###########################

def main():
    try:
        # Prepare dataset
        print("Preparing dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test, test_prices = prepare_dataset(
            n_samples=30000,  # Even more data
            window_size=20,
            horizon=5,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        print(f"Training set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        print(f"Test set: {X_test.shape}, {y_test.shape}")
        print(f"Test prices: {test_prices.shape}")
        
        # Check distribution of labels
        print(f"Label distribution in training set: {np.bincount(y_train)}")
        
        # Create datasets
        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        test_dataset = PriceDataset(X_test, y_test)
        
        # Create dataloaders
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]  # Number of features
        model = SimpleNetwork(input_size=input_size, n_classes=3).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        print("\n" + "="*50)
        print("Training model...")
        print("="*50)
        
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=15  # Increased epochs
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Evaluate model on test set
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        
        test_acc = test_correct / test_total
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Visualize a sample of the test data
        plt.figure(figsize=(12, 6))
        plt.plot(test_prices[:200])  # Show first 200 prices
        plt.title('Sample of Test Price Data')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
        
        # Run backtest
        print("\n" + "="*50)
        print("Running trading backtest...")
        print("="*50)
        
        backtest_results = backtest_strategy(
            model=trained_model,
            features=X_test,
            prices=test_prices,
            initial_capital=100000
        )
        
        # Analyze results
        analyze_backtest_results(backtest_results)
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()