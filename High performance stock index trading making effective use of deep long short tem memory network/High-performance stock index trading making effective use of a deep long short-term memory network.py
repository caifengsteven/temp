import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to simulate stock data
def simulate_stock_data(n_samples=2000, trend_cycles=5, vol_cycles=3):
    """
    Generate simulated stock price data with trends and volatility clustering
    
    Parameters:
    - n_samples: Number of days to simulate
    - trend_cycles: Number of trend cycles (up/down) in the data
    - vol_cycles: Number of volatility cycles
    
    Returns:
    - DataFrame with open, high, low, close, volume, and adjusted close prices
    """
    # Generate time index
    dates = pd.date_range(start='2010-01-01', periods=n_samples, freq='B')
    
    # Base price starting at 100
    base_price = 100
    
    # Generate trend component
    trend_cycle_length = n_samples // trend_cycles
    trend = np.zeros(n_samples)
    
    for i in range(trend_cycles):
        if i % 2 == 0:  # Upward trend
            trend[i*trend_cycle_length:(i+1)*trend_cycle_length] = np.linspace(0, 0.2, trend_cycle_length)
        else:  # Downward trend
            trend[i*trend_cycle_length:(i+1)*trend_cycle_length] = np.linspace(0.2, -0.1, trend_cycle_length)
    
    # Smooth the trend transitions
    from scipy.ndimage import gaussian_filter1d
    trend = gaussian_filter1d(trend, sigma=30)
    
    # Generate volatility component (volatility clustering)
    vol_cycle_length = n_samples // vol_cycles
    volatility = np.zeros(n_samples)
    
    for i in range(vol_cycles):
        if i % 2 == 0:  # Low volatility
            volatility[i*vol_cycle_length:(i+1)*vol_cycle_length] = np.linspace(0.005, 0.015, vol_cycle_length)
        else:  # High volatility
            volatility[i*vol_cycle_length:(i+1)*vol_cycle_length] = np.linspace(0.015, 0.005, vol_cycle_length)
    
    # Add random walk with the volatility
    random_walk = np.zeros(n_samples)
    random_walk[0] = 0
    
    for i in range(1, n_samples):
        random_walk[i] = random_walk[i-1] + np.random.normal(0, volatility[i])
        # Add mean reversion
        random_walk[i] = random_walk[i] - 0.05 * random_walk[i-1]
    
    # Combine trend and random walk components
    cumulative_returns = trend + random_walk
    price_series = base_price * np.exp(np.cumsum(cumulative_returns))
    
    # Generate open, high, low, close prices
    close_prices = price_series
    open_prices = close_prices * np.exp(np.random.normal(0, 0.005, n_samples))
    high_prices = np.maximum(close_prices, open_prices) * np.exp(np.random.uniform(0, 0.008, n_samples))
    low_prices = np.minimum(close_prices, open_prices) * np.exp(np.random.uniform(-0.008, 0, n_samples))
    
    # Generate volume (correlated with absolute returns and volatility)
    daily_returns = np.diff(np.log(close_prices), prepend=np.log(close_prices[0]))
    normalized_volumes = np.exp(10 + 3 * np.abs(daily_returns) + 2 * volatility + np.random.normal(0, 0.5, n_samples))
    
    # Create DataFrame with numeric index
    stock_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': normalized_volumes,
        'Adj Close': close_prices  # Assuming no dividends, so adj close = close
    }, index=range(n_samples))
    
    return stock_data

# LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=2, dropout=0.5, seq_length=10):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Linear layer to output the next value
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        
        # Predict for each time step
        out = self.fc(out)  # out: (batch_size, seq_length, 1)
        
        return out

class StockPredictor:
    def __init__(self, window_size=10, hidden_units=32, num_layers=2, dropout=0.5):
        self.window_size = window_size
        self.model = LSTMModel(
            input_size=6,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=window_size
        ).to(device)
        self.scaler = MinMaxScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def prepare_data(self, data):
        # Extract features
        features = np.column_stack([
            data['Adj Close'].values,
            data['Open'].values,
            data['Low'].values,
            data['High'].values,
            data['Close'].values,
            np.roll(data['Adj Close'].values, 1)  # Previous day's adj close
        ])
        
        # Fix first row
        features[0, 5] = features[0, 0]
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.window_size):
            X.append(scaled_features[i:i+self.window_size])
            y.append(scaled_features[i+1:i+self.window_size+1, 0])  # Predict adjusted close
        
        # Convert to tensors
        if len(X) > 0:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
            return X_tensor, y_tensor
        return None, None
    
    def train(self, data, epochs=5):
        X, y = self.prepare_data(data)
        if X is None:
            print("Not enough data for training")
            return
        
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict_next_day(self, data):
        if len(data) < self.window_size:
            return data['Adj Close'].iloc[-1]  # Return last known price if not enough data
        
        # Prepare input
        features = np.column_stack([
            data['Adj Close'].values,
            data['Open'].values,
            data['Low'].values,
            data['High'].values,
            data['Close'].values,
            np.roll(data['Adj Close'].values, 1)
        ])
        
        # Fix first row
        features[0, 5] = features[0, 0]
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get last window
        last_window = scaled_features[-self.window_size:]
        X = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
        
        # Get last prediction (next day)
        next_day_scaled = output[0, -1, 0].item()
        
        # Create a dummy row for inverse transform
        dummy_row = np.zeros(6)
        dummy_row[0] = next_day_scaled
        dummy_row = self.scaler.inverse_transform([dummy_row])[0]
        
        return dummy_row[0]  # Return predicted adj close

# Trading Strategy
class TradingStrategy:
    def __init__(self, num_bins=8):
        self.num_bins = num_bins
        self.returns_history = []
        self.cutoff_points = None
        self.bin_profits = {i: [] for i in range(1, num_bins + 1)}
        self.allocations = None
        self.position = 0
        self.buy_price = 0
        self.buy_bin = 0
    
    def update_distribution(self, predicted_return):
        self.returns_history.append(predicted_return)
        
        # Once we have enough history, calculate cutoffs
        if len(self.returns_history) >= 10:
            abs_returns = np.abs(self.returns_history)
            self.cutoff_points = [0]  # First bin is always at 0 (sell if return < 0)
            
            # Add remaining cutoff points based on distribution percentiles
            for i in range(1, self.num_bins):
                percentile = (i / (self.num_bins - 1)) * 60  # Top 60% as described in paper
                self.cutoff_points.append(np.percentile(abs_returns, percentile))
    
    def get_bin(self, predicted_return):
        if self.cutoff_points is None:
            return None  # Not enough history yet
        
        if predicted_return < self.cutoff_points[0]:
            return 1  # Sell bin
        
        for i in range(1, len(self.cutoff_points)):
            if i == len(self.cutoff_points) - 1 or predicted_return < self.cutoff_points[i+1]:
                return i + 1
        
        return self.num_bins  # Default to last bin
    
    def update_allocations(self):
        if self.allocations is None:
            # Initialize allocations (bin 1 is sell, others are buy initially)
            self.allocations = [None] + [1] * (self.num_bins - 1)
        
        # Update allocations based on historical profits
        for bin_idx in range(2, self.num_bins + 1):
            if self.bin_profits[bin_idx]:
                total_profit = sum(self.bin_profits[bin_idx])
                self.allocations[bin_idx] = 1 if total_profit > 0 else 0
    
    def decide_action(self, current_price, predicted_price, max_units=1):
        predicted_return = predicted_price / current_price - 1
        self.update_distribution(predicted_return)
        
        if self.cutoff_points is None:
            return 'hold', 0  # Not enough history yet
        
        bin_idx = self.get_bin(predicted_return)
        self.update_allocations()
        
        # Handle sell case (bin 1)
        if bin_idx == 1 and self.position > 0:
            # Record profit from this trade for the bin we bought in
            profit = current_price - self.buy_price
            self.bin_profits[self.buy_bin].append(profit)
            
            # Sell all position
            amount = self.position
            self.position = 0
            return 'sell', amount
        
        # Handle buy case
        if bin_idx > 1 and self.position == 0:
            allocation = self.allocations[bin_idx]
            if allocation > 0:
                # Buy
                amount = max_units * allocation
                self.position = amount
                self.buy_price = current_price
                self.buy_bin = bin_idx
                return 'buy', amount
        
        return 'hold', 0

# Backtest function
def backtest(data, window_size=10, hidden_units=32, num_layers=2, dropout=0.5,
             initial_capital=10000, max_allocation=1):
    # Initialize model and strategy
    predictor = StockPredictor(window_size, hidden_units, num_layers, dropout)
    strategy = TradingStrategy(num_bins=8)
    
    # Initialize portfolio values
    cash = initial_capital
    position = 0
    
    # Initialize results storage
    results = []
    
    # Initial training with first chunk of data
    train_data = data.iloc[:window_size*2]
    predictor.train(train_data, epochs=10)
    
    # Main backtest loop
    for i in range(window_size*2, len(data)-1):
        # Current price
        current_price = data.iloc[i]['Adj Close']
        
        # Periodically retrain the model
        if (i - window_size*2) % 5 == 0:
            predictor.train(data.iloc[:i], epochs=3)
        
        # Predict next day's price
        next_price = predictor.predict_next_day(data.iloc[:i])
        
        # Get trading decision
        action, amount = strategy.decide_action(current_price, next_price, max_allocation)
        
        # Execute trade
        if action == 'buy':
            cash -= amount * current_price
            position += amount
        elif action == 'sell':
            cash += amount * current_price
            position = 0
        
        # Calculate portfolio value
        holdings = position * current_price
        portfolio_value = cash + holdings
        
        # Record results
        results.append({
            'Date': i,
            'Price': current_price,
            'Predicted': next_price,
            'Action': action,
            'Position': position,
            'Cash': cash,
            'Portfolio': portfolio_value
        })
    
    return pd.DataFrame(results)

# Buy and Hold comparison
def buy_and_hold(data, initial_capital=10000):
    # Calculate how many units we can buy
    start_price = data.iloc[0]['Adj Close']
    units = initial_capital / start_price
    
    # Calculate portfolio value over time
    portfolio_values = units * data['Adj Close']
    
    return portfolio_values

# Evaluate performance
def evaluate_performance(results, buy_hold_values):
    # Last portfolio value
    final_port_value = results['Portfolio'].iloc[-1]
    initial_port_value = results['Portfolio'].iloc[0]
    
    # Cumulative returns
    cumulative_return = final_port_value / initial_port_value - 1
    bh_cumulative_return = buy_hold_values.iloc[-1] / buy_hold_values.iloc[0] - 1
    
    # Annualized returns (assuming 252 trading days in a year)
    days = len(results)
    annualized_return = (1 + cumulative_return) ** (252 / days) - 1
    bh_annualized_return = (1 + bh_cumulative_return) ** (252 / days) - 1
    
    # Volatility
    strategy_returns = results['Portfolio'].pct_change().dropna()
    strategy_volatility = strategy_returns.std() * np.sqrt(252)
    
    bh_returns = buy_hold_values.pct_change().dropna()
    bh_volatility = bh_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    strategy_sharpe = annualized_return / strategy_volatility if strategy_volatility > 0 else 0
    bh_sharpe = bh_annualized_return / bh_volatility if bh_volatility > 0 else 0
    
    # Max drawdown
    strategy_drawdown = (results['Portfolio'] / results['Portfolio'].cummax() - 1).min()
    bh_drawdown = (buy_hold_values / buy_hold_values.cummax() - 1).min()
    
    # Number of trades
    trades = results['Action'].value_counts()
    num_trades = trades.get('buy', 0) + trades.get('sell', 0)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Strategy': [
            cumulative_return,
            annualized_return,
            strategy_volatility,
            strategy_sharpe,
            strategy_drawdown,
            num_trades
        ],
        'Buy & Hold': [
            bh_cumulative_return,
            bh_annualized_return,
            bh_volatility,
            bh_sharpe,
            bh_drawdown,
            1  # Buy and hold makes one buy
        ]
    }, index=[
        'Cumulative Return',
        'Annualized Return',
        'Annualized Volatility',
        'Sharpe Ratio',
        'Max Drawdown',
        'Number of Trades'
    ])
    
    return comparison

# Run simulation and backtest
def run_simulation():
    print("Simulating stock data...")
    # Generate data
    data = simulate_stock_data(n_samples=300, trend_cycles=3, vol_cycles=2)
    
    print("Data simulation complete.")
    print(f"Generated {len(data)} days of stock data.")
    
    print("\nRunning backtest...")
    backtest_results = backtest(
        data,
        window_size=10,
        hidden_units=32,
        num_layers=2,
        dropout=0.5,
        initial_capital=10000,
        max_allocation=1
    )
    
    print("Backtest complete.")
    
    # Calculate buy and hold performance
    buy_hold_values = buy_and_hold(data.iloc[backtest_results['Date'].values], 10000)
    
    # Evaluate performance
    performance = evaluate_performance(backtest_results, buy_hold_values)
    
    print("\nPerformance comparison:")
    print(performance)
    
    # Visualize results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Asset price and trading actions
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results['Date'], backtest_results['Price'], label='Price')
    
    buy_indices = backtest_results[backtest_results['Action'] == 'buy']['Date']
    sell_indices = backtest_results[backtest_results['Action'] == 'sell']['Date']
    
    if len(buy_indices) > 0:
        buy_prices = backtest_results[backtest_results['Action'] == 'buy']['Price']
        plt.scatter(buy_indices, buy_prices, marker='^', color='green', s=100, label='Buy')
    
    if len(sell_indices) > 0:
        sell_prices = backtest_results[backtest_results['Action'] == 'sell']['Price']
        plt.scatter(sell_indices, sell_prices, marker='v', color='red', s=100, label='Sell')
    
    plt.title('Asset Price and Trading Actions')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot 2: Strategy vs Buy & Hold performance
    plt.subplot(2, 1, 2)
    plt.plot(backtest_results['Date'], backtest_results['Portfolio'], 
             label='Strategy Portfolio', color='blue')
    plt.plot(backtest_results['Date'], buy_hold_values.values, 
             label='Buy & Hold Portfolio', color='orange')
    plt.title('Strategy vs Buy & Hold Performance')
    plt.xlabel('Day')
    plt.ylabel('Portfolio Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trading_results.png')
    print("Results visualization saved to 'trading_results.png'")
    
    return data, backtest_results, performance, buy_hold_values

# Run simulation
if __name__ == "__main__":
    data, backtest_results, performance, buy_hold_values = run_simulation()