import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate simulated stock data
def generate_stock_data(num_days=1000, volatility=0.01, trend=0.0002, seasonality=0.1):
    """
    Generate simulated stock data with trend, seasonality, and random noise
    """
    # Initialize price at 100
    price = 100
    prices = [price]
    
    # Generate daily returns with trend, seasonality, and random noise
    for i in range(1, num_days):
        # Random component (noise)
        noise = np.random.normal(0, volatility)
        
        # Trend component
        trend_component = trend
        
        # Seasonality component (weekly pattern)
        seasonal_component = seasonality * np.sin(2 * np.pi * i / 20)
        
        # Calculate daily return
        daily_return = trend_component + seasonal_component + noise
        
        # Update price
        price = price * (1 + daily_return)
        prices.append(price)
    
    # Create dataframe
    dates = pd.date_range(start='2020-01-01', periods=num_days)
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    # Add Open, High, Low prices
    df['Open'] = df['Close'].shift(1)
    df.loc[0, 'Open'] = df.loc[0, 'Close'] * (1 - np.random.normal(0, volatility))
    
    # High is the maximum of Open and Close plus some random value
    df['High'] = df[['Open', 'Close']].max(axis=1) + df['Close'] * np.random.uniform(0, 0.01, num_days)
    
    # Low is the minimum of Open and Close minus some random value
    df['Low'] = df[['Open', 'Close']].min(axis=1) - df['Close'] * np.random.uniform(0, 0.01, num_days)
    
    # Add Volume (simulated)
    base_volume = 1000000  # Base daily volume
    df['Volume'] = base_volume + np.random.normal(0, base_volume * 0.2, num_days)
    df['Volume'] = df['Volume'].clip(lower=base_volume * 0.5)  # Ensure positive volume
    
    # Add Trading Value
    df['Amount'] = df['Close'] * df['Volume']
    
    # Add Turnover Rate (random between 1% and 3%)
    df['Turnover'] = np.random.uniform(0.01, 0.03, num_days)
    
    # Add Volume Ratio (random between 0.8 and 1.2)
    df['VolumeRatio'] = np.random.uniform(0.8, 1.2, num_days)
    
    # Add Price Ratios (PE, PB, PS)
    df['PE'] = np.random.uniform(15, 25, num_days)  # P/E ratio
    df['PB'] = np.random.uniform(1.5, 2.5, num_days)  # P/B ratio
    df['PS'] = np.random.uniform(2, 4, num_days)  # P/S ratio
    
    # Add Share Information (constant values)
    df['TotalShares'] = 10000000  # Total shares
    df['FloatShares'] = 8000000  # Float shares
    df['FreeFloatShares'] = 7000000  # Free float shares
    
    # Add Market Value Information
    df['TotalMarketValue'] = df['Close'] * df['TotalShares']
    df['CirculatingMarketValue'] = df['Close'] * df['FloatShares']
    
    return df

# Generate four simulated stocks with different characteristics
stock_data = {
    'Stock1': generate_stock_data(num_days=1000, volatility=0.01, trend=0.0003, seasonality=0.1),
    'Stock2': generate_stock_data(num_days=1000, volatility=0.015, trend=0.0001, seasonality=0.05),
    'Stock3': generate_stock_data(num_days=1000, volatility=0.02, trend=-0.0001, seasonality=0.15),
    'Stock4': generate_stock_data(num_days=1000, volatility=0.008, trend=0.0002, seasonality=0.08)
}

# Plot the simulated stock data
plt.figure(figsize=(15, 10))
for i, (name, data) in enumerate(stock_data.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(data['Date'], data['Close'])
    plt.title(f'{name} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
plt.tight_layout()
plt.savefig('simulated_stocks.png')
plt.show()

# Selective Scan State Space Layer (Mamba-like implementation)
class SelectiveScanStateSpace(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Parameters A, B, C, Delta for the state space model
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        
        # Projection layers
        self.proj_input = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_Delta = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish activation function
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x_proj = self.activation(self.proj_input(x))
        
        # Compute B, C, Delta (selective SSM parameters)
        B = self.proj_B(x_proj)  # (batch_size, seq_len, d_state)
        C = self.proj_C(x_proj)  # (batch_size, seq_len, d_state)
        Delta = torch.exp(self.proj_Delta(x_proj))  # (batch_size, seq_len, d_model)
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        # Output container
        output = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Manually perform the recurrent scan
        for t in range(seq_len):
            # Discretize A (using the diagonal approximation from the paper)
            A_discrete = torch.exp(Delta[:, t, :].unsqueeze(-1) * self.A)  # (batch_size, d_model, d_state)
            
            # Update state for each feature dimension
            for i in range(self.d_model):
                # Update state
                h = A_discrete[:, i, :] * h + B[:, t, :]
                
                # Compute output
                output[:, t, i] = torch.sum(C[:, t, :] * h, dim=1)
        
        return self.dropout(output)

# MambaStock model
class MambaStock(nn.Module):
    def __init__(self, input_dim, output_dim=1, d_model=64, d_state=16, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Stack of Selective Scan layers
        self.layers = nn.ModuleList([
            SelectiveScanStateSpace(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.activation = nn.Tanh()  # Constrain outputs to (-1, 1)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Apply Selective Scan layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        # Apply layer norm
        x = self.norm(x)
        
        # Project to output dimension and apply activation
        x = self.output_projection(x)
        x = self.activation(x)
        
        return x

# Prepare data for training and testing
def prepare_data(stock_df, seq_length=20, target_col='Close', split_ratio=0.8):
    """
    Prepare time series data for training and testing
    """
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Turnover', 
                'VolumeRatio', 'PE', 'PB', 'PS', 'TotalMarketValue', 'CirculatingMarketValue']
    
    # Extract features
    data = stock_df[features].values
    
    # Scale data
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler_X.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        # Calculate return rate for target
        next_price = stock_df[target_col].iloc[i+seq_length]
        current_price = stock_df[target_col].iloc[i+seq_length-1]
        return_rate = (next_price - current_price) / current_price
        # Constrain to [-1, 1] using tanh
        return_rate = np.tanh(return_rate * 10)  # Scale by 10 for better gradients
        y.append(return_rate)
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split data
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return train_dataset, test_dataset, scaler_X

# Train the model
def train_model(model, train_loader, test_loader, epochs=100, lr=0.01, device='cpu'):
    """
    Train the model and evaluate on test data
    """
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred[:, -1, :], y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                loss = criterion(y_pred[:, -1, :], y_batch)
                
                test_loss += loss.item()
        
        test_losses.append(test_loss / len(test_loader))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    
    return train_losses, test_losses

# Function to make predictions and evaluate model
def evaluate_model(model, test_loader, stock_df, seq_length, device='cpu'):
    """
    Make predictions and calculate performance metrics
    """
    model.eval()
    
    # Get all test data
    X_all = []
    y_all = []
    for X_batch, y_batch in test_loader:
        X_all.append(X_batch)
        y_all.append(y_batch)
    
    X_all = torch.cat(X_all, dim=0).to(device)
    y_all = torch.cat(y_all, dim=0).cpu().numpy()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_all)[:, -1, :].cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_all, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_all, y_pred)
    r2 = r2_score(y_all, y_pred)
    
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    
    # Get actual price data for comparison
    start_idx = len(stock_df) - len(y_pred) - 1
    actual_prices = stock_df['Close'].iloc[start_idx+1:].values
    
    # Convert predictions back to price
    # First, get the base prices (one day before each prediction)
    base_prices = stock_df['Close'].iloc[start_idx:-1].values
    
    # Inverse transform the tanh outputs to get return rates
    pred_return_rates = np.arctanh(y_pred) / 10  # Assuming we scaled by 10 in prepare_data
    
    # Calculate predicted prices
    pred_prices = base_prices * (1 + pred_return_rates.flatten())
    
    # Truncate to the same length
    min_len = min(len(actual_prices), len(pred_prices))
    actual_prices = actual_prices[:min_len]
    pred_prices = pred_prices[:min_len]
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(pred_prices, label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.show()
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'actual_prices': actual_prices,
        'pred_prices': pred_prices
    }

# Train and evaluate model for each stock
batch_size = 32
seq_length = 20
results = {}

for stock_name, stock_df in tqdm(stock_data.items(), desc="Processing stocks"):
    print(f"\nProcessing {stock_name}...")
    
    # Prepare data
    train_dataset, test_dataset, scaler = prepare_data(
        stock_df, seq_length=seq_length, target_col='Close', split_ratio=0.8
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Turnover', 
                          'VolumeRatio', 'PE', 'PB', 'PS', 'TotalMarketValue', 'CirculatingMarketValue']].shape[1]
    
    model = MambaStock(
        input_dim=input_dim,
        output_dim=1,
        d_model=64,
        d_state=16,
        n_layers=2,
        dropout=0.1
    ).to(device)
    
    # Train model
    print(f"Training model for {stock_name}...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, epochs=100, lr=0.01, device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{stock_name} - Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{stock_name}_training_loss.png')
    plt.show()
    
    # Evaluate model
    print(f"Evaluating model for {stock_name}...")
    eval_results = evaluate_model(model, test_loader, stock_df, seq_length, device=device)
    results[stock_name] = eval_results

# Compare performance across stocks
plt.figure(figsize=(15, 10))

# Plot MSE
plt.subplot(2, 2, 1)
stock_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in stock_names]
plt.bar(stock_names, mse_values)
plt.title('MSE Comparison')
plt.ylabel('MSE')
plt.grid(True, axis='y')

# Plot RMSE
plt.subplot(2, 2, 2)
rmse_values = [results[name]['RMSE'] for name in stock_names]
plt.bar(stock_names, rmse_values)
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.grid(True, axis='y')

# Plot MAE
plt.subplot(2, 2, 3)
mae_values = [results[name]['MAE'] for name in stock_names]
plt.bar(stock_names, mae_values)
plt.title('MAE Comparison')
plt.ylabel('MAE')
plt.grid(True, axis='y')

# Plot R2
plt.subplot(2, 2, 4)
r2_values = [results[name]['R2'] for name in stock_names]
plt.bar(stock_names, r2_values)
plt.title('R2 Comparison')
plt.ylabel('R2')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()

# Compare with baseline models
def train_baseline_lstm(train_loader, test_loader, input_dim, hidden_dim=64, num_layers=2, epochs=100, lr=0.01, device='cpu'):
    """
    Train a baseline LSTM model
    """
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.Tanh()
            
        def forward(self, x):
            # x shape: (batch_size, seq_len, input_dim)
            lstm_out, _ = self.lstm(x)
            # Take only the last time step output
            out = self.fc(lstm_out[:, -1, :])
            out = self.activation(out)
            return out
    
    # Initialize model
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}')
    
    # Evaluate
    model.eval()
    
    # Get all test data
    X_all = []
    y_all = []
    for X_batch, y_batch in test_loader:
        X_all.append(X_batch)
        y_all.append(y_batch)
    
    X_all = torch.cat(X_all, dim=0).to(device)
    y_all = torch.cat(y_all, dim=0).cpu().numpy()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_all).cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_all, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_all, y_pred)
    r2 = r2_score(y_all, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Compare MambaStock with LSTM baseline for one stock
stock_name = 'Stock1'  # Choose one stock for comparison
stock_df = stock_data[stock_name]

# Prepare data
train_dataset, test_dataset, scaler = prepare_data(
    stock_df, seq_length=seq_length, target_col='Close', split_ratio=0.8
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get input dimension
input_dim = stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Turnover', 
                      'VolumeRatio', 'PE', 'PB', 'PS', 'TotalMarketValue', 'CirculatingMarketValue']].shape[1]

# Train LSTM baseline
print(f"\nTraining LSTM baseline for {stock_name}...")
lstm_results = train_baseline_lstm(
    train_loader, test_loader, input_dim, hidden_dim=64, num_layers=2, epochs=100, lr=0.01, device=device
)

# Compare results
print("\nModel Comparison:")
print(f"MambaStock - MSE: {results[stock_name]['MSE']:.6f}, RMSE: {results[stock_name]['RMSE']:.6f}, MAE: {results[stock_name]['MAE']:.6f}, R2: {results[stock_name]['R2']:.6f}")
print(f"LSTM       - MSE: {lstm_results['MSE']:.6f}, RMSE: {lstm_results['RMSE']:.6f}, MAE: {lstm_results['MAE']:.6f}, R2: {lstm_results['R2']:.6f}")

# Plot comparison
models = ['MambaStock', 'LSTM']
mse_values = [results[stock_name]['MSE'], lstm_results['MSE']]
rmse_values = [results[stock_name]['RMSE'], lstm_results['RMSE']]
mae_values = [results[stock_name]['MAE'], lstm_results['MAE']]
r2_values = [results[stock_name]['R2'], lstm_results['R2']]

plt.figure(figsize=(15, 10))

# Plot MSE
plt.subplot(2, 2, 1)
plt.bar(models, mse_values)
plt.title('MSE Comparison')
plt.ylabel('MSE')
plt.grid(True, axis='y')

# Plot RMSE
plt.subplot(2, 2, 2)
plt.bar(models, rmse_values)
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.grid(True, axis='y')

# Plot MAE
plt.subplot(2, 2, 3)
plt.bar(models, mae_values)
plt.title('MAE Comparison')
plt.ylabel('MAE')
plt.grid(True, axis='y')

# Plot R2
plt.subplot(2, 2, 4)
plt.bar(models, r2_values)
plt.title('R2 Comparison')
plt.ylabel('R2')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()