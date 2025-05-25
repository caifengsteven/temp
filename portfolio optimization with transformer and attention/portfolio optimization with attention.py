import pdblp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
import xgboost as xgb
import os

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Bloomberg connection
con = pdblp.BCon(timeout=5000)
con.start()

# ================ 1. Model Components ================

# Time2Vec implementation
class Time2Vec(nn.Module):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
        
        # Linear part
        self.weights_linear = nn.Parameter(torch.randn(1, 1))
        self.bias_linear = nn.Parameter(torch.randn(1))
        
        # Periodic part
        self.weights_periodic = nn.Parameter(torch.randn(1, kernel_size))
        self.bias_periodic = nn.Parameter(torch.randn(kernel_size))
        
    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x[:, :, 0:1]
        
        # Linear part
        linear = x * self.weights_linear + self.bias_linear
        
        # Periodic part
        periodic = torch.sin(torch.matmul(x, self.weights_periodic) + self.bias_periodic)
        
        # Concatenate [linear, periodic]
        return torch.cat([linear, periodic], dim=-1)

# Gated Linear Unit (GLU)
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        
    def forward(self, x):
        # Split the tensor into two parts along the feature dimension
        a, b = torch.chunk(x, 2, dim=-1)
        return a * torch.sigmoid(b)

# Gated Residual Network (GRN)
class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRN, self).__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim * 2)  # *2 for GLU
        self.glu = GLU()
        
    def forward(self, x):
        residual = x
        
        # First dense layer with ELU activation
        out = self.fc1(x)
        out = self.elu(out)
        
        # Second dense layer followed by GLU
        out = self.fc2(out)
        out = self.glu(out)
        
        # Residual connection and layer normalization
        out = residual + out
        out = self.layer_norm(out)
        
        return out

# Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.grn = GRN(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # Self attention block
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # GRN block
        src = self.grn(src)
        
        return src

# Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.grn = GRN(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention block
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention block
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # GRN block
        tgt = self.grn(tgt)
        
        return tgt

# Complete Portfolio Transformer model
class PortfolioTransformer(nn.Module):
    def __init__(self, 
                 num_assets,
                 d_model=64, 
                 nhead=8, 
                 num_encoder_layers=4,
                 num_decoder_layers=4, 
                 hidden_dim=128,
                 time2vec_dim=10,
                 dropout=0.1):
        super(PortfolioTransformer, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_assets = num_assets
        self.time2vec_dim = time2vec_dim
        
        # Time embedding
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        
        # Input projection
        self.input_projection = nn.Linear(num_assets + time2vec_dim + 1, d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, hidden_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(d_model, num_assets)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def _create_time_encoding(self, seq_len, batch_size):
        # Create time steps: [0, 1, 2, ..., seq_len-1]
        time_steps = torch.arange(0, seq_len, device=device).unsqueeze(0).unsqueeze(-1)
        time_steps = time_steps.expand(batch_size, -1, -1).float()
        
        # Get time encoding
        time_encoding = self.time2vec(time_steps)
        return time_encoding
        
    def forward(self, src):
        # src: [batch_size, seq_len, num_assets]
        batch_size, seq_len, _ = src.shape
        
        # Get time encoding
        time_encoding = self._create_time_encoding(seq_len, batch_size)
        
        # Combine input features with time encoding
        # The shape of time_encoding is [batch_size, seq_len, time2vec_dim+1]
        time_encoding = time_encoding.expand(-1, -1, -1)
        src = torch.cat([src, time_encoding], dim=2)
        
        # Project to d_model
        src = self.input_projection(src)
        
        # Change to shape [seq_len, batch_size, d_model] for transformer layers
        src = src.permute(1, 0, 2)
        
        # Create mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(seq_len)
        
        # Encoder
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
        
        # Decoder
        output = src
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, tgt_mask=tgt_mask)
        
        # Get the last time step output
        output = output[-1]  # Shape: [batch_size, d_model]
        
        # Project to num_assets
        output = self.fc_out(output)  # Shape: [batch_size, num_assets]
        
        # Apply the sign and softmax as described in the paper
        weights = torch.sign(output) * torch.softmax(torch.abs(output), dim=1)
        
        return weights

# Custom dataset for financial time series
class FinancialDataset(Dataset):
    def __init__(self, returns_data, window_size):
        """
        Initialize dataset with financial returns
        
        Args:
            returns_data: DataFrame of asset returns
            window_size: Number of days to look back
        """
        self.returns_data = returns_data.values
        self.window_size = window_size
        self.T, self.num_assets = self.returns_data.shape
        
    def __len__(self):
        return self.T - self.window_size
    
    def __getitem__(self, idx):
        # Get window of historical returns
        x = self.returns_data[idx:idx+self.window_size]
        
        # Target is the next day's returns
        y = self.returns_data[idx+self.window_size]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Simplified Sharpe Loss without transaction costs for training
class SharpeLoss:
    def __call__(self, weights, returns):
        """
        Calculate negative Sharpe ratio without transaction costs (for training)
        
        Args:
            weights: Portfolio weights [batch_size, num_assets]
            returns: Asset returns [batch_size, num_assets]
        
        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Portfolio returns
        portfolio_returns = torch.sum(weights * returns, dim=1)
        
        # Calculate Sharpe ratio
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns, unbiased=False)
        
        # Negative Sharpe ratio (we want to maximize it)
        return -mean_return / (std_return + 1e-9)

# ================ 2. Training and Evaluation Functions ================

def train_portfolio_transformer(model, train_loader, val_loader, optimizer, num_epochs=100, patience=10):
    """
    Train the Portfolio Transformer model with simplified Sharpe loss
    
    Args:
        model: PortfolioTransformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
    
    Returns:
        Trained model
    """
    criterion = SharpeLoss()  # Simplified loss without transaction costs
    best_val_loss = float('inf')
    best_model = None
    no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            weights = model(data)
            
            loss = criterion(weights, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        train_loss /= batch_count
        
        # Validation
        model.eval()
        val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                weights = model(data)
                loss = criterion(weights, target)
                val_loss += loss.item()
                batch_count += 1
        
        val_loss /= batch_count
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def backtest_strategy(model, returns_data, window_size=50, rebalance_freq=1, cost_rate=0.0002):
    """
    Backtests the Portfolio Transformer strategy
    
    Args:
        model: Trained PortfolioTransformer model
        returns_data: DataFrame of asset returns
        window_size: Number of days to look back
        rebalance_freq: Frequency of rebalancing in days
        cost_rate: Transaction cost rate
    
    Returns:
        DataFrame with portfolio weights and performance metrics
    """
    model.eval()
    num_days = len(returns_data)
    num_assets = returns_data.shape[1]
    
    # Initialize arrays
    portfolio_weights = np.zeros((num_days, num_assets))
    portfolio_returns = np.zeros(num_days)
    
    # Initial positions
    prev_weights = None
    
    with torch.no_grad():
        for t in range(window_size, num_days, rebalance_freq):
            # Get historical window of returns
            historical_returns = returns_data.iloc[t-window_size:t].values
            
            # Convert to tensor
            historical_returns_tensor = torch.FloatTensor(historical_returns).unsqueeze(0).to(device)
            
            # Predict optimal weights
            weights = model(historical_returns_tensor).cpu().numpy()[0]
            
            # Store weights for the next rebalance_freq days
            for i in range(rebalance_freq):
                if t + i < num_days:
                    portfolio_weights[t + i] = weights
    
    # Calculate portfolio returns (with transaction costs)
    for t in range(window_size, num_days):
        if np.any(portfolio_weights[t-1]):
            # Calculate return for day t
            daily_return = np.sum(portfolio_weights[t-1] * returns_data.iloc[t].values)
            
            # Apply transaction costs
            if t > window_size and not np.array_equal(portfolio_weights[t-1], portfolio_weights[t-2]):
                transaction_costs = cost_rate * np.sum(np.abs(portfolio_weights[t-1] - portfolio_weights[t-2]))
                daily_return -= transaction_costs
                
            portfolio_returns[t] = daily_return
    
    # Create results DataFrame
    results = pd.DataFrame({
        'portfolio_return': portfolio_returns
    }, index=returns_data.index)
    
    # Add cumulative returns
    results['cumulative_return'] = (1 + results['portfolio_return']).cumprod() - 1
    
    # Add weights
    for i in range(num_assets):
        results[f'weight_{returns_data.columns[i]}'] = portfolio_weights[:, i]
    
    return results

# ================ 3. Benchmark Models ================

# Mean-Variance Optimization
def mean_variance_optimization(returns_data, window_size=50, cost_rate=0.0002):
    """
    Implements classical Markowitz mean-variance optimization
    """
    num_assets = returns_data.shape[1]
    num_days = len(returns_data)
    portfolio_weights = np.zeros((num_days, num_assets))
    portfolio_returns = np.zeros(num_days)
    
    def objective(weights, mean_returns, cov_matrix):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Negative Sharpe ratio (we want to maximize it)
        return -portfolio_return / (portfolio_volatility + 1e-9)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
    bounds = tuple((-1, 1) for _ in range(num_assets))
    
    for t in range(window_size, num_days):
        if t % 1 == 0:  # Daily rebalancing
            # Get historical window of returns
            historical_returns = returns_data.iloc[t-window_size:t]
            
            # Calculate mean returns and covariance
            mean_returns = historical_returns.mean().values
            cov_matrix = historical_returns.cov().values
            
            # Check if covariance matrix is positive definite
            try:
                # Initial guess: equal weights
                initial_weights = np.ones(num_assets) / num_assets
                
                # Optimize
                result = minimize(
                    objective,
                    initial_weights,
                    args=(mean_returns, cov_matrix),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                weights = result['x']
            except:
                # If optimization fails, use equal weights
                weights = np.ones(num_assets) / num_assets
                
            portfolio_weights[t] = weights
            
    # Calculate portfolio returns
    for t in range(window_size, num_days):
        if np.any(portfolio_weights[t-1]):
            daily_return = np.sum(portfolio_weights[t-1] * returns_data.iloc[t].values)
            
            # Apply transaction costs
            if t > window_size and not np.array_equal(portfolio_weights[t-1], portfolio_weights[t-2]):
                transaction_costs = cost_rate * np.sum(np.abs(portfolio_weights[t-1] - portfolio_weights[t-2]))
                daily_return -= transaction_costs
                
            portfolio_returns[t] = daily_return
            
    # Create results DataFrame
    results = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'cumulative_return': (1 + portfolio_returns).cumprod() - 1
    }, index=returns_data.index)
    
    return results

# XGBoost model
def xgboost_portfolio_optimization(returns_data, window_size=50, cost_rate=0.0002):
    """
    Implements portfolio optimization using XGBoost for returns prediction
    """
    num_assets = returns_data.shape[1]
    num_days = len(returns_data)
    portfolio_weights = np.zeros((num_days, num_assets))
    portfolio_returns = np.zeros(num_days)
    
    # Prepare features: historical returns for each asset
    for t in range(window_size, num_days - 1):
        # Training data up to time t
        X_train = []
        y_train = []
        
        for i in range(window_size, t):
            X_train.append(returns_data.iloc[i-window_size:i].values.flatten())
            y_train.append(returns_data.iloc[i].values)
        
        if len(X_train) > 0:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Train a separate model for each asset
            models = []
            predictions = np.zeros(num_assets)
            
            for asset in range(num_assets):
                try:
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                    model.fit(X_train, y_train[:, asset])
                    models.append(model)
                    
                    # Predict next day's return
                    X_predict = returns_data.iloc[t-window_size:t].values.flatten().reshape(1, -1)
                    predictions[asset] = model.predict(X_predict)[0]
                except:
                    # If model training fails, use zero prediction
                    predictions[asset] = 0
            
            # Determine weights based on predictions
            # Simple approach: allocate more to assets with higher predicted returns
            weights = np.zeros(num_assets)
            positive_preds = predictions > 0
            
            if np.any(positive_preds):
                weights[positive_preds] = predictions[positive_preds]
                weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else np.ones(num_assets) / num_assets
            else:
                # If all predictions are negative, go short on all assets
                weights = -predictions
                weights = -weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else np.ones(num_assets) / num_assets
                
            portfolio_weights[t] = weights
    
    # Calculate portfolio returns
    for t in range(window_size, num_days):
        if np.any(portfolio_weights[t-1]):
            daily_return = np.sum(portfolio_weights[t-1] * returns_data.iloc[t].values)
            
            # Apply transaction costs
            if t > window_size and not np.array_equal(portfolio_weights[t-1], portfolio_weights[t-2]):
                transaction_costs = cost_rate * np.sum(np.abs(portfolio_weights[t-1] - portfolio_weights[t-2]))
                daily_return -= transaction_costs
                
            portfolio_returns[t] = daily_return
    
    # Create results DataFrame
    results = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'cumulative_return': (1 + portfolio_returns).cumprod() - 1
    }, index=returns_data.index)
    
    return results

# LSTM implementation (simplified version of Zhang et al.)
class LSTMPortfolio(nn.Module):
    def __init__(self, num_assets, hidden_size=64):
        super(LSTMPortfolio, self).__init__()
        
        self.num_assets = num_assets
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(num_assets, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_assets)
        
    def forward(self, x):
        # x: [batch_size, seq_len, num_assets]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Project to num_assets
        output = self.fc(last_out)  # [batch_size, num_assets]
        
        # Apply the sign and softmax as described in the paper
        weights = torch.sign(output) * torch.softmax(torch.abs(output), dim=1)
        
        return weights

def lstm_portfolio_optimization(returns_data, window_size=50):
    """
    Trains and backtest the LSTM-based portfolio optimization from Zhang et al.
    """
    # Prepare dataset
    dataset = FinancialDataset(returns_data, window_size)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = LSTMPortfolio(returns_data.shape[1]).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_portfolio_transformer(model, train_loader, val_loader, optimizer, num_epochs=100, patience=10)
    
    # Backtest
    return backtest_strategy(model, returns_data, window_size)

# ================ 4. Performance Metrics ================ 

def calculate_max_drawdown(cum_returns):
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / (running_max + 1e-9)
    return drawdown.min()

def calculate_sortino_ratio(returns, risk_free=0, target=0):
    excess_returns = returns - risk_free
    downside_returns = excess_returns[excess_returns < target]
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
    sortino_ratio = (np.mean(excess_returns) * 252) / (downside_deviation if downside_deviation > 0 else 1e-6)
    return sortino_ratio

def calculate_performance_metrics(results):
    """
    Calculate performance metrics for a strategy
    
    Args:
        results: DataFrame with portfolio returns
    
    Returns:
        Dictionary of performance metrics
    """
    returns_series = results['portfolio_return'].dropna()
    cum_returns = results['cumulative_return'].dropna()
    
    # Annualized return
    ann_return = (1 + returns_series.mean()) ** 252 - 1
    
    # Annualized volatility
    ann_vol = returns_series.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Sortino ratio
    sortino = calculate_sortino_ratio(returns_series)
    
    # Maximum drawdown
    max_dd = calculate_max_drawdown(cum_returns)
    
    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else float('inf')
    
    # Percentage of positive returns
    pos_returns = (returns_series > 0).mean()
    
    return {
        'Returns': ann_return,
        'Vol.': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MDD': abs(max_dd),
        'Calmar': calmar,
        '% of +Ret': pos_returns
    }

# ================ 5. Main Testing Script ================

def analyze_covid_period(combined_results):
    """
    Analyze the performance during the COVID-19 period
    
    Args:
        combined_results: Dictionary of DataFrames with portfolio returns
    """
    # Define COVID period (Q1 2020)
    covid_start = '2020-01-01'
    covid_end = '2020-04-01'
    
    # Extract COVID period data
    covid_results = {}
    
    for name, results in combined_results.items():
        # Make sure the index is datetime before filtering
        if not isinstance(results.index, pd.DatetimeIndex):
            # Convert index to datetime if it's not already
            results = results.reset_index()
            results['index'] = pd.to_datetime(results['index'])
            results = results.set_index('index')

        # Filter to COVID period
        covid_data = results[(results.index >= covid_start) & (results.index <= covid_end)]
        if not covid_data.empty:
            covid_results[name] = covid_data
    
    # Only proceed if we have COVID data
    if not covid_results:
        print("\nNo data available for COVID-19 period analysis.")
        return None
    
    # Calculate COVID period metrics
    covid_metrics = {}
    
    for name, results in covid_results.items():
        metrics = calculate_performance_metrics(results)
        covid_metrics[name] = metrics
    
    # Create COVID metrics DataFrame
    covid_metrics_df = pd.DataFrame(covid_metrics).T
    print("\nCOVID-19 Period (Q1 2020) Performance:")
    print(covid_metrics_df)
    covid_metrics_df.to_csv('results/covid_metrics.csv')
    
    # Plot COVID period cumulative returns
    plt.figure(figsize=(12, 6))
    
    for name, results in covid_results.items():
        plt.plot(results['cumulative_return'], label=name)
    
    plt.title('Cumulative Returns During COVID-19 Crisis (Q1 2020)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/covid_returns.png')
    
    # Plot 12-month rolling Sharpe ratio during 2019-2020
    rolling_window = 252  # 12 months
    plt.figure(figsize=(12, 6))
    
    for name, full_results in combined_results.items():
        # Make sure the index is datetime
        if not isinstance(full_results.index, pd.DatetimeIndex):
            # Convert index to datetime if it's not already
            full_results = full_results.reset_index()
            full_results['index'] = pd.to_datetime(full_results['index'])
            full_results = full_results.set_index('index')
            
        # Get data from 2019-2020
        period_data = full_results[(full_results.index >= '2019-01-01') & 
                                   (full_results.index <= '2020-12-31')]
        
        # Only proceed if we have enough data
        if len(period_data) > rolling_window:
            returns_series = period_data['portfolio_return'].dropna()
            
            # Calculate rolling Sharpe ratio
            rolling_returns = returns_series.rolling(rolling_window).mean() * 252
            rolling_vol = returns_series.rolling(rolling_window).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / (rolling_vol + 1e-9)  # Avoid division by zero
            
            plt.plot(rolling_sharpe, label=name)
    
    plt.title('12-Month Rolling Sharpe Ratio (2019-2020)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    
    # Add COVID-19 period highlight if we have data in that range
    try:
        plt.axvspan(pd.to_datetime(covid_start), pd.to_datetime(covid_end), color='gray', alpha=0.3, label='COVID-19 Q1')
    except:
        pass
        
    plt.legend()
    plt.grid(True)
    plt.savefig('results/covid_rolling_sharpe.png')
    
    return covid_metrics_df

def main():
    # Create directory for results
    os.makedirs('results', exist_ok=True)
    
    # Retrieve data from Bloomberg
    tickers = ['AGG US Equity', 'DBC US Equity', 'VIX Index', 'VTI US Equity']
    start_date = '20060101'
    end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    # Get price data from Bloomberg
    df = con.bdh(tickers, 'PX_LAST', start_date, end_date)
    
    # Calculate daily returns
    returns = df.pct_change().dropna()
    
    # Ensure index is datetime
    returns.index = pd.to_datetime(returns.index)
    
    # Define test years (as per paper)
    test_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
    
    # Dictionary to store results
    results = {}
    all_metrics = {}
    
    for year in test_years:
        print(f"\nTraining and testing for year {year}...")
        
        # Training data: all data up to the end of previous year
        train_data = returns[returns.index < f'{year}-01-01']
        
        # Test data: current year
        test_data = returns[(returns.index >= f'{year}-01-01') & 
                           (returns.index < f'{year+1}-01-01')]
        
        print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        window_size = 50  # Historical window size
        
        # ---- Portfolio Transformer ----
        print("\nTraining Portfolio Transformer...")
        
        # Create datasets
        train_dataset = FinancialDataset(train_data, window_size)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Create model
        pt_model = PortfolioTransformer(
            num_assets=train_data.shape[1],
            d_model=64,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            hidden_dim=128,
            time2vec_dim=10,
            dropout=0.1
        ).to(device)
        
        # Define optimizer
        optimizer = optim.Adam(pt_model.parameters(), lr=0.001)
        
        # Train the model
        pt_model = train_portfolio_transformer(
            pt_model, 
            train_loader, 
            val_loader, 
            optimizer, 
            num_epochs=100, 
            patience=10
        )
        
        # Backtest on test data
        print("Backtesting Portfolio Transformer...")
        pt_results = backtest_strategy(pt_model, test_data, window_size)
        
        # ---- Benchmark Models ----
        print("\nRunning Mean-Variance Optimization...")
        mv_results = mean_variance_optimization(test_data, window_size)
        
        print("Running XGBoost...")
        xgb_results = xgboost_portfolio_optimization(test_data, window_size)
        
        print("Running LSTM...")
        lstm_results = lstm_portfolio_optimization(test_data, window_size)
        
        # Store results
        results[year] = {
            'Portfolio Transformer': pt_results,
            'Mean-Variance': mv_results,
            'XGBoost': xgb_results,
            'LSTM': lstm_results
        }
        
        # Calculate and store metrics
        year_metrics = {}
        for model_name, model_results in results[year].items():
            metrics = calculate_performance_metrics(model_results)
            year_metrics[model_name] = metrics
            
            print(f"{model_name} {year} Results:")
            print(f"  Annual Return: {metrics['Returns']:.4f}")
            print(f"  Annual Volatility: {metrics['Vol.']:.4f}")
            print(f"  Sharpe Ratio: {metrics['Sharpe']:.4f}")
        
        all_metrics[year] = year_metrics
        
        # Plot and save year results
        plt.figure(figsize=(12, 6))
        for model_name, model_results in results[year].items():
            plt.plot(model_results['cumulative_return'], label=model_name)
        plt.title(f'Cumulative Returns for {year}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/returns_{year}.png')
    
    # Combine all years for overall performance
    combined_results = {}
    for model_name in ['Portfolio Transformer', 'Mean-Variance', 'XGBoost', 'LSTM']:
        model_results_list = []
        for year in test_years:
            if model_name in results[year]:
                model_results_list.append(results[year][model_name])
        if model_results_list:
            model_results = pd.concat(model_results_list)
            combined_results[model_name] = model_results
    
    # Calculate overall metrics
    overall_metrics = {}
    for model_name, model_results in combined_results.items():
        metrics = calculate_performance_metrics(model_results)
        overall_metrics[model_name] = metrics
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(overall_metrics).T
    print("\nOverall Performance Metrics:")
    print(metrics_df)
    metrics_df.to_csv('results/overall_metrics.csv')
    
    # Plot overall cumulative returns
    plt.figure(figsize=(12, 6))
    for model_name, model_results in combined_results.items():
        plt.plot(model_results['cumulative_return'], label=model_name)
    plt.title('Overall Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/overall_returns.png')
    
    # COVID-19 analysis
    try:
        covid_metrics_df = analyze_covid_period(combined_results)
    except Exception as e:
        print(f"Error during COVID-19 analysis: {e}")
        covid_metrics_df = None
    
    return metrics_df, combined_results, all_metrics

if __name__ == "__main__":
    try:
        metrics_df, combined_results, all_metrics = main()
    except Exception as e:
        print(f"Error in main execution: {e}")