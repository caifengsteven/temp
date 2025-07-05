import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

class ParallelRetention(nn.Module):
    """Implementation of Parallel Retention mechanism as described in the paper"""
    def __init__(self, dim, num_heads=4, dropout=0.1, decay_factor=0.95):
        super(ParallelRetention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.decay_factor = decay_factor
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Group normalization
        self.norm = nn.GroupNorm(self.num_heads, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, D]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, D]
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, D]
        
        # Create causal mask with decay factor
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    mask[i, j] = self.decay_factor ** (i - j)
                else:
                    mask[i, j] = 0.0
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, L, L]
        scores = scores * mask.unsqueeze(0).unsqueeze(0)  # Apply causal mask with decay
        
        # Apply attention to values
        output = torch.matmul(scores, v)  # [B, H, L, D]
        
        # Reshape and project output
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # [B, L, H*D]
        output = self.out_proj(output)  # [B, L, D]
        
        # Apply group normalization
        output = output.permute(0, 2, 1)  # [B, D, L]
        output = self.norm(output)
        output = output.permute(0, 2, 1)  # [B, L, D]
        
        return output

class MultiRelationalDiffusion(nn.Module):
    """Implementation of Multi-relational Graph Diffusion as described in the paper"""
    def __init__(self, in_channels, out_channels, num_relations, expansion_steps=5):
        super(MultiRelationalDiffusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.expansion_steps = expansion_steps
        
        # Learnable weight coefficients for each relation and expansion step
        self.alpha = nn.Parameter(torch.ones(num_relations, expansion_steps) / expansion_steps)
        
        # Learnable transition matrices
        self.transition = nn.Parameter(torch.rand(num_relations, expansion_steps, in_channels, out_channels))
        
        # 1x1 convolution for merging relation-specific representations
        self.conv1x1 = nn.Conv2d(num_relations, 1, kernel_size=1)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x, adj_list):
        """
        Forward pass of multi-relational diffusion
        
        Args:
            x: Node features [num_nodes, in_channels]
            adj_list: List of adjacency matrices for each relation [num_relations, num_nodes, num_nodes]
        
        Returns:
            Diffused node representations [num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Normalize alpha coefficients for each relation
        alpha_normalized = F.softmax(self.alpha, dim=1)
        
        # Initialize output tensor for each relation
        relation_outputs = []
        
        for r in range(self.num_relations):
            adj = adj_list[:, r, :, :]  # [batch_size, num_nodes, num_nodes]
            
            # Compute diffusion matrix for this relation
            diffusion_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)
            
            for k in range(self.expansion_steps):
                # Compute weighted transition matrix
                weighted_transition = alpha_normalized[r, k] * adj
                diffusion_matrix += weighted_transition
            
            # Apply diffusion to node features
            output_r = torch.bmm(diffusion_matrix, x)  # [batch_size, num_nodes, in_channels]
            
            # Project to output dimension
            output_r = F.linear(output_r, self.transition[r].mean(dim=0))  # [batch_size, num_nodes, out_channels]
            
            relation_outputs.append(output_r)
        
        # Stack relation-specific outputs [batch_size, num_relations, num_nodes, out_channels]
        stacked_outputs = torch.stack(relation_outputs, dim=1)
        
        # Apply 1x1 convolution to merge relation-specific representations
        merged_output = self.conv1x1(stacked_outputs.permute(0, 1, 3, 2)).squeeze(1).permute(0, 2, 1)
        
        # Apply activation function
        output = self.activation(merged_output)
        
        return output

class MGDPR(nn.Module):
    """
    Multi-relational Graph Diffusion Neural Network with Parallel Retention
    as described in the paper.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, lookback_window,
                 num_layers=3, expansion_steps=5, num_heads=4, dropout=0.1, decay_factor=0.95):
        super(MGDPR, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.lookback_window = lookback_window
        self.num_layers = num_layers
        
        # Initial feature embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-relational diffusion layers
        self.diffusion_layers = nn.ModuleList([
            MultiRelationalDiffusion(
                hidden_dim, hidden_dim, num_relations, expansion_steps
            ) for _ in range(num_layers)
        ])
        
        # Parallel retention layers
        self.retention_layers = nn.ModuleList([
            ParallelRetention(
                hidden_dim, num_heads, dropout, decay_factor
            ) for _ in range(num_layers)
        ])
        
        # Layer transformation weights
        self.transform_weights1 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.transform_weights2 = nn.ModuleList([
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.transform_biases1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)
        ])
        
        self.transform_biases2 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, adj_list):
        """
        Forward pass of MGDPR
        
        Args:
            x: Node features [batch_size, num_nodes, lookback_window, input_dim]
            adj_list: Multi-relational adjacency matrices [batch_size, num_relations, num_nodes, num_nodes]
        
        Returns:
            Node classifications [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, lookback, _ = x.shape
        
        # Reshape x to [batch_size*num_nodes, lookback, input_dim]
        x_flat = x.reshape(batch_size*num_nodes, lookback, -1)
        
        # Initial embedding
        h0 = self.embedding(x_flat)  # [batch_size*num_nodes, lookback, hidden_dim]
        h0 = h0.reshape(batch_size, num_nodes, lookback, -1)
        
        # Take the last timestep features for graph diffusion
        h0_last = h0[:, :, -1, :]  # [batch_size, num_nodes, hidden_dim]
        
        # Initialize h and h_prime for the retention layer
        h = h0_last
        h_prime = torch.zeros_like(h0_last)
        
        # Apply multi-relational diffusion and parallel retention layers
        for l in range(self.num_layers):
            # Graph diffusion
            h_diffused = self.diffusion_layers[l](h, adj_list)
            
            # Reshape for retention
            h_for_retention = h_diffused.reshape(batch_size*num_nodes, 1, -1)
            
            # Parallel retention
            h_retained = self.retention_layers[l](h_for_retention)
            h_retained = h_retained.reshape(batch_size, num_nodes, -1)
            
            # Combine diffused and retained representations (Eq. 4 in the paper)
            h_combined = torch.cat((h_retained, h_prime), dim=-1)
            h_prime = F.relu(self.transform_weights1[l](h_prime) + self.transform_biases1[l])
            h = F.relu(self.transform_weights2[l](h_combined) + self.transform_biases2[l])
        
        # Output layer
        output = self.output(h)  # [batch_size, num_nodes, output_dim]
        
        return output

def create_dynamic_stock_graph(stock_data, lookback_window):
    """
    Create dynamic multi-relational stock graphs based on entropy and signal energy
    as described in section 4.1 of the paper.
    
    Args:
        stock_data: Dictionary with keys as stock symbols and values as DataFrames with OHLCV data
        lookback_window: Number of days to look back for computing relations
    
    Returns:
        Multi-relational adjacency matrices for each timestep
    """
    num_stocks = len(stock_data)
    num_days = len(list(stock_data.values())[0])
    num_relations = 5  # Open, High, Low, Close, Volume
    
    # Initialize adjacency matrices
    adj_matrices = np.zeros((num_days - lookback_window, num_relations, num_stocks, num_stocks))
    
    # Loop through each day (starting after lookback window)
    for t in range(lookback_window, num_days):
        # Loop through each relation (OHLCV)
        for r, indicator in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
            # Get time series for all stocks for this indicator
            time_series = {}
            for i, (symbol, data) in enumerate(stock_data.items()):
                time_series[symbol] = data[indicator][t-lookback_window:t].values
            
            # Compute entropy and energy for each stock's time series
            entropy = {}
            energy = {}
            for symbol, ts in time_series.items():
                # Compute signal energy as in Eq. 2
                energy[symbol] = np.sum(np.abs(ts)**2)
                
                # Compute probability distribution (histogram)
                hist, _ = np.histogram(ts, bins='auto', density=True)
                # Add small epsilon to avoid log(0)
                hist = hist + 1e-10
                hist = hist / np.sum(hist)
                
                # Compute entropy as in Eq. 2
                entropy[symbol] = -np.sum(hist * np.log(hist))
            
            # Compute adjacency matrix entries based on entropy and energy ratio
            for i, symbol_i in enumerate(stock_data.keys()):
                for j, symbol_j in enumerate(stock_data.keys()):
                    if i != j:  # No self-loops
                        # Implement Eq. 2 from the paper
                        energy_ratio = energy[symbol_i] / energy[symbol_j]
                        entropy_diff = entropy[symbol_i] - entropy[symbol_j]
                        adj_matrices[t-lookback_window, r, i, j] = energy_ratio * np.exp(entropy_diff)
            
            # Normalize the adjacency matrix
            row_sums = adj_matrices[t-lookback_window, r].sum(axis=1, keepdims=True)
            adj_matrices[t-lookback_window, r] = adj_matrices[t-lookback_window, r] / (row_sums + 1e-10)
    
    return adj_matrices

def generate_stock_labels(stock_data, threshold=0.01):
    """
    Generate labels for stock movement prediction:
    1 for price increase greater than threshold
    0 for price change within threshold
    -1 for price decrease greater than threshold
    
    Args:
        stock_data: Dictionary with keys as stock symbols and values as DataFrames with OHLCV data
        threshold: Movement threshold
    
    Returns:
        Labels for each stock and each day
    """
    num_stocks = len(stock_data)
    num_days = len(list(stock_data.values())[0])
    
    # Initialize labels
    labels = np.zeros((num_days - 1, num_stocks))
    
    # Loop through each stock
    for i, (symbol, data) in enumerate(stock_data.items()):
        # Calculate daily returns
        returns = data['Close'].pct_change().values[1:]
        
        # Assign labels based on threshold
        labels[:, i] = np.where(returns > threshold, 1, np.where(returns < -threshold, -1, 0))
    
    return labels

def prepare_dataset(stock_data, lookback_window, prediction_step=1):
    """
    Prepare dataset for training MGDPR.
    
    Args:
        stock_data: Dictionary with keys as stock symbols and values as DataFrames with OHLCV data
        lookback_window: Number of days to look back
        prediction_step: Number of days ahead to predict
    
    Returns:
        Features, adjacency matrices, and labels
    """
    # Create dynamic stock graphs
    adj_matrices = create_dynamic_stock_graph(stock_data, lookback_window)
    
    # Generate labels
    all_labels = generate_stock_labels(stock_data)
    
    # Prepare features
    num_stocks = len(stock_data)
    num_days = len(list(stock_data.values())[0])
    num_relations = 5  # OHLCV
    
    # Initialize features
    features = np.zeros((num_days - lookback_window - prediction_step, num_stocks, lookback_window, num_relations))
    
    # Loop through each day
    for t in range(lookback_window, num_days - prediction_step):
        # Loop through each stock
        for i, (symbol, data) in enumerate(stock_data.items()):
            # Extract features for each relation
            for r, indicator in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                # Normalize within lookback window
                window_data = data[indicator][t-lookback_window:t].values
                min_val = window_data.min()
                max_val = window_data.max()
                if max_val > min_val:  # Avoid division by zero
                    normalized_data = (window_data - min_val) / (max_val - min_val)
                else:
                    normalized_data = np.zeros_like(window_data)
                
                features[t-lookback_window, i, :, r] = normalized_data
    
    # Get labels for prediction
    labels = all_labels[lookback_window+prediction_step-1:]
    
    # Make sure dimensions match
    assert features.shape[0] == adj_matrices.shape[0] == labels.shape[0]
    
    return features, adj_matrices, labels

def download_stock_data(symbols, start_date, end_date):
    """
    Download stock data for multiple symbols using yfinance
    
    Args:
        symbols: List of stock symbols
        start_date: Start date for data download
        end_date: End date for data download
    
    Returns:
        Dictionary with symbols as keys and DataFrames as values
    """
    stock_data = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                stock_data[symbol] = data
                print(f"Downloaded data for {symbol}: {len(data)} days")
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
    
    return stock_data

def train_model(model, train_features, train_adj, train_labels, val_features, val_adj, val_labels, 
                epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
    """
    Train the MGDPR model
    
    Args:
        model: MGDPR model
        train_features, train_adj, train_labels: Training data
        val_features, val_adj, val_labels: Validation data
        epochs, batch_size, learning_rate: Training parameters
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Convert data to PyTorch tensors
    train_features = torch.FloatTensor(train_features).to(device)
    train_adj = torch.FloatTensor(train_adj).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    
    val_features = torch.FloatTensor(val_features).to(device)
    val_adj = torch.FloatTensor(val_adj).to(device)
    val_labels = torch.LongTensor(val_labels).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Number of batches
    num_samples = train_features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        # Shuffle indices
        indices = torch.randperm(num_samples)
        
        for batch in range(num_batches):
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_features = train_features[batch_indices]
            batch_adj = train_adj[batch_indices]
            batch_labels = train_labels[batch_indices]
            
            # Forward pass
            outputs = model(batch_features, batch_adj)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, 3), batch_labels.view(-1) + 1)  # +1 to shift from [-1,0,1] to [0,1,2]
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predictions = torch.max(outputs, dim=-1)
            correct = (predictions == batch_labels).float().sum()
            epoch_acc += correct / (batch_labels.numel())
        
        # Calculate epoch metrics
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features, val_adj)
            val_loss = criterion(val_outputs.view(-1, 3), val_labels.view(-1) + 1)
            
            _, val_predictions = torch.max(val_outputs, dim=-1)
            val_correct = (val_predictions == val_labels).float().sum()
            val_acc = val_correct / (val_labels.numel())
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history

def evaluate_model(model, test_features, test_adj, test_labels, device='cpu'):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained MGDPR model
        test_features, test_adj, test_labels: Test data
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Convert data to PyTorch tensors
    test_features = torch.FloatTensor(test_features).to(device)
    test_adj = torch.FloatTensor(test_adj).to(device)
    test_labels = torch.LongTensor(test_labels).to(device)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(test_features, test_adj)
        _, predictions = torch.max(outputs, dim=-1)
    
    # Convert to numpy for metrics calculation
    predictions = predictions.cpu().numpy()
    true_labels = test_labels.cpu().numpy()
    
    # Calculate metrics
    acc = accuracy_score(true_labels.flatten(), predictions.flatten())
    mcc = matthews_corrcoef(true_labels.flatten(), predictions.flatten())
    f1 = f1_score(true_labels.flatten(), predictions.flatten(), average='weighted')
    
    return {
        'accuracy': acc,
        'mcc': mcc,
        'f1_score': f1
    }

def simulate_stock_data(num_stocks=10, num_days=500, lookback_window=21):
    """
    Generate simulated stock data for testing
    
    Args:
        num_stocks: Number of stocks to simulate
        num_days: Number of days to simulate
        lookback_window: Historical lookback window size
    
    Returns:
        Dictionary with simulated stock data
    """
    # Initial price for each stock
    initial_prices = np.random.uniform(50, 200, num_stocks)
    
    # Create stock symbols
    symbols = [f'STOCK_{i}' for i in range(num_stocks)]
    
    # Dictionary to store stock data
    stock_data = {}
    
    # Generate correlated stock movements
    # Create correlation matrix (stocks in same sectors are more correlated)
    num_sectors = 3
    sectors = np.random.randint(0, num_sectors, num_stocks)
    
    correlation = np.zeros((num_stocks, num_stocks))
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i == j:
                correlation[i, j] = 1.0
            elif sectors[i] == sectors[j]:
                correlation[i, j] = np.random.uniform(0.5, 0.8)
            else:
                correlation[i, j] = np.random.uniform(0.1, 0.3)
    
    # Cholesky decomposition for correlated random numbers
    L = np.linalg.cholesky(correlation)
    
    # Generate prices
    for i, symbol in enumerate(symbols):
        # Base parameters for this stock
        volatility = np.random.uniform(0.01, 0.03)
        drift = np.random.uniform(0.0001, 0.0005)
        
        # Initialize price arrays
        prices = np.zeros(num_days)
        prices[0] = initial_prices[i]
        
        # Generate other prices (Open, High, Low, Volume)
        open_prices = np.zeros(num_days)
        high_prices = np.zeros(num_days)
        low_prices = np.zeros(num_days)
        volumes = np.zeros(num_days)
        
        # Set initial values
        open_prices[0] = prices[0]
        high_prices[0] = prices[0] * 1.01
        low_prices[0] = prices[0] * 0.99
        volumes[0] = np.random.normal(100000, 20000)
        
        # Generate random returns
        random_returns = np.random.normal(0, 1, (num_stocks, num_days-1))
        correlated_returns = np.dot(L, random_returns)
        
        # Generate price series
        for t in range(1, num_days):
            # Daily return with correlation
            daily_return = drift + volatility * correlated_returns[i, t-1]
            
            # Update close price
            prices[t] = prices[t-1] * (1 + daily_return)
            
            # Generate related prices
            # Open price is affected by previous close
            open_prices[t] = prices[t-1] * (1 + np.random.normal(0, 0.005))
            
            # High and low prices depend on open and close
            price_range = abs(prices[t] - open_prices[t]) + prices[t] * np.random.uniform(0.005, 0.015)
            if open_prices[t] > prices[t]:
                high_prices[t] = open_prices[t] + price_range * np.random.uniform(0, 0.5)
                low_prices[t] = prices[t] - price_range * np.random.uniform(0, 0.5)
            else:
                high_prices[t] = prices[t] + price_range * np.random.uniform(0, 0.5)
                low_prices[t] = open_prices[t] - price_range * np.random.uniform(0, 0.5)
            
            # Volume has some correlation with price movement
            vol_base = 100000 + 20000 * abs(daily_return) / volatility
            volumes[t] = np.random.normal(vol_base, vol_base * 0.2)
        
        # Create DataFrame for this stock
        stock_data[symbol] = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': prices,
            'Volume': volumes
        })
    
    return stock_data

def plot_results(history, metrics):
    """
    Plot training history and evaluation metrics
    
    Args:
        history: Training history dictionary
        metrics: Evaluation metrics dictionary
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(2, 2, 3)
    metrics_values = [metrics['accuracy'], metrics['mcc'], metrics['f1_score']]
    metrics_labels = ['Accuracy', 'MCC', 'F1-Score']
    
    plt.bar(metrics_labels, metrics_values)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('mgdpr_results.png')
    plt.show()

def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    lookback_window = 21
    hidden_dim = 64
    num_layers = 2
    expansion_steps = 5
    num_heads = 4
    batch_size = 16
    learning_rate = 0.001
    epochs = 50
    
    # Generate simulated stock data
    print("Generating simulated stock data...")
    stock_data = simulate_stock_data(num_stocks=20, num_days=500, lookback_window=lookback_window)
    
    # Alternative: Download real stock data
    # Uncomment to use real data instead of simulated data
    """
    # Define stock symbols (e.g., NASDAQ stocks)
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX',
               'CMCSA', 'INTC', 'CSCO', 'PEP', 'AVGO', 'TXN', 'COST', 'QCOM', 'TMUS', 'AMGN']
    
    # Download data
    start_date = '2018-01-01'
    end_date = '2020-01-01'
    stock_data = download_stock_data(symbols, start_date, end_date)
    """
    
    # Prepare dataset
    print("Preparing dataset...")
    features, adj_matrices, labels = prepare_dataset(stock_data, lookback_window)
    
    # Split data into train, validation, and test sets
    num_samples = features.shape[0]
    train_size = int(0.7 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_features = features[:train_size]
    train_adj = adj_matrices[:train_size]
    train_labels = labels[:train_size]
    
    val_features = features[train_size:train_size+val_size]
    val_adj = adj_matrices[train_size:train_size+val_size]
    val_labels = labels[train_size:train_size+val_size]
    
    test_features = features[train_size+val_size:]
    test_adj = adj_matrices[train_size+val_size:]
    test_labels = labels[train_size+val_size:]
    
    print(f"Data shapes - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
    
    # Initialize model
    num_stocks = features.shape[1]
    num_relations = features.shape[3]
    
    model = MGDPR(
        input_dim=1,  # Single feature value for each timestep and relation
        hidden_dim=hidden_dim,
        output_dim=3,  # Three classes: -1, 0, 1
        num_relations=num_relations,
        lookback_window=lookback_window,
        num_layers=num_layers,
        expansion_steps=expansion_steps,
        num_heads=num_heads
    )
    
    # Train model
    print("Training model...")
    trained_model, history = train_model(
        model, 
        train_features, 
        train_adj, 
        train_labels,
        val_features,
        val_adj,
        val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(trained_model, test_features, test_adj, test_labels, device=device)
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Plot results
    plot_results(history, metrics)

if __name__ == "__main__":
    main()