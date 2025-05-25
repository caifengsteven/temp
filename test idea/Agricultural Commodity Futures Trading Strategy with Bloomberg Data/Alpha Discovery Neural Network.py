import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
from datetime import datetime, timedelta
import random
import math

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to generate synthetic stock market data
def generate_synthetic_data(n_stocks=500, n_days=500, n_features=5):
    """
    Generate synthetic stock market data
    
    Parameters:
    n_stocks: Number of stocks
    n_days: Number of trading days
    n_features: Number of features per stock (OHLCV)
    
    Returns:
    data: Dictionary containing the generated data
    """
    # Generate base market movement (common factor)
    market_return = np.zeros(n_days)
    market_vol = 0.01  # Daily market volatility
    market_return[0] = 0
    
    # Generate random market movements with some trend and mean reversion
    for i in range(1, n_days):
        # Add some mean reversion and momentum
        market_return[i] = 0.7 * market_return[i-1] + np.random.normal(0, market_vol)
    
    # Accumulate returns to create market price
    market_price = 1000 * np.exp(np.cumsum(market_return))
    
    # Generate individual stock prices with correlation to market
    stock_data = {}
    future_returns = {}
    
    for stock_id in range(n_stocks):
        # Stock specific parameters
        stock_beta = np.random.uniform(0.5, 1.5)  # Beta to market
        stock_alpha = np.random.normal(0, 0.0002)  # Daily alpha
        stock_vol = np.random.uniform(0.01, 0.03)  # Idiosyncratic volatility
        
        # Generate stock return series
        stock_return = np.zeros(n_days)
        stock_return[0] = np.random.normal(0, 0.01)
        
        for i in range(1, n_days):
            # Stock return is a function of market return (beta), alpha, and idiosyncratic return
            market_component = stock_beta * market_return[i]
            idiosyncratic_component = np.random.normal(0, stock_vol)
            stock_return[i] = stock_alpha + market_component + idiosyncratic_component
        
        # Create price series from returns
        price = 100 * np.exp(np.cumsum(stock_return))
        
        # Generate OHLCV data
        open_price = price.copy()
        close_price = price.copy()
        
        # High and low are variations around the price
        daily_range = price * np.random.uniform(0.01, 0.03, n_days)
        high_price = price + daily_range / 2
        low_price = price - daily_range / 2
        
        # Volume is somewhat correlated with volatility
        volume = np.exp(np.random.normal(10, 1, n_days)) * (1 + 5 * np.abs(stock_return))
        
        # Store OHLCV in the stock_data dictionary
        stock_data[stock_id] = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
        
        # Calculate future 5-day returns for each day
        future_5d_return = np.zeros(n_days)
        for i in range(n_days - 5):
            future_5d_return[i] = (close_price[i + 5] / close_price[i]) - 1
        
        future_returns[stock_id] = future_5d_return
    
    # Create a dictionary to store all data
    data = {
        'market_price': market_price,
        'stock_data': stock_data,
        'future_returns': future_returns
    }
    
    return data

# Generate prior knowledge factors (traditional technical indicators)
def generate_prior_knowledge(stock_data, n_factors=50):
    """
    Generate traditional technical indicators as prior knowledge
    
    Parameters:
    stock_data: Dictionary of stock data
    n_factors: Number of factors to generate
    
    Returns:
    factors: Dictionary of technical factors
    """
    factors = {}
    n_stocks = len(stock_data)
    n_days = len(stock_data[0]['close'])
    
    # Initialize factors with zeros
    for i in range(n_factors):
        factors[i] = np.zeros((n_stocks, n_days))
    
    # For each stock
    for stock_id in range(n_stocks):
        open_price = stock_data[stock_id]['open']
        high_price = stock_data[stock_id]['high']
        low_price = stock_data[stock_id]['low']
        close_price = stock_data[stock_id]['close']
        volume = stock_data[stock_id]['volume']
        
        # Factor 1: Price momentum (returns over different periods)
        for i, period in enumerate([1, 3, 5, 10, 20, 30]):
            for j in range(period, n_days):
                factors[i][stock_id, j] = (close_price[j] / close_price[j - period]) - 1
        
        # Factor 6-10: Moving averages
        for i, period in enumerate([5, 10, 20, 30, 60]):
            for j in range(period, n_days):
                factors[i + 6][stock_id, j] = np.mean(close_price[j - period:j])
        
        # Factor 11-15: Relative strength index (RSI)
        for i, period in enumerate([5, 10, 14, 20, 30]):
            for j in range(period, n_days):
                gains = np.maximum(0, np.diff(close_price[j - period:j + 1]))
                losses = np.maximum(0, -np.diff(close_price[j - period:j + 1]))
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss == 0:
                    factors[i + 11][stock_id, j] = 100
                else:
                    rs = avg_gain / avg_loss
                    factors[i + 11][stock_id, j] = 100 - (100 / (1 + rs))
        
        # Factor 16-20: Bollinger Bands
        for i, period in enumerate([10, 20, 30, 40, 50]):
            for j in range(period, n_days):
                ma = np.mean(close_price[j - period:j])
                std = np.std(close_price[j - period:j])
                upper_band = ma + 2 * std
                lower_band = ma - 2 * std
                factors[i + 16][stock_id, j] = (close_price[j] - lower_band) / (upper_band - lower_band)
        
        # Factor 21-25: MACD
        for i, (fast, slow) in enumerate([(12, 26), (8, 17), (10, 20), (15, 30), (5, 10)]):
            for j in range(slow, n_days):
                ema_fast = np.mean(close_price[j - fast:j])  # Simplified EMA
                ema_slow = np.mean(close_price[j - slow:j])  # Simplified EMA
                factors[i + 21][stock_id, j] = ema_fast - ema_slow
        
        # Factor 26-30: Volume indicators
        for i, period in enumerate([5, 10, 20, 30, 60]):
            for j in range(period, n_days):
                factors[i + 26][stock_id, j] = volume[j] / np.mean(volume[j - period:j])
        
        # Factor 31-35: Price volatility
        for i, period in enumerate([5, 10, 20, 30, 60]):
            for j in range(period, n_days):
                factors[i + 31][stock_id, j] = np.std(close_price[j - period:j]) / np.mean(close_price[j - period:j])
        
        # Factor 36-40: High-Low range
        for i, period in enumerate([5, 10, 20, 30, 60]):
            for j in range(period, n_days):
                factors[i + 36][stock_id, j] = np.mean((high_price[j - period:j] - low_price[j - period:j]) / close_price[j - period:j])
        
        # Factor 41-45: Gap indicators
        for i, period in enumerate([1, 2, 3, 4, 5]):
            for j in range(period, n_days):
                factors[i + 41][stock_id, j] = (open_price[j] / close_price[j - 1]) - 1
        
        # Factor 46-50: Custom combinations
        # Factor 46: Momentum * Volatility
        factors[46][stock_id, 10:] = factors[0][stock_id, 10:] * factors[31][stock_id, 10:]
        
        # Factor 47: RSI * Volume
        factors[47][stock_id, 14:] = factors[13][stock_id, 14:] * factors[26][stock_id, 14:] / 100
        
        # Factor 48: MACD * BB
        factors[48][stock_id, 26:] = factors[21][stock_id, 26:] * factors[16][stock_id, 26:]
        
        # Factor 49: Moving average crossover
        factors[49][stock_id, 60:] = factors[6][stock_id, 60:] - factors[10][stock_id, 60:]
    
    return factors

# Create PyTorch dataset for ADN
class StockDataset(Dataset):
    def __init__(self, stock_data, future_returns, lookback=30, train_indices=None):
        """
        Dataset for stock price prediction
        
        Parameters:
        stock_data: Dictionary of stock price data
        future_returns: Dictionary of future 5-day returns
        lookback: Number of days to look back for features
        train_indices: Indices of training samples (stock_id, day)
        """
        self.stock_data = stock_data
        self.future_returns = future_returns
        self.lookback = lookback
        
        # If train_indices is not provided, create indices for all valid samples
        if train_indices is None:
            n_stocks = len(stock_data)
            n_days = len(stock_data[0]['close'])
            
            self.indices = []
            for stock_id in range(n_stocks):
                for day in range(lookback, n_days - 5):  # Ensure there are 5 days for the future return
                    self.indices.append((stock_id, day))
        else:
            self.indices = train_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stock_id, day = self.indices[idx]
        
        # Get the last 'lookback' days of data
        ohlcv_data = np.zeros((self.lookback, 5))
        for i in range(self.lookback):
            d = day - self.lookback + i
            ohlcv_data[i, 0] = self.stock_data[stock_id]['open'][d]
            ohlcv_data[i, 1] = self.stock_data[stock_id]['high'][d]
            ohlcv_data[i, 2] = self.stock_data[stock_id]['low'][d]
            ohlcv_data[i, 3] = self.stock_data[stock_id]['close'][d]
            ohlcv_data[i, 4] = self.stock_data[stock_id]['volume'][d]
        
        # Get the future 5-day return
        future_return = self.future_returns[stock_id][day]
        
        return torch.tensor(ohlcv_data, dtype=torch.float32), torch.tensor(future_return, dtype=torch.float32), torch.tensor([stock_id, day], dtype=torch.long)

# Create dataset with factors
class FactorDataset(Dataset):
    def __init__(self, factors, future_returns, factor_indices=None, day_range=None):
        """
        Dataset for factor-based prediction
        
        Parameters:
        factors: Dictionary of factors
        future_returns: Dictionary of future 5-day returns
        factor_indices: List of factor indices to use
        day_range: Tuple of (start_day, end_day) for slicing
        """
        self.factors = factors
        self.future_returns = future_returns
        
        if factor_indices is None:
            self.factor_indices = list(factors.keys())
        else:
            self.factor_indices = factor_indices
        
        # Get dimensions
        n_stocks = factors[0].shape[0]
        n_days = factors[0].shape[1] if day_range is None else day_range[1] - day_range[0]
        start_day = 0 if day_range is None else day_range[0]
        end_day = factors[0].shape[1] if day_range is None else day_range[1]
        
        # Create list of valid samples
        self.samples = []
        for day in range(start_day, end_day):
            for stock_id in range(n_stocks):
                # Check if future return is available and not NaN
                if day < len(future_returns[stock_id]) - 5 and not np.isnan(future_returns[stock_id][day]):
                    self.samples.append((stock_id, day))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        stock_id, day = self.samples[idx]
        
        # Get factor values for this stock on this day
        factor_values = np.array([self.factors[f][stock_id, day] for f in self.factor_indices])
        
        # Get future return
        future_return = self.future_returns[stock_id][day]
        
        return torch.tensor(factor_values, dtype=torch.float32), torch.tensor(future_return, dtype=torch.float32), torch.tensor([stock_id, day], dtype=torch.long)

# Alpha Discovery Network base class
class ADN(nn.Module):
    def __init__(self, prior_knowledge=None):
        super(ADN, self).__init__()
        self.prior_knowledge = prior_knowledge
    
    def calculate_ic(self, predictions, future_returns, day_indices):
        """
        Calculate Information Coefficient (Spearman rank correlation)
        grouped by trading day
        
        Parameters:
        predictions: Tensor of model predictions
        future_returns: Tensor of actual future returns
        day_indices: Tensor of day indices for each sample
        
        Returns:
        ic: Average IC across days
        """
        # Convert to numpy
        predictions = predictions.detach().cpu().numpy()
        future_returns = future_returns.detach().cpu().numpy()
        day_indices = day_indices.detach().cpu().numpy()
        
        # Group by day
        unique_days = np.unique(day_indices)
        
        ics = []
        for day in unique_days:
            mask = day_indices == day
            if np.sum(mask) > 5:  # Need at least a few samples to calculate correlation
                day_pred = predictions[mask]
                day_returns = future_returns[mask]
                # Calculate Spearman rank correlation
                correlation, _ = spearmanr(day_pred, day_returns, nan_policy='omit')
                if not np.isnan(correlation):
                    ics.append(correlation)
        
        # Return average IC
        return np.mean(ics) if ics else 0

# Fully Connected Network
class FCN(ADN):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2, prior_knowledge=None):
        super(FCN, self).__init__(prior_knowledge)
        
        # Create network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten input if it's multidimensional
        x_flat = x.view(x.size(0), -1)
        return self.network(x_flat).squeeze(-1)

# CNN-based model (LeNet)
class LeNet(ADN):
    def __init__(self, input_channels=5, lookback=30, dropout=0.2, prior_knowledge=None):
        super(LeNet, self).__init__(prior_knowledge)
        
        # Modified LeNet for 1D time series
        self.conv1 = nn.Conv1d(input_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after convolutions and pooling
        size_after_conv = ((lookback - 4) // 2 - 4) // 2
        self.fc1 = nn.Linear(16 * size_after_conv, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input shape: (batch_size, lookback, channels)
        # Reshape to (batch_size, channels, lookback) for 1D convolution
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)

# LSTM-based model
class LSTM(ADN):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, dropout=0.2, prior_knowledge=None):
        super(LSTM, self).__init__(prior_knowledge)
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Use the final hidden state
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Final prediction
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)

# Masked initialization with prior knowledge
def masked_init_with_prior(model, prior_knowledge_factors, mask_ratio=0.3):
    """
    Initialize model with prior knowledge and apply masking
    
    Parameters:
    model: The neural network model
    prior_knowledge_factors: List of prior knowledge factors
    mask_ratio: Ratio of connections to mask
    
    Returns:
    model: Updated model
    """
    if prior_knowledge_factors is None or len(prior_knowledge_factors) == 0:
        return model
    
    # For simplicity, we'll only initialize the first layer of the network
    if isinstance(model, FCN):
        first_layer = model.network[0]
    elif isinstance(model, LeNet):
        first_layer = model.conv1
    elif isinstance(model, LSTM):
        first_layer = model.lstm
    else:
        raise ValueError("Unsupported model type")
    
    # Create a mask tensor (1 = keep, 0 = mask)
    if hasattr(first_layer, 'weight'):
        mask = torch.ones_like(first_layer.weight.data)
        
        # Randomly set some connections to zero
        num_params = mask.numel()
        mask_indices = np.random.choice(num_params, size=int(num_params * mask_ratio), replace=False)
        mask.view(-1)[mask_indices] = 0
        
        # Apply the mask
        first_layer.weight.data = first_layer.weight.data * mask
    
    return model

# Train a single model
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, early_stopping=10, device=None):
    """
    Train a single model
    
    Parameters:
    model: Neural network model
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    criterion: Loss function
    optimizer: Optimizer
    num_epochs: Maximum number of epochs
    early_stopping: Number of epochs for early stopping
    device: Device to use for training
    
    Returns:
    model: Trained model
    history: Training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_ic': [],
        'val_ic': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_ics = []
        
        for inputs, targets, indices in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate IC
            ic = model.calculate_ic(outputs, targets, indices[:, 1])
            train_ics.append(ic)
        
        train_loss /= len(train_loader)
        train_ic = np.mean(train_ics)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ics = []
        
        with torch.no_grad():
            for inputs, targets, indices in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Calculate IC
                ic = model.calculate_ic(outputs, targets, indices[:, 1])
                val_ics.append(ic)
        
        val_loss /= len(val_loader)
        val_ic = np.mean(val_ics)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ic'].append(train_ic)
        history['val_ic'].append(val_ic)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train IC: {train_ic:.4f} | Val IC: {val_ic:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model weights
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                # Load best weights
                model.load_state_dict(best_weights)
                break
    
    # Load best weights if not loaded already
    if patience_counter < early_stopping:
        model.load_state_dict(best_weights)
    
    return model, history

# Generate multiple ADN models
def generate_adns(n_models, model_type, train_loader, val_loader, 
                 prior_knowledge=None, device=None, **model_kwargs):
    """
    Generate multiple ADN models
    
    Parameters:
    n_models: Number of models to generate
    model_type: Type of model to generate (FCN, LeNet, LSTM)
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    prior_knowledge: Prior knowledge factors
    device: Device to use for training
    model_kwargs: Additional model parameters
    
    Returns:
    models: List of trained models
    histories: List of training histories
    """
    models = []
    histories = []
    
    for i in range(n_models):
        print(f'Training model {i+1}/{n_models}')
        
        # Create model
        if model_type == 'FCN':
            model = FCN(prior_knowledge=prior_knowledge, **model_kwargs).to(device)
        elif model_type == 'LeNet':
            model = LeNet(prior_knowledge=prior_knowledge, **model_kwargs).to(device)
        elif model_type == 'LSTM':
            model = LSTM(prior_knowledge=prior_knowledge, **model_kwargs).to(device)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        
        # Apply masked initialization with prior knowledge
        model = masked_init_with_prior(model, prior_knowledge)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader, 
            criterion, optimizer, 
            num_epochs=50, early_stopping=10, device=device
        )
        
        models.append(model)
        histories.append(history)
    
    return models, histories

# Evaluate models on test set
def evaluate_models(models, test_loader, device):
    """
    Evaluate models on test set
    
    Parameters:
    models: List of trained models
    test_loader: DataLoader for test data
    device: Device to use for evaluation
    
    Returns:
    results: Dictionary of evaluation results
    """
    results = {
        'predictions': [],
        'ics': []
    }
    
    for model in models:
        model.eval()
        model_preds = []
        day_ics = []
        
        with torch.no_grad():
            for inputs, targets, indices in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                # Store predictions
                model_preds.append(outputs.cpu().numpy())
                
                # Calculate IC by day
                ic = model.calculate_ic(outputs, targets, indices[:, 1])
                day_ics.append(ic)
        
        # Concatenate predictions
        model_preds = np.concatenate(model_preds)
        
        results['predictions'].append(model_preds)
        results['ics'].append(np.mean(day_ics))
    
    return results

# Calculate diversity of features
def calculate_diversity(features, method='cross_entropy'):
    """
    Calculate diversity between features
    
    Parameters:
    features: List of feature values (each a numpy array)
    method: Method to calculate diversity ('cross_entropy' or 'correlation')
    
    Returns:
    diversity: Diversity score
    """
    n_features = len(features)
    
    if method == 'cross_entropy':
        # Calculate cross entropy between feature distributions
        distance_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    # Normalize features to probability distributions
                    f1 = features[i]
                    f2 = features[j]
                    
                    # Apply softmax
                    f1_softmax = np.exp(f1) / np.sum(np.exp(f1))
                    f2_softmax = np.exp(f2) / np.sum(np.exp(f2))
                    
                    # Calculate cross entropy
                    cross_entropy = -np.sum(f1_softmax * np.log(f2_softmax + 1e-10))
                    distance_matrix[i, j] = cross_entropy
        
        # Cluster the distance matrix
        n_clusters = min(5, n_features // 2)  # Use at most 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        # Calculate average distance between cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        diversity = 0
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                diversity += np.linalg.norm(cluster_centers[i] - cluster_centers[j])
        
        diversity /= (n_clusters * (n_clusters - 1) / 2)
    
    elif method == 'correlation':
        # Calculate pairwise correlations
        corr_matrix = np.corrcoef(features)
        
        # Average of absolute off-diagonal correlations
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        diversity = 1 - np.mean(np.abs(corr_matrix[mask]))
    
    else:
        raise ValueError(f'Unknown diversity method: {method}')
    
    return diversity

# Genetic Programming for comparison
class GP:
    def __init__(self, population_size=100, generations=50, tournament_size=5, 
                 mutation_rate=0.2, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Define operators
        self.operators = ['+', '-', '*', '/', 'log', 'exp', 'square', 'sqrt', 'abs']
        
        # Define terminals (indices of input features)
        self.terminals = ['x[0]', 'x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']
        
        # Initialize population
        self.population = []
        
    def generate_random_tree(self, max_depth=3, depth=0):
        """Generate a random expression tree"""
        # At maximum depth or randomly at lower depths, return a terminal
        if depth >= max_depth or (depth > 0 and np.random.random() < 0.4):
            return np.random.choice(self.terminals)
        
        # Otherwise choose an operator
        op = np.random.choice(self.operators)
        
        if op in ['+', '-', '*', '/']:
            left = self.generate_random_tree(max_depth, depth + 1)
            right = self.generate_random_tree(max_depth, depth + 1)
            return f"({left} {op} {right})"
        else:
            expr = self.generate_random_tree(max_depth, depth + 1)
            return f"{op}({expr})"
    
    def initialize_population(self):
        """Initialize a random population"""
        self.population = [self.generate_random_tree() for _ in range(self.population_size)]
    
    def evaluate_expression(self, expr, x):
        """Evaluate a mathematical expression"""
        # Create a safe evaluation environment
        safe_dict = {
            'x': x,
            'abs': np.abs,
            'log': lambda x: np.log(np.abs(x) + 1e-10),
            'exp': lambda x: np.exp(np.minimum(x, 10)),  # Prevent overflow
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'square': lambda x: x * x
        }
        
        try:
            return eval(expr, {"__builtins__": {}}, safe_dict)
        except:
            return np.zeros_like(x[0])
    
    def calculate_fitness(self, expr, X, y, day_indices):
        """Calculate fitness (IC) of an expression"""
        try:
            # Evaluate expression for all samples
            preds = self.evaluate_expression(expr, X)
            
            # Group by day and calculate IC
            unique_days = np.unique(day_indices)
            
            ics = []
            for day in unique_days:
                mask = day_indices == day
                if np.sum(mask) > 5:  # Need at least a few samples to calculate correlation
                    day_pred = preds[mask]
                    day_returns = y[mask]
                    
                    # Check for valid predictions
                    if np.all(np.isfinite(day_pred)) and np.std(day_pred) > 0:
                        # Calculate Spearman rank correlation
                        correlation, _ = spearmanr(day_pred, day_returns, nan_policy='omit')
                        if not np.isnan(correlation):
                            ics.append(correlation)
            
            # Return average IC
            fitness = np.mean(ics) if ics else -1
            
            return fitness
        except:
            return -1
    
    def tournament_selection(self, fitnesses):
        """Select individual using tournament selection"""
        indices = np.random.choice(len(fitnesses), self.tournament_size, replace=False)
        selected_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return self.population[selected_idx]
    
    def crossover(self, expr1, expr2):
        """Perform crossover between two expressions"""
        # Simple string-based crossover (not ideal but works for demonstration)
        if np.random.random() < self.crossover_rate:
            # Convert to strings and split at a random point
            str1 = str(expr1)
            str2 = str(expr2)
            
            # Find all balanced parenthesis subexpressions
            def find_subexpressions(expr):
                subexprs = []
                stack = []
                for i, char in enumerate(expr):
                    if char == '(':
                        stack.append(i)
                    elif char == ')' and stack:
                        start = stack.pop()
                        if not stack:  # Only consider top-level expressions
                            subexprs.append((start, i+1))
                return subexprs
            
            subexpr1 = find_subexpressions(str1)
            subexpr2 = find_subexpressions(str2)
            
            if subexpr1 and subexpr2:
                # Choose random subexpressions
                start1, end1 = subexpr1[np.random.randint(len(subexpr1))]
                start2, end2 = subexpr2[np.random.randint(len(subexpr2))]
                
                # Swap subexpressions
                new_expr1 = str1[:start1] + str2[start2:end2] + str1[end1:]
                new_expr2 = str2[:start2] + str1[start1:end1] + str2[end2:]
                
                return new_expr1, new_expr2
        
        return expr1, expr2
    
    def mutate(self, expr):
        """Mutate an expression"""
        if np.random.random() < self.mutation_rate:
            # Convert to string
            str_expr = str(expr)
            
            # Choose a mutation type
            mutation_type = np.random.choice(['operator', 'terminal', 'subtree'])
            
            if mutation_type == 'operator':
                # Replace an operator
                for op in self.operators:
                    if op in str_expr:
                        new_op = np.random.choice([o for o in self.operators if o != op])
                        str_expr = str_expr.replace(op, new_op, 1)
                        break
            
            elif mutation_type == 'terminal':
                # Replace a terminal
                for term in self.terminals:
                    if term in str_expr:
                        new_term = np.random.choice([t for t in self.terminals if t != term])
                        str_expr = str_expr.replace(term, new_term, 1)
                        break
            
            elif mutation_type == 'subtree':
                # Replace a subtree with a new random tree
                subexpr = self.generate_random_tree(max_depth=2)
                
                # Find a random position to insert
                if '(' in str_expr:
                    pos = str_expr.find('(')
                    end_pos = pos
                    count = 1
                    while count > 0 and end_pos < len(str_expr) - 1:
                        end_pos += 1
                        if str_expr[end_pos] == '(':
                            count += 1
                        elif str_expr[end_pos] == ')':
                            count -= 1
                    
                    str_expr = str_expr[:pos] + subexpr + str_expr[end_pos+1:]
            
            return str_expr
        
        return expr
    
    def evolve(self, X, y, day_indices):
        """Evolve the population"""
        self.initialize_population()
        
        # Evaluate initial population
        fitnesses = [self.calculate_fitness(expr, X, y, day_indices) for expr in self.population]
        
        best_fitness = max(fitnesses)
        best_expr = self.population[np.argmax(fitnesses)]
        
        for generation in range(self.generations):
            new_population = []
            
            # Elitism - keep the best individual
            elite_idx = np.argmax(fitnesses)
            new_population.append(self.population[elite_idx])
            
            # Generate new individuals
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(fitnesses)
                parent2 = self.tournament_selection(fitnesses)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            self.population = new_population
            
            # Evaluate new population
            fitnesses = [self.calculate_fitness(expr, X, y, day_indices) for expr in self.population]
            
            # Update best individual
            gen_best_fitness = max(fitnesses)
            gen_best_expr = self.population[np.argmax(fitnesses)]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_expr = gen_best_expr
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f'Generation {generation+1}/{self.generations} | Best Fitness: {best_fitness:.4f}')
        
        # Return the best expression
        return best_expr, best_fitness

# Helper function to run experiments
def run_experiment(synthetic_data, model_type='FCN', n_models=50, use_prior=True, use_gp=True):
    """
    Run experiment with a specific model type
    
    Parameters:
    synthetic_data: Dictionary of synthetic data
    model_type: Type of model to use ('FCN', 'LeNet', 'LSTM')
    n_models: Number of models to train
    use_prior: Whether to use prior knowledge
    use_gp: Whether to run GP for comparison
    
    Returns:
    results: Dictionary of results
    """
    start_time = time.time()
    
    # Extract data
    stock_data = synthetic_data['stock_data']
    future_returns = synthetic_data['future_returns']
    
    # Generate prior knowledge factors
    pk_factors = generate_prior_knowledge(stock_data) if use_prior else None
    
    # Get data dimensions
    n_stocks = len(stock_data)
    n_days = len(stock_data[0]['close'])
    lookback = 30
    
    # Split data into train, validation, and test sets
    train_days = int(n_days * 0.6)
    val_days = int(n_days * 0.2)
    test_days = n_days - train_days - val_days
    
    train_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                     for day in range(lookback, train_days - 5)]
    
    val_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                   for day in range(train_days, train_days + val_days - 5)]
    
    test_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                    for day in range(train_days + val_days, n_days - 5)]
    
    # Create datasets
    train_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=train_indices)
    val_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=val_indices)
    test_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=test_indices)
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    model_params = {
        'FCN': {
            'input_dim': lookback * 5,  # OHLCV * lookback
            'hidden_dims': [64, 32],
            'dropout': 0.2
        },
        'LeNet': {
            'input_channels': 5,  # OHLCV
            'lookback': lookback,
            'dropout': 0.2
        },
        'LSTM': {
            'input_dim': 5,  # OHLCV
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.2
        }
    }
    
    # Train models
    models, histories = generate_adns(
        n_models=n_models,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        prior_knowledge=pk_factors,
        device=device,
        **model_params[model_type]
    )
    
    # Evaluate models on test set
    test_results = evaluate_models(models, test_loader, device)
    
    # Calculate average IC
    adn_avg_ic = np.mean(test_results['ics'])
    
    # Calculate diversity
    adn_predictions = test_results['predictions']
    adn_diversity = calculate_diversity(adn_predictions, method='correlation')
    
    # Total time
    adn_time = time.time() - start_time
    
    results = {
        'model_type': model_type,
        'adn_avg_ic': adn_avg_ic,
        'adn_diversity': adn_diversity,
        'adn_time': adn_time,
        'adn_models': models,
        'adn_histories': histories,
        'adn_predictions': adn_predictions
    }
    
    # Run GP for comparison if requested
    if use_gp:
        gp_start_time = time.time()
        
        # Format data for GP
        X_tensor, y_tensor, indices_tensor = next(iter(DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False)))
        
        X = X_tensor.numpy()
        y = y_tensor.numpy()
        day_indices = indices_tensor[:, 1].numpy()
        
        # Reshape X for GP
        X_reshaped = []
        for i in range(5):  # OHLCV
            X_reshaped.append(X[:, -1, i])  # Use only the most recent value for simplicity
        
        # Add some simple derived features
        X_reshaped.append((X[:, -1, 3] / X[:, -5, 3]) - 1)  # 5-day return
        
        # Run GP
        gp = GP(population_size=50, generations=30)
        gp_expressions = []
        gp_fitnesses = []
        
        for i in range(n_models):
            expr, fitness = gp.evolve(X_reshaped, y, day_indices)
            gp_expressions.append(expr)
            gp_fitnesses.append(fitness)
        
        # Evaluate GP expressions
        gp_predictions = []
        for expr in gp_expressions:
            preds = gp.evaluate_expression(expr, X_reshaped)
            gp_predictions.append(preds)
        
        # Calculate average IC
        gp_avg_ic = np.mean(gp_fitnesses)
        
        # Calculate diversity
        gp_diversity = calculate_diversity(gp_predictions, method='correlation')
        
        # Total time
        gp_time = time.time() - gp_start_time
        
        # Add GP results
        results.update({
            'gp_avg_ic': gp_avg_ic,
            'gp_diversity': gp_diversity,
            'gp_time': gp_time,
            'gp_expressions': gp_expressions,
            'gp_predictions': gp_predictions
        })
    
    return results

# Create and evaluate a portfolio strategy
def portfolio_strategy(predictions, future_returns, day_indices, top_pct=0.3, strategy_type='adn'):
    """
    Create and evaluate a portfolio strategy
    
    Parameters:
    predictions: Matrix of predictions (models x samples)
    future_returns: Vector of actual future returns
    day_indices: Vector of day indices
    top_pct: Percentage of top stocks to select
    strategy_type: Type of strategy ('adn', 'gp', or 'combined')
    
    Returns:
    results: Dictionary of strategy results
    """
    # Convert to numpy
    predictions = np.array(predictions)
    future_returns = np.array(future_returns)
    day_indices = np.array(day_indices)
    
    # Combine predictions (average across models)
    if len(predictions.shape) > 1:
        combined_pred = np.mean(predictions, axis=0)
    else:
        combined_pred = predictions
    
    # Group by day
    unique_days = np.unique(day_indices)
    n_days = len(unique_days)
    
    # Track strategy performance
    strategy_returns = np.zeros(n_days - 1)  # Skip the first day (no previous day's prediction)
    
    for i, day in enumerate(unique_days[:-1]):
        # Get predictions for current day
        day_mask = day_indices == day
        day_preds = combined_pred[day_mask]
        
        # Get returns for the next trading day
        next_day = unique_days[i + 1]
        next_day_mask = day_indices == next_day
        next_day_returns = future_returns[next_day_mask]
        
        # Select top and bottom stocks
        n_stocks = len(day_preds)
        top_k = int(n_stocks * top_pct)
        
        # Get indices of top and bottom stocks
        top_indices = np.argsort(day_preds)[-top_k:]
        bottom_indices = np.argsort(day_preds)[:top_k]
        
        # Calculate strategy return for the day (long top, short bottom)
        long_return = np.mean(next_day_returns[top_indices]) if len(top_indices) > 0 else 0
        short_return = -np.mean(next_day_returns[bottom_indices]) if len(bottom_indices) > 0 else 0
        strategy_returns[i] = (long_return + short_return) / 2
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Calculate other performance metrics
    total_return = cumulative_returns[-1]
    
    # Calculate max drawdown
    peak = 0
    max_drawdown = 0
    
    for i in range(len(cumulative_returns)):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
        else:
            drawdown = (peak - cumulative_returns[i]) / (1 + peak)
            max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio (assuming zero risk-free rate)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
    
    # Store results
    results = {
        'strategy_type': strategy_type,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }
    
    return results

# Compare models and strategies
def compare_strategies(adn_models, gp_expressions, test_dataset, prior_knowledge_factors=None):
    """
    Compare different strategies
    
    Parameters:
    adn_models: List of ADN models
    gp_expressions: List of GP expressions
    test_dataset: Test dataset
    prior_knowledge_factors: Prior knowledge factors
    
    Returns:
    results: Dictionary of comparison results
    """
    # Get test data
    X_tensor, y_tensor, indices_tensor = next(iter(DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False)))
    
    X = X_tensor.numpy()
    y = y_tensor.numpy()
    day_indices = indices_tensor[:, 1].numpy()
    
    # Get predictions from ADN models
    adn_predictions = []
    for model in adn_models:
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor.to(device))
            adn_predictions.append(outputs.cpu().numpy())
    
    # Get predictions from GP expressions if available
    gp_predictions = []
    if gp_expressions:
        # Reshape X for GP
        X_reshaped = []
        for i in range(5):  # OHLCV
            X_reshaped.append(X[:, -1, i])  # Use only the most recent value for simplicity
        
        # Add some simple derived features
        X_reshaped.append((X[:, -1, 3] / X[:, -5, 3]) - 1)  # 5-day return
        
        # Evaluate GP expressions
        gp = GP()  # Create a dummy GP object for evaluation
        for expr in gp_expressions:
            preds = gp.evaluate_expression(expr, X_reshaped)
            gp_predictions.append(preds)
    
    # Get predictions from prior knowledge factors if available
    pk_predictions = []
    if prior_knowledge_factors is not None:
        for factor_id in range(len(prior_knowledge_factors)):
            # Extract factor values for test days
            factor_values = []
            for idx in range(len(day_indices)):
                stock_id, day = indices_tensor[idx]
                factor_values.append(prior_knowledge_factors[factor_id][stock_id.item(), day.item()])
            pk_predictions.append(np.array(factor_values))
    
    # Create and evaluate strategies
    strategies = []
    
    # ADN strategy
    if adn_predictions:
        adn_strategy = portfolio_strategy(adn_predictions, y, day_indices, strategy_type='ADN')
        strategies.append(adn_strategy)
    
    # GP strategy
    if gp_predictions:
        gp_strategy = portfolio_strategy(gp_predictions, y, day_indices, strategy_type='GP')
        strategies.append(gp_strategy)
    
    # Prior knowledge strategy
    if pk_predictions:
        pk_strategy = portfolio_strategy(pk_predictions, y, day_indices, strategy_type='PK')
        strategies.append(pk_strategy)
    
    # Combined strategy (ADN + PK)
    if adn_predictions and pk_predictions:
        # Select top 25 from each based on individual IC
        adn_ics = []
        for preds in adn_predictions:
            ic = np.mean([spearmanr(preds[day_indices == day], y[day_indices == day])[0] 
                         for day in np.unique(day_indices)])
            adn_ics.append(ic)
        
        pk_ics = []
        for preds in pk_predictions:
            ic = np.mean([spearmanr(preds[day_indices == day], y[day_indices == day])[0] 
                         for day in np.unique(day_indices)])
            pk_ics.append(ic)
        
        # Select top predictions
        top_adn = [adn_predictions[i] for i in np.argsort(adn_ics)[-25:]]
        top_pk = [pk_predictions[i] for i in np.argsort(pk_ics)[-25:]]
        
        combined_predictions = top_adn + top_pk
        combined_strategy = portfolio_strategy(combined_predictions, y, day_indices, strategy_type='Combined')
        strategies.append(combined_strategy)
    
    # GP-PK strategy
    if gp_predictions and pk_predictions:
        gp_pk_predictions = gp_predictions + pk_predictions
        gp_pk_strategy = portfolio_strategy(gp_pk_predictions, y, day_indices, strategy_type='GP-PK')
        strategies.append(gp_pk_strategy)
    
    # Create summary table
    summary_data = {
        'Strategy': [s['strategy_type'] for s in strategies],
        'Total Return': [s['total_return'] for s in strategies],
        'Max Drawdown': [s['max_drawdown'] for s in strategies],
        'Sharpe Ratio': [s['sharpe_ratio'] for s in strategies]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    results = {
        'strategies': strategies,
        'summary': summary_df,
        'adn_predictions': adn_predictions,
        'gp_predictions': gp_predictions,
        'pk_predictions': pk_predictions
    }
    
    return results

# Main function to run all experiments
def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(n_stocks=100, n_days=370)  # Smaller dataset for quicker run time
    
    # Extract data
    stock_data = synthetic_data['stock_data']
    future_returns = synthetic_data['future_returns']
    
    # Get data dimensions
    n_stocks = len(stock_data)
    n_days = len(stock_data[0]['close'])
    lookback = 30
    
    # Split data
    train_days = int(n_days * 0.6)
    val_days = int(n_days * 0.2)
    test_days = n_days - train_days - val_days
    
    # Generate prior knowledge factors
    print("Generating prior knowledge factors...")
    pk_factors = generate_prior_knowledge(stock_data)
    
    # Create indices for each set
    train_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                     for day in range(lookback, train_days - 5)]
    
    val_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                   for day in range(train_days, train_days + val_days - 5)]
    
    test_indices = [(stock_id, day) for stock_id in range(n_stocks) 
                    for day in range(train_days + val_days, n_days - 5)]
    
    # Create datasets
    train_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=train_indices)
    val_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=val_indices)
    test_dataset = StockDataset(stock_data, future_returns, lookback=lookback, train_indices=test_indices)
    
    # Run experiments for each model type
    model_types = ['FCN', 'LeNet', 'LSTM']
    experiment_results = {}
    
    for model_type in model_types:
        print(f"\nRunning experiment with {model_type}...")
        result = run_experiment(
            synthetic_data=synthetic_data,
            model_type=model_type,
            n_models=15,  # Reduce number of models for quicker run time
            use_prior=True,
            use_gp=(model_type == 'FCN')  # Only run GP comparison once
        )
        experiment_results[model_type] = result
    
    # Compare model performance
    print("\nModel Performance Comparison:")
    comparison_data = {
        'Model': [],
        'IC': [],
        'Diversity': [],
        'Training Time (h)': []
    }
    
    for model_type, result in experiment_results.items():
        comparison_data['Model'].append(model_type)
        comparison_data['IC'].append(result['adn_avg_ic'])
        comparison_data['Diversity'].append(result['adn_diversity'])
        comparison_data['Training Time (h)'].append(result['adn_time'] / 3600)  # Convert to hours
    
    # Add GP if available
    if 'gp_avg_ic' in experiment_results['FCN']:
        comparison_data['Model'].append('GP')
        comparison_data['IC'].append(experiment_results['FCN']['gp_avg_ic'])
        comparison_data['Diversity'].append(experiment_results['FCN']['gp_diversity'])
        comparison_data['Training Time (h)'].append(experiment_results['FCN']['gp_time'] / 3600)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # Compare strategies
    print("\nComparing trading strategies...")
    
    # Get models and expressions
    fcn_models = experiment_results['FCN']['adn_models']
    gp_expressions = experiment_results['FCN'].get('gp_expressions', [])
    
    # Compare strategies
    strategy_results = compare_strategies(fcn_models, gp_expressions, test_dataset, pk_factors)
    
    # Print strategy comparison
    print("\nStrategy Performance Comparison:")
    print(strategy_results['summary'])
    
    # Plot strategy cumulative returns
    plt.figure(figsize=(12, 6))
    
    for strategy in strategy_results['strategies']:
        plt.plot(strategy['cumulative_returns'], label=f"{strategy['strategy_type']} (SR={strategy['sharpe_ratio']:.2f})")
    
    plt.xlabel('Trading Day')
    plt.ylabel('Cumulative Return')
    plt.title('Strategy Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Compare IC distributions
    plt.figure(figsize=(12, 6))
    
    # Calculate IC distributions for each day
    if strategy_results['adn_predictions']:
        adn_ics = []
        for day in np.unique(test_indices[-1][1]):
            day_mask = np.array([idx[1] for idx in test_indices]) == day
            if np.sum(day_mask) > 5:
                day_y = np.array([future_returns[idx[0]][idx[1]] for idx in test_indices])[day_mask]
                day_pred = np.mean([p[day_mask] for p in strategy_results['adn_predictions']], axis=0)
                ic, _ = spearmanr(day_pred, day_y)
                if not np.isnan(ic):
                    adn_ics.append(ic)
        
        plt.plot(adn_ics, label='ADN IC')
    
    if strategy_results['gp_predictions']:
        gp_ics = []
        for day in np.unique(test_indices[-1][1]):
            day_mask = np.array([idx[1] for idx in test_indices]) == day
            if np.sum(day_mask) > 5:
                day_y = np.array([future_returns[idx[0]][idx[1]] for idx in test_indices])[day_mask]
                day_pred = np.mean([p[day_mask] for p in strategy_results['gp_predictions']], axis=0)
                ic, _ = spearmanr(day_pred, day_y)
                if not np.isnan(ic):
                    gp_ics.append(ic)
        
        plt.plot(gp_ics, label='GP IC')
    
    if strategy_results['pk_predictions']:
        pk_ics = []
        for day in np.unique(test_indices[-1][1]):
            day_mask = np.array([idx[1] for idx in test_indices]) == day
            if np.sum(day_mask) > 5:
                day_y = np.array([future_returns[idx[0]][idx[1]] for idx in test_indices])[day_mask]
                day_pred = np.mean([p[day_mask] for p in strategy_results['pk_predictions']], axis=0)
                ic, _ = spearmanr(day_pred, day_y)
                if not np.isnan(ic):
                    pk_ics.append(ic)
        
        plt.plot(pk_ics, label='PK IC')
    
    plt.xlabel('Trading Day')
    plt.ylabel('Information Coefficient')
    plt.title('Daily Information Coefficient')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize feature diversity with t-SNE
    if (strategy_results['adn_predictions'] and 
        strategy_results['gp_predictions'] and 
        strategy_results['pk_predictions']):
        
        try:
            from sklearn.manifold import TSNE
            
            # Sample predictions from each source
            adn_sample = strategy_results['adn_predictions'][:10]
            gp_sample = strategy_results['gp_predictions'][:10]
            pk_sample = strategy_results['pk_predictions'][:10]
            
            # Combine predictions
            all_preds = []
            labels = []
            
            for p in adn_sample:
                all_preds.append(p)
                labels.append('ADN')
            
            for p in gp_sample:
                all_preds.append(p)
                labels.append('GP')
            
            for p in pk_sample:
                all_preds.append(p)
                labels.append('PK')
            
            # Create correlation matrix
            corr_matrix = np.zeros((len(all_preds), len(all_preds)))
            
            for i in range(len(all_preds)):
                for j in range(len(all_preds)):
                    corr, _ = spearmanr(all_preds[i], all_preds[j])
                    corr_matrix[i, j] = np.abs(corr) if not np.isnan(corr) else 0
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=min(5, len(all_preds)-1), random_state=42)
            embedding = tsne.fit_transform(corr_matrix)
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            colors = {'ADN': 'blue', 'GP': 'red', 'PK': 'green'}
            
            for i, label in enumerate(labels):
                plt.scatter(embedding[i, 0], embedding[i, 1], color=colors[label], label=label if label not in plt.gca().get_legend_handles_labels()[1] else '')
            
            plt.title('Feature Diversity Visualization (t-SNE)')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        except ImportError:
            print("scikit-learn not available for t-SNE visualization")

if __name__ == "__main__":
    main()