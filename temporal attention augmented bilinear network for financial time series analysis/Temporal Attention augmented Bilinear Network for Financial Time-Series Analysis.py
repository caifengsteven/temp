import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import time
import math
import json
import os
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# Try to import Bloomberg API
try:
    import pdblp
    HAS_BLOOMBERG = True
    print("Bloomberg API is available.")
except ImportError:
    HAS_BLOOMBERG = False
    print("Bloomberg API is not available. Will use simulated data for real-time.")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


#######################
# MODEL IMPLEMENTATION
#######################

class BilinearLayer(nn.Module):
    """
    Bilinear Layer (BL) as described in the paper.
    Maps an input X of size D x T to Y of size D' x T'
    """
    def __init__(self, input_dim_first, input_dim_second, output_dim_first, output_dim_second):
        super(BilinearLayer, self).__init__()
        self.W1 = nn.Parameter(torch.Tensor(output_dim_first, input_dim_first))
        self.W2 = nn.Parameter(torch.Tensor(input_dim_second, output_dim_second))
        self.B = nn.Parameter(torch.Tensor(output_dim_first, output_dim_second))
        self.reset_parameters()
        
    def reset_parameters(self):
        # He initialization
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.B, -bound, bound)
        
    def forward(self, x):
        # x has shape (batch_size, D, T)
        batch_size = x.size(0)
        
        # Perform W1 * X
        temp = torch.bmm(self.W1.expand(batch_size, -1, -1), x)
        
        # Perform temp * W2
        out = torch.bmm(temp, self.W2.expand(batch_size, -1, -1))
        
        # Add bias
        out = out + self.B.unsqueeze(0).expand(batch_size, -1, -1)
        
        return out


class TemporalAttentionBilinearLayer(nn.Module):
    """
    Temporal Attention augmented Bilinear Layer (TABL)
    """
    def __init__(self, input_dim_first, input_dim_second, output_dim_first, output_dim_second, activation=F.relu):
        super(TemporalAttentionBilinearLayer, self).__init__()
        
        # Transformation matrices
        self.W1 = nn.Parameter(torch.Tensor(output_dim_first, input_dim_first))
        self.W = nn.Parameter(torch.Tensor(input_dim_second, input_dim_second))
        self.W2 = nn.Parameter(torch.Tensor(input_dim_second, output_dim_second))
        self.B = nn.Parameter(torch.Tensor(output_dim_first, output_dim_second))
        
        # Attention weight parameter (λ in the paper)
        self.lambda_param = nn.Parameter(torch.Tensor(1))
        
        # Store dimensions
        self.input_dim_first = input_dim_first
        self.input_dim_second = input_dim_second
        self.output_dim_first = output_dim_first
        self.output_dim_second = output_dim_second
        
        # Activation function
        self.activation = activation
        
        # Initialize parameters
        self.reset_parameters(input_dim_second)
        
    def reset_parameters(self, T):
        # He initialization for W1 and W2
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        # Initialize W with 1/T on diagonal as described in the paper
        nn.init.constant_(self.W, 0)
        diag_indices = torch.arange(self.W.size(0))
        self.W.data[diag_indices, diag_indices] = 1/T
        
        # Initialize bias
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.B, -bound, bound)
        
        # Initialize lambda to 0.5 (middle of [0,1])
        nn.init.constant_(self.lambda_param, 0.5)
    
    def forward(self, x):
        # x has shape (batch_size, D, T)
        batch_size = x.size(0)
        
        # Step 1: W1 * X - Eq. (7) in the paper
        X_bar = torch.bmm(self.W1.expand(batch_size, -1, -1), x)  # (batch_size, D', T)
        
        # Step 2: Calculate attention weights - Eq. (8) in the paper
        # Use the original input x for attention calculation
        E = torch.bmm(x, self.W.expand(batch_size, -1, -1))  # (batch_size, D, T)
        
        # Step 3: Apply softmax to get attention weights - Eq. (9) in the paper
        # Apply softmax along the time dimension (axis=2)
        A = F.softmax(E, dim=2)  # (batch_size, D, T)
        
        # Step 4: Apply attention with lambda - Eq. (10) in the paper
        # Clamp lambda to [0, 1]
        lambda_clamped = torch.clamp(self.lambda_param, 0, 1)
        
        # Initialize X_tilde with same shape as X_bar
        X_tilde = torch.zeros_like(X_bar)
        
        # Create attention mask aligned with X_bar
        # For each feature in D', we need to select appropriate attention weights
        for i in range(batch_size):
            for j in range(self.output_dim_first):
                # Apply attention weights to each feature
                # Modulo ensures we cycle through input features if output_dim > input_dim
                attention_idx = j % self.input_dim_first
                X_tilde[i, j] = lambda_clamped * X_bar[i, j] * A[i, attention_idx] + (1 - lambda_clamped) * X_bar[i, j]
        
        # Step 5: X_tilde * W2 + B - Eq. (11) in the paper
        out = torch.bmm(X_tilde, self.W2.expand(batch_size, -1, -1))
        out = out + self.B.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply activation
        out = self.activation(out)
        
        return out, A


class TABL_Network(nn.Module):
    """
    Network with Temporal Attention augmented Bilinear Layers
    Implements the C(TABL) configuration from the paper with 2 hidden layers
    """
    def __init__(self, input_dim_first, input_dim_second, num_classes, hidden_dim_first=32, hidden_dim_second=40):
        super(TABL_Network, self).__init__()
        
        # Store dimensions for debugging
        self.input_dim_first = input_dim_first
        self.input_dim_second = input_dim_second
        self.hidden_dim_first = hidden_dim_first
        self.hidden_dim_second = hidden_dim_second
        
        # First hidden BL layer
        self.hidden1 = BilinearLayer(input_dim_first, input_dim_second, hidden_dim_first, hidden_dim_second)
        
        # Second hidden BL layer
        self.hidden2 = BilinearLayer(hidden_dim_first, hidden_dim_second, hidden_dim_first, hidden_dim_second)
        
        # TABL layer for classification with num_classes classes
        self.tabl_layer = TemporalAttentionBilinearLayer(hidden_dim_first, hidden_dim_second, num_classes, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)  # Increased dropout for better regularization
        
        # Activation function
        self.activation = F.relu
        
    def forward(self, x):
        # First hidden layer
        x = self.activation(self.hidden1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.activation(self.hidden2(x))
        x = self.dropout(x)
        
        # TABL layer
        out, attention = self.tabl_layer(x)
        
        # Reshape output to (batch_size, num_classes)
        out = out.squeeze(-1)
        
        return out, attention


#######################
# DATA HANDLING
#######################

def fetch_lob_data(ticker, levels=10, start_date=None, end_date=None, intervals=10):
    """
    Generate simulated LOB data for training the model
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    levels : int
        Number of LOB levels to fetch
    start_date, end_date : str
        Date range to fetch data for
    intervals : int
        Number of time intervals to fetch
        
    Returns:
    --------
    numpy.ndarray
        LOB data of shape (n_samples, 4*levels, intervals)
        Each sample contains prices and volumes for 'levels' LOB levels
        over 'intervals' time steps.
    """
    # Generate simulated LOB data
    print("Generating simulated LOB data for training...")
    
    # Number of samples to generate
    n_samples = 1000
    
    # Generate simulated LOB data with more balanced classes
    data = np.zeros((n_samples, 4*levels, intervals))
    
    # Generate labels with more balanced distribution
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Decide the trend type for this sample
        trend_type = i % 3 - 1  # -1, 0, 1 in sequence for balance
        labels[i] = trend_type
        
        # Start with a mid price
        mid_price = 100 + np.random.normal(0, 5)
        
        # Generate temporal variations for mid price
        mid_prices = np.zeros(intervals)
        mid_prices[0] = mid_price
        
        # Set trend based on label
        if trend_type == -1:  # Downward
            trend = -0.01
        elif trend_type == 0:  # Stationary
            trend = 0
        else:  # Upward
            trend = 0.01
        
        # Add random fluctuations to create a time series
        for t in range(1, intervals):
            # Random walk with trend
            mid_prices[t] = mid_prices[t-1] * (1 + trend + np.random.normal(0, 0.001))
        
        # Generate bid and ask prices based on mid prices
        for t in range(intervals):
            current_mid = mid_prices[t]
            
            # Generate spreads for each level
            spreads = np.exp(np.linspace(0, 1, levels)) - 1
            
            # Bid prices (descending from mid price)
            for j in range(levels):
                data[i, j, t] = current_mid * (1 - 0.0005 - 0.001 * spreads[j])
            
            # Ask prices (ascending from mid price)
            for j in range(levels):
                data[i, levels + j, t] = current_mid * (1 + 0.0005 + 0.001 * spreads[j])
            
            # Generate volumes with patterns based on trend
            # Bid volumes
            if trend_type == -1:  # Downward trend: higher ask volumes
                data[i, 2*levels:3*levels, t] = np.random.exponential(80, levels)
                data[i, 3*levels:4*levels, t] = np.random.exponential(120, levels)
            elif trend_type == 1:  # Upward trend: higher bid volumes
                data[i, 2*levels:3*levels, t] = np.random.exponential(120, levels)
                data[i, 3*levels:4*levels, t] = np.random.exponential(80, levels)
            else:  # Stationary: balanced volumes
                data[i, 2*levels:3*levels, t] = np.random.exponential(100, levels)
                data[i, 3*levels:4*levels, t] = np.random.exponential(100, levels)
    
    # Shuffle the data to avoid sequential patterns
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    return data, labels


def prepare_data(data, labels, val_split=0.15, test_split=0.15):
    """
    Prepare data for training
    
    Parameters:
    -----------
    data : numpy.ndarray
        LOB data of shape (n_samples, features, time_steps)
    labels : numpy.ndarray
        Labels of shape (n_samples,)
    val_split, test_split : float
        Proportion of data to use for validation and test
        
    Returns:
    --------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
        DataLoaders for training, validation and test sets
    """
    # Convert labels from {-1, 0, 1} to {0, 1, 2}
    labels = labels + 1
    
    # Determine split sizes
    n_samples = data.shape[0]
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val
    
    # Split the data
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]
    
    train_labels = labels[:n_train]
    val_labels = labels[n_train:n_train+n_val]
    test_labels = labels[n_train+n_val:]
    
    # Normalize the data (important for convergence)
    # We'll normalize each channel separately
    channel_means = []
    channel_stds = []
    
    for i in range(data.shape[1]):
        # Calculate mean and std from training data
        channel_mean = np.mean(train_data[:, i, :])
        channel_std = np.std(train_data[:, i, :])
        
        channel_means.append(channel_mean)
        channel_stds.append(max(channel_std, 1e-6))
        
        # Apply normalization to all splits
        train_data[:, i, :] = (train_data[:, i, :] - channel_mean) / max(channel_std, 1e-6)
        val_data[:, i, :] = (val_data[:, i, :] - channel_mean) / max(channel_std, 1e-6)
        test_data[:, i, :] = (test_data[:, i, :] - channel_mean) / max(channel_std, 1e-6)
    
    # Save normalization parameters
    norm_params = {
        'means': channel_means,
        'stds': channel_stds
    }
    
    # Save the normalization parameters for later use
    with open('norm_params.json', 'w') as f:
        json.dump({
            'means': channel_means,
            'stds': channel_stds
        }, f)
    
    # Convert to PyTorch tensors
    train_data_tensor = torch.FloatTensor(train_data)
    val_data_tensor = torch.FloatTensor(val_data)
    test_data_tensor = torch.FloatTensor(test_data)
    
    train_labels_tensor = torch.LongTensor(train_labels)
    val_labels_tensor = torch.LongTensor(val_labels)
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, norm_params


def train_model(model, train_loader, val_loader, device, epochs=100):
    """
    Train the model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    train_loader, val_loader : torch.utils.data.DataLoader
        DataLoaders for training and validation sets
    device : torch.device
        Device to train on
    epochs : int
        Number of epochs to train for
        
    Returns:
    --------
    model : torch.nn.Module
        Trained model
    history : dict
        Training history
    """
    # Move model to device
    model = model.to(device)
    
    # Create optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create weighted loss function to handle class imbalance
    # Count class frequencies in training set
    class_counts = torch.zeros(3, dtype=torch.float32)
    for _, labels in train_loader:
        for cls in range(3):
            class_counts[cls] += (labels == cls).sum().item()
    
    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3  # Normalize
    class_weights = class_weights.to(device)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # Training loop
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to predictions
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch statistics
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Convert outputs to predictions
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch statistics
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        # Update learning rate
        scheduler.step(val_f1)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Early stopping based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_tabl_model.pt')
        else:
            patience_counter += 1
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Train F1: {train_f1:.4f} | '
              f'Val F1: {val_f1:.4f}')
        
        # Print per-class metrics for validation
        val_precision = precision_score(val_targets, val_preds, average=None, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average=None, zero_division=0)
        val_f1_per_class = f1_score(val_targets, val_preds, average=None, zero_division=0)
        
        for i, cls in enumerate(['Down', 'Stationary', 'Up']):
            print(f"  {cls}: Prec={val_precision[i]:.4f}, Recall={val_recall[i]:.4f}, F1={val_f1_per_class[i]:.4f}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('best_tabl_model.pt'))
    
    return model, history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    test_loader : torch.utils.data.DataLoader
        DataLoader for test set
    device : torch.device
        Device to evaluate on
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Track statistics
    test_preds = []
    test_targets = []
    attention_patterns = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, attention = model(inputs)
            
            # Convert outputs to predictions
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            
            # Save attention patterns
            attention_patterns.append(attention.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, average='macro', zero_division=0)
    recall = recall_score(test_targets, test_preds, average='macro', zero_division=0)
    f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    class_precision = precision_score(test_targets, test_preds, average=None, zero_division=0)
    class_recall = recall_score(test_targets, test_preds, average=None, zero_division=0)
    class_f1 = f1_score(test_targets, test_preds, average=None, zero_division=0)
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'attention_patterns': attention_patterns,
        'predictions': test_preds,
        'targets': test_targets
    }
    
    return metrics


#######################
# BLOOMBERG DATA ACCESS
#######################

def connect_to_bloomberg():
    """
    Connect to Bloomberg API
    
    Returns:
    --------
    conn : pdblp.BCon
        Bloomberg connection object
    """
    if not HAS_BLOOMBERG:
        print("Bloomberg API not available. Will use simulated data.")
        return None
        
    try:
        # Connect to Bloomberg terminal
        print("Connecting to Bloomberg...")
        conn = pdblp.BCon(timeout=5000)
        conn.start()
        print("Successfully connected to Bloomberg.")
        return conn
    except Exception as e:
        print(f"Error connecting to Bloomberg: {e}")
        return None


def get_realtime_lob_data(conn, ticker, levels=10):
    """
    Get real-time LOB data from Bloomberg
    
    Parameters:
    -----------
    conn : pdblp.BCon
        Bloomberg connection
    ticker : str
        Ticker symbol
    levels : int
        Number of LOB levels to fetch
        
    Returns:
    --------
    lob_data : dict
        Current LOB data
    """
    if conn is None:
        # Use simulated data if Bloomberg is not available
        return simulate_realtime_lob_data(ticker, levels)
        
    try:
        # Fields to request for each level
        fields = []
        for i in range(1, levels + 1):
            fields.extend([
                f'BID{i}', f'ASK{i}',  # Prices
                f'BID_SIZE{i}', f'ASK_SIZE{i}'  # Volumes
            ])
        
        # Make request to Bloomberg
        data = conn.ref(ticker, fields)
        
        if data is None or data.empty:
            print("No data returned from Bloomberg. Using simulated data.")
            return simulate_realtime_lob_data(ticker, levels)
        
        # Process response
        bid_prices = [data[f'BID{i}'].iloc[0] for i in range(1, levels + 1)]
        ask_prices = [data[f'ASK{i}'].iloc[0] for i in range(1, levels + 1)]
        bid_volumes = [data[f'BID_SIZE{i}'].iloc[0] for i in range(1, levels + 1)]
        ask_volumes = [data[f'ASK_SIZE{i}'].iloc[0] for i in range(1, levels + 1)]
        
        # Calculate mid-price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Format data
        return {
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
            'mid_price': mid_price,
            'source': 'bloomberg'
        }
        
    except Exception as e:
        print(f"Error fetching LOB data from Bloomberg: {e}")
        return simulate_realtime_lob_data(ticker, levels)


def simulate_realtime_lob_data(ticker, levels=10):
    """
    Simulate real-time LOB data when Bloomberg is not available
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    levels : int
        Number of LOB levels to fetch
        
    Returns:
    --------
    lob_data : dict
        Simulated LOB data
    """
    # Get the base price for the ticker (this could be cached/persisted)
    base_price = 100.0  # Default
    
    # For known tickers, use more realistic prices
    ticker_prices = {
        'SPY': 450.0,
        'AAPL': 180.0,
        'MSFT': 360.0,
        'AMZN': 140.0,
        'GOOGL': 130.0,
        'META': 450.0,
        'TSLA': 200.0,
        'NVDA': 800.0
    }
    
    if ticker in ticker_prices:
        base_price = ticker_prices[ticker]
    
    # Add small random variation to the price
    current_price = base_price * (1 + np.random.normal(0, 0.0005))
    
    # Calculate spread
    spread = current_price * 0.0005  # 5 bps spread
    
    # Calculate bid/ask prices
    bid_price = current_price - spread/2
    ask_price = current_price + spread/2
    
    # Generate spreads for each level
    spreads = np.exp(np.linspace(0, 1, levels)) - 1
    
    # Generate bid and ask prices
    bid_prices = [bid_price * (1 - 0.0002 * spreads[i]) for i in range(levels)]
    ask_prices = [ask_price * (1 + 0.0002 * spreads[i]) for i in range(levels)]
    
    # Generate volumes (decreasing with level)
    base_volume = 1000
    volume_decay = np.exp(-np.linspace(0, 2, levels))
    
    bid_volumes = [int(base_volume * volume_decay[i] * (1 + np.random.normal(0, 0.1))) for i in range(levels)]
    ask_volumes = [int(base_volume * volume_decay[i] * (1 + np.random.normal(0, 0.1))) for i in range(levels)]
    
    return {
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'bid_volumes': bid_volumes,
        'ask_volumes': ask_volumes,
        'mid_price': current_price,
        'source': 'simulated'
    }


#######################
# TRADING SYSTEM
#######################

class TABLTradingStrategy:
    """
    Trading strategy using TABL model predictions
    """
    def __init__(self, model, device, norm_params, lookback_periods=10, position_size=1.0, stop_loss=0.01, take_profit=0.02):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained TABL model
        device : torch.device
            Device to run inference on
        norm_params : dict
            Normalization parameters
        lookback_periods : int
            Number of time periods to look back (should match model input)
        position_size : float
            Size of position to take (relative to account)
        stop_loss : float
            Stop loss level (as percentage)
        take_profit : float
            Take profit level (as percentage)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.norm_params = norm_params
        self.lookback_periods = lookback_periods
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Data buffer to store historical data
        self.buffer = deque(maxlen=lookback_periods)
        
        # Current position
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.entry_price = None
        self.trade_history = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
    def process_lob_data(self, lob_data):
        """
        Process LOB data into model input format
        
        Parameters:
        -----------
        lob_data : dict
            LOB data with bid/ask prices and volumes
            
        Returns:
        --------
        features : numpy.ndarray
            Processed features
        """
        # Extract features from lob_data
        bid_prices = np.array(lob_data['bid_prices'])
        ask_prices = np.array(lob_data['ask_prices'])
        bid_volumes = np.array(lob_data['bid_volumes'])
        ask_volumes = np.array(lob_data['ask_volumes'])
        
        # Combine into feature vector (shape: 4*levels)
        features = np.concatenate([bid_prices, ask_prices, bid_volumes, ask_volumes])
        
        return features
    
    def update(self, lob_data):
        """
        Update strategy with new market data
        
        Parameters:
        -----------
        lob_data : dict
            Current limit order book data
            
        Returns:
        --------
        action : dict
            Trading action to take
        """
        # Process the LOB data into model input format
        features = self.process_lob_data(lob_data)
        
        # Add to buffer
        self.buffer.append(features)
        
        # If buffer is not full yet, return hold action
        if len(self.buffer) < self.lookback_periods:
            return {
                'type': 'HOLD',
                'size': 0,
                'reason': 'Insufficient data',
                'signal': 0,
                'confidence': 0,
                'prediction': None
            }
        
        # Get current mid price
        current_price = lob_data['mid_price']
        
        # Check if we need to close existing position (stop loss/take profit)
        action = self._check_exit_conditions(current_price)
        if action['type'] != 'HOLD':
            return action
            
        # Prepare data for prediction
        prediction = self._make_prediction()
        
        # No prediction available
        if prediction is None:
            return {
                'type': 'HOLD',
                'size': 0,
                'reason': 'No prediction',
                'signal': 0,
                'confidence': 0,
                'prediction': None
            }
        
        # Extract signal and confidence
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Only take trades with sufficient confidence
        confidence_threshold = 0.6
        
        if confidence >= confidence_threshold:
            if signal == 1 and self.position <= 0:  # Bullish signal
                # Close any short position
                if self.position < 0:
                    self._close_position(current_price, 'Signal change')
                
                # Open long position
                action = {
                    'type': 'BUY',
                    'size': self.position_size,
                    'reason': f'Bullish signal (conf: {confidence:.2f})',
                    'signal': signal,
                    'confidence': confidence,
                    'prediction': prediction
                }
                self._open_position(1, current_price)
                return action
                
            elif signal == -1 and self.position >= 0:  # Bearish signal
                # Close any long position
                if self.position > 0:
                    self._close_position(current_price, 'Signal change')
                
                # Open short position
                action = {
                    'type': 'SELL',
                    'size': self.position_size,
                    'reason': f'Bearish signal (conf: {confidence:.2f})',
                    'signal': signal,
                    'confidence': confidence,
                    'prediction': prediction
                }
                self._open_position(-1, current_price)
                return action
        
        # Default is hold
        return {
            'type': 'HOLD',
            'size': 0,
            'reason': f'No signal with sufficient confidence (signal: {signal}, conf: {confidence:.2f})',
            'signal': signal,
            'confidence': confidence,
            'prediction': prediction
        }
    
    def _make_prediction(self):
        """
        Make a prediction using the TABL model
        
        Returns:
        --------
        prediction : dict
            Prediction results
        """
        try:
            # Prepare data for model
            data = np.array(list(self.buffer))
            
            # Reshape to (1, features, time)
            data = data.T.reshape(1, -1, self.lookback_periods)
            
            # Normalize data
            for i in range(data.shape[1]):
                if i < len(self.norm_params['means']):
                    data[:, i, :] = (data[:, i, :] - self.norm_params['means'][i]) / self.norm_params['stds'][i]
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs, attention = self.model(data_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predicted class
                _, predicted_class = torch.max(outputs, 1)
                
                # Convert from {0, 1, 2} to {-1, 0, 1}
                signal = predicted_class.item() - 1
                
                # Get confidence
                confidence = probabilities[0, predicted_class.item()].item()
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy()[0],
                    'attention': attention.cpu().numpy()[0]
                }
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def _check_exit_conditions(self, current_price):
        """
        Check if we need to exit the position
        
        Parameters:
        -----------
        current_price : float
            Current price
            
        Returns:
        --------
        action : dict
            Trading action
        """
        # Default action is hold
        action = {
            'type': 'HOLD',
            'size': 0,
            'reason': 'No exit condition met',
            'signal': 0,
            'confidence': 0,
            'prediction': None
        }
        
        # If no position, return hold
        if self.position == 0 or self.entry_price is None:
            return action
        
        # Calculate PnL
        pnl_pct = (current_price / self.entry_price - 1) * self.position
        
        if pnl_pct <= -self.stop_loss:
            # Stop loss hit
            action = {
                'type': 'CLOSE',
                'size': self.position_size,
                'reason': f'Stop loss ({pnl_pct:.2%})',
                'signal': 0,
                'confidence': 1.0,
                'prediction': None
            }
            self._close_position(current_price, 'Stop loss')
            return action
        
        if pnl_pct >= self.take_profit:
            # Take profit hit
            action = {
                'type': 'CLOSE',
                'size': self.position_size,
                'reason': f'Take profit ({pnl_pct:.2%})',
                'signal': 0,
                'confidence': 1.0,
                'prediction': None
            }
            self._close_position(current_price, 'Take profit')
            return action
            
        return action
    
    def _open_position(self, direction, price):
        """
        Open a new position
        
        Parameters:
        -----------
        direction : int
            Direction of position (-1 for short, 1 for long)
        price : float
            Entry price
        """
        self.position = direction
        self.entry_price = price
        self.total_trades += 1
        
        self.trade_history.append({
            'timestamp': datetime.datetime.now(),
            'action': 'BUY' if direction > 0 else 'SELL',
            'price': price,
            'size': self.position_size,
            'pnl': None  # Explicitly set to None for new positions
        })
    
    def _close_position(self, price, reason):
        """
        Close the current position
        
        Parameters:
        -----------
        price : float
            Exit price
        reason : str
            Reason for closing
        """
        if self.position == 0 or self.entry_price is None:
            return
            
        pnl = (price / self.entry_price - 1) * self.position * self.position_size
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'timestamp': datetime.datetime.now(),
            'action': 'CLOSE',
            'price': price,
            'size': self.position_size,
            'pnl': pnl,
            'reason': reason
        })
        
        self.position = 0
        self.entry_price = None
        
    def get_performance_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }
            
        # Calculate metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average profit and loss - Fix for None values
        profits = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) is not None and t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) is not None and t.get('pnl', 0) < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }
        
    def reset(self):
        """Reset the strategy"""
        self.buffer.clear()
        self.position = 0
        self.entry_price = None
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0


#######################
# BACKTESTING
#######################

def generate_backtest_data(ticker, n_samples=1000, levels=10, intervals=10, trend=None):
    """
    Generate simulated data for backtesting
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    n_samples : int
        Number of samples to generate
    levels : int
        Number of LOB levels
    intervals : int
        Number of time intervals
    trend : float or None
        Overall trend direction
        
    Returns:
    --------
    backtest_data : list
        List of LOB data samples for backtesting
    """
    print(f"Generating simulated backtest data for {ticker}...")
    
    # Get base price for ticker
    base_price = 100.0
    ticker_prices = {
        'SPY': 450.0,
        'AAPL': 180.0,
        'MSFT': 360.0,
        'AMZN': 140.0,
        'GOOGL': 130.0,
        'META': 450.0,
        'TSLA': 200.0,
        'NVDA': 800.0
    }
    
    if ticker in ticker_prices:
        base_price = ticker_prices[ticker]
    
    # Overall trend component
    if trend is None:
        trend = np.random.choice([-0.005, 0, 0.005]) # Random trend direction
    
    # Generate mid price time series
    mid_prices = np.zeros(n_samples)
    mid_prices[0] = base_price
    
    # Add price movements
    for i in range(1, n_samples):
        # Random walk with trend and some mean reversion
        random_component = np.random.normal(0, 0.002)
        trend_component = trend
        reversion_component = 0.01 * (base_price - mid_prices[i-1]) / base_price
        
        mid_prices[i] = mid_prices[i-1] * (1 + random_component + trend_component + reversion_component)
    
    # Generate backtest data
    backtest_data = []
    
    for i in range(n_samples):
        current_mid = mid_prices[i]
        
        # Generate spreads for each level
        spreads = np.exp(np.linspace(0, 1, levels)) - 1
        
        # Calculate bid/ask
        spread_bps = 5  # 5 bps spread
        spread_amount = current_mid * (spread_bps / 10000)
        
        bid_price = current_mid - spread_amount/2
        ask_price = current_mid + spread_amount/2
        
        # Generate bid and ask prices
        bid_prices = [bid_price * (1 - 0.0002 * spreads[j]) for j in range(levels)]
        ask_prices = [ask_price * (1 + 0.0002 * spreads[j]) for j in range(levels)]
        
        # Generate volumes with patterns
        base_volume = 1000
        volume_decay = np.exp(-np.linspace(0, 2, levels))
        
        bid_volumes = [int(base_volume * volume_decay[j] * (1 + np.random.normal(0, 0.1))) for j in range(levels)]
        ask_volumes = [int(base_volume * volume_decay[j] * (1 + np.random.normal(0, 0.1))) for j in range(levels)]
        
        # Store data
        backtest_data.append({
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
            'mid_price': current_mid,
            'timestamp': datetime.datetime.now() + datetime.timedelta(seconds=i),
            'source': 'backtest'
        })
    
    return backtest_data


def backtest_strategy(model, device, norm_params, backtest_data, lookback_periods=10, initial_capital=100000):
    """
    Backtest the TABL trading strategy
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained TABL model
    device : torch.device
        Device to run inference on
    norm_params : dict
        Normalization parameters
    backtest_data : list
        List of LOB data samples
    lookback_periods : int
        Number of periods to look back for prediction
    initial_capital : float
        Initial capital for backtesting
        
    Returns:
    --------
    results : dict
        Backtest results
    """
    # Initialize strategy
    strategy = TABLTradingStrategy(
        model=model,
        device=device,
        norm_params=norm_params,
        lookback_periods=lookback_periods,
        position_size=0.1,  # Use 10% of capital per trade
        stop_loss=0.01,     # 1% stop loss
        take_profit=0.02    # 2% take profit
    )
    
    # Initialize portfolio
    portfolio = {
        'cash': initial_capital,
        'position': 0,
        'equity': [],
        'timestamps': [],
        'trades': []
    }
    
    # Run backtest
    print(f"Running backtest on {len(backtest_data)} samples...")
    
    for i, lob_data in enumerate(backtest_data):
        # Skip first 'lookback_periods' data points
        if i < lookback_periods:
            # Initialize equity curve
            portfolio['equity'].append(initial_capital)
            portfolio['timestamps'].append(lob_data['timestamp'])
            continue
            
        # Get current price
        current_price = lob_data['mid_price']
        
        # Update strategy
        action = strategy.update(lob_data)
        
        # Execute trading action
        if action['type'] == 'BUY':
            # Calculate how many shares we can buy
            shares = (portfolio['cash'] * action['size']) / current_price
            portfolio['cash'] -= shares * current_price
            portfolio['position'] += shares
            
            # Record trade
            portfolio['trades'].append({
                'timestamp': lob_data['timestamp'],
                'type': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': shares * current_price,
                'reason': action['reason']
            })
            
            print(f"[{lob_data['timestamp']}] BUY: {shares:.2f} shares at ${current_price:.2f} - {action['reason']}")
            
        elif action['type'] == 'SELL':
            # Calculate how many shares to sell short
            shares = (portfolio['cash'] * action['size']) / current_price
            portfolio['cash'] += shares * current_price
            portfolio['position'] -= shares
            
            # Record trade
            portfolio['trades'].append({
                'timestamp': lob_data['timestamp'],
                'type': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': shares * current_price,
                'reason': action['reason']
            })
            
            print(f"[{lob_data['timestamp']}] SELL: {shares:.2f} shares at ${current_price:.2f} - {action['reason']}")
            
        elif action['type'] == 'CLOSE':
            # Close the position
            close_value = portfolio['position'] * current_price
            portfolio['cash'] += close_value
            
            # Record trade
            portfolio['trades'].append({
                'timestamp': lob_data['timestamp'],
                'type': 'CLOSE',
                'price': current_price,
                'shares': abs(portfolio['position']),
                'value': abs(close_value),
                'reason': action['reason']
            })
            
            print(f"[{lob_data['timestamp']}] CLOSE: {abs(portfolio['position']):.2f} shares at ${current_price:.2f} - {action['reason']}")
            
            portfolio['position'] = 0
        
        # Calculate current equity
        equity = portfolio['cash'] + portfolio['position'] * current_price
        portfolio['equity'].append(equity)
        portfolio['timestamps'].append(lob_data['timestamp'])
    
    # Calculate performance metrics
    equity_array = np.array(portfolio['equity'])
    returns = np.diff(equity_array) / equity_array[:-1]
    
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)  # Annualized
    max_drawdown = np.max(np.maximum.accumulate(equity_array) - equity_array) / np.max(equity_array)
    
    # Get strategy metrics
    strategy_metrics = strategy.get_performance_metrics()
    
    results = {
        'initial_capital': initial_capital,
        'final_equity': portfolio['equity'][-1],
        'total_return': (portfolio['equity'][-1] / initial_capital - 1) * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_metrics': strategy_metrics,
        'equity_curve': portfolio['equity'],
        'timestamps': portfolio['timestamps'],
        'trades': portfolio['trades']
    }
    
    print(f"\nBacktest Results:")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {strategy_metrics['win_rate']:.2f}")
    
    return results


def plot_backtest_results(results):
    """
    Plot backtest results
    
    Parameters:
    -----------
    results : dict
        Backtest results
    """
    # Convert timestamps to datetime if they're strings
    if isinstance(results['timestamps'][0], str):
        timestamps = [datetime.datetime.fromisoformat(ts) for ts in results['timestamps']]
    else:
        timestamps = results['timestamps']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(timestamps, results['equity_curve'])
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    
    # Plot drawdown
    equity_array = np.array(results['equity_curve'])
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (running_max - equity_array) / running_max * 100
    
    ax2.fill_between(timestamps, drawdown, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    
    # Add buy/sell markers on the equity curve
    buys = [(t['timestamp'], t['price']) for t in results['trades'] if t['type'] == 'BUY']
    sells = [(t['timestamp'], t['price']) for t in results['trades'] if t['type'] == 'SELL']
    closes = [(t['timestamp'], t['price']) for t in results['trades'] if t['type'] == 'CLOSE']
    
    if buys:
        buy_timestamps, buy_prices = zip(*buys)
        # Find the closest equity value for each buy timestamp
        buy_equities = [results['equity_curve'][timestamps.index(t) if t in timestamps else -1] for t in buy_timestamps]
        ax1.scatter(buy_timestamps, buy_equities, color='green', marker='^', s=100, label='Buy')
    
    if sells:
        sell_timestamps, sell_prices = zip(*sells)
        # Find the closest equity value for each sell timestamp
        sell_equities = [results['equity_curve'][timestamps.index(t) if t in timestamps else -1] for t in sell_timestamps]
        ax1.scatter(sell_timestamps, sell_equities, color='red', marker='v', s=100, label='Sell')
    
    if closes:
        close_timestamps, close_prices = zip(*closes)
        # Find the closest equity value for each close timestamp
        close_equities = [results['equity_curve'][timestamps.index(t) if t in timestamps else -1] for t in close_timestamps]
        ax1.scatter(close_timestamps, close_equities, color='blue', marker='o', s=100, label='Close')
    
    ax1.legend()
    
    # Add key metrics as text
    textstr = '\n'.join((
        f"Initial Capital: ${results['initial_capital']:.2f}",
        f"Final Equity: ${results['final_equity']:.2f}",
        f"Total Return: {results['total_return']:.2f}%",
        f"Sharpe Ratio: {results['sharpe_ratio']:.2f}",
        f"Max Drawdown: {results['max_drawdown']*100:.2f}%",
        f"Win Rate: {results['trade_metrics']['win_rate']*100:.2f}%",
        f"Total Trades: {results['trade_metrics']['total_trades']}"
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.show()


#######################
# LIVE TRADING
#######################

def run_live_trading(model, device, norm_params, ticker, run_duration=3600, broker_api=None):
    """
    Run live trading with the TABL model
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained TABL model
    device : torch.device
        Device to run inference on
    norm_params : dict
        Normalization parameters
    ticker : str
        Ticker symbol to trade
    run_duration : int
        Duration to run in seconds
    broker_api : object, optional
        Broker API for executing trades
        
    Returns:
    --------
    results : dict
        Trading results
    """
    # Connect to Bloomberg
    bbg_conn = connect_to_bloomberg()
    
    # Initialize strategy
    strategy = TABLTradingStrategy(
        model=model,
        device=device,
        norm_params=norm_params,
        lookback_periods=10,
        position_size=0.1,  # Use 10% of capital per trade
        stop_loss=0.01,     # 1% stop loss
        take_profit=0.02    # 2% take profit
    )
    
    # Portfolio tracking
    portfolio = {
        'cash': 100000,  # Initial capital
        'position': 0,
        'equity': [],
        'timestamps': [],
        'trades': []
    }
    
    # Trading loop
    start_time = time.time()
    print(f"Starting live trading for {ticker}...")
    print(f"Will run for {run_duration/60:.1f} minutes")
    
    try:
        while time.time() - start_time < run_duration:
            # Get current timestamp
            current_time = datetime.datetime.now()
            
            # Get market data
            lob_data = get_realtime_lob_data(bbg_conn, ticker)
            current_price = lob_data['mid_price']
            
            # Update strategy
            action = strategy.update(lob_data)
            
            # Execute trading action
            if action['type'] == 'BUY':
                # Calculate how many shares we can buy
                shares = (portfolio['cash'] * action['size']) / current_price
                portfolio['cash'] -= shares * current_price
                portfolio['position'] += shares
                
                # Record trade
                portfolio['trades'].append({
                    'timestamp': current_time,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price,
                    'reason': action['reason']
                })
                
                print(f"[{current_time}] BUY: {shares:.2f} shares at ${current_price:.2f} - {action['reason']}")
                
                # Execute with broker if available
                if broker_api:
                    try:
                        broker_api.place_order(ticker, 'BUY', shares, current_price)
                    except Exception as e:
                        print(f"Error executing trade with broker: {e}")
                
            elif action['type'] == 'SELL':
                # Calculate how many shares to sell short
                shares = (portfolio['cash'] * action['size']) / current_price
                portfolio['cash'] += shares * current_price
                portfolio['position'] -= shares
                
                # Record trade
                portfolio['trades'].append({
                    'timestamp': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price,
                    'reason': action['reason']
                })
                
                print(f"[{current_time}] SELL: {shares:.2f} shares at ${current_price:.2f} - {action['reason']}")
                
                # Execute with broker if available
                if broker_api:
                    try:
                        broker_api.place_order(ticker, 'SELL', shares, current_price)
                    except Exception as e:
                        print(f"Error executing trade with broker: {e}")
                
            elif action['type'] == 'CLOSE':
                # Close the position
                close_value = portfolio['position'] * current_price
                portfolio['cash'] += close_value
                
                # Record trade
                portfolio['trades'].append({
                    'timestamp': current_time,
                    'type': 'CLOSE',
                    'price': current_price,
                    'shares': abs(portfolio['position']),
                    'value': abs(close_value),
                    'reason': action['reason']
                })
                
                print(f"[{current_time}] CLOSE: {abs(portfolio['position']):.2f} shares at ${current_price:.2f} - {action['reason']}")
                
                # Execute with broker if available
                if broker_api:
                    try:
                        order_type = 'BUY' if portfolio['position'] < 0 else 'SELL'
                        broker_api.place_order(ticker, order_type, abs(portfolio['position']), current_price)
                    except Exception as e:
                        print(f"Error executing trade with broker: {e}")
                
                portfolio['position'] = 0
            
            # Calculate current equity
            equity = portfolio['cash'] + portfolio['position'] * current_price
            portfolio['equity'].append(equity)
            portfolio['timestamps'].append(current_time)
            
            # Wait for next update (1 second)
            time.sleep(1)
            
            # Print status every 60 seconds
            if int(time.time() - start_time) % 60 == 0:
                elapsed = int(time.time() - start_time)
                remaining = run_duration - elapsed
                print(f"Status: Elapsed {elapsed} seconds, Remaining {remaining} seconds. Current equity: ${equity:.2f}")
    
    except KeyboardInterrupt:
        print("\nTrading interrupted by user")
    
    finally:
        # Close Bloomberg connection
        if bbg_conn:
            try:
                bbg_conn.stop()
            except:
                pass
        
        # Close any open positions
        if portfolio['position'] != 0:
            print(f"Closing final position: {portfolio['position']:.2f} shares")
            
            # Get final price
            try:
                final_lob_data = get_realtime_lob_data(None, ticker)  # Use simulated data
                final_price = final_lob_data['mid_price']
            except:
                final_price = current_price  # Use last known price
                
            # Close position in portfolio
            close_value = portfolio['position'] * final_price
            portfolio['cash'] += close_value
                
            # Execute with broker if available
            if broker_api:
                try:
                    order_type = 'BUY' if portfolio['position'] < 0 else 'SELL'
                    broker_api.place_order(ticker, order_type, abs(portfolio['position']), final_price)
                except Exception as e:
                    print(f"Error executing final trade with broker: {e}")
                    
            portfolio['position'] = 0
        
        # Calculate performance metrics
        strategy_metrics = strategy.get_performance_metrics()
        
        if portfolio['equity']:
            equity_array = np.array(portfolio['equity'])
            initial_capital = 100000
            final_equity = portfolio['equity'][-1]
            
            # Calculate metrics only if we have sufficient data
            if len(equity_array) > 1:
                returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)  # Annualized
                max_drawdown = np.max(np.maximum.accumulate(equity_array) - equity_array) / np.max(equity_array)
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                
            results = {
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return': (final_equity / initial_capital - 1) * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trade_metrics': strategy_metrics,
                'equity_curve': portfolio['equity'],
                'timestamps': portfolio['timestamps'],
                'trades': portfolio['trades']
            }
            
            print("\nTrading Results:")
            print(f"Initial Capital: ${initial_capital:.2f}")
            print(f"Final Equity: ${final_equity:.2f}")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"Win Rate: {strategy_metrics['win_rate']*100:.2f}%")
            print(f"Total Trades: {strategy_metrics['total_trades']}")
            
            # Plot results if we have data
            if len(portfolio['equity']) > 2:
                plot_backtest_results(results)
                
            return results
        else:
            print("No trading activity to report.")
            return None


#######################
# MAIN FUNCTION
#######################

def main():
    """Main entry point for the TABL trading system"""
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ask if user wants to train a new model or load existing
    if os.path.exists('best_tabl_model.pt') and os.path.exists('norm_params.json'):
        load_model = input("Found existing model. Load it? (Y/n): ").lower() != 'n'
    else:
        load_model = False
    
    # Load or train model
    if load_model:
        print("Loading existing model...")
        
        # Load normalization parameters
        with open('norm_params.json', 'r') as f:
            norm_params = json.load(f)
        
        # Create model
        model = TABL_Network(
            input_dim_first=40,   # Number of features (40 for 10 LOB levels)
            input_dim_second=10,  # Number of time steps
            num_classes=3         # 3 classes: down, stationary, up
        )
        
        # Load weights
        model.load_state_dict(torch.load('best_tabl_model.pt', map_location=device))
        print("Model loaded successfully.")
        
    else:
        print("Training new model...")
        
        # Get LOB data
        data, labels = fetch_lob_data(ticker="SPY", levels=10, intervals=10)
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Class distribution: {np.bincount(labels + 1)}")
        
        # Prepare data
        train_loader, val_loader, test_loader, norm_params = prepare_data(data, labels)
        print(f"Train set: {len(train_loader.dataset)} samples")
        print(f"Validation set: {len(val_loader.dataset)} samples")
        print(f"Test set: {len(test_loader.dataset)} samples")
        
        # Create model
        model = TABL_Network(
            input_dim_first=data.shape[1],   # Number of features (40 for 10 LOB levels)
            input_dim_second=data.shape[2],  # Number of time steps
            num_classes=3                    # 3 classes: down, stationary, up
        )
        print(model)
        
        # Train model
        model, history = train_model(model, train_loader, val_loader, device, epochs=30)
        
        # Plot training history
        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(3, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(3, 1, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(3, 1, 3)
        plt.plot(history['train_f1'], label='Train F1 Score')
        plt.plot(history['val_f1'], label='Validation F1 Score')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)
        
        # Print metrics
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        
        print("\nPer-class metrics:")
        for i, label in enumerate(['Down', 'Stationary', 'Up']):
            print(f"{label}: Precision={metrics['class_precision'][i]:.4f}, Recall={metrics['class_recall'][i]:.4f}, F1={metrics['class_f1'][i]:.4f}")
    
    # Select mode
    print("\nSelect mode:")
    print("1. Backtest strategy")
    print("2. Live trading (simulated)")
    print("3. Exit")
    
    mode = input("Enter choice (1-3): ")
    
    if mode == "1":
        # Backtest mode
        ticker = input("Enter ticker symbol (default: SPY): ") or "SPY"
        
        # Generate backtest data
        backtest_data = generate_backtest_data(ticker=ticker, n_samples=1000)
        
        # Run backtest
        results = backtest_strategy(model, device, norm_params, backtest_data)
        
        # Plot results
        plot_backtest_results(results)
        
    elif mode == "2":
        # Live trading mode
        ticker = input("Enter ticker symbol (default: SPY): ") or "SPY"
        duration = int(input("Enter trading duration in minutes (default: 10): ") or "10")
        
        # Run live trading (simulated)
        run_live_trading(model, device, norm_params, ticker, run_duration=duration*60)
        
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()