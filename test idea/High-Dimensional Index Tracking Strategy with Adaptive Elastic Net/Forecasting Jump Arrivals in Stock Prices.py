import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LimitOrderBookSimulator:
    """
    Simulates limit order book data for testing jump prediction algorithms
    """
    
    def __init__(self, n_levels=10, n_features=144, n_time_steps=120, volatility=0.01, 
                 price_impact=0.01, mean_reversion=0.05, jump_intensity=0.03):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        n_levels : int
            Number of price levels to simulate in the order book
        n_features : int
            Number of features to generate (based on the paper's LOB features)
        n_time_steps : int
            Number of time steps per sample
        volatility : float
            Volatility parameter for price movement
        price_impact : float
            Impact parameter for order flow on price
        mean_reversion : float
            Mean reversion parameter for price
        jump_intensity : float
            Probability of a jump occurring
        """
        self.n_levels = n_levels
        self.n_features = n_features
        self.n_time_steps = n_time_steps
        self.volatility = volatility
        self.price_impact = price_impact
        self.mean_reversion = mean_reversion
        self.jump_intensity = jump_intensity
        
    def generate_limit_order_book(self, n_samples=1000, with_jumps=True):
        """
        Generate simulated limit order book data
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        with_jumps : bool
            Whether to include jumps in the simulation
            
        Returns:
        --------
        lob_data : numpy.ndarray
            Simulated limit order book data
        labels : numpy.ndarray
            Jump occurrence labels
        """
        # Initialize data arrays
        lob_data = np.zeros((n_samples, self.n_time_steps, self.n_features))
        labels = np.zeros(n_samples)
        
        # Generate base mid-price process
        for i in range(n_samples):
            # Initialize LOB features
            mid_price = 100.0  # Starting mid-price
            prices = np.zeros((self.n_time_steps, 1))
            
            # Price and volume features for each level
            ask_prices = np.zeros((self.n_time_steps, self.n_levels))
            ask_volumes = np.zeros((self.n_time_steps, self.n_levels))
            bid_prices = np.zeros((self.n_time_steps, self.n_levels))
            bid_volumes = np.zeros((self.n_time_steps, self.n_levels))
            
            # Additional features (spreads, derivatives, etc.)
            spreads = np.zeros((self.n_time_steps, self.n_levels))
            mid_prices = np.zeros((self.n_time_steps, 1))
            
            # Order flow (for time-sensitive features)
            order_intensities = np.zeros((self.n_time_steps, 6))  # 6 types of intensities as in the paper
            
            # Initial spreads increase with level
            spread_base = 0.01
            
            # Will there be a jump in this sample?
            jump_occurs = with_jumps and np.random.rand() < self.jump_intensity
            if jump_occurs:
                # Jump will occur at random time after 80% of the sample
                jump_time = np.random.randint(int(self.n_time_steps * 0.8), self.n_time_steps)
                jump_size = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)  # Random jump size
                labels[i] = 1
            
            # Simulate time series
            for t in range(self.n_time_steps):
                # Store current mid price
                prices[t, 0] = mid_price
                mid_prices[t, 0] = mid_price
                
                # Generate order book levels
                for level in range(self.n_levels):
                    # Spreads increase with level
                    current_spread = spread_base * (level + 1)
                    spreads[t, level] = current_spread
                    
                    # Set prices for this level
                    ask_prices[t, level] = mid_price + current_spread
                    bid_prices[t, level] = mid_price - current_spread
                    
                    # Set volumes with some randomness
                    ask_volumes[t, level] = np.random.exponential(100) / (level + 1)
                    bid_volumes[t, level] = np.random.exponential(100) / (level + 1)
                
                # Generate order intensities (simplified)
                for j in range(6):
                    order_intensities[t, j] = np.random.poisson(20)
                
                # Update mid price for next step with mean reversion and random noise
                price_change = (100 - mid_price) * self.mean_reversion + np.random.normal(0, self.volatility)
                
                # Add jump if it's the right time
                if jump_occurs and t == jump_time:
                    price_change += jump_size
                
                mid_price += price_change
            
            # Combine all features
            sample_data = np.zeros((self.n_time_steps, self.n_features))
            
            # First set of features (basic LOB data - prices and volumes at each level)
            feature_idx = 0
            for level in range(self.n_levels):
                sample_data[:, feature_idx] = ask_prices[:, level]
                feature_idx += 1
                sample_data[:, feature_idx] = ask_volumes[:, level]
                feature_idx += 1
                sample_data[:, feature_idx] = bid_prices[:, level]
                feature_idx += 1
                sample_data[:, feature_idx] = bid_volumes[:, level]
                feature_idx += 1
            
            # Time-insensitive features
            # Spreads
            for level in range(self.n_levels):
                sample_data[:, feature_idx] = spreads[:, level]
                feature_idx += 1
            
            # Mid-prices
            sample_data[:, feature_idx] = mid_prices[:, 0]
            feature_idx += 1
            
            # Price differences (approximation)
            for level in range(self.n_levels-1):
                sample_data[:, feature_idx] = ask_prices[:, level+1] - ask_prices[:, level]
                feature_idx += 1
                sample_data[:, feature_idx] = bid_prices[:, level] - bid_prices[:, level+1]
                feature_idx += 1
            
            # Volume and price means
            sample_data[:, feature_idx] = np.mean(ask_prices, axis=1)
            feature_idx += 1
            sample_data[:, feature_idx] = np.mean(bid_prices, axis=1)
            feature_idx += 1
            sample_data[:, feature_idx] = np.mean(ask_volumes, axis=1)
            feature_idx += 1
            sample_data[:, feature_idx] = np.mean(bid_volumes, axis=1)
            feature_idx += 1
            
            # Time-sensitive features
            # Derivatives (approximations)
            for level in range(self.n_levels):
                # Price derivatives
                if t > 0:
                    sample_data[1:, feature_idx] = np.diff(ask_prices[:, level])
                feature_idx += 1
                if t > 0:
                    sample_data[1:, feature_idx] = np.diff(bid_prices[:, level])
                feature_idx += 1
                
                # Volume derivatives
                if t > 0:
                    sample_data[1:, feature_idx] = np.diff(ask_volumes[:, level])
                feature_idx += 1
                if t > 0:
                    sample_data[1:, feature_idx] = np.diff(bid_volumes[:, level])
                feature_idx += 1
            
            # Order intensities
            for j in range(6):
                sample_data[:, feature_idx] = order_intensities[:, j]
                feature_idx += 1
            
            # Fill remaining features with random data if we haven't reached n_features
            while feature_idx < self.n_features:
                sample_data[:, feature_idx] = np.random.randn(self.n_time_steps)
                feature_idx += 1
            
            # Store this sample
            lob_data[i] = sample_data
        
        return lob_data, labels
    
    def detect_jumps(self, prices, window_size=120, significance=0.01):
        """
        Detect jumps using Lee and Mykland (2008) algorithm
        
        Parameters:
        -----------
        prices : numpy.ndarray
            Price time series
        window_size : int
            Window size for the jump test
        significance : float
            Significance level for the jump test
            
        Returns:
        --------
        jumps : numpy.ndarray
            Binary array indicating jump occurrences
        """
        n = len(prices)
        jumps = np.zeros(n)
        
        # Cannot detect jumps at the beginning due to window size requirement
        if n <= window_size:
            return jumps
        
        # Calculate returns
        returns = np.diff(np.log(prices)) 
        
        # Initialize bipower variation with additional protection against zeros
        bipower_variation = np.zeros(n)
        for i in range(window_size + 1, n):
            # Calculate bipower variation using previous window_size observations
            abs_returns = np.abs(returns[i-window_size:i-1])
            product_terms = abs_returns[:-1] * abs_returns[1:]
            bipower_variation[i] = np.mean(product_terms) * (np.pi/2)
            
            # Protection against zero or very small values
            if bipower_variation[i] < 1e-8:
                bipower_variation[i] = np.std(returns[i-window_size:i-1])**2
        
        # Calculate jump statistic
        jump_stat = np.zeros(n)
        for i in range(window_size + 1, n):
            if bipower_variation[i] > 0:
                jump_stat[i] = returns[i-1] / np.sqrt(bipower_variation[i])
        
        # Calculate critical value
        c = np.sqrt(2 * np.log(n))
        beta = (1 / c) * (np.log(np.pi) + np.log(np.log(n))) / (2 * c)
        critical_value = c - beta
        
        # Identify jumps
        for i in range(window_size + 1, n):
            if np.abs(jump_stat[i]) > critical_value:
                jumps[i] = 1
        
        return jumps


class LOBDataset(Dataset):
    """
    PyTorch Dataset for limit order book data
    """
    
    def __init__(self, X, y, normalize=True):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, time_steps, features)
        y : numpy.ndarray
            Target labels
        normalize : bool
            Whether to normalize the data
        """
        self.X = X
        self.y = y
        
        # Normalize each feature for each sample independently
        if normalize:
            self.X = self.normalize_data(self.X)
        
        # Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    
    def normalize_data(self, X):
        """
        Normalize input data as described in the paper
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, time_steps, features)
            
        Returns:
        --------
        X_norm : numpy.ndarray
            Normalized input data
        """
        X_norm = X.copy()
        n_samples, time_steps, n_features = X.shape
        
        for i in range(n_samples):
            for j in range(n_features):
                feature_values = X_norm[i, :, j]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                
                # Avoid division by zero
                if std > 0:
                    X_norm[i, :, j] = (feature_values - mean) / std
                else:
                    X_norm[i, :, j] = 0.0
        
        return X_norm
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch model definitions
class MLP(nn.Module):
    """
    Multi-Layer Perceptron model for jump prediction
    """
    
    def __init__(self, input_shape):
        """
        Initialize MLP model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (time_steps, features)
        """
        super(MLP, self).__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 40)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(40, 40)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(40, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network model for jump prediction
    """
    
    def __init__(self, input_shape):
        """
        Initialize CNN model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (time_steps, features)
        """
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=16, kernel_size=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size of convolutional layers
        def conv_output_size(size, kernel_size, pool_size):
            return ((size - kernel_size + 1) // pool_size)
        
        # Calculate final output size after all conv layers
        conv_output = input_shape[0]
        conv_output = conv_output_size(conv_output, 4, 2)
        conv_output = conv_output_size(conv_output, 3, 2)
        conv_output = conv_output_size(conv_output, 3, 2)
        
        # Ensure we have at least one feature
        conv_output = max(1, conv_output)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * conv_output, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        # Permute input from [batch, time_steps, features] to [batch, features, time_steps]
        x = x.permute(0, 2, 1)
        
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class LSTM(nn.Module):
    """
    Long Short-Term Memory model for jump prediction
    """
    
    def __init__(self, input_shape):
        """
        Initialize LSTM model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (time_steps, features)
        """
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_shape[1],
            hidden_size=40,
            batch_first=True,
            dropout=0.5
        )
        
        self.fc1 = nn.Linear(40, 40)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(40, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # We take only the last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


class FeatureAttention(nn.Module):
    """
    Feature-wise attention mechanism for time series data
    """
    
    def __init__(self, time_steps):
        """
        Initialize feature attention mechanism
        
        Parameters:
        -----------
        time_steps : int
            Number of time steps in the input
        """
        super(FeatureAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(time_steps, 1),
            nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, feature_dim)
            
        Returns:
        --------
        weighted_x : torch.Tensor
            Weighted input tensor of same shape as input
        attention_weights : torch.Tensor
            Attention weights of shape (batch_size, feature_dim)
        """
        # Permute to (batch_size, feature_dim, time_steps)
        x_permuted = x.permute(0, 2, 1)
        
        # Calculate attention scores
        scores = self.attention(x_permuted).squeeze(2)  # (batch_size, feature_dim)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)  # (batch_size, feature_dim)
        
        # Expand weights to match input dimensions
        weights_expanded = weights.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch_size, time_steps, feature_dim)
        
        # Permute back to original order
        weighted_x = x * weights_expanded  # (batch_size, time_steps, feature_dim)
        
        return weighted_x, weights


class CNN_LSTM_Attention(nn.Module):
    """
    CNN-LSTM-Attention model for jump prediction as described in the paper
    """
    
    def __init__(self, input_shape):
        """
        Initialize CNN-LSTM-Attention model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (time_steps, features)
        """
        super(CNN_LSTM_Attention, self).__init__()
        
        self.feature_attention = FeatureAttention(input_shape[0])
        
        # Convolutional layer
        self.conv = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate output size after conv and pool
        conv_output = input_shape[0] - 5 + 1  # after conv
        conv_output = conv_output // 2  # after pool
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=40,
            batch_first=True,
            dropout=0.5
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(40, 40)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(40, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Store dimensions for attention visualization
        self.input_shape = input_shape
        self.attention_weights = None
    
    def forward(self, x):
        """Forward pass"""
        # Feature attention
        x_weighted, self.attention_weights = self.feature_attention(x)
        
        # Permute from [batch, time_steps, features] to [batch, features, time_steps]
        x = x_weighted.permute(0, 2, 1)
        
        # Convolutional layer
        x = self.activation(self.conv(x))
        x = self.pool(x)
        
        # Permute back to [batch, time_steps, features] for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # We take only the last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x
    
    def get_attention_weights(self):
        """Return the last computed attention weights"""
        return self.attention_weights


class JumpPredictor:
    """
    Class for predicting jumps in stock prices using PyTorch models
    """
    
    def __init__(self):
        """Initialize the jump predictor"""
        self.models = {}
        self.history = {}
        self.attention_weights = {}
    
    def train_model(self, model_name, model, train_loader, val_loader, 
                   epochs=50, learning_rate=0.001, pos_weight=None):
        """
        Train a PyTorch model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : torch.nn.Module
            PyTorch model to train
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        pos_weight : float or None
            Positive class weight for imbalanced data
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Move model to device
        model = model.to(device)
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Handle class imbalance with custom weighted loss
                if pos_weight is not None:
                    # Add small epsilon to avoid log(0)
                    eps = 1e-12
                    # Custom weighted BCE loss
                    loss = -pos_weight * targets * torch.log(outputs + eps) - (1 - targets) * torch.log(1 - outputs + eps)
                    loss = loss.mean()
                else:
                    # Standard BCE loss
                    loss = nn.functional.binary_cross_entropy(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Handle class imbalance with custom weighted loss
                    if pos_weight is not None:
                        # Add small epsilon to avoid log(0)
                        eps = 1e-12
                        # Custom weighted BCE loss
                        loss = -pos_weight * targets * torch.log(outputs + eps) - (1 - targets) * torch.log(1 - outputs + eps)
                        loss = loss.mean()
                    else:
                        # Standard BCE loss
                        loss = nn.functional.binary_cross_entropy(outputs, targets)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs} - '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Store model and history
        self.models[model_name] = model
        self.history[model_name] = history
        
        return history
    
    def train_models(self, X_train, y_train, X_val, y_val, batch_size=32, 
                    epochs=50, models_to_train=None, pos_weight=None):
        """
        Train multiple models
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation data
        y_val : numpy.ndarray
            Validation labels
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        models_to_train : list or None
            List of model names to train, or None to train all models
        pos_weight : float or None
            Positive class weight for imbalanced data
            
        Returns:
        --------
        self : JumpPredictor
            Self reference for method chaining
        """
        # Create datasets and dataloaders
        train_dataset = LOBDataset(X_train, y_train)
        val_dataset = LOBDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Get input shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Define the models to train
        if models_to_train is None:
            models_to_train = ['mlp', 'cnn', 'lstm', 'cnn_lstm_attention']
        
        # Train each model
        for model_name in models_to_train:
            print(f"\nTraining {model_name} model...")
            
            # Create model
            if model_name == 'mlp':
                model = MLP(input_shape)
            elif model_name == 'cnn':
                model = CNN(input_shape)
            elif model_name == 'lstm':
                model = LSTM(input_shape)
            elif model_name == 'cnn_lstm_attention':
                model = CNN_LSTM_Attention(input_shape)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Train model
            self.train_model(
                model_name, 
                model, 
                train_loader, 
                val_loader, 
                epochs=epochs, 
                pos_weight=pos_weight
            )
        
        return self
    
    def evaluate_models(self, X_test, y_test, batch_size=32, threshold=0.5):
        """
        Evaluate trained models on test data
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test data
        y_test : numpy.ndarray
            Test labels
        batch_size : int
            Batch size for evaluation
        threshold : float
            Classification threshold
            
        Returns:
        --------
        results : dict
            Dictionary of performance metrics for each model
        """
        # Create dataset and dataloader
        test_dataset = LOBDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name} model...")
            
            # Move model to device
            model = model.to(device)
            model.eval()
            
            # Get predictions
            all_preds = []
            all_targets = []
            attention_weights = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    # Move data to device
                    inputs = inputs.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    preds = (outputs > threshold).float().cpu().numpy()
                    
                    # Store predictions and targets
                    all_preds.extend(preds.flatten())
                    all_targets.extend(targets.numpy().flatten())
                    
                    # Store attention weights if available
                    if model_name == 'cnn_lstm_attention':
                        attention_weights.append(model.get_attention_weights().cpu().numpy())
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Store attention weights
            if model_name == 'cnn_lstm_attention' and len(attention_weights) > 0:
                self.attention_weights[model_name] = np.vstack(attention_weights)
            
            # Calculate metrics
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            kappa = cohen_kappa_score(all_targets, all_preds)
            
            # Store results
            results[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'kappa': kappa
            }
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Cohen's Kappa: {kappa:.4f}")
        
        return results
    
    def plot_training_history(self):
        """
        Plot training history for all trained models
        """
        for model_name, history in self.history.items():
            plt.figure(figsize=(12, 5))
            
            # Plot training & validation loss values
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'])
            plt.plot(history['val_loss'])
            plt.title(f'{model_name} - Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            # Plot training & validation accuracy values
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'])
            plt.plot(history['val_acc'])
            plt.title(f'{model_name} - Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            plt.tight_layout()
            plt.show()
    
    def plot_attention_weights(self, X_samples, y_samples, sample_indices=None, n_samples=3):
        """
        Plot attention weights for sample inputs to visualize feature importance
        
        Parameters:
        -----------
        X_samples : numpy.ndarray
            Input samples to visualize
        y_samples : numpy.ndarray
            Target labels for the samples
        sample_indices : list or None
            Indices of specific samples to visualize, or None to select randomly
        n_samples : int
            Number of samples to visualize if sample_indices is None
        """
        # Check if CNN-LSTM-Attention model is available
        if 'cnn_lstm_attention' not in self.models:
            print("CNN-LSTM-Attention model not available")
            return
        
        # Select samples to visualize
        if sample_indices is None:
            sample_indices = np.random.choice(len(X_samples), size=n_samples, replace=False)
        
        # Create dataset for the selected samples
        sample_dataset = LOBDataset(X_samples[sample_indices], y_samples[sample_indices])
        sample_loader = DataLoader(sample_dataset, batch_size=1)
        
        # Get model
        model = self.models['cnn_lstm_attention'].to(device)
        model.eval()
        
        # For each selected sample
        for i, (inputs, targets) in enumerate(sample_loader):
            # Get sample index
            sample_idx = sample_indices[i]
            
            # Move input to device
            inputs = inputs.to(device)
            
            # Forward pass to get attention weights
            with torch.no_grad():
                outputs = model(inputs)
                attention_weights = model.get_attention_weights().cpu().numpy()[0]
            
            # Get prediction
            pred = (outputs > 0.5).float().cpu().numpy()[0][0]
            true_label = targets.numpy()[0]
            
            # Plot sample heatmap and attention weights
            plt.figure(figsize=(15, 6))
            
            # Plot sample data
            plt.subplot(1, 2, 1)
            sns.heatmap(inputs[0, :, :20].cpu().numpy(), cmap='viridis', 
                        xticklabels=5, yticklabels=10)
            plt.title(f'Sample {sample_idx} - First 20 Features Data\n'
                      f'True: {true_label}, Pred: {pred}')
            plt.xlabel('Feature')
            plt.ylabel('Time Step')
            
            # Plot attention weights
            plt.subplot(1, 2, 2)
            sns.barplot(x=np.arange(20), y=attention_weights[:20])
            plt.title(f'Sample {sample_idx} - First 20 Features Attention Weights')
            plt.xlabel('Feature')
            plt.ylabel('Attention Weight')
            
            plt.tight_layout()
            plt.show()


# Trading Environment and Strategy Classes
class TradingEnvironment:
    """
    Simulates a trading environment for testing strategies based on jump predictions
    """
    
    def __init__(self, prices, commission=0.001, slippage=0.0005, initial_balance=10000.0):
        """
        Initialize the trading environment
        
        Parameters:
        -----------
        prices : numpy.ndarray
            Time series of asset prices
        commission : float
            Trading commission as a fraction of trade value
        slippage : float
            Simulated slippage as a fraction of trade value
        initial_balance : float
            Initial account balance
        """
        self.prices = prices
        self.commission = commission
        self.slippage = slippage
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.returns = []
        self.current_step = 0
    
    def step(self, action):
        """
        Take a trading action
        
        Parameters:
        -----------
        action : int
            Trading action: 1 (buy), 0 (hold), -1 (sell)
            
        Returns:
        --------
        reward : float
            Reward from the action
        done : bool
            Whether the episode is done
        info : dict
            Additional information
        """
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Calculate transaction costs
        transaction_cost = 0.0
        
        # Execute trade based on action
        if action == 1 and self.position <= 0:  # Buy
            # Close short position if exists
            if self.position < 0:
                trade_value = abs(self.position) * current_price
                transaction_cost += trade_value * self.commission
                transaction_cost += trade_value * self.slippage
                self.position = 0
            
            # Calculate how many shares we can buy with current balance
            max_shares = self.balance / (current_price * (1 + self.commission + self.slippage))
            shares_to_buy = max_shares  # Buy with all available balance
            
            # Update position and balance
            trade_value = shares_to_buy * current_price
            transaction_cost += trade_value * self.commission
            transaction_cost += trade_value * self.slippage
            self.position += shares_to_buy
            self.balance -= (trade_value + transaction_cost)
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'action': 'buy',
                'shares': shares_to_buy,
                'value': trade_value,
                'cost': transaction_cost
            })
            
        elif action == -1 and self.position >= 0:  # Sell
            # Close long position if exists
            if self.position > 0:
                trade_value = self.position * current_price
                transaction_cost += trade_value * self.commission
                transaction_cost += trade_value * self.slippage
                
                # Update balance
                self.balance += (trade_value - transaction_cost)
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'price': current_price,
                    'action': 'sell',
                    'shares': self.position,
                    'value': trade_value,
                    'cost': transaction_cost
                })
                
                # Reset position
                self.position = 0
            
            # Short selling (uncomment if desired)
            # max_shares = self.balance / (current_price * (1 + self.commission + self.slippage))
            # shares_to_sell = max_shares
            # self.position -= shares_to_sell
            # trade_value = shares_to_sell * current_price
            # transaction_cost += trade_value * self.commission
            # transaction_cost += trade_value * self.slippage
            # self.balance += (trade_value - transaction_cost)
        
        # Calculate portfolio value (balance + position value)
        portfolio_value = self.balance
        if self.position > 0:
            portfolio_value += self.position * current_price
        elif self.position < 0:
            portfolio_value -= abs(self.position) * current_price
        
        # Calculate return
        if len(self.equity_curve) > 0:
            period_return = (portfolio_value / self.equity_curve[-1]) - 1
        else:
            period_return = 0.0
            
        self.returns.append(period_return)
        self.equity_curve.append(portfolio_value)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        return period_return, done, {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance
        }
    
    def run_strategy(self, actions):
        """
        Run a complete trading strategy
        
        Parameters:
        -----------
        actions : list
            List of trading actions for each time step
            
        Returns:
        --------
        performance : dict
            Dictionary of strategy performance metrics
        """
        self.reset()
        
        total_returns = []
        
        for action in actions:
            reward, done, info = self.step(action)
            total_returns.append(reward)
            
            if done:
                break
        
        # Calculate performance metrics
        total_return = (self.equity_curve[-1] / self.initial_balance) - 1
        sharpe_ratio = np.mean(self.returns) / np.std(self.returns) * np.sqrt(252) if np.std(self.returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        
        performance = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return performance
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve"""
        # Convert equity curve to numpy array
        equity = np.array(self.equity_curve)
        
        # Calculate the running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate the drawdown
        drawdown = (running_max - equity) / running_max
        
        # Return the maximum drawdown
        return np.max(drawdown)
    
    def plot_equity_curve(self):
        """Plot the equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Time Step')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()


class JumpPredictionStrategy:
    """
    Trading strategy based on jump predictions
    """
    
    def __init__(self, model, window_size=120, threshold=0.5, entry_threshold=0.7, exit_threshold=0.3):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        model : torch.nn.Module
            Trained model for jump prediction
        window_size : int
            Size of the input window for prediction
        threshold : float
            Classification threshold
        entry_threshold : float
            Threshold for trade entry (higher confidence)
        exit_threshold : float
            Threshold for trade exit (lower confidence)
        """
        self.model = model
        self.window_size = window_size
        self.threshold = threshold
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def generate_signals(self, X_data, device):
        """
        Generate trading signals based on model predictions
        
        Parameters:
        -----------
        X_data : numpy.ndarray
            Input data for predictions
        device : torch.device
            Device to run the model on
            
        Returns:
        --------
        signals : list
            List of trading signals (1: buy, 0: hold, -1: sell)
        """
        # Create dataset
        dataset = LOBDataset(X_data, np.zeros(len(X_data)))
        loader = DataLoader(dataset, batch_size=64)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(device)
        self.model.eval()
        
        # Generate predictions
        all_preds = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())
        
        # Generate signals based on predictions
        signals = []
        position = 0  # 0: no position, 1: long position
        
        for pred in all_preds:
            if position == 0 and pred > self.entry_threshold:
                # Enter long position
                signals.append(1)
                position = 1
            elif position == 1 and pred < self.exit_threshold:
                # Exit long position
                signals.append(-1)
                position = 0
            else:
                # Hold current position
                signals.append(0)
        
        return signals


def generate_price_series(n_steps=1000, volatility=0.01, drift=0.0001, jump_intensity=0.03, jump_size_mean=0.02):
    """
    Generate a simulated price series with jumps
    
    Parameters:
    -----------
    n_steps : int
        Number of time steps
    volatility : float
        Volatility parameter
    drift : float
        Drift parameter
    jump_intensity : float
        Probability of a jump occurring
    jump_size_mean : float
        Mean size of jumps
        
    Returns:
    --------
    prices : numpy.ndarray
        Simulated price series
    jump_locations : numpy.ndarray
        Binary array indicating jump occurrences
    """
    # Initialize price series
    prices = np.zeros(n_steps)
    prices[0] = 100.0  # Starting price
    
    # Initialize jump locations
    jump_locations = np.zeros(n_steps)
    
    # Generate price series
    for t in range(1, n_steps):
        # Regular price movement (GBM)
        price_change = prices[t-1] * (drift + volatility * np.random.normal())
        
        # Add jump component
        if np.random.rand() < jump_intensity:
            jump_size = np.random.normal(jump_size_mean, jump_size_mean)
            price_change += prices[t-1] * jump_size
            jump_locations[t] = 1
        
        # Update price
        prices[t] = prices[t-1] + price_change
    
    return prices, jump_locations


def test_jump_prediction_strategy(model, X_test, device, n_steps=1000):
    """
    Test the jump prediction strategy
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model for jump prediction
    X_test : numpy.ndarray
        Test data for predictions
    device : torch.device
        Device to run the model on
    n_steps : int
        Number of time steps to simulate
        
    Returns:
    --------
    strategy_performance : dict
        Dictionary of strategy performance metrics
    benchmark_performance : dict
        Dictionary of benchmark performance metrics
    """
    # Generate simulated price series
    prices, jump_locations = generate_price_series(n_steps)
    
    # Initialize strategy
    strategy = JumpPredictionStrategy(model, entry_threshold=0.6, exit_threshold=0.4)
    
    # Generate trading signals
    signals = strategy.generate_signals(X_test[:n_steps], device)
    
    # Initialize trading environment
    env = TradingEnvironment(prices)
    
    # Run strategy
    strategy_performance = env.run_strategy(signals)
    
    # Plot equity curve
    env.plot_equity_curve()
    
    # Run buy-and-hold benchmark
    benchmark_signals = [1] + [0] * (n_steps - 1)  # Buy at start and hold
    env.reset()
    benchmark_performance = env.run_strategy(benchmark_signals)
    
    # Print performance comparison
    print("\nStrategy Performance:")
    print(f"Total Return: {strategy_performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {strategy_performance['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {strategy_performance['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {len(strategy_performance['trades'])}")
    
    print("\nBuy-and-Hold Performance:")
    print(f"Total Return: {benchmark_performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {benchmark_performance['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {benchmark_performance['max_drawdown']*100:.2f}%")
    
    # Plot price series with jumps and trades
    plt.figure(figsize=(14, 7))
    
    # Plot price series
    plt.plot(prices, label='Price', color='blue')
    
    # Mark jumps
    jump_indices = np.where(jump_locations == 1)[0]
    plt.scatter(jump_indices, prices[jump_indices], color='red', marker='*', s=100, label='Jumps')
    
    # Mark trades
    for trade in strategy_performance['trades']:
        if trade['action'] == 'buy':
            plt.scatter(trade['step'], prices[trade['step']], color='green', marker='^', s=100)
        elif trade['action'] == 'sell':
            plt.scatter(trade['step'], prices[trade['step']], color='black', marker='v', s=100)
    
    plt.title('Price Series with Jumps and Trades')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return strategy_performance, benchmark_performance


def evaluate_multiple_strategies(model, X_test, device, n_simulations=5, n_steps=1000):
    """
    Evaluate strategies across multiple price simulations
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model for jump prediction
    X_test : numpy.ndarray
        Test data for predictions
    device : torch.device
        Device to run the model on
    n_simulations : int
        Number of simulations to run
    n_steps : int
        Number of time steps in each simulation
        
    Returns:
    --------
    results : dict
        Dictionary of aggregated performance metrics
    """
    # Initialize results
    strategy_returns = []
    strategy_sharpes = []
    strategy_drawdowns = []
    strategy_trades = []
    
    benchmark_returns = []
    benchmark_sharpes = []
    benchmark_drawdowns = []
    
    # Run simulations
    for i in range(n_simulations):
        print(f"\nSimulation {i+1}/{n_simulations}")
        
        # Test strategy
        strategy_perf, benchmark_perf = test_jump_prediction_strategy(model, X_test, device, n_steps)
        
        # Collect results
        strategy_returns.append(strategy_perf['total_return'])
        strategy_sharpes.append(strategy_perf['sharpe_ratio'])
        strategy_drawdowns.append(strategy_perf['max_drawdown'])
        strategy_trades.append(len(strategy_perf['trades']))
        
        benchmark_returns.append(benchmark_perf['total_return'])
        benchmark_sharpes.append(benchmark_perf['sharpe_ratio'])
        benchmark_drawdowns.append(benchmark_perf['max_drawdown'])
    
    # Compile results
    results = {
        'strategy': {
            'returns': {
                'mean': np.mean(strategy_returns),
                'std': np.std(strategy_returns),
                'min': np.min(strategy_returns),
                'max': np.max(strategy_returns)
            },
            'sharpe': {
                'mean': np.mean(strategy_sharpes),
                'std': np.std(strategy_sharpes),
                'min': np.min(strategy_sharpes),
                'max': np.max(strategy_sharpes)
            },
            'drawdown': {
                'mean': np.mean(strategy_drawdowns),
                'std': np.std(strategy_drawdowns),
                'min': np.min(strategy_drawdowns),
                'max': np.max(strategy_drawdowns)
            },
            'trades': {
                'mean': np.mean(strategy_trades),
                'std': np.std(strategy_trades),
                'min': np.min(strategy_trades),
                'max': np.max(strategy_trades)
            }
        },
        'benchmark': {
            'returns': {
                'mean': np.mean(benchmark_returns),
                'std': np.std(benchmark_returns),
                'min': np.min(benchmark_returns),
                'max': np.max(benchmark_returns)
            },
            'sharpe': {
                'mean': np.mean(benchmark_sharpes),
                'std': np.std(benchmark_sharpes),
                'min': np.min(benchmark_sharpes),
                'max': np.max(benchmark_sharpes)
            },
            'drawdown': {
                'mean': np.mean(benchmark_drawdowns),
                'std': np.std(benchmark_drawdowns),
                'min': np.min(benchmark_drawdowns),
                'max': np.max(benchmark_drawdowns)
            }
        }
    }
    
    # Print summary
    print("\n--- Strategy Performance Summary ---")
    print(f"Average Return: {results['strategy']['returns']['mean']*100:.2f}% (±{results['strategy']['returns']['std']*100:.2f}%)")
    print(f"Average Sharpe: {results['strategy']['sharpe']['mean']:.4f} (±{results['strategy']['sharpe']['std']:.4f})")
    print(f"Average Max Drawdown: {results['strategy']['drawdown']['mean']*100:.2f}% (±{results['strategy']['drawdown']['std']*100:.2f}%)")
    print(f"Average Trades: {results['strategy']['trades']['mean']:.1f} (±{results['strategy']['trades']['std']:.1f})")
    
    print("\n--- Buy-and-Hold Performance Summary ---")
    print(f"Average Return: {results['benchmark']['returns']['mean']*100:.2f}% (±{results['benchmark']['returns']['std']*100:.2f}%)")
    print(f"Average Sharpe: {results['benchmark']['sharpe']['mean']:.4f} (±{results['benchmark']['sharpe']['std']:.4f})")
    print(f"Average Max Drawdown: {results['benchmark']['drawdown']['mean']*100:.2f}% (±{results['benchmark']['drawdown']['std']*100:.2f}%)")
    
    return results


def run_trading_strategy_test(predictor, X_test, y_test):
    """
    Run tests of the trading strategy using the trained model
    
    Parameters:
    -----------
    predictor : JumpPredictor
        Trained jump predictor
    X_test : numpy.ndarray
        Test data
    y_test : numpy.ndarray
        Test labels
        
    Returns:
    --------
    results : dict
        Dictionary of performance results
    """
    # Check if CNN-LSTM-Attention model is available
    if 'cnn_lstm_attention' not in predictor.models:
        print("CNN-LSTM-Attention model not available for strategy testing")
        return None
    
    # Get the model
    model = predictor.models['cnn_lstm_attention']
    
    # Evaluate strategy across multiple simulations
    results = evaluate_multiple_strategies(model, X_test, device, n_simulations=3, n_steps=750)
    
    return results


# Function to compare model performance
def compare_model_performance(results, metric='f1'):
    """
    Compare and visualize model performance
    
    Parameters:
    -----------
    results : dict
        Dictionary of performance metrics for each model
    metric : str
        Metric to compare ('precision', 'recall', 'f1', or 'kappa')
        
    Returns:
    --------
    best_model : str
        Name of the best performing model
    """
    model_names = list(results.keys())
    metric_values = [results[model][metric] for model in model_names]
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison by {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return best model name
    best_idx = np.argmax(metric_values)
    return model_names[best_idx]


# Main execution code
def run_simulation(n_samples=5000, train_ratio=0.7, val_ratio=0.15, epochs=20, batch_size=32, test_trading=True):
    """
    Run a complete simulation of the jump prediction task
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    train_ratio : float
        Ratio of data for training
    val_ratio : float
        Ratio of data for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    test_trading : bool
        Whether to test the trading strategy
        
    Returns:
    --------
    results : dict
        Dictionary of performance metrics for each model
    predictor : JumpPredictor
        Trained jump predictor
    """
    # Initialize simulator
    simulator = LimitOrderBookSimulator()
    
    # Generate data
    print("Generating simulated limit order book data...")
    X, y = simulator.generate_limit_order_book(n_samples=n_samples)
    
    # Data split
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Get class distribution information
    print(f"Training set: {np.sum(y_train)} jumps out of {len(y_train)} samples ({np.mean(y_train)*100:.2f}%)")
    print(f"Validation set: {np.sum(y_val)} jumps out of {len(y_val)} samples ({np.mean(y_val)*100:.2f}%)")
    print(f"Test set: {np.sum(y_test)} jumps out of {len(y_test)} samples ({np.mean(y_test)*100:.2f}%)")
    
    # Calculate positive class weight to handle imbalance - inverse of class frequency
    pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Initialize predictor
    predictor = JumpPredictor()
    
    # Train models
    predictor.train_models(
        X_train, y_train, 
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs,
        pos_weight=pos_weight
    )
    
    # Plot training history
    predictor.plot_training_history()
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test, batch_size=batch_size)
    
    # Compare model performance
    best_model = compare_model_performance(results, metric='f1')
    print(f"Best model by F1 score: {best_model}")
    
    # Visualize attention weights for the CNN-LSTM-Attention model
    if 'cnn_lstm_attention' in predictor.models:
        # Find indices of correctly predicted jumps
        test_dataset = LOBDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        all_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = predictor.models['cnn_lstm_attention'](inputs)
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds.flatten())
        
        all_preds = np.array(all_preds)
        correct_jump_indices = np.where((y_test == 1) & (all_preds == 1))[0]
        
        if len(correct_jump_indices) > 0:
            print("Visualizing attention weights for correctly predicted jumps...")
            predictor.plot_attention_weights(X_test, y_test, 
                                           sample_indices=correct_jump_indices[:2], 
                                           n_samples=2)
    
    # Output table of all metrics for all models
    metrics_df = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1 Score', "Cohen's Kappa"])
    
    for i, (model_name, metrics) in enumerate(results.items()):
        metrics_df.loc[i] = [
            model_name,
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['kappa']
        ]
    
    print("\nModel Performance Comparison:")
    print(metrics_df.to_string(index=False))
    
    # Test trading strategy
    if test_trading and 'cnn_lstm_attention' in predictor.models:
        print("\n--- Testing Trading Strategy ---")
        trading_results = run_trading_strategy_test(predictor, X_test, y_test)
    
    return results, predictor


# Run the complete simulation
if __name__ == "__main__":
    results, predictor = run_simulation(n_samples=5000, epochs=20, batch_size=64, test_trading=True)