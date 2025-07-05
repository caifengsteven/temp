import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters for simulation
n_assets = 10  # Number of assets
n_days = 100  # Number of days
steps_per_day = 14  # Number of 30-minute intervals per day (6.5 trading hours)
n_steps = n_days * steps_per_day  # Total number of steps
n_lags = 10  # Number of lags to include in the model
vol_of_vol_scale = 0.2  # Scale parameter for volatility of volatility
corr_scale = 0.5  # Scale parameter for correlations
lookahead = 1  # Number of steps to forecast ahead

class SimulatedAssetData:
    """
    Class to simulate asset price, volatility, and volatility of volatility data
    with correlations between assets.
    """
    def __init__(self, n_assets, n_steps, steps_per_day, vol_of_vol_scale, corr_scale):
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.steps_per_day = steps_per_day
        self.vol_of_vol_scale = vol_of_vol_scale
        self.corr_scale = corr_scale

    def generate_correlation_matrix(self):
        """Generate a random correlation matrix for assets"""
        # Start with a random matrix
        A = np.random.randn(self.n_assets, self.n_assets)
        # Make it symmetric
        A = (A + A.T) / 2
        # Add a large diagonal to ensure positive definiteness
        A = A + self.n_assets * np.eye(self.n_assets)
        # Convert to correlation matrix
        D = np.sqrt(np.diag(1 / np.diag(A)))
        corr_matrix = D @ A @ D
        # Scale off-diagonal elements
        np.fill_diagonal(corr_matrix, 1)
        off_diag_mask = ~np.eye(self.n_assets, dtype=bool)
        corr_matrix[off_diag_mask] *= self.corr_scale
        return corr_matrix

    def generate_vol_of_vol_matrix(self):
        """Generate a volatility of volatility matrix"""
        # Generate a base volatility of volatility for each asset
        base_vol_of_vol = np.abs(np.random.randn(self.n_assets)) * self.vol_of_vol_scale
        # Generate a random correlation structure for vol of vol
        vol_corr = self.generate_correlation_matrix()
        # Create a covariance matrix for vol of vol
        vol_of_vol_matrix = np.outer(base_vol_of_vol, base_vol_of_vol) * vol_corr
        return vol_of_vol_matrix

    def simulate_volatility_paths(self):
        """
        Simulate correlated volatility paths for assets with
        stochastic volatility of volatility
        """
        # Initialize volatility paths
        vols = np.zeros((self.n_steps, self.n_assets))
        vols[0] = np.abs(np.random.randn(self.n_assets)) * 0.01 + 0.02  # Initial volatility

        # Generate asset correlation matrix
        asset_corr = self.generate_correlation_matrix()
        
        # Generate volatility of volatility matrix
        vol_of_vol_matrix = self.generate_vol_of_vol_matrix()
        
        # Simulate volatility paths with mean reversion
        mean_reversion = 0.98  # Mean reversion parameter
        long_term_mean = 0.02  # Long-term mean volatility
        
        for t in range(1, self.n_steps):
            # Check if it's a new trading day
            new_day = (t % self.steps_per_day == 0)
            
            # Stronger mean reversion at the start of a new day
            if new_day:
                day_mean_reversion = 0.8
            else:
                day_mean_reversion = mean_reversion
            
            # Generate correlated innovations for volatility of volatility
            vol_innovations = np.random.multivariate_normal(
                np.zeros(self.n_assets), 
                vol_of_vol_matrix
            )
            
            # Update volatilities with mean reversion and stochastic vol of vol
            vols[t] = day_mean_reversion * (vols[t-1] - long_term_mean) + long_term_mean + vol_innovations
            vols[t] = np.maximum(vols[t], 0.001)  # Ensure volatility is positive
        
        return vols
    
    def compute_spot_volatility(self, vols):
        """Compute spot volatility from volatility paths"""
        # In a real setting, this would estimate spot volatility from price paths
        # using Fourier methods as described in the paper.
        # For our simulation, we'll use the generated volatility paths directly.
        return vols
    
    def compute_spot_covolatility(self, vols):
        """Compute spot covolatility between assets"""
        # Initialize covolatility tensor
        n_pairs = (self.n_assets * (self.n_assets - 1)) // 2
        covols = np.zeros((self.n_steps, n_pairs))
        
        # Generate asset correlation matrix
        asset_corr = self.generate_correlation_matrix()
        
        # For each time step
        for t in range(self.n_steps):
            pair_idx = 0
            for i in range(self.n_assets):
                for j in range(i+1, self.n_assets):
                    # Covolatility is correlation times product of volatilities
                    covols[t, pair_idx] = asset_corr[i, j] * vols[t, i] * vols[t, j]
                    pair_idx += 1
        
        return covols
    
    def compute_volatility_of_volatility(self, vols):
        """Compute volatility of volatility for each asset"""
        # Initialize vol of vol array
        vol_of_vols = np.zeros((self.n_steps, self.n_assets))
        
        # Window size for rolling estimation of vol of vol
        window = min(self.steps_per_day, 5)
        
        # For each asset
        for i in range(self.n_assets):
            # For each time step (after enough history)
            for t in range(window, self.n_steps):
                # Compute standard deviation of volatility in the rolling window
                vol_of_vols[t, i] = np.std(vols[t-window:t, i])
        
        return vol_of_vols
    
    def compute_covolatility_of_volatility(self, vols):
        """Compute covolatility of volatility between assets"""
        # Initialize covol of vol tensor
        n_pairs = (self.n_assets * (self.n_assets - 1)) // 2
        covol_of_vols = np.zeros((self.n_steps, n_pairs))
        
        # Window size for rolling estimation
        window = min(self.steps_per_day, 5)
        
        # For each time step (after enough history)
        for t in range(window, self.n_steps):
            pair_idx = 0
            for i in range(self.n_assets):
                for j in range(i+1, self.n_assets):
                    # Compute covariance of volatilities in the rolling window
                    covol_of_vols[t, pair_idx] = np.cov(
                        vols[t-window:t, i], 
                        vols[t-window:t, j]
                    )[0, 1]
                    pair_idx += 1
        
        return covol_of_vols
    
    def simulate_data(self):
        """Simulate all data needed for the SpotV2Net model"""
        # Simulate volatility paths
        vols = self.simulate_volatility_paths()
        
        # Compute spot volatility (diagonal of covariance matrix)
        spot_vols = self.compute_spot_volatility(vols)
        
        # Compute spot covolatility (off-diagonal of covariance matrix)
        spot_covols = self.compute_spot_covolatility(vols)
        
        # Compute volatility of volatility
        vol_of_vols = self.compute_volatility_of_volatility(vols)
        
        # Compute covolatility of volatility
        covol_of_vols = self.compute_covolatility_of_volatility(vols)
        
        return {
            'spot_vols': spot_vols,
            'spot_covols': spot_covols,
            'vol_of_vols': vol_of_vols,
            'covol_of_vols': covol_of_vols
        }

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer implementation as described in the paper.
    """
    def __init__(self, in_features, out_features, n_heads, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.alpha = alpha  # LeakyReLU negative slope
        self.concat = concat
        
        # Weight matrices
        self.W = nn.Parameter(torch.zeros(n_heads, in_features, out_features))
        # Edge feature transformation
        self.U = nn.Parameter(torch.zeros(n_heads, out_features))
        # Attention weights
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features + 1))
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h, edge_features):
        """
        Forward pass of GAT layer.
        
        Args:
            h: Node features [batch_size, n_nodes, in_features]
            edge_features: Edge features [batch_size, n_nodes, n_nodes, edge_dim]
            
        Returns:
            Node features after attention [batch_size, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = h.size()
        
        # Apply weight matrix to node features for each head
        # [batch_size, n_heads, n_nodes, out_features]
        Wh = torch.matmul(h.unsqueeze(1), self.W)
        
        # Prepare for attention computation
        # Repeat for each node as source and target
        # [batch_size, n_heads, n_nodes, n_nodes, out_features]
        Wh_i = Wh.unsqueeze(3).repeat(1, 1, 1, n_nodes, 1)
        Wh_j = Wh.unsqueeze(2).repeat(1, 1, n_nodes, 1, 1)
        
        # Transform edge features
        # [batch_size, n_nodes, n_nodes, n_heads]
        edge_transform = torch.matmul(edge_features, self.U.t())
        
        # Concatenate for attention computation
        # [batch_size, n_heads, n_nodes, n_nodes, 2*out_features + 1]
        attn_input = torch.cat([Wh_i, Wh_j, edge_transform.unsqueeze(1)], dim=-1)
        
        # Compute attention coefficients
        # [batch_size, n_heads, n_nodes, n_nodes]
        e = F.leaky_relu(torch.sum(attn_input * self.a.view(1, self.n_heads, 1, 1, -1), dim=-1), self.alpha)
        
        # Apply softmax to get attention weights
        # [batch_size, n_heads, n_nodes, n_nodes]
        attention = F.softmax(e, dim=3)
        attention = self.dropout(attention)
        
        # Apply attention to get new node features
        # [batch_size, n_heads, n_nodes, out_features]
        h_prime = torch.matmul(attention, Wh)
        
        # Combine heads
        if self.concat:
            # Concatenate heads
            # [batch_size, n_nodes, n_heads * out_features]
            h_prime = h_prime.transpose(1, 2).contiguous().view(batch_size, n_nodes, -1)
        else:
            # Average heads
            # [batch_size, n_nodes, out_features]
            h_prime = h_prime.mean(dim=1)
        
        return h_prime

class SpotV2Net(nn.Module):
    """
    Implementation of SpotV2Net for multivariate intraday spot volatility forecasting.
    """
    def __init__(self, n_assets, n_lags, hidden_dim, n_heads, dropout=0.1, alpha=0.2):
        super(SpotV2Net, self).__init__()
        self.n_assets = n_assets
        self.n_lags = n_lags
        
        # Calculate dimensions
        self.node_feature_dim = n_lags * (1 + (n_assets - 1))  # Volatility + covols with other assets
        self.edge_feature_dim = n_lags * 3  # Vol of vol for 2 assets + covol of vol
        
        # GAT layers
        self.gat1 = GATLayer(
            in_features=self.node_feature_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True
        )
        
        self.gat2 = GATLayer(
            in_features=hidden_dim * n_heads,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            concat=False
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, lookahead)
        
    def forward(self, node_features, edge_features):
        """
        Forward pass of SpotV2Net.
        
        Args:
            node_features: Node features [batch_size, n_nodes, node_feature_dim]
            edge_features: Edge features [batch_size, n_nodes, n_nodes, edge_feature_dim]
            
        Returns:
            Volatility predictions [batch_size, n_nodes, lookahead]
        """
        # First GAT layer
        x = self.gat1(node_features, edge_features)
        
        # Apply ReLU
        x = F.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_features)
        
        # Apply ReLU
        x = F.relu(x)
        
        # Output layer
        output = self.output(x)
        
        return output

class SpotV2NetWithoutEdges(nn.Module):
    """
    Version of SpotV2Net without edge features for comparison.
    """
    def __init__(self, n_assets, n_lags, hidden_dim, n_heads, dropout=0.1, alpha=0.2):
        super(SpotV2NetWithoutEdges, self).__init__()
        self.n_assets = n_assets
        self.n_lags = n_lags
        
        # Calculate dimensions
        self.node_feature_dim = n_lags * (1 + (n_assets - 1))  # Volatility + covols with other assets
        
        # Create dummy edge features (ones)
        self.dummy_edge_features = nn.Parameter(torch.ones(1, n_assets, n_assets, 1), requires_grad=False)
        
        # GAT layers
        self.gat1 = GATLayer(
            in_features=self.node_feature_dim,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True
        )
        
        self.gat2 = GATLayer(
            in_features=hidden_dim * n_heads,
            out_features=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            concat=False
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, lookahead)
        
    def forward(self, node_features, _):
        """
        Forward pass of SpotV2Net without edge features.
        
        Args:
            node_features: Node features [batch_size, n_nodes, node_feature_dim]
            _: Ignored edge features
            
        Returns:
            Volatility predictions [batch_size, n_nodes, lookahead]
        """
        batch_size = node_features.size(0)
        dummy_edges = self.dummy_edge_features.expand(batch_size, -1, -1, -1)
        
        # First GAT layer
        x = self.gat1(node_features, dummy_edges)
        
        # Apply ReLU
        x = F.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, dummy_edges)
        
        # Apply ReLU
        x = F.relu(x)
        
        # Output layer
        output = self.output(x)
        
        return output

class HARSpotModel:
    """
    Heterogeneous Auto-Regressive (HAR) model for spot volatility forecasting.
    This is a panel version of the HAR model that includes cross-asset effects.
    """
    def __init__(self, n_assets):
        self.n_assets = n_assets
        self.params = None
        
    def fit(self, X, y):
        """
        Fit the HAR-Spot model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values [n_samples]
        """
        # Add constant term
        X_with_const = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Solve using OLS
        self.params = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
    def predict(self, X):
        """
        Make predictions with the HAR-Spot model.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predictions [n_samples]
        """
        # Add constant term
        X_with_const = np.hstack([np.ones((X.shape[0], 1)), X])
        
        return X_with_const @ self.params
    
def prepare_har_features(spot_vols, n_assets, n_lags):
    """
    Prepare features for the HAR-Spot model.
    
    Args:
        spot_vols: Spot volatility time series [n_steps, n_assets]
        n_assets: Number of assets
        n_lags: Number of lags to consider
        
    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Target values [n_samples]
    """
    # We need at least n_lags + lookahead steps of data
    n_steps = spot_vols.shape[0]
    n_samples = n_steps - n_lags - lookahead
    
    # Features include:
    # 1. Current volatility for each asset
    # 2. Average volatility of past 7 steps (half-day) for each asset
    # 3. Average volatility of past 14 steps (day) for each asset
    # Repeated for all assets to capture cross-asset effects
    n_features = n_assets * 3
    
    X = np.zeros((n_samples * n_assets, n_features))
    y = np.zeros(n_samples * n_assets)
    
    for i in range(n_samples):
        for j in range(n_assets):
            # Index in the feature matrix
            idx = i * n_assets + j
            
            # Target: volatility lookahead steps ahead
            y[idx] = spot_vols[i + n_lags + lookahead, j]
            
            # Current volatility features for all assets
            X[idx, 0:n_assets] = spot_vols[i + n_lags, :]
            
            # Half-day average volatility features for all assets
            half_day_window = min(7, n_lags)
            X[idx, n_assets:2*n_assets] = np.mean(
                spot_vols[i + n_lags - half_day_window + 1:i + n_lags + 1, :], 
                axis=0
            )
            
            # Full-day average volatility features for all assets
            full_day_window = min(14, n_lags)
            X[idx, 2*n_assets:3*n_assets] = np.mean(
                spot_vols[i + n_lags - full_day_window + 1:i + n_lags + 1, :], 
                axis=0
            )
    
    return X, y

def prepare_data_for_spotv2net(data, n_assets, n_lags, split_idx):
    """
    Prepare data for the SpotV2Net model.
    
    Args:
        data: Dictionary containing simulated data
        n_assets: Number of assets
        n_lags: Number of lags to consider
        split_idx: Index to split data into train and test sets
        
    Returns:
        Dictionary containing prepared data for training and testing
    """
    spot_vols = data['spot_vols']
    spot_covols = data['spot_covols']
    vol_of_vols = data['vol_of_vols']
    covol_of_vols = data['covol_of_vols']
    
    n_steps = spot_vols.shape[0]
    n_samples = n_steps - n_lags - lookahead
    
    # Initialize tensors for node and edge features
    node_features = np.zeros((n_samples, n_assets, n_lags * (1 + (n_assets - 1))))
    edge_features = np.zeros((n_samples, n_assets, n_assets, n_lags * 3))
    targets = np.zeros((n_samples, n_assets, lookahead))
    
    # Fill node and edge features
    for i in range(n_samples):
        # For each asset (node)
        for j in range(n_assets):
            # Node features: own volatility lags
            node_features[i, j, :n_lags] = spot_vols[i:i+n_lags, j][::-1]
            
            # Node features: covolatility lags with other assets
            covolStart = n_lags
            for k in range(n_assets):
                if k == j:
                    continue
                
                # Find the covolatility index
                if j < k:
                    covol_idx = (j * (2 * n_assets - j - 1)) // 2 + (k - j - 1)
                else:
                    covol_idx = (k * (2 * n_assets - k - 1)) // 2 + (j - k - 1)
                
                node_features[i, j, covolStart:covolStart+n_lags] = spot_covols[i:i+n_lags, covol_idx][::-1]
                covolStart += n_lags
            
            # Target: future volatility
            targets[i, j, :] = spot_vols[i+n_lags:i+n_lags+lookahead, j]
        
        # For each edge
        for j in range(n_assets):
            for k in range(n_assets):
                if j == k:
                    # Self-loops: use own vol of vol
                    edge_features[i, j, k, :n_lags] = vol_of_vols[i:i+n_lags, j][::-1]
                    edge_features[i, j, k, n_lags:2*n_lags] = vol_of_vols[i:i+n_lags, j][::-1]
                    edge_features[i, j, k, 2*n_lags:3*n_lags] = vol_of_vols[i:i+n_lags, j][::-1]
                else:
                    # Regular edges: vol of vol for both nodes and their covol of vol
                    edge_features[i, j, k, :n_lags] = vol_of_vols[i:i+n_lags, j][::-1]
                    edge_features[i, j, k, n_lags:2*n_lags] = vol_of_vols[i:i+n_lags, k][::-1]
                    
                    # Find the covol of vol index
                    if j < k:
                        covol_idx = (j * (2 * n_assets - j - 1)) // 2 + (k - j - 1)
                    else:
                        covol_idx = (k * (2 * n_assets - k - 1)) // 2 + (j - k - 1)
                    
                    edge_features[i, j, k, 2*n_lags:3*n_lags] = covol_of_vols[i:i+n_lags, covol_idx][::-1]
    
    # Split into train and test sets
    train_node_features = node_features[:split_idx]
    train_edge_features = edge_features[:split_idx]
    train_targets = targets[:split_idx]
    
    test_node_features = node_features[split_idx:]
    test_edge_features = edge_features[split_idx:]
    test_targets = targets[split_idx:]
    
    # Convert to PyTorch tensors
    train_node_features = torch.FloatTensor(train_node_features)
    train_edge_features = torch.FloatTensor(train_edge_features)
    train_targets = torch.FloatTensor(train_targets)
    
    test_node_features = torch.FloatTensor(test_node_features)
    test_edge_features = torch.FloatTensor(test_edge_features)
    test_targets = torch.FloatTensor(test_targets)
    
    return {
        'train': {
            'node_features': train_node_features,
            'edge_features': train_edge_features,
            'targets': train_targets
        },
        'test': {
            'node_features': test_node_features,
            'edge_features': test_edge_features,
            'targets': test_targets
        }
    }

def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def compute_qlike(y_true, y_pred):
    """Compute QLIKE loss"""
    ratio = y_true / y_pred
    return np.mean(ratio - np.log(ratio) - 1)

def train_spotv2net(model, data, n_epochs, batch_size, lr):
    """
    Train the SpotV2Net model.
    
    Args:
        model: SpotV2Net model
        data: Dictionary containing prepared data
        n_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Unpack data
    train_node_features = data['train']['node_features']
    train_edge_features = data['train']['edge_features']
    train_targets = data['train']['targets']
    
    # Create DataLoader
    train_dataset = TensorDataset(train_node_features, train_edge_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': []
    }
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_node_features, batch_edge_features, batch_targets in train_loader:
            # Forward pass
            outputs = model(batch_node_features, batch_edge_features)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss for this epoch
        epoch_loss /= len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}')
    
    return model, history

def evaluate_models(data, n_assets, n_lags, split_idx):
    """
    Evaluate different models for spot volatility forecasting.
    
    Args:
        data: Dictionary containing simulated data
        n_assets: Number of assets
        n_lags: Number of lags to consider
        split_idx: Index to split data into train and test sets
        
    Returns:
        Dictionary containing evaluation results
    """
    # Prepare data for SpotV2Net
    prepared_data = prepare_data_for_spotv2net(data, n_assets, n_lags, split_idx)
    
    # Initialize models
    spotv2net = SpotV2Net(
        n_assets=n_assets,
        n_lags=n_lags,
        hidden_dim=32,
        n_heads=4,
        dropout=0.1,
        alpha=0.2
    )
    
    spotv2net_no_edges = SpotV2NetWithoutEdges(
        n_assets=n_assets,
        n_lags=n_lags,
        hidden_dim=32,
        n_heads=4,
        dropout=0.1,
        alpha=0.2
    )
    
    # Train SpotV2Net models
    print("Training SpotV2Net...")
    spotv2net, spotv2net_history = train_spotv2net(
        model=spotv2net,
        data=prepared_data,
        n_epochs=50,
        batch_size=32,
        lr=0.001
    )
    
    print("Training SpotV2Net without edge features...")
    spotv2net_no_edges, spotv2net_no_edges_history = train_spotv2net(
        model=spotv2net_no_edges,
        data=prepared_data,
        n_epochs=50,
        batch_size=32,
        lr=0.001
    )
    
    # Prepare data for HAR-Spot model
    spot_vols = data['spot_vols']
    X, y = prepare_har_features(spot_vols, n_assets, n_lags)
    
    # Split into train and test sets
    train_samples = split_idx * n_assets
    X_train, X_test = X[:train_samples], X[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
    
    # Train HAR-Spot model
    print("Training HAR-Spot model...")
    har_spot = HARSpotModel(n_assets)
    har_spot.fit(X_train, y_train)
    
    # Evaluate models
    # SpotV2Net
    spotv2net.eval()
    with torch.no_grad():
        test_node_features = prepared_data['test']['node_features']
        test_edge_features = prepared_data['test']['edge_features']
        test_targets = prepared_data['test']['targets']
        
        spotv2net_preds = spotv2net(test_node_features, test_edge_features).numpy()
        spotv2net_no_edges_preds = spotv2net_no_edges(test_node_features, test_edge_features).numpy()
    
    # HAR-Spot
    har_spot_preds = har_spot.predict(X_test)
    
    # Reshape predictions for comparison
    har_spot_preds_reshaped = har_spot_preds.reshape(-1, n_assets)
    test_targets_reshaped = test_targets.reshape(-1, n_assets).numpy()
    spotv2net_preds_reshaped = spotv2net_preds.reshape(-1, n_assets)
    spotv2net_no_edges_preds_reshaped = spotv2net_no_edges_preds.reshape(-1, n_assets)
    
    # Compute metrics
    har_spot_mse = compute_mse(test_targets_reshaped, har_spot_preds_reshaped)
    spotv2net_mse = compute_mse(test_targets_reshaped, spotv2net_preds_reshaped)
    spotv2net_no_edges_mse = compute_mse(test_targets_reshaped, spotv2net_no_edges_preds_reshaped)
    
    har_spot_qlike = compute_qlike(test_targets_reshaped.flatten(), har_spot_preds_reshaped.flatten())
    spotv2net_qlike = compute_qlike(test_targets_reshaped.flatten(), spotv2net_preds_reshaped.flatten())
    spotv2net_no_edges_qlike = compute_qlike(test_targets_reshaped.flatten(), spotv2net_no_edges_preds_reshaped.flatten())
    
    # Print results
    print("\nTest MSE:")
    print(f"HAR-Spot: {har_spot_mse:.6f}")
    print(f"SpotV2Net without edge features: {spotv2net_no_edges_mse:.6f}")
    print(f"SpotV2Net: {spotv2net_mse:.6f}")
    
    print("\nTest QLIKE:")
    print(f"HAR-Spot: {har_spot_qlike:.6f}")
    print(f"SpotV2Net without edge features: {spotv2net_no_edges_qlike:.6f}")
    print(f"SpotV2Net: {spotv2net_qlike:.6f}")
    
    return {
        'har_spot': {
            'mse': har_spot_mse,
            'qlike': har_spot_qlike,
            'preds': har_spot_preds_reshaped
        },
        'spotv2net_no_edges': {
            'mse': spotv2net_no_edges_mse,
            'qlike': spotv2net_no_edges_qlike,
            'preds': spotv2net_no_edges_preds_reshaped,
            'history': spotv2net_no_edges_history
        },
        'spotv2net': {
            'mse': spotv2net_mse,
            'qlike': spotv2net_qlike,
            'preds': spotv2net_preds_reshaped,
            'history': spotv2net_history
        },
        'true_values': test_targets_reshaped
    }

def visualize_results(data, results):
    """
    Visualize the results of the evaluation.
    
    Args:
        data: Dictionary containing simulated data
        results: Dictionary containing evaluation results
    """
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(results['spotv2net']['history']['train_loss'], label='SpotV2Net')
    plt.plot(results['spotv2net_no_edges']['history']['train_loss'], label='SpotV2Net without edge features')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Plot MSE and QLIKE comparison
    plt.figure(figsize=(12, 6))
    
    models = ['HAR-Spot', 'SpotV2Net without\nedge features', 'SpotV2Net']
    mse_values = [results['har_spot']['mse'], results['spotv2net_no_edges']['mse'], results['spotv2net']['mse']]
    qlike_values = [results['har_spot']['qlike'], results['spotv2net_no_edges']['qlike'], results['spotv2net']['qlike']]
    
    plt.subplot(1, 2, 1)
    plt.bar(models, mse_values, color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('MSE')
    plt.title('MSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.bar(models, qlike_values, color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('QLIKE')
    plt.title('QLIKE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('metrics_comparison.png')
    plt.close()
    
    # Plot prediction examples for a few assets
    assets_to_plot = min(4, n_assets)
    test_steps = results['true_values'].shape[0]
    
    plt.figure(figsize=(15, 10))
    for i in range(assets_to_plot):
        plt.subplot(2, 2, i+1)
        plt.plot(results['true_values'][:, i], label='True', linewidth=2)
        plt.plot(results['har_spot']['preds'][:, i], label='HAR-Spot', linestyle='--')
        plt.plot(results['spotv2net_no_edges']['preds'][:, i], label='SpotV2Net-NE', linestyle='-.')
        plt.plot(results['spotv2net']['preds'][:, i], label='SpotV2Net', linestyle=':')
        plt.xlabel('Time Step')
        plt.ylabel('Spot Volatility')
        plt.title(f'Asset {i+1}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.close()
    
    # Visualize volatility spillover effects
    # Create a correlation matrix based on the covol of vol values
    covol_of_vols = data['covol_of_vols']
    mean_covol_of_vols = np.mean(covol_of_vols, axis=0)
    
    # Reconstruct full matrix from upper triangular form
    spillover_matrix = np.zeros((n_assets, n_assets))
    idx = 0
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            spillover_matrix[i, j] = mean_covol_of_vols[idx]
            spillover_matrix[j, i] = mean_covol_of_vols[idx]
            idx += 1
    
    # Plot spillover matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(spillover_matrix, annot=True, cmap='coolwarm', fmt='.2e')
    plt.title('Volatility Spillover Effects (Co-volatility of Volatility)')
    plt.xlabel('Asset')
    plt.ylabel('Asset')
    plt.savefig('spillover_matrix.png')
    plt.close()
    
    # Create and visualize the graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_assets):
        G.add_node(i)
    
    # Add edges based on spillover matrix
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                # Use absolute value for edge weight since we care about magnitude
                G.add_edge(i, j, weight=abs(spillover_matrix[i, j]))
    
    # Calculate node size based on volatility
    mean_vols = np.mean(data['spot_vols'], axis=0)
    node_sizes = 1000 * mean_vols / np.max(mean_vols)
    
    # Calculate edge width based on spillover strength
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    normalized_weights = [w / max_weight for w in edge_weights]
    
    # Plot the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    edges = nx.draw_networkx_edges(
        G, 
        pos, 
        width=[w * 5 for w in normalized_weights],
        edge_color=edge_weights,
        edge_cmap=plt.cm.coolwarm,
        arrowstyle='-|>',
        arrowsize=10,
        alpha=0.7
    )
    
    # Add edge colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Spillover Strength')
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, {i: f"Asset {i+1}" for i in range(n_assets)})
    
    plt.title('Volatility Spillover Network')
    plt.axis('off')
    plt.savefig('spillover_network.png')
    plt.close()

# Simulate data
print("Simulating data...")
simulator = SimulatedAssetData(
    n_assets=n_assets,
    n_steps=n_steps,
    steps_per_day=steps_per_day,
    vol_of_vol_scale=vol_of_vol_scale,
    corr_scale=corr_scale
)
data = simulator.simulate_data()

# Plot example of simulated data
plt.figure(figsize=(15, 10))

# Plot spot volatility for a few assets
plt.subplot(2, 2, 1)
for i in range(min(5, n_assets)):
    plt.plot(data['spot_vols'][:, i], label=f'Asset {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Spot Volatility')
plt.title('Simulated Spot Volatility')
plt.legend()
plt.grid(True)

# Plot spot co-volatility for a few pairs
plt.subplot(2, 2, 2)
for i in range(min(5, n_assets*(n_assets-1)//2)):
    plt.plot(data['spot_covols'][:, i], label=f'Pair {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Spot Co-volatility')
plt.title('Simulated Spot Co-volatility')
plt.legend()
plt.grid(True)

# Plot volatility of volatility for a few assets
plt.subplot(2, 2, 3)
for i in range(min(5, n_assets)):
    plt.plot(data['vol_of_vols'][:, i], label=f'Asset {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Volatility of Volatility')
plt.title('Simulated Volatility of Volatility')
plt.legend()
plt.grid(True)

# Plot co-volatility of volatility for a few pairs
plt.subplot(2, 2, 4)
for i in range(min(5, n_assets*(n_assets-1)//2)):
    plt.plot(data['covol_of_vols'][:, i], label=f'Pair {i+1}')
plt.xlabel('Time Step')
plt.ylabel('Co-volatility of Volatility')
plt.title('Simulated Co-volatility of Volatility')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('simulated_data.png')
plt.close()

# Set split index for train/test
split_idx = int(0.8 * (n_steps - n_lags - lookahead))

# Evaluate models
results = evaluate_models(data, n_assets, n_lags, split_idx)

# Visualize results
visualize_results(data, results)

print("Evaluation complete! Visualization files saved.")