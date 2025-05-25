import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#######################################
# Generate Simulation Data
#######################################

def generate_simulation_data(T=180, n=200, m=100):
    """
    Generate simulated data with time-varying relationships.
    
    Args:
        T: Number of time periods (months)
        n: Number of observations per time period (stocks)
        m: Number of features per observation
        
    Returns:
        X: Features of shape (T, n, m)
        r: Target values of shape (T, n)
    """
    print("Generating simulation data...")
    
    # Initialize data containers
    X = np.zeros((T, n, m))
    r = np.zeros((T, n))
    
    # Generate feature values - standard normal distribution
    for t in range(T):
        X[t] = np.random.normal(0, 1, (n, m))
    
    # Generate time-varying latent relationships
    v = np.zeros((T, m))
    v[0] = np.random.normal(0, 1, m)  # Initial values
    
    for t in range(1, T):
        # Latent relationship follows a Wiener process
        delta = np.random.normal(0, 1, m)
        v[t] = 0.95 * v[t-1] + 0.05 * delta
    
    # Generate target values with non-linear relationship
    for t in range(T):
        for i in range(n):
            r[t, i] = np.sum(np.tanh(X[t, i] * v[t])) + np.random.normal(0, 1)
    
    print("Data generation complete.")
    return X, r

#######################################
# Model Definitions
#######################################

class FinancialNeuralNet(nn.Module):
    def __init__(self, input_dim, l1_penalty=1e-4):
        super(FinancialNeuralNet, self).__init__()
        
        self.l1_penalty = l1_penalty
        
        # Define the network architecture as in the paper
        self.layer1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.layer3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.output = nn.Linear(8, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First hidden layer
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second hidden layer
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Third hidden layer
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.output(x)
        
        return x
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_penalty * l1_loss

#######################################
# Early Stopping Implementation
#######################################

def early_stopping(model, train_loader, val_loader, optimizer, max_epochs=1000, 
                  patience=5, tolerance=0.001):
    """
    Implementation of early stopping as outlined in Algorithm 1 in the paper.
    
    Returns:
        best_epochs: Number of optimal training epochs
        best_model_state: Model state with best validation performance
    """
    model.train()
    best_model_state = model.state_dict().copy()
    best_loss = float('inf')
    best_epochs = 0
    q = 0  # patience counter
    
    # Calculate initial validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            val_loss += torch.mean((outputs.flatten() - y_batch) ** 2).item() * len(y_batch)
    
    val_loss /= len(val_loader.dataset)
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict().copy()
        best_epochs = 0
    
    for epoch in range(1, max_epochs + 1):
        # Train for one epoch
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Compute loss
            mse_loss = torch.mean((outputs.flatten() - y_batch) ** 2)
            l1_loss = model.get_l1_loss()
            loss = mse_loss + l1_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += torch.mean((outputs.flatten() - y_batch) ** 2).item() * len(y_batch)
        
        val_loss /= len(val_loader.dataset)
        
        # Check if this is the best model so far
        if val_loss < best_loss:
            improvement = best_loss - val_loss
            if improvement >= tolerance:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epochs = epoch
                q = 0  # Reset patience counter
            else:
                q += 1  # Minor improvement, increment patience counter
        else:
            q += 1  # No improvement, increment patience counter
        
        # Check if patience exceeded
        if q >= patience:
            break
    
    return best_epochs, best_model_state

#######################################
# DNN with Expanding Window (Pooled Approach)
#######################################

def train_dnn_expanding(X, r, refit_interval=10, l1_penalty=1e-4, learning_rate=0.001,
                       batch_size=50, patience=5, max_epochs=1000):
    """
    Train a neural network using the expanding window (pooled) approach.
    Model is retrained every refit_interval time periods.
    """
    T, n, m = X.shape
    
    # Initialize containers for predictions and performance metrics
    predictions = np.zeros((T, n))
    r2_metrics = np.zeros(T)
    rank_corr_metrics = np.zeros(T)
    
    # Start training from period 60 (to match OES which needs historical data)
    start_period = 60
    
    print("Training DNN with expanding window...")
    
    for t in tqdm(range(start_period, T)):
        # Determine if model should be retrained
        if t == start_period or t % refit_interval == 0:
            # Expand the training window
            X_expanded = X[:t].reshape(-1, m)
            r_expanded = r[:t].flatten()
            
            # Split into training (90%) and validation (10%)
            num_samples = len(X_expanded)
            indices = np.random.permutation(num_samples)
            split_idx = int(0.9 * num_samples)
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train, y_train = X_expanded[train_indices], r_expanded[train_indices]
            X_val, y_val = X_expanded[val_indices], r_expanded[val_indices]
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Create and train model
            model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Implement early stopping
            best_epochs, best_model_state = early_stopping(
                model, train_loader, val_loader, optimizer,
                max_epochs=max_epochs, patience=patience, tolerance=0.001
            )
            
            # Load best model
            model.load_state_dict(best_model_state)
        
        # Make predictions for current period
        model.eval()
        with torch.no_grad():
            X_current = torch.FloatTensor(X[t]).to(device)
            pred = model(X_current).cpu().numpy().flatten()
        
        predictions[t] = pred
        
        # Calculate performance metrics
        r2_metrics[t] = r2_score(r[t], pred)
        rank_corr_metrics[t] = spearmanr(r[t], pred)[0]
    
    # Calculate pooled R² for the out-of-sample period
    y_true = r[start_period:].flatten()
    y_pred = predictions[start_period:].flatten()
    pooled_r2 = r2_score(y_true, y_pred)
    
    # Calculate mean metrics (excluding NaN values)
    mean_r2 = np.nanmean(r2_metrics[start_period:])
    mean_rank_corr = np.nanmean(rank_corr_metrics[start_period:])
    
    results = {
        'predictions': predictions,
        'r2_metrics': r2_metrics,
        'rank_corr_metrics': rank_corr_metrics,
        'pooled_r2': pooled_r2,
        'mean_r2': mean_r2,
        'mean_rank_corr': mean_rank_corr
    }
    
    return results

#######################################
# Online Early Stopping Implementation
#######################################

def train_oes(X, r, l1_penalty=1e-4, learning_rate=0.001, batch_size=50, 
             patience=5, max_epochs=1000):
    """
    Train a neural network using the Online Early Stopping (OES) approach.
    """
    T, n, m = X.shape
    
    # Initialize containers for predictions and performance metrics
    predictions = np.zeros((T, n))
    r2_metrics = np.zeros(T)
    rank_corr_metrics = np.zeros(T)
    tau_values = np.zeros(T)  # To store optimal number of epochs
    
    # Start from period 60 to match DNN's starting point
    start_period = 60
    
    # Initialize model
    base_model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
    base_model_state = base_model.state_dict().copy()
    
    print("Training with Online Early Stopping...")
    
    for t in tqdm(range(start_period, T)):
        if t == start_period:
            # First iteration, initialize τ to 0
            tau_est = 0
            
            # Train model on period t-1
            model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
            model.load_state_dict(base_model_state)
            
            X_train = torch.FloatTensor(X[t-1]).to(device)
            y_train = torch.FloatTensor(r[t-1]).to(device)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train for a few epochs
            model.train()
            for epoch in range(10):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    mse_loss = torch.mean((outputs.flatten() - y_batch) ** 2)
                    l1_loss = model.get_l1_loss()
                    loss = mse_loss + l1_loss
                    loss.backward()
                    optimizer.step()
            
            # Save weights for t
            theta_t = model.state_dict().copy()
            
        else:
            # Load weights from t-2
            model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
            model.load_state_dict(base_model_state)
            
            # Prepare data for t-2 (training) and t-1 (validation)
            X_train = torch.FloatTensor(X[t-2]).to(device)
            y_train = torch.FloatTensor(r[t-2]).to(device)
            
            X_val = torch.FloatTensor(X[t-1]).to(device)
            y_val = torch.FloatTensor(r[t-1]).to(device)
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Use early stopping to find optimal τ
            tau_opt, best_model_state = early_stopping(
                model, train_loader, val_loader, optimizer,
                max_epochs=max_epochs, patience=patience, tolerance=0.001
            )
            
            # Update τ estimate (moving average)
            if t > start_period + 1:
                tau_values[t-2] = tau_opt
                tau_est = np.mean(tau_values[:t-1][tau_values[:t-1] > 0])
            else:
                tau_est = tau_opt
                tau_values[t-2] = tau_opt
            
            # Save t-1 weights
            theta_t_minus_1 = best_model_state
            
            # Now train on t-1 data for τ_est epochs
            model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
            model.load_state_dict(theta_t_minus_1)
            
            X_train = torch.FloatTensor(X[t-1]).to(device)
            y_train = torch.FloatTensor(r[t-1]).to(device)
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train for estimated optimal steps
            model.train()
            for epoch in range(int(tau_est + 0.5)):  # Round to nearest integer
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    mse_loss = torch.mean((outputs.flatten() - y_batch) ** 2)
                    l1_loss = model.get_l1_loss()
                    loss = mse_loss + l1_loss
                    loss.backward()
                    optimizer.step()
            
            # Save weights for t
            theta_t = model.state_dict().copy()
        
        # Make predictions for period t
        model = FinancialNeuralNet(input_dim=m, l1_penalty=l1_penalty).to(device)
        model.load_state_dict(theta_t)
        model.eval()
        
        with torch.no_grad():
            X_current = torch.FloatTensor(X[t]).to(device)
            pred = model(X_current).cpu().numpy().flatten()
        
        predictions[t] = pred
        
        # Calculate performance metrics
        r2_metrics[t] = r2_score(r[t], pred)
        rank_corr_metrics[t] = spearmanr(r[t], pred)[0]
        
        # Save base weights for next iteration
        base_model_state = theta_t if t == start_period else theta_t_minus_1
    
    # Calculate pooled R² for the out-of-sample period
    y_true = r[start_period:].flatten()
    y_pred = predictions[start_period:].flatten()
    pooled_r2 = r2_score(y_true, y_pred)
    
    # Calculate mean metrics (excluding NaN values)
    mean_r2 = np.nanmean(r2_metrics[start_period:])
    mean_rank_corr = np.nanmean(rank_corr_metrics[start_period:])
    
    results = {
        'predictions': predictions,
        'r2_metrics': r2_metrics,
        'rank_corr_metrics': rank_corr_metrics,
        'tau_values': tau_values,
        'pooled_r2': pooled_r2,
        'mean_r2': mean_r2,
        'mean_rank_corr': mean_rank_corr
    }
    
    return results

#######################################
# Evaluation and Comparison
#######################################

def evaluate_models(dnn_results, oes_results, X, r, start_period=60):
    """
    Evaluate and compare DNN and OES models
    """
    # Compare performance metrics
    print("\n======= Performance Comparison =======")
    print(f"{'Metric':<20} {'DNN':<15} {'OES':<15}")
    print(f"{'-'*50}")
    print(f"{'Pooled R² (%):':<20} {dnn_results['pooled_r2']*100:<15.2f} {oes_results['pooled_r2']*100:<15.2f}")
    print(f"{'Mean R² (%):':<20} {dnn_results['mean_r2']*100:<15.2f} {oes_results['mean_r2']*100:<15.2f}")
    print(f"{'Mean Rank Corr (%):':<20} {dnn_results['mean_rank_corr']*100:<15.2f} {oes_results['mean_rank_corr']*100:<15.2f}")
    
    # Plot R² over time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(start_period, len(dnn_results['r2_metrics'])), 
             dnn_results['r2_metrics'][start_period:], label='DNN')
    plt.plot(range(start_period, len(oes_results['r2_metrics'])), 
             oes_results['r2_metrics'][start_period:], label='OES')
    plt.title('R² Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    # Plot Rank Correlation over time
    plt.subplot(1, 2, 2)
    plt.plot(range(start_period, len(dnn_results['rank_corr_metrics'])), 
             dnn_results['rank_corr_metrics'][start_period:], label='DNN')
    plt.plot(range(start_period, len(oes_results['rank_corr_metrics'])), 
             oes_results['rank_corr_metrics'][start_period:], label='OES')
    plt.title('Rank Correlation Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('Rank Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot OES τ values
    plt.figure(figsize=(10, 4))
    valid_tau = oes_results['tau_values'][oes_results['tau_values'] > 0]
    valid_tau_indices = np.where(oes_results['tau_values'] > 0)[0]
    plt.plot(valid_tau_indices, valid_tau, '-o', markersize=3)
    plt.title('Optimal Number of Training Epochs (τ) Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('τ')
    plt.grid(True)
    plt.show()
    
    # Calculate decile performance
    def get_decile_performance(predictions, actual_returns, start_period):
        decile_returns = np.zeros((10, len(predictions) - start_period))
        
        for t in range(start_period, len(predictions)):
            # Sort stocks into deciles based on predictions
            sorted_indices = np.argsort(predictions[t])
            num_per_decile = len(sorted_indices) // 10
            
            for d in range(10):
                if d < 9:
                    # For deciles 0-8
                    decile_indices = sorted_indices[d * num_per_decile:(d + 1) * num_per_decile]
                else:
                    # For the last decile, include any remaining stocks
                    decile_indices = sorted_indices[d * num_per_decile:]
                
                decile_returns[d, t - start_period] = np.mean(actual_returns[t, decile_indices])
        
        avg_decile_returns = np.mean(decile_returns, axis=1)
        return avg_decile_returns
    
    # Get decile performance
    dnn_decile_returns = get_decile_performance(dnn_results['predictions'], r, start_period)
    oes_decile_returns = get_decile_performance(oes_results['predictions'], r, start_period)
    
    # Plot decile returns
    plt.figure(figsize=(10, 6))
    deciles = np.arange(1, 11)
    
    plt.bar(deciles - 0.2, dnn_decile_returns, width=0.4, label='DNN', color='skyblue')
    plt.bar(deciles + 0.2, oes_decile_returns, width=0.4, label='OES', color='salmon')
    
    plt.title('Average Returns by Decile')
    plt.xlabel('Decile (1 = Lowest Predicted Returns, 10 = Highest)')
    plt.ylabel('Average Return')
    plt.xticks(deciles)
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()
    
    # Print decile returns
    print("\n======= Decile Performance =======")
    print(f"{'Decile':<10} {'DNN Returns':<15} {'OES Returns':<15}")
    print(f"{'-'*40}")
    for i in range(10):
        print(f"{i+1:<10} {dnn_decile_returns[i]:<15.4f} {oes_decile_returns[i]:<15.4f}")
    
    print(f"\nP10-P1 Spread (DNN): {dnn_decile_returns[9] - dnn_decile_returns[0]:.4f}")
    print(f"P10-P1 Spread (OES): {oes_decile_returns[9] - oes_decile_returns[0]:.4f}")

#######################################
# Main Execution
#######################################

def main():
    # Generate simulation data
    X, r = generate_simulation_data(T=180, n=200, m=100)
    
    # Train DNN with expanding window
    start_time = time.time()
    dnn_results = train_dnn_expanding(
        X, r, 
        refit_interval=10,
        l1_penalty=1e-4, 
        learning_rate=0.01,
        batch_size=50
    )
    dnn_time = time.time() - start_time
    print(f"DNN training completed in {dnn_time:.2f} seconds")
    
    # Train with Online Early Stopping
    start_time = time.time()
    oes_results = train_oes(
        X, r,
        l1_penalty=1e-3,  # Higher regularization as per paper
        learning_rate=0.001,  # Lower learning rate as per paper
        batch_size=50
    )
    oes_time = time.time() - start_time
    print(f"OES training completed in {oes_time:.2f} seconds")
    
    # Evaluate and compare models
    evaluate_models(dnn_results, oes_results, X, r)
    
    return dnn_results, oes_results, X, r

if __name__ == "__main__":
    dnn_results, oes_results, X, r = main()