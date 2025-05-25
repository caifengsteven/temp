import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxopt
from cvxopt import matrix, solvers
import datetime as dt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Suppress solver output
solvers.options['show_progress'] = False

class StochasticIndexTracker:
    """
    Improved implementation of stochastic neural network approach for index tracking
    with cardinality constraints.
    """
    def __init__(self, K, learning_rate=0.005, epochs=3000, temp_min=0.01, temp_decay=0.9995):
        """
        Parameters:
        K (int): Maximum number of assets to include in the portfolio
        learning_rate (float): Learning rate for optimization
        epochs (int): Number of training epochs
        temp_min (float): Minimum temperature for annealing
        temp_decay (float): Temperature decay rate
        """
        self.K = K
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.S = None
        self.w_tilde = None
        self.training = True
        self.selected_assets = None
        
    def fit(self, X, y, verbose=True):
        """
        Fit the model to the training data with improved optimization.
        """
        D, N = X.shape
        
        # Initialize parameters with better scaling
        self.S = torch.nn.Parameter(torch.randn(self.K, N) / np.sqrt(N))
        self.w_tilde = torch.nn.Parameter(torch.zeros(N))
        
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Set training mode
        self.training = True
        
        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam([
            {'params': self.S, 'weight_decay': 1e-4},
            {'params': self.w_tilde, 'weight_decay': 1e-4}
        ], lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=100, verbose=verbose
        )
        
        # Initialize temperature
        temperature = 1.0
        
        # Keep track of best model
        best_loss = float('inf')
        best_S = None
        best_w_tilde = None
        
        # Training loop
        pbar = tqdm(range(self.epochs)) if verbose else range(self.epochs)
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Forward pass with current temperature
            w = self._generate_weights(temperature)
            
            # Compute tracking error
            y_pred = torch.matmul(X_tensor, w.reshape(-1, 1))
            loss = torch.mean((y_pred - y_tensor) ** 2)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_([self.S, self.w_tilde], max_norm=1.0)
            
            optimizer.step()
            
            # Update temperature with decay
            temperature = max(self.temp_min, temperature * self.temp_decay)
            
            # Update learning rate
            scheduler.step(loss)
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_S = self.S.detach().clone()
                best_w_tilde = self.w_tilde.detach().clone()
            
            # Update progress bar
            if verbose and epoch % 100 == 0:
                pbar.set_description(f"Loss: {loss.item():.8f}, Temp: {temperature:.4f}")
        
        # Restore best model
        self.S.data = best_S
        self.w_tilde.data = best_w_tilde
        
        # Set to evaluation mode after training
        self.training = False
        
        # Store selected assets
        self.selected_assets = self.get_selected_assets()
        
        return self
    
    def _generate_weights(self, temperature):
        """
        Generate portfolio weights with improved stability.
        """
        # Generate selection probabilities with temperature annealing
        pi = torch.softmax(self.S / temperature, dim=1)
        
        # Sample asset selections
        z = torch.zeros(self.K, pi.shape[1], device=pi.device)
        
        if self.training:
            # Training mode: use Gumbel-Softmax with straight-through estimator
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(pi)))
            gumbel_logits = (torch.log(pi) + gumbel_noise) / temperature
            
            # Softmax in forward pass for gradient computation
            soft_samples = torch.softmax(gumbel_logits, dim=1)
            
            # Straight-through trick: use hard samples in forward pass but soft samples for gradient
            _, indices = torch.max(gumbel_logits, dim=1)
            hard_samples = torch.zeros_like(soft_samples).scatter_(1, indices.unsqueeze(1), 1.0)
            
            # Use straight-through estimator: hard in forward, soft in backward
            z = hard_samples.detach() - soft_samples.detach() + soft_samples
        else:
            # Evaluation mode: deterministic selection of top assets
            _, indices = torch.max(pi, dim=1)
            for i, idx in enumerate(indices):
                z[i, idx] = 1.0
        
        # Sum up to get the final selection mask (remove duplicates)
        z_sum = torch.clamp(torch.sum(z, dim=0), 0, 1)
        
        # Generate weights with softmax for better stability
        w_masked = torch.exp(self.w_tilde) * z_sum
        
        # Add small epsilon to avoid division by zero
        w_sum = torch.sum(w_masked) + 1e-10
        w_normalized = w_masked / w_sum
        
        return w_normalized
    
    def predict(self, X):
        """
        Predict index values using the trained model.
        """
        with torch.no_grad():
            self.training = False
            X_tensor = torch.FloatTensor(X)
            w = self._generate_weights(temperature=self.temp_min)
            y_pred = torch.matmul(X_tensor, w.reshape(-1, 1))
            return y_pred.numpy()
    
    def get_portfolio_weights(self):
        """
        Get the optimized portfolio weights.
        """
        with torch.no_grad():
            self.training = False
            w = self._generate_weights(temperature=self.temp_min)
            return w.numpy()
    
    def get_selected_assets(self):
        """
        Get the indices of selected assets.
        """
        with torch.no_grad():
            self.training = False
            pi = torch.softmax(self.S, dim=1)
            
            # Get top index for each row
            _, indices = torch.max(pi, dim=1)
            selected_assets = set(indices.cpu().numpy())
            
            return list(selected_assets)
    
    def post_process(self, X, y):
        """
        Post-process the results using quadratic programming.
        """
        if self.selected_assets is None:
            self.selected_assets = self.get_selected_assets()
            
        selected_assets = self.selected_assets
        X_selected = X[:, selected_assets]
        
        # Solve constrained regression problem with QP
        D, K_selected = X_selected.shape
        
        if K_selected == 0:
            # No assets selected, return equal weights for all assets
            return np.ones(X.shape[1]) / X.shape[1]
        
        # Setup quadratic programming problem
        P = 2 * np.dot(X_selected.T, X_selected)
        q = -2 * np.dot(X_selected.T, y)
        G = -np.eye(K_selected)
        h = np.zeros(K_selected)
        A = np.ones((1, K_selected))
        b = np.ones(1)
        
        # Add small value to diagonal of P for numerical stability
        P = P + 1e-8 * np.eye(K_selected)
        
        # Convert to cvxopt matrices
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        try:
            # Solve QP problem
            sol = solvers.qp(P, q, G, h, A, b)
            
            # Extract solution
            w_qp = np.array(sol['x']).flatten()
            
            # Map back to original asset space
            w_refined = np.zeros(X.shape[1])
            for i, idx in enumerate(selected_assets):
                w_refined[idx] = w_qp[i]
            
            return w_refined
            
        except Exception as e:
            print(f"QP solver failed: {e}")
            # Fall back to equal weights for selected assets
            w_refined = np.zeros(X.shape[1])
            for idx in selected_assets:
                w_refined[idx] = 1.0 / len(selected_assets)
            
            return w_refined


def generate_better_synthetic_data(D=750, N=100, K_true=5, noise_level=0.01):
    """
    Generate better synthetic data for testing index tracking methods.
    """
    # Generate K_true base factors
    base_factors = np.random.randn(D, K_true)
    
    # Generate true weights (equal weights)
    w_true_base = np.ones(K_true) / K_true
    
    # Generate index based on factors
    y = np.dot(base_factors, w_true_base)
    
    # Make y positive to avoid negative values in cumulative returns
    y = y - np.min(y) + 0.001
    
    # Generate factor loadings for each asset
    factor_loadings = np.zeros((N, K_true))
    
    # Each asset is primarily affected by one factor
    assets_per_factor = N // K_true
    remainder = N % K_true
    
    idx = 0
    w_true = np.zeros(N)
    
    for i in range(K_true):
        num_assets = assets_per_factor + (1 if i < remainder else 0)
        
        # Create assets with primary exposure to this factor
        for j in range(num_assets):
            # Primary factor has loading between 0.6 and 1.0
            primary_loading = 0.6 + 0.4 * np.random.rand()
            factor_loadings[idx, i] = primary_loading
            
            # Other factors have smaller loadings
            for k in range(K_true):
                if k != i:
                    factor_loadings[idx, k] = 0.1 * np.random.rand()
            
            idx += 1
        
        # Set one asset in each group to have non-zero weight
        group_start = i * assets_per_factor + min(i, remainder)
        w_true[group_start] = w_true_base[i]
    
    # Generate asset returns
    X = np.zeros((D, N))
    for i in range(N):
        # Systematic component from factors
        X[:, i] = np.dot(base_factors, factor_loadings[i])
        
        # Add idiosyncratic component
        X[:, i] += noise_level * np.random.randn(D)
    
    # Add small positive drift to returns to make them more realistic
    X += 0.0002
    
    return X, y, w_true


def simulate_better_market_data(D=750, N=100, K_sectors=10, include_factors=True):
    """
    Generate more realistic market data for testing index tracking methods.
    """
    # Create market factor with positive drift
    market_drift = 0.0005  # 5bp daily drift
    market_vol = 0.01     # 1% daily volatility
    market_factor = np.random.normal(market_drift, market_vol, D)
    
    # Create sector factors
    sector_vol = 0.015
    sector_factors = np.random.normal(0, sector_vol, (D, K_sectors))
    
    # Create specific factors if requested
    if include_factors:
        # Size factor (small minus big)
        size_factor = np.random.normal(0, 0.008, D)
        
        # Value factor (high B/M minus low B/M)
        value_factor = np.random.normal(0.0002, 0.01, D)
        
        # Momentum factor
        momentum_factor = np.random.normal(0.0003, 0.012, D)
        
        # Volatility factor
        volatility_factor = np.random.normal(-0.0001, 0.015, D)
        
        factors = np.column_stack([
            market_factor, 
            sector_factors, 
            size_factor, 
            value_factor, 
            momentum_factor, 
            volatility_factor
        ])
    else:
        factors = np.column_stack([market_factor, sector_factors])
    
    # Generate factor exposures for each asset
    num_factors = factors.shape[1]
    factor_exposures = np.zeros((N, num_factors))
    
    # Assign assets to sectors
    sectors = np.random.randint(0, K_sectors, N)
    
    # Generate size characteristics (market cap)
    size = np.exp(np.random.normal(10, 2, N))
    log_size = np.log(size)
    market_caps = size.copy()
    
    # Generate other asset characteristics
    book_to_market = np.random.lognormal(0, 0.7, N)
    momentum_scores = np.random.normal(0, 1, N)
    volatilities = np.random.uniform(0.01, 0.03, N)
    
    # Set factor exposures
    for i in range(N):
        # Market exposure between 0.8 and 1.2
        factor_exposures[i, 0] = 0.8 + 0.4 * np.random.rand()
        
        # Sector exposure (one-hot)
        sector_idx = sectors[i] + 1  # +1 because market is at index 0
        factor_exposures[i, sector_idx] = 0.6 + 0.4 * np.random.rand()
        
        if include_factors:
            # Size exposure (negative relationship with log_size)
            size_idx = K_sectors + 1
            factor_exposures[i, size_idx] = -0.2 + 0.4 * (log_size[i] - np.mean(log_size)) / np.std(log_size)
            
            # Value exposure
            value_idx = K_sectors + 2
            factor_exposures[i, value_idx] = 0.3 * (book_to_market[i] - np.mean(book_to_market)) / np.std(book_to_market)
            
            # Momentum exposure
            momentum_idx = K_sectors + 3
            factor_exposures[i, momentum_idx] = 0.25 * momentum_scores[i]
            
            # Volatility exposure
            vol_idx = K_sectors + 4
            factor_exposures[i, vol_idx] = 0.2 * (volatilities[i] - np.mean(volatilities)) / np.std(volatilities)
    
    # Generate asset returns
    X = np.zeros((D, N))
    for i in range(N):
        # Systematic return from factors
        X[:, i] = np.dot(factors, factor_exposures[i])
        
        # Add idiosyncratic return
        X[:, i] += volatilities[i] * np.random.randn(D)
    
    # Generate index as market-cap weighted portfolio
    weights = market_caps / np.sum(market_caps)
    y = np.dot(X, weights)
    
    return X, y, market_caps


def solve_qp_robust(X, y, regularization=1e-8):
    """
    Solve the constrained regression problem using quadratic programming
    with added regularization for better numerical stability.
    """
    D, N = X.shape
    
    # Setup quadratic programming problem
    P = 2 * np.dot(X.T, X) + regularization * np.eye(N)
    q = -2 * np.dot(X.T, y)
    G = -np.eye(N)
    h = np.zeros(N)
    A = np.ones((1, N))
    b = np.ones(1)
    
    # Convert to cvxopt matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    
    try:
        # Solve QP problem
        sol = solvers.qp(P, q, G, h, A, b)
        
        # Extract solution
        w = np.array(sol['x']).flatten()
        
        return w
    except:
        # If QP fails, use equal weights
        return np.ones(N) / N


def forward_selection_improved(X, y, K):
    """
    Improved forward selection for index tracking.
    """
    D, N = X.shape
    selected = []
    
    # Initial error with no assets
    best_overall_error = float('inf')
    
    for _ in range(K):
        best_error = float('inf')
        best_idx = -1
        
        for i in range(N):
            if i in selected:
                continue
                
            # Try adding this asset
            current = selected + [i]
            X_sub = X[:, current]
            
            try:
                # Solve QP for this subset
                w_sub = solve_qp_robust(X_sub, y)
                
                # Calculate tracking error
                y_pred = np.dot(X_sub, w_sub)
                error = np.mean((y_pred - y) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_idx = i
            except:
                # Skip if QP fails
                continue
        
        # If we found a better asset, add it
        if best_idx != -1:
            selected.append(best_idx)
            
            # Update best overall error
            if best_error < best_overall_error:
                best_overall_error = best_error
            else:
                # If error increased, we might want to stop
                break
    
    # Solve QP with selected assets
    if len(selected) > 0:
        X_selected = X[:, selected]
        w_selected = solve_qp_robust(X_selected, y)
        
        # Map back to original asset space
        w = np.zeros(N)
        for i, idx in enumerate(selected):
            w[idx] = w_selected[i]
    else:
        # If no assets selected, use equal weights
        w = np.ones(N) / N
    
    return w


def backward_selection_improved(X, y, K):
    """
    Improved backward selection for index tracking.
    """
    D, N = X.shape
    
    # Start with all assets
    selected = list(range(N))
    
    # Initial weights
    try:
        w = solve_qp_robust(X, y)
    except:
        # If QP fails, use equal weights
        w = np.ones(N) / N
    
    while len(selected) > K:
        # Find asset with smallest weight
        min_weight = float('inf')
        min_idx = -1
        
        for i, idx in enumerate(selected):
            if w[idx] < min_weight:
                min_weight = w[idx]
                min_idx = i
        
        if min_idx != -1:
            # Remove this asset
            selected.pop(min_idx)
            
            # Update weights
            X_sub = X[:, selected]
            try:
                w_sub = solve_qp_robust(X_sub, y)
                
                # Map back to original space
                w = np.zeros(N)
                for i, idx in enumerate(selected):
                    w[idx] = w_sub[i]
            except:
                # If QP fails, use equal weights for remaining assets
                w = np.zeros(N)
                for idx in selected:
                    w[idx] = 1.0 / len(selected)
        else:
            # No assets to remove
            break
    
    return w


def largest_market_cap_improved(X, y, K, market_caps):
    """
    Improved largest market capitalization method for index tracking.
    """
    D, N = X.shape
    
    # Select K assets with largest market cap
    selected = np.argsort(market_caps)[-K:]
    
    # Solve QP with selected assets
    X_selected = X[:, selected]
    
    try:
        w_selected = solve_qp_robust(X_selected, y)
    except:
        # If QP fails, use market cap weights
        caps_selected = market_caps[selected]
        w_selected = caps_selected / np.sum(caps_selected)
    
    # Map back to original asset space
    w = np.zeros(N)
    for i, idx in enumerate(selected):
        w[idx] = w_selected[i]
    
    return w


def plot_weights(w, w_true=None, title='Portfolio Weights'):
    """
    Plot portfolio weights.
    
    Parameters:
    w (numpy.ndarray): Portfolio weights
    w_true (numpy.ndarray, optional): True weights for comparison
    title (str): Plot title
    """
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(w)), w)
    if w_true is not None:
        plt.bar(range(len(w_true)), w_true, alpha=0.5, color='red')
        plt.legend(['Estimated Weights', 'True Weights'])
    plt.title(title)
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.show()


def plot_tracking(X, y, weights_dict, title='Index Tracking Performance'):
    """
    Plot index tracking performance for multiple methods.
    
    Parameters:
    X (numpy.ndarray): Asset returns, shape (D, N)
    y (numpy.ndarray): Index returns, shape (D,)
    weights_dict (dict): Dictionary mapping method names to portfolio weights
    title (str): Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot index
    cumulative_y = np.cumsum(y)
    plt.plot(cumulative_y, linewidth=2, label='Index')
    
    # Plot tracking portfolios
    for method_name, w in weights_dict.items():
        y_pred = np.dot(X, w)
        cumulative_pred = np.cumsum(y_pred)
        plt.plot(cumulative_pred, label=f'{method_name} (K={np.sum(w > 1e-5)})')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot tracking errors
    plt.figure(figsize=(15, 8))
    
    for method_name, w in weights_dict.items():
        y_pred = np.dot(X, w)
        # Avoid division by zero
        y_nonzero = np.where(np.abs(y) > 1e-10, y, 1e-10)
        tracking_error = (y_pred - y) / np.abs(y_nonzero)
        plt.plot(tracking_error, label=f'{method_name} Error')
    
    plt.title('Tracking Errors')
    plt.xlabel('Time')
    plt.ylabel('Percentage Error')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.show()


def evaluate_portfolio(X, y, w):
    """
    Evaluate the performance of a portfolio.
    
    Parameters:
    X (numpy.ndarray): Asset returns, shape (D, N)
    y (numpy.ndarray): Index returns, shape (D,)
    w (numpy.ndarray): Portfolio weights, shape (N,)
    
    Returns:
    dict: Performance metrics
    """
    # Predict index values
    y_pred = np.dot(X, w)
    
    # Calculate tracking error
    tracking_error = np.mean((y_pred - y) ** 2)
    
    # Calculate volatility
    volatility = np.std(y_pred)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = np.mean(y_pred) / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumsum(y_pred)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / (peak + 1e-10)
    max_drawdown = np.max(drawdown)
    
    # Count number of assets used (non-zero weights)
    num_assets = np.sum(w > 1e-5)
    
    return {
        'tracking_error': tracking_error,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_assets': num_assets
    }


def compare_metrics(X, y, weights_dict):
    """
    Compare performance metrics across methods.
    
    Parameters:
    X (numpy.ndarray): Asset returns, shape (D, N)
    y (numpy.ndarray): Index returns, shape (D,)
    weights_dict (dict): Dictionary mapping method names to portfolio weights
    """
    # Calculate metrics for index
    y_index = y
    vol_index = np.std(y_index)
    sharpe_index = np.mean(y_index) / vol_index if vol_index > 0 else 0
    
    cumulative_index = np.cumsum(y_index)
    peak_index = np.maximum.accumulate(cumulative_index)
    drawdown_index = (peak_index - cumulative_index) / (peak_index + 1e-10)
    max_drawdown_index = np.max(drawdown_index)
    
    # Calculate metrics for each method
    metrics = {}
    for method_name, w in weights_dict.items():
        metrics[method_name] = evaluate_portfolio(X, y, w)
    
    # Create plots for comparing metrics
    metrics_to_plot = ['tracking_error', 'volatility', 'sharpe_ratio', 'max_drawdown']
    titles = ['Tracking Error', 'Volatility', 'Sharpe Ratio', 'Maximum Drawdown']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        values = [metrics[method][metric] for method in weights_dict.keys()]
        
        # For volatility, sharpe ratio, and max drawdown, also show index values
        if metric == 'volatility':
            reference = vol_index
        elif metric == 'sharpe_ratio':
            reference = sharpe_index
        elif metric == 'max_drawdown':
            reference = max_drawdown_index
        else:
            reference = None
        
        ax = axes[i]
        bars = ax.bar(list(weights_dict.keys()), values)
        
        if reference is not None:
            ax.axhline(y=reference, color='r', linestyle='--', label='Index')
            
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xticklabels(list(weights_dict.keys()), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def run_improved_toy_example():
    """Run an improved toy example."""
    print("Running improved toy example...")
    
    # Generate better synthetic data
    D = 750
    N = 100
    K_true = 5
    
    X, y, w_true = generate_better_synthetic_data(D, N, K_true)
    
    # Train our stochastic neural network model
    model = StochasticIndexTracker(K=K_true, learning_rate=0.01, epochs=2000)
    model.fit(X, y)
    
    # Get portfolio weights
    w_snn = model.get_portfolio_weights()
    
    # Post-process the results
    w_snn_pp = model.post_process(X, y)
    
    # Compare with standard QP
    w_qp = solve_qp_robust(X, y)
    
    # Plot results
    plot_weights(w_qp, title='QP Without Cardinality Constraint')
    plot_weights(w_snn, title='Stochastic Neural Network')
    plot_weights(w_snn_pp, title='Stochastic Neural Network with Post-Processing')
    
    # Evaluate tracking performance
    print("\nPerformance Metrics:")
    print("QP Without Cardinality Constraint:")
    print(evaluate_portfolio(X, y, w_qp))
    print("\nStochastic Neural Network:")
    print(evaluate_portfolio(X, y, w_snn))
    print("\nStochastic Neural Network with Post-Processing:")
    print(evaluate_portfolio(X, y, w_snn_pp))
    
    # Plot tracking performance
    plot_tracking(X, y, {
        'QP': w_qp,
        'SNN': w_snn,
        'SNN+PP': w_snn_pp
    }, title='Toy Example: Index Tracking Performance')


def run_improved_market_simulation():
    """Run an improved market simulation."""
    print("\nRunning improved market simulation...")
    
    # Generate more realistic market data
    D = 750
    N = 50
    K_sectors = 5
    
    X, y, market_caps = simulate_better_market_data(D, N, K_sectors, include_factors=True)
    
    # Split data into training and testing sets
    train_size = int(0.7 * D)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test different portfolio sizes
    K_values = [5, 10, 15]
    
    for K in K_values:
        print(f"\nTesting with K={K}")
        
        # Train stochastic neural network model
        model = StochasticIndexTracker(K=K, learning_rate=0.005, epochs=2000)
        model.fit(X_train, y_train)
        
        # Get portfolio weights
        w_snn = model.get_portfolio_weights()
        
        # Post-process the results
        w_snn_pp = model.post_process(X_train, y_train)
        
        # Compare with baseline methods
        w_forward = forward_selection_improved(X_train, y_train, K)
        w_backward = backward_selection_improved(X_train, y_train, K)
        w_market_cap = largest_market_cap_improved(X_train, y_train, K, market_caps)
        
        # Evaluate on test set
        methods = {
            'Forward Selection': w_forward,
            'Backward Selection': w_backward,
            'Largest Market Cap': w_market_cap,
            'Stochastic NN': w_snn,
            'Stochastic NN+PP': w_snn_pp
        }
        
        print("\nTest Set Performance:")
        for method_name, w in methods.items():
            metrics = evaluate_portfolio(X_test, y_test, w)
            print(f"{method_name}:")
            print(f"  Tracking Error: {metrics['tracking_error']:.8f}")
            print(f"  Volatility: {metrics['volatility']:.4f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
            print(f"  Num Assets: {metrics['num_assets']}")
        
        # Plot tracking performance on test set
        plot_tracking(X_test, y_test, methods, title=f'Market Simulation: Index Tracking Performance (K={K})')
        
        # Compare performance metrics across methods
        compare_metrics(X_test, y_test, methods)


def run_improved_backtesting():
    """Run improved backtesting to simulate a realistic investing scenario."""
    print("\nRunning improved backtesting simulation...")
    
    # Generate market data over a longer period
    D = 500
    N = 50
    K_sectors = 5
    
    X, y, market_caps = simulate_better_market_data(D, N, K_sectors, include_factors=True)
    
    # Set portfolio size
    K = 10
    
    # Parameters for backtesting
    lookback_period = 125
    rebalance_period = 21
    
    # Prepare for backtesting
    start_time = lookback_period
    end_time = D
    
    # Lists to store portfolio values
    portfolio_values = {
        'Index': [1.0],
        'Forward Selection': [1.0],
        'Backward Selection': [1.0],
        'Largest Market Cap': [1.0],
        'Stochastic NN': [1.0],
        'Stochastic NN+PP': [1.0]
    }
    
    # Run backtesting
    for t in range(start_time, end_time, rebalance_period):
        print(f"Backtesting period: {t-lookback_period}-{t} (training) | {t}-{min(t+rebalance_period, end_time)} (testing)")
        
        # Training data
        X_train = X[t-lookback_period:t]
        y_train = y[t-lookback_period:t]
        
        # Testing data
        if t + rebalance_period <= end_time:
            X_test = X[t:t+rebalance_period]
            y_test = y[t:t+rebalance_period]
        else:
            X_test = X[t:end_time]
            y_test = y[t:end_time]
        
        # Train models
        model = StochasticIndexTracker(K=K, learning_rate=0.005, epochs=1000)
        model.fit(X_train, y_train, verbose=False)
        
        # Get portfolio weights
        w_snn = model.get_portfolio_weights()
        w_snn_pp = model.post_process(X_train, y_train)
        w_forward = forward_selection_improved(X_train, y_train, K)
        w_backward = backward_selection_improved(X_train, y_train, K)
        w_market_cap = largest_market_cap_improved(X_train, y_train, K, market_caps)
        
        # Calculate returns during the testing period
        index_return = np.sum(y_test)
        forward_return = np.sum(np.dot(X_test, w_forward))
        backward_return = np.sum(np.dot(X_test, w_backward))
        market_cap_return = np.sum(np.dot(X_test, w_market_cap))
        snn_return = np.sum(np.dot(X_test, w_snn))
        snn_pp_return = np.sum(np.dot(X_test, w_snn_pp))
        
        # Update portfolio values
        portfolio_values['Index'].append(portfolio_values['Index'][-1] * (1 + index_return))
        portfolio_values['Forward Selection'].append(portfolio_values['Forward Selection'][-1] * (1 + forward_return))
        portfolio_values['Backward Selection'].append(portfolio_values['Backward Selection'][-1] * (1 + backward_return))
        portfolio_values['Largest Market Cap'].append(portfolio_values['Largest Market Cap'][-1] * (1 + market_cap_return))
        portfolio_values['Stochastic NN'].append(portfolio_values['Stochastic NN'][-1] * (1 + snn_return))
        portfolio_values['Stochastic NN+PP'].append(portfolio_values['Stochastic NN+PP'][-1] * (1 + snn_pp_return))
    
    # Plot portfolio values
    plt.figure(figsize=(15, 8))
    for method, values in portfolio_values.items():
        plt.plot(values, label=method)
    
    plt.title(f'Backtesting Results (K={K})')
    plt.xlabel('Rebalancing Period')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate final performance metrics
    print("\nFinal Backtesting Performance:")
    for method in portfolio_values.keys():
        if method == 'Index':
            continue
        
        values = portfolio_values[method]
        returns = np.diff(values) / values[:-1]
        
        # Calculate metrics
        total_return = values[-1] / values[0] - 1
        annualized_return = (1 + total_return) ** (252 / D) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)
        
        # Tracking error to index
        index_values = portfolio_values['Index']
        index_returns = np.diff(index_values) / index_values[:-1]
        tracking_error = np.sqrt(np.mean((returns - index_returns) ** 2)) * np.sqrt(252)
        
        print(f"\n{method}:")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Annualized Return: {annualized_return:.4f}")
        print(f"  Volatility: {volatility:.4f}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.4f}")
        print(f"  Tracking Error: {tracking_error:.4f}")
    
    # Plot tracking error over time
    plt.figure(figsize=(15, 8))
    
    index_values = portfolio_values['Index']
    
    for method, values in portfolio_values.items():
        if method == 'Index':
            continue
        
        # Calculate percentage tracking error
        tracking_error = [(values[i] / index_values[i] - 1) * 100 for i in range(len(values))]
        plt.plot(tracking_error, label=method)
    
    plt.title('Tracking Error Over Time')
    plt.xlabel('Rebalancing Period')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.show()


def main():
    print("Testing Improved Index Tracking with Cardinality Constraints using Stochastic Neural Networks")
    
    # Run improved toy example
    run_improved_toy_example()
    
    # Run improved market simulation
    run_improved_market_simulation()
    
    # Run improved backtesting
    run_improved_backtesting()


if __name__ == "__main__":
    main()