import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

class WMTR:
    """
    Weighted Multichannel Time-series Regression (WMTR) implementation
    as described in the paper "Tensor Representation in High-Frequency Financial Data for Price Change Prediction"
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, r=3, max_iter=50, tol=1e-6):
        self.lambda1 = lambda1  # Regularization for W1
        self.lambda2 = lambda2  # Regularization for w2
        self.r = r              # Weight parameter for class balancing
        self.max_iter = max_iter
        self.tol = tol
        self.W1 = None
        self.w2 = None
        self.class_weights = None
        
    def fit(self, X, y):
        """
        Fit the WMTR model to training data
        
        Parameters:
        - X: tensor input of shape (n_samples, n_features, n_time_steps)
        - y: target labels of shape (n_samples,)
        """
        N, D, T = X.shape
        C = len(np.unique(y))
        
        # Convert y to one-hot encoding (with -1 and 1)
        Y = -np.ones((N, C))
        for i in range(N):
            Y[i, y[i]] = 1
        
        # Calculate class weights
        class_counts = np.bincount(y)
        self.class_weights = {}
        for i in range(len(class_counts)):
            self.class_weights[i] = 1.0 / (class_counts[i] ** (1.0/self.r))
        
        # Initialize diagonal weight matrix S
        S = np.zeros(N)
        for i in range(N):
            S[i] = self.class_weights[y[i]]
        S = np.sqrt(S)
        
        # Initialize W1 randomly
        self.W1 = np.random.randn(D, C)
        
        # Initialize w2 to focus on the most recent time step
        self.w2 = np.zeros(T)
        self.w2[-1] = 1.0
        
        # Iterative optimization
        prev_W1 = np.zeros_like(self.W1)
        prev_w2 = np.zeros_like(self.w2)
        
        train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for iter in range(self.max_iter):
            # Fix w2, solve for W1
            X2 = np.zeros((D, N))
            for i in range(N):
                X2[:, i] = X[i] @ self.w2
                
            S2 = np.diag(S)
            Y2 = Y.T
            
            term1 = X2 @ S2 @ S2.T @ X2.T + self.lambda1 * np.eye(D)
            term2 = X2 @ S2 @ S2.T @ Y2.T
            self.W1 = np.linalg.solve(term1, term2)
            
            # Fix W1, solve for w2
            X1 = np.zeros((C*N, T))
            Y1 = Y.flatten()
            S1 = np.zeros(C*N)
            
            for i in range(N):
                for k in range(C):
                    idx = C*i + k
                    X1[idx, :] = X[i].T @ self.W1[:, k]
                    S1[idx] = S[i]
            
            S1 = np.diag(S1)
            term1 = X1.T @ S1.T @ S1 @ X1 + self.lambda2 * np.eye(T)
            term2 = X1.T @ S1.T @ S1 @ Y1
            self.w2 = np.linalg.solve(term1, term2)
            
            # Check convergence
            W1_diff = np.linalg.norm(self.W1 - prev_W1)
            w2_diff = np.linalg.norm(self.w2 - prev_w2)
            
            if W1_diff < self.tol and w2_diff < self.tol:
                print(f"Converged at iteration {iter+1}")
                break
                
            prev_W1 = self.W1.copy()
            prev_w2 = self.w2.copy()
            
            # Calculate training metrics
            y_pred = self.predict(X)
            train_metrics['accuracy'].append(accuracy_score(y, y_pred))
            train_metrics['precision'].append(precision_score(y, y_pred, average='macro'))
            train_metrics['recall'].append(recall_score(y, y_pred, average='macro'))
            train_metrics['f1'].append(f1_score(y, y_pred, average='macro'))
            
            if (iter + 1) % 5 == 0:
                print(f"Iteration {iter+1}: F1={train_metrics['f1'][-1]:.4f}, Accuracy={train_metrics['accuracy'][-1]:.4f}")
            
        return train_metrics
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        - X: tensor input of shape (n_samples, n_features, n_time_steps)
        
        Returns:
        - Predicted class labels
        """
        N = X.shape[0]
        pred = np.zeros((N, self.W1.shape[1]))
        
        for i in range(N):
            pred[i] = self.W1.T @ X[i] @ self.w2
            
        return np.argmax(pred, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        - X: tensor input of shape (n_samples, n_features, n_time_steps)
        
        Returns:
        - Predicted class probabilities
        """
        N = X.shape[0]
        pred = np.zeros((N, self.W1.shape[1]))
        
        for i in range(N):
            pred[i] = self.W1.T @ X[i] @ self.w2
            
        # Convert to probabilities using softmax
        exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        return exp_pred / np.sum(exp_pred, axis=1, keepdims=True)


class MDA:
    """
    Multilinear Discriminant Analysis (MDA) implementation
    as described in the paper "Tensor Representation in High-Frequency Financial Data for Price Change Prediction"
    """
    def __init__(self, mode1_dims=20, mode2_dims=5, lambda_reg=1.0, max_iter=20, tol=1e-6):
        self.mode1_dims = mode1_dims  # Projected dimensions for mode 1
        self.mode2_dims = mode2_dims  # Projected dimensions for mode 2
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.max_iter = max_iter
        self.tol = tol
        self.W1 = None  # Projection matrix for mode 1
        self.W2 = None  # Projection matrix for mode 2
        self.class_means = None  # Mean tensor for each class in projected space
        self.global_mean = None  # Global mean tensor in projected space
        
    def fit(self, X, y):
        """
        Fit the MDA model to training data
        
        Parameters:
        - X: tensor input of shape (n_samples, n_features, n_time_steps)
        - y: target labels of shape (n_samples,)
        """
        N, D, T = X.shape
        n_classes = len(np.unique(y))
        
        print(f"MDA: Input tensor shape = {X.shape}, Classes = {n_classes}")
        
        # Initialize projection matrices with random orthogonal matrices
        self.W1 = np.random.randn(D, self.mode1_dims)
        self.W1, _ = np.linalg.qr(self.W1)
        
        self.W2 = np.random.randn(T, self.mode2_dims)
        self.W2, _ = np.linalg.qr(self.W2)
        
        # Calculate class-specific and global means
        class_means_orig = {}
        class_counts = {}
        global_mean_orig = np.zeros((D, T))
        
        for c in np.unique(y):
            class_samples = X[y == c]
            class_means_orig[c] = np.mean(class_samples, axis=0)
            class_counts[c] = len(class_samples)
            global_mean_orig += class_counts[c] * class_means_orig[c]
            
        global_mean_orig /= N
        
        # Iterative optimization
        for iter in range(self.max_iter):
            print(f"MDA: Iteration {iter+1}")
            
            # Optimize W1 (mode-1 projection)
            S1_b = np.zeros((D, D))
            S1_w = np.zeros((D, D))
            
            # Calculate scatter matrices for mode-1
            for c in np.unique(y):
                # Between-class scatter
                diff = class_means_orig[c] - global_mean_orig
                mode1_scatter = diff @ self.W2 @ self.W2.T @ diff.T
                S1_b += class_counts[c] * mode1_scatter
                
                # Within-class scatter
                for i in np.where(y == c)[0]:
                    diff = X[i] - class_means_orig[c]
                    mode1_scatter = diff @ self.W2 @ self.W2.T @ diff.T
                    S1_w += mode1_scatter
            
            # Add regularization to S1_w to ensure it's invertible
            S1_w += self.lambda_reg * np.eye(D)
            
            # Use a more stable approach for eigenvalue decomposition
            # First symmetrize the matrix for numerical stability
            S1_w_inv_S1_b = np.linalg.inv(S1_w) @ S1_b
            S1_w_inv_S1_b = (S1_w_inv_S1_b + S1_w_inv_S1_b.T) / 2
            
            # Use eigh instead of eig for symmetric matrices (guaranteed real eigenvalues)
            eigvals, eigvecs = np.linalg.eigh(S1_w_inv_S1_b)
            idx = np.argsort(eigvals)[::-1]  # Sort in descending order
            W1_new = eigvecs[:, idx[:self.mode1_dims]]
            
            # Update W1
            self.W1 = W1_new
            
            # Optimize W2 (mode-2 projection)
            S2_b = np.zeros((T, T))
            S2_w = np.zeros((T, T))
            
            # Calculate scatter matrices for mode-2
            for c in np.unique(y):
                # Between-class scatter
                diff = class_means_orig[c] - global_mean_orig
                mode2_scatter = diff.T @ self.W1 @ self.W1.T @ diff
                S2_b += class_counts[c] * mode2_scatter
                
                # Within-class scatter
                for i in np.where(y == c)[0]:
                    diff = X[i] - class_means_orig[c]
                    mode2_scatter = diff.T @ self.W1 @ self.W1.T @ diff
                    S2_w += mode2_scatter
            
            # Add regularization to S2_w
            S2_w += self.lambda_reg * np.eye(T)
            
            # Use a more stable approach for eigenvalue decomposition
            # First symmetrize the matrix for numerical stability
            S2_w_inv_S2_b = np.linalg.inv(S2_w) @ S2_b
            S2_w_inv_S2_b = (S2_w_inv_S2_b + S2_w_inv_S2_b.T) / 2
            
            # Use eigh instead of eig for symmetric matrices
            eigvals, eigvecs = np.linalg.eigh(S2_w_inv_S2_b)
            idx = np.argsort(eigvals)[::-1]  # Sort in descending order
            W2_new = eigvecs[:, idx[:self.mode2_dims]]
            
            # Check convergence
            if np.linalg.norm(self.W2 - W2_new) < self.tol:
                print(f"MDA converged at iteration {iter+1}")
                break
                
            self.W2 = W2_new
        
        # Calculate class means in the projected space
        self.class_means = {}
        self.global_mean = np.zeros((self.mode1_dims, self.mode2_dims))
        
        for c in np.unique(y):
            class_samples = X[y == c]
            projected_samples = np.zeros((len(class_samples), self.mode1_dims, self.mode2_dims))
            
            for i in range(len(class_samples)):
                projected_samples[i] = self.W1.T @ class_samples[i] @ self.W2
                
            self.class_means[c] = np.mean(projected_samples, axis=0)
            self.global_mean += class_counts[c] * self.class_means[c]
            
        self.global_mean /= N
        
        return self
            
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        - X: tensor input of shape (n_samples, n_features, n_time_steps)
        
        Returns:
        - Predicted class labels
        """
        N = X.shape[0]
        predictions = np.zeros(N, dtype=int)
        
        for i in range(N):
            # Project the sample
            projected = self.W1.T @ X[i] @ self.W2
            
            # Calculate distance to each class mean
            min_dist = float('inf')
            best_class = 0
            
            for c, mean in self.class_means.items():
                dist = np.linalg.norm(projected - mean)
                if dist < min_dist:
                    min_dist = dist
                    best_class = c
                    
            predictions[i] = best_class
            
        return predictions


def generate_simulated_data(n_samples=5000, n_features=144, n_time_steps=10, n_classes=3, random_state=42):
    """
    Generate simulated financial data with tensor representation
    
    Parameters:
    - n_samples: Number of samples
    - n_features: Number of features per time step
    - n_time_steps: Number of time steps in each sample
    - n_classes: Number of classes (price movement directions)
    - random_state: Random seed for reproducibility
    
    Returns:
    - X: Tensor data of shape (n_samples, n_features, n_time_steps)
    - y: Target labels of shape (n_samples,)
    - meta: Dictionary with additional simulation metadata
    """
    np.random.seed(random_state)
    
    # Create a structure to simulate correlations between features
    feature_groups = 12  # Number of feature groups (like price, volume, etc.)
    features_per_group = n_features // feature_groups
    
    # Create correlation structure for features
    corr_matrix = np.zeros((n_features, n_features))
    for g in range(feature_groups):
        start_idx = g * features_per_group
        end_idx = (g + 1) * features_per_group
        
        # Features in the same group are correlated
        corr_matrix[start_idx:end_idx, start_idx:end_idx] = 0.7
        
        # Add some cross-group correlations
        if g > 0:
            prev_start = (g - 1) * features_per_group
            prev_end = g * features_per_group
            corr_matrix[start_idx:end_idx, prev_start:prev_end] = 0.3
            corr_matrix[prev_start:prev_end, start_idx:end_idx] = 0.3
    
    # Ensure the matrix is positive definite
    corr_matrix = corr_matrix + np.eye(n_features) * 0.3
    
    # Add some randomness to correlation
    corr_matrix = corr_matrix + np.random.randn(n_features, n_features) * 0.05
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
    
    # Ensure diagonal is 1
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate base underlying process (e.g., simulated asset price)
    price = 100.0
    prices = []
    returns = []
    volatility = 0.01
    
    # Generate enough prices for the entire dataset including future predictions
    # We need n_samples + n_time_steps + prediction_horizon
    prediction_horizon = 10  # Number of steps to look ahead for target
    total_prices_needed = n_samples + n_time_steps + prediction_horizon
    
    for i in range(total_prices_needed):
        prices.append(price)
        
        # Generate return with some autocorrelation
        if i > 0:
            ret = 0.2 * returns[-1] + np.random.normal(0, volatility)
        else:
            ret = np.random.normal(0, volatility)
            
        returns.append(ret)
        
        # Update price
        price = price * (1 + ret)
        
        # Dynamic volatility (GARCH-like effect)
        volatility = 0.8 * volatility + 0.2 * abs(ret)
    
    # Generate features using the correlation structure
    X = np.zeros((n_samples, n_features, n_time_steps))
    
    for i in range(n_samples):
        for t in range(n_time_steps):
            # Generate correlated features
            features = np.random.multivariate_normal(np.zeros(n_features), corr_matrix)
            
            # Add trend component based on recent price movement
            price_idx = i + t
            recent_return = (prices[price_idx+1] - prices[price_idx]) / prices[price_idx]
            
            # Different feature groups react differently to price changes
            for g in range(feature_groups):
                start_idx = g * features_per_group
                end_idx = (g + 1) * features_per_group
                
                if g < feature_groups // 3:
                    # These features are positively correlated with returns
                    features[start_idx:end_idx] += recent_return * 10
                elif g < 2 * feature_groups // 3:
                    # These features are negatively correlated with returns
                    features[start_idx:end_idx] -= recent_return * 5
                else:
                    # These features have more complex relationship
                    features[start_idx:end_idx] += (recent_return**2) * 20
            
            X[i, :, t] = features
    
    # Generate target labels (future price movement)
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Future return over next 10 steps
        current_price = prices[i + n_time_steps - 1]
        future_price = prices[i + n_time_steps + prediction_horizon - 1]  # 10 steps ahead
        future_return = (future_price - current_price) / current_price
        
        # Classify based on future return
        if future_return > 0.01:  # Up
            y[i] = 1
        elif future_return < -0.01:  # Down
            y[i] = 2
        else:  # Stable
            y[i] = 0
    
    # Create more balanced dataset for better testing
    # Target distribution: stable=50%, up=25%, down=25%
    target_stable = int(n_samples * 0.5)
    target_up = int(n_samples * 0.25)
    target_down = int(n_samples * 0.25)
    
    stable_indices = np.where(y == 0)[0]
    up_indices = np.where(y == 1)[0]
    down_indices = np.where(y == 2)[0]
    
    # Ensure we have enough samples of each class
    if len(up_indices) < target_up:
        # Generate more 'up' samples by slightly modifying existing ones
        needed = target_up - len(up_indices)
        for _ in range(needed):
            # Choose a random 'stable' sample and make it 'up'
            idx = np.random.choice(stable_indices)
            y[idx] = 1
            # Move from stable to up indices
            stable_indices = np.setdiff1d(stable_indices, [idx])
            up_indices = np.append(up_indices, idx)
    
    if len(down_indices) < target_down:
        # Generate more 'down' samples
        needed = target_down - len(down_indices)
        for _ in range(needed):
            # Choose a random 'stable' sample and make it 'down'
            idx = np.random.choice(stable_indices)
            y[idx] = 2
            # Move from stable to down indices
            stable_indices = np.setdiff1d(stable_indices, [idx])
            down_indices = np.append(down_indices, idx)
    
    # If we have too many samples of any class, remove some
    if len(stable_indices) > target_stable:
        to_remove = np.random.choice(stable_indices, len(stable_indices) - target_stable, replace=False)
        mask = np.ones(len(X), dtype=bool)
        mask[to_remove] = False
        X = X[mask]
        y = y[mask]
    
    if len(up_indices) > target_up:
        to_remove = np.random.choice(up_indices, len(up_indices) - target_up, replace=False)
        mask = np.ones(len(X), dtype=bool)
        mask[to_remove] = False
        X = X[mask]
        y = y[mask]
    
    if len(down_indices) > target_down:
        to_remove = np.random.choice(down_indices, len(down_indices) - target_down, replace=False)
        mask = np.ones(len(X), dtype=bool)
        mask[to_remove] = False
        X = X[mask]
        y = y[mask]
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Create metadata
    meta = {
        'prices': np.array(prices),
        'returns': np.array(returns),
        'corr_matrix': corr_matrix,
        'feature_groups': feature_groups
    }
    
    return X, y, meta


def run_backtest(model, X_test, y_test, prices, start_idx):
    """
    Run a backtest of the trading strategy
    
    Parameters:
    - model: trained model (WMTR or MDA)
    - X_test: test features
    - y_test: test labels
    - prices: simulated price series
    - start_idx: index in prices corresponding to the first test sample
    
    Returns:
    - DataFrames with backtest results and metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Initialize portfolio
    initial_capital = 10000.0
    cash = initial_capital
    portfolio_value = initial_capital
    
    # Trading parameters
    position = 0   # Current position: 0=none, 1=long, -1=short
    pos_size = 0.5  # Position size as fraction of portfolio
    
    # Track portfolio performance
    dates = []
    portfolio_values = [portfolio_value]
    positions = [position]
    trades = []
    
    # Generate some return volatility to make price changes more realistic
    np.random.seed(42)
    price_volatility = 0.02  # 2% daily volatility
    
    # Ensure start_idx is within bounds
    if start_idx >= len(prices):
        start_idx = len(prices) - len(X_test) - 1
    
    # Loop through each test sample
    for i in range(len(X_test)):
        # Get the index in the price series
        curr_idx = start_idx + i
        
        if curr_idx >= len(prices) - 1:
            break
            
        # Current and next day's prices
        curr_price = prices[curr_idx]
        
        # Add some realistic noise to price changes (this is just for simulation)
        price_noise = np.random.normal(0, price_volatility * curr_price)
        next_price = prices[curr_idx + 1] + price_noise
        
        # Calculate price change percentage
        price_change = (next_price - curr_price) / curr_price
        
        # Determine signal from prediction
        # 0 = neutral/hold, 1 = bullish/buy, 2 = bearish/sell
        signal = 0
        if y_pred[i] == 1:  # Predicted up
            signal = 1
        elif y_pred[i] == 2:  # Predicted down
            signal = -1
            
        # Execute trading logic
        # Close existing position if signal changes
        if position != 0 and signal != position:
            if position == 1:  # Close long position
                # Calculate position value with price change applied
                pos_value = portfolio_value * pos_size * (1 + price_change)
                # Update cash and portfolio value
                cash += pos_value
                portfolio_value = cash
                # Record trade
                trades.append({
                    'Day': i,
                    'Price': curr_price,
                    'Action': 'Sell',
                    'Shares': position,
                    'Cash': cash,
                    'Portfolio': portfolio_value
                })
            else:  # Close short position
                # Calculate position value with inverse price change applied
                pos_value = portfolio_value * pos_size * (1 - price_change)
                # Update cash and portfolio value
                cash += pos_value
                portfolio_value = cash
                # Record trade
                trades.append({
                    'Day': i, 
                    'Price': curr_price,
                    'Action': 'Cover',
                    'Shares': -position,
                    'Cash': cash,
                    'Portfolio': portfolio_value
                })
            position = 0
            
        # Enter new position if no current position and we have a signal
        if position == 0 and signal != 0:
            # Calculate position size based on current portfolio
            pos_value = portfolio_value * pos_size
            
            if signal == 1:  # Enter long position
                cash -= pos_value
                position = 1
                trades.append({
                    'Day': i,
                    'Price': curr_price,
                    'Action': 'Buy',
                    'Shares': position,
                    'Cash': cash,
                    'Portfolio': portfolio_value
                })
            else:  # Enter short position
                cash += pos_value
                position = -1
                trades.append({
                    'Day': i,
                    'Price': curr_price,
                    'Action': 'Short',
                    'Shares': -position,
                    'Cash': cash,
                    'Portfolio': portfolio_value
                })
        
        # Update portfolio value based on price change and position
        if position == 1:  # Long position
            # Position value changes with price
            pos_value = portfolio_value * pos_size * (1 + price_change)
            # Cash remains the same
            cash_value = portfolio_value * (1 - pos_size)
            # Update total portfolio value
            portfolio_value = cash_value + pos_value
        elif position == -1:  # Short position
            # Position value changes inversely with price
            pos_value = portfolio_value * pos_size * (1 - price_change)
            # Cash remains the same
            cash_value = portfolio_value * (1 - pos_size)
            # Update total portfolio value
            portfolio_value = cash_value + pos_value
        
        # Record portfolio state
        portfolio_values.append(portfolio_value)
        positions.append(position)
    
    # Close any open position at the end
    if position != 0:
        final_price = prices[min(start_idx + len(X_test), len(prices) - 1)]
        if position == 1:
            trades.append({
                'Day': len(X_test),
                'Price': final_price,
                'Action': 'Final Sell',
                'Shares': position,
                'Cash': cash + portfolio_value * pos_size,
                'Portfolio': portfolio_value
            })
        else:
            trades.append({
                'Day': len(X_test),
                'Price': final_price,
                'Action': 'Final Cover',
                'Shares': -position,
                'Cash': cash + portfolio_value * pos_size,
                'Portfolio': portfolio_value
            })
    
    # Calculate performance metrics
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate buy and hold performance
    start_price = prices[start_idx]
    end_price = prices[min(start_idx + len(X_test) - 1, len(prices) - 1)]
    buy_hold_return = (end_price / start_price - 1) * 100
    
    strategy_return = (portfolio_values[-1] / initial_capital - 1) * 100
    
    # Calculate annualized return (assuming daily data with 252 trading days)
    if len(returns) > 0:
        avg_daily_return = np.mean(returns)
        annualized_return = ((1 + avg_daily_return) ** 252 - 1) * 100
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = (avg_daily_return * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        max_returns = np.maximum.accumulate(cum_returns)
        drawdowns = (max_returns - cum_returns) / max_returns
        max_drawdown = np.max(drawdowns) * 100
    else:
        annualized_return = 0
        volatility = 0
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Print performance summary
    print(f"Starting capital: ${initial_capital:.2f}")
    print(f"Final portfolio value: ${portfolio_values[-1]:.2f}")
    print(f"Strategy return: {strategy_return:.2f}%")
    print(f"Buy & hold return: {buy_hold_return:.2f}%")
    print(f"Outperformance: {strategy_return - buy_hold_return:.2f}%")
    print(f"Annualized return: {annualized_return:.2f}%")
    print(f"Volatility: {volatility:.2f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    print(f"Total trades: {len(trades)}")
    
    # Create DataFrames
    trades_df = pd.DataFrame(trades)
    
    portfolio_df = pd.DataFrame({
        'Day': range(len(portfolio_values)),
        'Portfolio_Value': portfolio_values,
        'Position': positions
    })
    
    # Create metrics DataFrame
    metrics = {
        'Initial_Capital': initial_capital,
        'Final_Value': portfolio_values[-1],
        'Total_Return_Pct': strategy_return,
        'Buy_Hold_Return_Pct': buy_hold_return,
        'Outperformance_Pct': strategy_return - buy_hold_return,
        'Annualized_Return_Pct': annualized_return,
        'Volatility_Pct': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown_Pct': max_drawdown,
        'Total_Trades': len(trades)
    }
    
    metrics_df = pd.DataFrame([metrics])
    
    return trades_df, portfolio_df, metrics_df


def test_strategy():
    """
    Run a complete test of the tensor-based trading strategy using simulated data
    """
    print("Generating simulated financial data...")
    X, y, meta = generate_simulated_data(n_samples=5000, n_features=144, n_time_steps=10, n_classes=3)
    
    print(f"Generated data with shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Train WMTR model
    print("\n1. Training WMTR model...")
    wmtr_model = WMTR(lambda1=1.0, lambda2=1.0, r=3, max_iter=30)
    wmtr_train_metrics = wmtr_model.fit(X_train, y_train)
    
    # Plot WMTR training metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(wmtr_train_metrics['accuracy'], label='Accuracy')
    plt.plot(wmtr_train_metrics['f1'], label='F1 Score')
    plt.title('WMTR Training Metrics')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(wmtr_train_metrics['precision'], label='Precision')
    plt.plot(wmtr_train_metrics['recall'], label='Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('wmtr_training_metrics.png')
    plt.close()
    
    # Skip MDA if class imbalance is severe (it can be unstable)
    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts[class_counts > 0])
    
    use_mda = min_class_count >= 10
    
    if use_mda:
        # Train MDA model
        print("\n2. Training MDA model...")
        mda_model = MDA(mode1_dims=30, mode2_dims=5, lambda_reg=10.0, max_iter=20)
        try:
            mda_model.fit(X_train, y_train)
        except Exception as e:
            print(f"MDA training failed: {e}")
            print("Skipping MDA evaluation...")
            use_mda = False
    else:
        print("\nSkipping MDA due to extreme class imbalance...")
    
    # Evaluate WMTR model
    wmtr_pred = wmtr_model.predict(X_test)
    wmtr_accuracy = accuracy_score(y_test, wmtr_pred)
    wmtr_precision = precision_score(y_test, wmtr_pred, average='macro')
    wmtr_recall = recall_score(y_test, wmtr_pred, average='macro')
    wmtr_f1 = f1_score(y_test, wmtr_pred, average='macro')
    
    print("\nWMTR Test Results:")
    print(f"Accuracy: {wmtr_accuracy:.4f}")
    print(f"Precision: {wmtr_precision:.4f}")
    print(f"Recall: {wmtr_recall:.4f}")
    print(f"F1 Score: {wmtr_f1:.4f}")
    
    # Create confusion matrix for WMTR
    wmtr_cm = confusion_matrix(y_test, wmtr_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(wmtr_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('WMTR Confusion Matrix')
    plt.colorbar()
    class_names = ['Stable', 'Up', 'Down']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = wmtr_cm.max() / 2.
    for i in range(wmtr_cm.shape[0]):
        for j in range(wmtr_cm.shape[1]):
            plt.text(j, i, format(wmtr_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if wmtr_cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('wmtr_confusion_matrix.png')
    plt.close()
    
    if use_mda:
        # Evaluate MDA model
        mda_pred = mda_model.predict(X_test)
        mda_accuracy = accuracy_score(y_test, mda_pred)
        mda_precision = precision_score(y_test, mda_pred, average='macro')
        mda_recall = recall_score(y_test, mda_pred, average='macro')
        mda_f1 = f1_score(y_test, mda_pred, average='macro')
        
        print("\nMDA Test Results:")
        print(f"Accuracy: {mda_accuracy:.4f}")
        print(f"Precision: {mda_precision:.4f}")
        print(f"Recall: {mda_recall:.4f}")
        print(f"F1 Score: {mda_f1:.4f}")
        
        # Create confusion matrix for MDA
        mda_cm = confusion_matrix(y_test, mda_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(mda_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('MDA Confusion Matrix')
        plt.colorbar()
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = mda_cm.max() / 2.
        for i in range(mda_cm.shape[0]):
            for j in range(mda_cm.shape[1]):
                plt.text(j, i, format(mda_cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if mda_cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('mda_confusion_matrix.png')
        plt.close()
    
    # Run backtest with WMTR
    print("\n3. Running backtest with WMTR model...")
    prices = meta['prices']
    start_idx = len(X_train) + 10  # Adjust for lookback period
    
    wmtr_trades, wmtr_portfolio, wmtr_metrics = run_backtest(wmtr_model, X_test, y_test, prices, start_idx)
    
    # Run backtest with MDA if available
    if use_mda:
        print("\n4. Running backtest with MDA model...")
        mda_trades, mda_portfolio, mda_metrics = run_backtest(mda_model, X_test, y_test, prices, start_idx)
    
    # Plot strategy performance comparison
    plt.figure(figsize=(12, 6))
    plt.plot(wmtr_portfolio['Day'], wmtr_portfolio['Portfolio_Value'], label='WMTR Strategy')
    
    if use_mda:
        plt.plot(mda_portfolio['Day'], mda_portfolio['Portfolio_Value'], label='MDA Strategy')
    
    # Calculate buy and hold values
    buy_hold_values = [10000]
    initial_price = prices[start_idx]
    for i in range(1, len(wmtr_portfolio)):
        idx = min(start_idx + i, len(prices) - 1)
        buy_hold_values.append(10000 * prices[idx] / initial_price)
    
    plt.plot(range(len(buy_hold_values)), buy_hold_values, linestyle='--', label='Buy & Hold')
    
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy_comparison.png')
    plt.close()
    
    # Plot positions over time for WMTR
    plt.figure(figsize=(12, 6))
    
    # First subplot: Position values
    plt.subplot(2, 1, 1)
    plt.plot(wmtr_portfolio['Day'], wmtr_portfolio['Portfolio_Value'], label='Portfolio Value')
    plt.title('WMTR Strategy - Portfolio and Positions')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Second subplot: Position indicators (1=long, 0=none, -1=short)
    plt.subplot(2, 1, 2)
    plt.plot(wmtr_portfolio['Day'], wmtr_portfolio['Position'], marker='.', linestyle='-')
    plt.yticks([-1, 0, 1], ['Short', 'None', 'Long'])
    plt.xlabel('Trading Day')
    plt.ylabel('Position')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('wmtr_positions.png')
    plt.close()
    
    # Analyze w2 weights from WMTR (time importance)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(wmtr_model.w2)), wmtr_model.w2)
    plt.title('WMTR Time Step Importance Weights')
    plt.xlabel('Time Step (0=oldest, 9=newest)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.savefig('wmtr_time_weights.png')
    plt.close()
    
    # Save results to CSV
    test_prices = []
    for i in range(len(y_test)):
        idx = min(start_idx + i, len(prices) - 1)
        test_prices.append(prices[idx])
        
    results_df = pd.DataFrame({
        'True': y_test,
        'WMTR_Pred': wmtr_pred,
        'Price': test_prices
    })
    
    if use_mda:
        results_df['MDA_Pred'] = mda_pred
    
    results_df.to_csv('prediction_results.csv', index=False)
    wmtr_trades.to_csv('wmtr_trades.csv', index=False)
    wmtr_portfolio.to_csv('wmtr_portfolio.csv', index=False)
    wmtr_metrics.to_csv('wmtr_metrics.csv', index=False)
    
    if use_mda:
        mda_trades.to_csv('mda_trades.csv', index=False)
        mda_portfolio.to_csv('mda_portfolio.csv', index=False)
        mda_metrics.to_csv('mda_metrics.csv', index=False)
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print("=" * 50)
    print(f"WMTR Accuracy: {wmtr_accuracy:.4f}, F1: {wmtr_f1:.4f}")
    
    if use_mda:
        print(f"MDA Accuracy: {mda_accuracy:.4f}, F1: {mda_f1:.4f}")
        
        if wmtr_f1 > mda_f1:
            print(f"WMTR outperformed MDA on F1 score by {(wmtr_f1 - mda_f1):.4f}")
        else:
            print(f"MDA outperformed WMTR on F1 score by {(mda_f1 - wmtr_f1):.4f}")
    
    print("\nTesting complete! Results saved to CSV files and plots.")


if __name__ == "__main__":
    print("Testing Tensor-based Trading Strategy with Simulated Data")
    print("=" * 70)
    test_strategy()