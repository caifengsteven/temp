import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import pdblp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
USE_SYNTHETIC_DATA = True  # Set to False to use real Bloomberg data

class QPLayer(Function):
    """
    Custom PyTorch layer for solving and differentiating through 
    quadratic programming problems.
    """
    
    @staticmethod
    def forward(ctx, Q, p, A, b, G, h, alpha, gamma1, gamma2, E, D):
        """
        Solve a QP of the form:
        min  0.5 * x^T Q x + p^T x + alpha * gamma1 * ||E x||_1 + (1-alpha) * gamma2/2 * ||D x||_2^2
        s.t. A x = b
             G x <= h
        """
        n = Q.shape[0]
        
        # Add L2 regularization to Q
        Q_reg = Q.clone() + (1 - alpha) * gamma2 * D.t() @ D
        
        # Convert to numpy for CVXPY
        Q_reg_np = Q_reg.detach().numpy()
        p_np = p.detach().numpy()
        
        # For L1 term, we need to reformulate
        if alpha > 0 and gamma1 > 0:
            # Reformulate with positive and negative parts: x = x+ - x-
            x_var = cp.Variable(2*n)
            
            # Convert tensors to numpy
            E_np = E.detach().numpy()
            
            # Compute alpha * gamma1 scalar value
            alpha_gamma1 = alpha.item() * gamma1.item()
            
            # Augmented matrices for reformulation
            Q_aug = np.block([
                [Q_reg_np, -Q_reg_np],
                [-Q_reg_np, Q_reg_np]
            ])
            
            # Create p_aug using numpy arrays consistently
            p_term = alpha_gamma1 * np.sum(E_np.T, axis=1)
            p_aug = np.concatenate([
                p_np + p_term,
                -p_np + p_term
            ])
            
            if A is not None:
                A_np = A.detach().numpy()
                A_aug = np.block([A_np, -A_np])
            else:
                A_aug = None
                
            if G is not None:
                G_np = G.detach().numpy()
                G_aug = np.block([G_np, -G_np])
            else:
                G_aug = None
                
            # Non-negativity constraints for x+ and x-
            I_n = np.eye(2*n)
            if G_aug is not None:
                G_aug = np.vstack([G_aug, -I_n])
                h_np = h.detach().numpy()
                h_aug = np.concatenate([h_np, np.zeros(2*n)])
            else:
                G_aug = -I_n
                h_aug = np.zeros(2*n)
            
            # Solve the QP
            constraints = []
            if A_aug is not None and b is not None:
                b_np = b.detach().numpy()
                constraints.append(cp.matmul(A_aug, x_var) == b_np)
            if G_aug is not None:
                constraints.append(cp.matmul(G_aug, x_var) <= h_aug)
                
            prob = cp.Problem(
                cp.Minimize(0.5 * cp.quad_form(x_var, Q_aug) + p_aug @ x_var),
                constraints
            )
            
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"Warning: QP solver status: {prob.status}")
                    # Fall back to a simpler QP without regularization
                    x_sol = np.zeros(n)
                else:
                    # Extract x = x+ - x-
                    x_sol = x_var.value[:n] - x_var.value[n:]
            except Exception as e:
                print(f"QP solver error: {e}")
                x_sol = np.zeros(n)
        else:
            # Standard QP without L1 term
            x_var = cp.Variable(n)
            
            # Build constraints
            constraints = []
            if A is not None and b is not None:
                A_np = A.detach().numpy()
                b_np = b.detach().numpy()
                constraints.append(A_np @ x_var == b_np)
            if G is not None and h is not None:
                G_np = G.detach().numpy()
                h_np = h.detach().numpy()
                constraints.append(G_np @ x_var <= h_np)
            
            # Solve the QP
            prob = cp.Problem(
                cp.Minimize(0.5 * cp.quad_form(x_var, Q_reg_np) + p_np @ x_var),
                constraints
            )
            
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"Warning: QP solver status: {prob.status}")
                    # Fall back to a simpler QP without regularization
                    x_sol = np.zeros(n)
                else:
                    x_sol = x_var.value
            except Exception as e:
                print(f"QP solver error: {e}")
                x_sol = np.zeros(n)
        
        # Save variables for backward pass
        x_tensor = torch.tensor(x_sol, dtype=torch.float32)
        ctx.save_for_backward(x_tensor, Q, p, A, b, G, h, 
                              alpha, gamma1, gamma2, E, D)
        
        return x_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        x, Q, p, A, b, G, h, alpha, gamma1, gamma2, E, D = ctx.saved_tensors
        
        # Compute Jacobian-vector product through implicit differentiation
        # of the KKT conditions
        
        # This is a simplified approach - for a real implementation,
        # we'd need to properly account for active constraints and the L1 term
        
        # Regularized Q
        Q_reg = Q + (1 - alpha) * gamma2 * D.t() @ D
        
        # Approximate gradient through L1 term using quadratic approximation
        # as described in the paper
        if alpha > 0 and gamma1 > 0:
            # Add L1 regularization approximation using diagonal weight matrix
            Ex = E @ x
            tau = 1e-4  # small constant to avoid division by zero
            W = torch.diag(1.0 / (torch.abs(Ex) + tau)) @ E.t() @ E
            Q_reg = Q_reg + alpha * gamma1 * W
        
        # Make sure Q_reg is well-conditioned
        jitter = 1e-6
        min_eig = torch.linalg.eigvalsh(Q_reg).min().item()
        if min_eig < jitter:
            Q_reg = Q_reg + (jitter - min_eig) * torch.eye(Q_reg.shape[0])
        
        # Compute descent direction by solving the linear system
        try:
            dx = torch.linalg.solve(Q_reg, -grad_output)
        except:
            # Fallback if solve fails
            dx = -grad_output / torch.trace(Q_reg)
        
        # Return gradients for all inputs (None for those we don't need)
        return dx, None, None, None, None, None, None, None, None, None, None

class RegularizedMVOModel(nn.Module):
    """
    Regularized Mean-Variance Optimization model with learnable regularization parameters.
    """
    def __init__(self, n_assets, n_features, prediction_dim=1, alpha=0.5, 
                 parameterized=True, reg_type='EN'):
        """
        Initialize the model.
        
        Args:
            n_assets: Number of assets
            n_features: Number of features per prediction dimension
            prediction_dim: Dimension of prediction features (1=univariate, 2=multivariate)
            alpha: Weight between L1 and L2 terms (0 = only L2, 1 = only L1)
            parameterized: Whether to use parameterized matrices E and D
            reg_type: Regularization type ('L1', 'L2', 'L2-COV', 'EN')
        """
        super(RegularizedMVOModel, self).__init__()
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.prediction_dim = prediction_dim
        self.alpha = alpha
        self.parameterized = parameterized
        self.reg_type = reg_type
        
        # Prediction model parameters
        self.beta = nn.Parameter(torch.randn(n_features, n_assets) * 0.1)
        
        # Regularization parameters
        if reg_type in ['L1', 'EN']:
            self.log_gamma1 = nn.Parameter(torch.tensor(0.0))  # L1 regularization strength
        else:
            self.log_gamma1 = nn.Parameter(torch.tensor(-10.0))  # Very small value
            
        if reg_type in ['L2', 'L2-COV', 'EN']:
            self.log_gamma2 = nn.Parameter(torch.tensor(0.0))  # L2 regularization strength
        else:
            self.log_gamma2 = nn.Parameter(torch.tensor(-10.0))  # Very small value
        
        # Parameterized matrices
        if parameterized:
            if reg_type in ['L1', 'EN']:
                self.theta_E = nn.Parameter(torch.zeros(n_assets))
            if reg_type in ['L2', 'EN']:
                self.theta_D = nn.Parameter(torch.zeros(n_assets))
                
    def forward(self, X, V, delta=10.0, A=None, b=None, G=None, h=None):
        """
        Forward pass of the model.
        
        Args:
            X: Feature matrix (batch_size, n_features)
            V: Covariance matrix (n_assets, n_assets)
            delta: Risk aversion parameter
            A, b, G, h: Constraints matrices and vectors
            
        Returns:
            z: Optimal portfolio weights
            y_pred: Predicted returns
        """
        batch_size = X.shape[0]
        
        # Predict returns
        y_pred = X @ self.beta  # (batch_size, n_assets)
        
        # Get regularization parameters
        gamma1 = 10.0 ** self.log_gamma1
        gamma2 = 10.0 ** self.log_gamma2
        
        # Create regularization matrices
        if self.parameterized:
            if self.reg_type in ['L1', 'EN']:
                E = torch.diag(torch.sigmoid(self.theta_E))
            else:
                E = torch.eye(self.n_assets)
                
            if self.reg_type in ['L2', 'EN']:
                D = torch.diag(torch.sigmoid(self.theta_D))
            elif self.reg_type == 'L2-COV':
                # For L2-COV, use a more sophisticated D based on feature covariance
                # In a real implementation, this would be based on the covariance model
                D = torch.eye(self.n_assets)  # Simplified version
            else:
                D = torch.eye(self.n_assets)
        else:
            E = torch.eye(self.n_assets)
            D = torch.eye(self.n_assets)
        
        # Prepare for QP solver
        # Process each example in the batch
        z_batch = []
        for i in range(batch_size):
            Q = delta * V
            p = -y_pred[i]
            
            # Solve the regularized QP
            z = QPLayer.apply(Q, p, A, b, G, h, 
                             torch.tensor(self.alpha), gamma1, gamma2, E, D)
            z_batch.append(z)
            
        z_batch = torch.stack(z_batch)
        
        return z_batch, y_pred

def get_bloomberg_data(tickers, start_date, end_date):
    """Get price data from Bloomberg"""
    
    if USE_SYNTHETIC_DATA:
        # Generate synthetic data if Bloomberg is not available
        return generate_synthetic_data(tickers, start_date, end_date)
    
    try:
        # Initialize Bloomberg connection
        con = pdblp.BCon(timeout=10000)
        con.start()
        
        print(f"Getting data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Get price data
        field = 'PX_LAST'
        prices_data = {}
        
        for ticker in tickers:
            print(f"Requesting data for {ticker}...")
            df = con.bdh(ticker, field, start_date, end_date)
            
            if df is not None and not df.empty:
                if isinstance(df, pd.DataFrame):
                    if df.shape[1] == 1:  # If we have only one column
                        prices_data[ticker] = df.iloc[:, 0]
                    else:
                        print(f"Multiple columns found: {df.columns}")
                        # Try to find the right column
                        if field in df.columns:
                            prices_data[ticker] = df[field]
                        elif df.columns.nlevels > 1 and field in df.columns.levels[0]:
                            prices_data[ticker] = df[field].iloc[:, 0]
                        else:
                            print(f"Unexpected data format for {ticker}")
            else:
                print(f"Empty or None data returned for {ticker}")
        
        # Close connection
        con.stop()
        
        # Convert to DataFrame
        if prices_data:
            prices = pd.DataFrame(prices_data)
            return prices
        else:
            print("No valid data returned from Bloomberg")
            return generate_synthetic_data(tickers, start_date, end_date)
            
    except Exception as e:
        print(f"Error retrieving Bloomberg data: {e}")
        print("Falling back to synthetic data")
        return generate_synthetic_data(tickers, start_date, end_date)

def generate_synthetic_data(tickers, start_date, end_date):
    """Generate synthetic price data for testing"""
    print(f"Generating synthetic data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate dates
    dates = pd.date_range(start=start, end=end, freq='B')
    
    # Generate correlated synthetic price data
    n_days = len(dates)
    n_assets = len(tickers)
    
    # Create correlation matrix (introduce some positive and negative correlations)
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Randomly assign correlations between -0.7 and 0.7
            corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(-0.7, 0.7)
    
    # Ensure correlation matrix is positive definite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix += (-min_eig + 0.01) * np.eye(n_assets)
    
    # Create price paths
    prices = np.zeros((n_days, n_assets))
    prices[0] = 100  # Starting prices
    
    # Annualized parameters
    annual_returns = np.random.normal(0.05, 0.02, n_assets)  # Mean annual return 5%
    annual_vols = np.random.uniform(0.15, 0.35, n_assets)    # Annual volatilities 15%-35%
    
    # Daily parameters
    daily_returns = annual_returns / 252
    daily_vols = annual_vols / np.sqrt(252)
    
    # Generate multivariate normal returns
    cov_matrix = np.outer(daily_vols, daily_vols) * corr_matrix
    returns = np.random.multivariate_normal(daily_returns, cov_matrix, n_days-1)
    
    # Generate price paths
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i-1])
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    
    return df

def calculate_features(prices, window=252):
    """Calculate features for return prediction"""
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate 252-day trend (average return)
    trend = returns.rolling(window=window).mean()
    
    # Calculate carry (if we had front and back month futures data)
    # In this synthetic case, we'll create a proxy that's a function of trend
    # and some noise
    carry = trend + 0.001 * np.random.randn(*trend.shape)
    
    # Create a multi-asset feature DataFrame
    features_data = {}
    
    for asset in prices.columns:
        features_data[f'{asset}_trend'] = trend[asset]
        features_data[f'{asset}_carry'] = carry[asset]
        
    features = pd.DataFrame(features_data)
    
    return returns, features

def prepare_training_data(returns, features, forward_days=5):
    """Prepare training data with forward returns"""
    # Calculate forward returns (average of next forward_days days)
    forward_returns = returns.rolling(window=forward_days).mean().shift(-forward_days)
    
    # Align data
    valid_indices = features.dropna().index.intersection(forward_returns.dropna().index)
    
    if len(valid_indices) == 0:
        print("Warning: No valid data points found after alignment.")
        # Use a less restrictive approach
        valid_indices = features.dropna().index
    
    X = features.loc[valid_indices].values
    y = forward_returns.loc[valid_indices].fillna(0).values  # Fill NaNs with zeros
    
    return X, y, valid_indices

def evaluate_portfolio(weights, returns, risk_free_rate=0.0):
    """Calculate portfolio performance metrics"""
    # Calculate portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)
    
    # Calculate performance metrics
    mean_return = portfolio_returns.mean() * 252  # Annualized return
    volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate drawdowns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()
    
    # Calculate value at risk (95%)
    var = np.percentile(portfolio_returns, 5)
    
    # Calculate mean-variance cost
    delta = 10.0  # Risk aversion parameter
    mvo_cost = -mean_return + (delta/2) * (volatility**2)
    
    metrics = {
        'Mean': mean_return,
        'Volatility': volatility,
        'Sharpe': sharpe_ratio,
        'VaR': var,
        'Avg DD': avg_drawdown,
        'Max DD': max_drawdown,
        'MVO Cost': mvo_cost
    }
    
    return metrics, portfolio_returns

def train_model(model, X_train, y_train, V_train, epochs=100, lr=0.01,
                A=None, b=None, G=None, h=None):
    """Train the regularized MVO model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    V_tensor = torch.tensor(V_train, dtype=torch.float32)
    
    if A is not None:
        A_tensor = torch.tensor(A, dtype=torch.float32)
        b_tensor = torch.tensor(b, dtype=torch.float32)
    else:
        A_tensor = None
        b_tensor = None
        
    if G is not None:
        G_tensor = torch.tensor(G, dtype=torch.float32)
        h_tensor = torch.tensor(h, dtype=torch.float32)
    else:
        G_tensor = None
        h_tensor = None
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        z_pred, y_pred = model(X_tensor, V_tensor, delta=10.0, 
                               A=A_tensor, b=b_tensor, G=G_tensor, h=h_tensor)
        
        # Calculate MVO cost
        # c(z, y) = -z^T y + (delta/2) * z^T V z
        delta = 10.0
        mvo_cost = -torch.bmm(z_pred.unsqueeze(1), y_tensor.unsqueeze(2)).squeeze()
        mvo_cost += 0.5 * delta * torch.bmm(
            torch.bmm(z_pred.unsqueeze(1), V_tensor.unsqueeze(0).repeat(z_pred.shape[0], 1, 1)),
            z_pred.unsqueeze(2)
        ).squeeze()
        
        # Mean over batch
        loss = mvo_cost.mean()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
        losses.append(loss.item())
    
    return losses

def test_strategy(model, X_test, V_test, test_returns, test_dates,
                 A=None, b=None, G=None, h=None):
    """Test the trained model on out-of-sample data"""
    model.eval()
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    V_tensor = torch.tensor(V_test, dtype=torch.float32)
    
    if A is not None:
        A_tensor = torch.tensor(A, dtype=torch.float32)
        b_tensor = torch.tensor(b, dtype=torch.float32)
    else:
        A_tensor = None
        b_tensor = None
        
    if G is not None:
        G_tensor = torch.tensor(G, dtype=torch.float32)
        h_tensor = torch.tensor(h, dtype=torch.float32)
    else:
        G_tensor = None
        h_tensor = None
    
    with torch.no_grad():
        # Get portfolio weights
        z_pred, y_pred = model(X_tensor, V_tensor, delta=10.0,
                              A=A_tensor, b=b_tensor, G=G_tensor, h=h_tensor)
    
    # Convert to numpy
    weights = z_pred.numpy()
    
    # Create DataFrame of weights
    weights_df = pd.DataFrame(
        weights, 
        index=test_dates,
        columns=test_returns.columns
    )
    
    # Evaluate portfolio performance
    metrics, portfolio_returns = evaluate_portfolio(weights_df, test_returns)
    
    # Create portfolio returns series
    portfolio_returns_series = pd.Series(
        portfolio_returns.values,
        index=test_returns.index
    )
    
    return weights_df, metrics, portfolio_returns_series

def run_experiment(prices, window_size=2*252, test_size=252, 
                  prediction_type='univariate', constraint_type='unconstrained'):
    """Run a complete experiment with training and testing"""
    # Calculate returns and features
    returns, features = calculate_features(prices)
    
    # Get asset names
    asset_names = prices.columns
    n_assets = len(asset_names)
    
    # Determine prediction dimension based on type
    if prediction_type == 'univariate':
        # Use only trend features
        feature_cols = [f'{asset}_trend' for asset in asset_names]
        prediction_dim = 1
    else:  # multivariate
        # Use both trend and carry features
        feature_cols = [f'{asset}_trend' for asset in asset_names] + \
                      [f'{asset}_carry' for asset in asset_names]
        prediction_dim = 2
    
    # Select features
    selected_features = features[feature_cols]
    
    # Set up constraints based on type
    if constraint_type == 'unconstrained':
        # No constraints
        A = None
        b = None
        G = None
        h = None
    else:  # constrained
        # Market neutral constraint: sum of weights = 0
        A = np.ones((1, n_assets))
        b = np.zeros(1)
        
        # Position limits: -0.25 <= weights <= 0.25
        G = np.vstack([np.eye(n_assets), -np.eye(n_assets)])
        h = np.concatenate([0.25 * np.ones(n_assets), 0.25 * np.ones(n_assets)])
    
    # Define model configurations to test
    model_configs = [
        # Nominal models (no regularization)
        {'name': 'Nominal OLS', 'reg_type': 'L2', 'alpha': 0, 'parameterized': False},
        
        # User-defined regularization models
        {'name': 'OLS-L2', 'reg_type': 'L2', 'alpha': 0, 'parameterized': False},
        {'name': 'OLS-L2-COV', 'reg_type': 'L2-COV', 'alpha': 0, 'parameterized': False},
        {'name': 'OLS-L1', 'reg_type': 'L1', 'alpha': 1, 'parameterized': False},
        {'name': 'OLS-EN', 'reg_type': 'EN', 'alpha': 0.5, 'parameterized': False},
        
        # Parameterized regularization models
        {'name': 'OLS-L2-P', 'reg_type': 'L2', 'alpha': 0, 'parameterized': True},
        {'name': 'OLS-L1-P', 'reg_type': 'L1', 'alpha': 1, 'parameterized': True},
        {'name': 'OLS-EN-P', 'reg_type': 'EN', 'alpha': 0.5, 'parameterized': True}
    ]
    
    # Store results
    results = {}
    portfolio_returns = {}
    
    # Make sure we have enough data
    if len(returns) < window_size + test_size:
        window_size = max(252, len(returns) // 2 - test_size)
        print(f"Adjusting window size to {window_size} due to limited data")
    
    # Walk-forward testing
    start_idx = window_size
    end_idx = len(returns) - test_size
    
    # We'll do just one walk-forward period for simplicity
    # In a real implementation, we'd iterate through multiple periods
    train_end_idx = end_idx
    test_end_idx = min(train_end_idx + test_size, len(returns))
    
    print(f"Training period: {returns.index[start_idx]} to {returns.index[train_end_idx-1]}")
    print(f"Testing period: {returns.index[train_end_idx]} to {returns.index[test_end_idx-1] if test_end_idx-1 < len(returns) else 'end'}")
    
    # Prepare training data
    X_train, y_train, train_dates = prepare_training_data(
        returns.iloc[start_idx:train_end_idx],
        selected_features.iloc[start_idx:train_end_idx]
    )
    
    # Calculate training covariance
    V_train = returns.iloc[start_idx:train_end_idx].cov().values
    
    # Prepare test data
    test_slice = slice(train_end_idx, test_end_idx)
    X_test, y_test, test_dates = prepare_training_data(
        returns.iloc[test_slice],
        selected_features.iloc[test_slice]
    )
    
    # Calculate test covariance (would normally use training covariance in practice)
    V_test = V_train
    
    # Test returns for performance evaluation
    test_returns_data = returns.loc[test_dates]
    
    # Number of features for the model
    n_features = X_train.shape[1]
    
    # Train and test each model
    for config in model_configs:
        print(f"\nTraining model: {config['name']}")
        
        # Initialize model
        model = RegularizedMVOModel(
            n_assets=n_assets,
            n_features=n_features,
            prediction_dim=prediction_dim,
            alpha=config['alpha'],
            parameterized=config['parameterized'],
            reg_type=config['reg_type']
        )
        
        # Train model
        try:
            losses = train_model(
                model, X_train, y_train, V_train, 
                epochs=50, lr=0.01,  # Reduced epochs for faster execution
                A=A, b=b, G=G, h=h
            )
            
            # Test model
            weights, metrics, port_returns = test_strategy(
                model, X_test, V_test, test_returns_data, test_dates,
                A=A, b=b, G=G, h=h
            )
            
            # Store results
            results[config['name']] = metrics
            portfolio_returns[config['name']] = port_returns
            
            # Print metrics
            print(f"Results for {config['name']}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error during model training/testing: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next model
            continue
    
    # Compare results
    if results:
        results_df = pd.DataFrame(results).T
        
        # Plot portfolio values
        plt.figure(figsize=(12, 6))
        for name, returns_series in portfolio_returns.items():
            cumulative_returns = (1 + returns_series).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values, label=name)
        
        plt.title(f'Portfolio Performance: {prediction_type}, {constraint_type}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'portfolio_performance_{prediction_type}_{constraint_type}.png')
        plt.show()
        
        return results_df, portfolio_returns
    else:
        print("No valid results to return")
        return pd.DataFrame(), {}

def main():
    # Define parameters
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Define tickers - these can be adjusted based on what's available
    # We'll use 24 commodity futures markets as in the paper
    tickers = [
        # Energy
        'CL=F', 'HO=F', 'QS=F', 'XB=F',
        # Grains
        'BO=F', 'C=F', 'KW=F', 'S=F', 'SM=F', 'W=F',
        # Livestock
        'FC=F', 'LC=F', 'LH=F',
        # Metals
        'GC=F', 'HG=F', 'PA=F', 'PL=F', 'SI=F',
        # Softs
        'CC=F', 'CT=F', 'DF=F', 'KC=F', 'RS=F', 'SB=F'
    ]
    
    # Use short names for plotting
    short_names = [t.split('=')[0] for t in tickers]
    
    try:
        # Get price data
        prices = get_bloomberg_data(tickers, start_date, end_date)
        
        # Rename columns to short names for clarity
        prices.columns = short_names
        
        # Run experiments
        print("\n\n==== Experiment 1: Unconstrained MVO with Univariate Prediction ====")
        results1, port_returns1 = run_experiment(
            prices, 
            prediction_type='univariate',
            constraint_type='unconstrained'
        )
        
        print("\n\n==== Experiment 2: Unconstrained MVO with Multivariate Prediction ====")
        results2, port_returns2 = run_experiment(
            prices, 
            prediction_type='multivariate',
            constraint_type='unconstrained'
        )
        
        print("\n\n==== Experiment 3: Constrained MVO with Univariate Prediction ====")
        results3, port_returns3 = run_experiment(
            prices, 
            prediction_type='univariate',
            constraint_type='constrained'
        )
        
        print("\n\n==== Experiment 4: Constrained MVO with Multivariate Prediction ====")
        results4, port_returns4 = run_experiment(
            prices, 
            prediction_type='multivariate',
            constraint_type='constrained'
        )
        
        # Display summary of results
        print("\n\nSummary of Results (Sharpe Ratios):")
        try:
            summary = pd.DataFrame({
                'Unconstrained-Uni': results1['Sharpe'] if not results1.empty else [],
                'Unconstrained-Multi': results2['Sharpe'] if not results2.empty else [],
                'Constrained-Uni': results3['Sharpe'] if not results3.empty else [],
                'Constrained-Multi': results4['Sharpe'] if not results4.empty else []
            })
            print(summary)
            
            # Save results to CSV
            if not results1.empty:
                results1.to_csv('results_unconstrained_univariate.csv')
            if not results2.empty:
                results2.to_csv('results_unconstrained_multivariate.csv')
            if not results3.empty:
                results3.to_csv('results_constrained_univariate.csv')
            if not results4.empty:
                results4.to_csv('results_constrained_multivariate.csv')
        except Exception as e:
            print(f"Error creating summary: {e}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()