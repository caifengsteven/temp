import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gamma
from scipy.stats import norm
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ForwardVarianceCurveNetwork(nn.Module):
    """Neural network to model the forward variance curve"""
    def __init__(self, hidden_layers=3, hidden_dim=100):
        super(ForwardVarianceCurveNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, hidden_dim))
        
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.layers.append(nn.Linear(hidden_dim, 1))
        
        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, t):
        """
        Forward pass
        t: time inputs [batch_size, 1]
        """
        x = t
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.layers[-1](x)
        x = nn.functional.softplus(x)  # Ensure positive output
        return x

class ModifiedSOE:
    """Modified Summation of Exponentials method for simulating rough Bergomi model"""
    def __init__(self, H=0.07, eta=1.9, rho=-0.9, T=1.0, n=2000, N=20):
        """
        Initialize the mSOE scheme
        H: Hurst parameter in (0, 0.5)
        eta: volatility of volatility
        rho: correlation parameter in (-1, 0)
        T: time horizon
        n: number of time steps
        N: number of terms in the mSOE approximation
        """
        self.H = H
        self.eta = eta
        self.rho = rho
        self.T = T
        self.n = n
        self.dt = T / n
        self.N = N
        
        # Compute alpha_j and lambda_j using approach B
        self.alpha_j, self.lambda_j = self._compute_params_approach_B()
        
        # Compute covariance matrix for the Gaussian random vector
        self.cov_matrix = self._compute_cov_matrix()
        
    def _compute_params_approach_B(self):
        """Compute parameters for mSOE scheme using approach B"""
        # This is a simplified implementation - in practice, would use the full algorithm from the paper
        # We're using the example parameters from Table 1 in the paper for H=0.07 and N=20
        alpha_j = np.array([
            0.26118, 0.19002, 0.13840, 0.10717, 0.11366, 0.14757, 0.35898, 0.49341,
            0.67818, 0.93214, 1.28121, 1.76098, 2.42043, 3.32681, 4.57262, 6.28495,
            8.63850, 11.87339, 16.31967, 22.43096
        ])
        
        lambda_j = np.array([
            0.47726, 0.22777, 0.108690, 5.11098e-2, 1.93668e-2, 2.04153e-3, 1.0,
            2.09531, 4.390310, 9.19906, 1.92749, 40.38675, 84.62266, 177.31051,
            371.52005, 778.44877, 1631.08960, 3417.63439, 7160.99521, 15004.4875
        ])
        
        return alpha_j, lambda_j
    
    def _compute_cov_matrix(self):
        """Compute covariance matrix for the Gaussian random vector"""
        N = self.N
        dt = self.dt
        lambda_j = self.lambda_j
        H = self.H
        
        # Initialize covariance matrix
        cov = np.zeros((N + 2, N + 2))
        
        # Fill in the covariance matrix
        # First row/column (Brownian increment)
        cov[0, 0] = dt
        
        # First row/column with other components
        for l in range(1, N + 1):
            cov[0, l] = cov[l, 0] = (1.0 - np.exp(-lambda_j[l-1] * dt)) / lambda_j[l-1]
        
        # Last row/column (local part of the Volterra process)
        cov[0, N+1] = cov[N+1, 0] = np.sqrt(2.0 * H) * dt**(H + 0.5) / (H + 0.5)
        
        # Covariance between exponential components
        for k in range(1, N + 1):
            for l in range(1, N + 1):
                cov[k, l] = (1.0 - np.exp(-(lambda_j[k-1] + lambda_j[l-1]) * dt)) / (lambda_j[k-1] + lambda_j[l-1])
        
        # Covariance between exponential components and local part
        for l in range(1, N + 1):
            # This is a simplification of the lower incomplete gamma function
            # In practice, would use scipy.special.gammainc
            cov[l, N+1] = cov[N+1, l] = np.sqrt(2.0 * H) * (1.0 - np.exp(-lambda_j[l-1] * dt)) / (lambda_j[l-1]**(H + 0.5))
        
        # Variance of local part
        cov[N+1, N+1] = dt**(2 * H)
        
        return cov
    
    def simulate(self, xi_0, num_paths=10000):
        """
        Simulate paths of the rough Bergomi model
        xi_0: function that maps time to initial forward variance
        num_paths: number of paths to simulate
        
        Returns:
        S: stock price paths [num_paths, n+1]
        V: variance paths [num_paths, n+1]
        """
        n = self.n
        dt = self.dt
        N = self.N
        H = self.H
        eta = self.eta
        rho = self.rho
        alpha_j = self.alpha_j
        lambda_j = self.lambda_j
        
        # Initialize paths
        S = np.ones((num_paths, n + 1))
        V = np.zeros((num_paths, n + 1))
        
        # Initialize I_j_F (history components)
        I_j_F = np.zeros((num_paths, N))
        
        # Initial variance
        for i in range(num_paths):
            V[i, 0] = xi_0(0)
        
        # Perform Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(self.cov_matrix)
        
        # Simulate paths
        for i in tqdm(range(n), desc="Simulating paths"):
            # Generate correlated Gaussian random vector
            Z = np.random.normal(0, 1, (num_paths, N + 2))
            X = Z @ L.T
            
            # Extract components
            dW = X[:, 0]  # Brownian increment for W
            dW_perp = np.random.normal(0, np.sqrt(dt), num_paths)  # Independent Brownian increment for W'
            dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp  # Brownian increment for Z
            
            # Update history components I_j_F
            for j in range(N):
                if i == 0:
                    I_j_F[:, j] = 0
                else:
                    I_j_F[:, j] = np.exp(-lambda_j[j] * dt) * (I_j_F[:, j] + X[:, j+1])
            
            # Compute history part I_F
            I_F = np.sqrt(2.0 * H) * np.sum(alpha_j * I_j_F, axis=1)
            
            # Compute local part I_N
            I_N = np.sqrt(2.0 * H) * X[:, N+1]
            
            # Update variance process
            t_next = (i + 1) * dt
            V[:, i+1] = xi_0(t_next) * np.exp(eta * (I_F + I_N) - 0.5 * eta**2 * t_next**(2 * H))
            
            # Update stock price
            S[:, i+1] = S[:, i] * np.exp(-0.5 * V[:, i] * dt + np.sqrt(V[:, i]) * dZ)
        
        return S, V

def black_scholes_call(S0, K, r, sigma, T):
    """
    Calculate Black-Scholes price for a European call option
    S0: initial stock price
    K: strike price
    r: risk-free rate
    sigma: volatility
    T: time to maturity
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_implied_volatility(market_price, S0, K, r, T, max_iterations=100, precision=1e-8):
    """
    Calculate implied volatility using Newton-Raphson method
    market_price: observed option price
    S0: initial stock price
    K: strike price
    r: risk-free rate
    T: time to maturity
    max_iterations: maximum number of iterations
    precision: desired precision
    """
    # Initial guess
    sigma = 0.2
    
    for i in range(max_iterations):
        # Calculate price and vega
        price = black_scholes_call(S0, K, r, sigma, T)
        
        # Check if precision is reached
        if abs(price - market_price) < precision:
            return sigma
        
        # Calculate vega (derivative of price with respect to sigma)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        
        # Newton-Raphson update
        if abs(vega) < 1e-10:  # Avoid division by zero
            sigma = sigma * 1.5
        else:
            sigma = sigma - (price - market_price) / vega
        
        # Ensure sigma stays positive
        sigma = max(0.001, min(sigma, 5.0))
    
    # If max iterations reached, return last estimate
    return sigma

def wasserstein_1_distance(X, Y):
    """
    Calculate Wasserstein-1 distance between two empirical distributions
    X, Y: samples from the distributions
    """
    # Sort the samples
    X_sorted = np.sort(X)
    Y_sorted = np.sort(Y)
    
    # Calculate Wasserstein-1 distance
    return np.mean(np.abs(X_sorted - Y_sorted))

def train_neural_sde(forward_variance_model, target_xi_0, S_target, num_epochs=100, batch_size=4096, learning_rate=1e-4):
    """
    Train the neural SDE model to learn the forward variance curve
    forward_variance_model: neural network model for the forward variance curve
    target_xi_0: target forward variance curve function
    S_target: target stock price paths [num_paths, n+1]
    """
    # Set model to training mode
    forward_variance_model.train()
    
    # Define optimizer
    optimizer = optim.Adam(forward_variance_model.parameters(), lr=learning_rate)
    
    # Define scheduler for learning rate decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Initialize mSOE simulator
    msoe = ModifiedSOE()
    n = msoe.n
    T = msoe.T
    
    # Total number of paths
    num_paths = S_target.shape[0]
    
    # Split data into training and testing
    train_size = int(0.8192 * num_paths)
    S_train = S_target[:train_size]
    S_test = S_target[train_size:]
    
    # Initialize lists to track progress
    train_losses = []
    test_losses = []
    option_errors = []
    
    # Define time grid
    t_grid = np.linspace(0, T, n+1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(train_size)
        S_train_shuffled = S_train[indices]
        
        # Mini-batch training
        total_loss = 0
        num_batches = 0
        
        for i in range(0, train_size, batch_size):
            # Get batch
            batch_indices = indices[i:min(i+batch_size, train_size)]
            S_batch = S_train_shuffled[i:min(i+batch_size, train_size)]
            
            # Forward variance function based on neural network
            def nn_xi_0(t):
                t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
                return forward_variance_model(t_tensor).item()
            
            # Simulate paths using the current neural network model
            S_pred, _ = msoe.simulate(nn_xi_0, num_paths=len(S_batch))
            
            # Calculate Wasserstein-1 distance between target and predicted terminal distributions
            loss = wasserstein_1_distance(S_batch[:, -1], S_pred[:, -1])
            
            # Backpropagation
            optimizer.zero_grad()
            loss_tensor = torch.tensor(loss, requires_grad=True, device=device)
            loss_tensor.backward()
            optimizer.step()
            
            total_loss += loss
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        def nn_xi_0_test(t):
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
            return forward_variance_model(t_tensor).item()
        
        S_test_pred, _ = msoe.simulate(nn_xi_0_test, num_paths=len(S_test))
        test_loss = wasserstein_1_distance(S_test[:, -1], S_test_pred[:, -1])
        test_losses.append(test_loss)
        
        # Calculate option pricing error
        # We'll compute the error for a range of strikes
        strikes = np.linspace(0.8, 1.2, 5)
        max_option_error = 0
        
        for K in strikes:
            # Target option price (discounted expected payoff)
            target_payoffs = np.maximum(S_target[:, -1] - K, 0)
            target_price = np.mean(target_payoffs)
            
            # Predicted option price
            pred_payoffs = np.maximum(S_test_pred[:, -1] - K, 0)
            pred_price = np.mean(pred_payoffs)
            
            # Update maximum error
            max_option_error = max(max_option_error, abs(target_price - pred_price))
        
        option_errors.append(max_option_error)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}, Max Option Error: {max_option_error:.6f}")
    
    return train_losses, test_losses, option_errors

def test_mSOE_scheme():
    """Test the mSOE scheme for simulating the rough Bergomi model"""
    # Define parameters
    H = 0.07
    eta = 1.9
    rho = -0.9
    T = 1.0
    n = 2000
    N_values = [2, 4, 8, 16, 32]
    num_paths = 1000  # Reduced for quicker testing
    S0 = 1.0
    xi_0 = lambda t: 0.2352  # Constant forward variance curve
    
    # Strikes for option pricing
    log_moneyness = np.linspace(-0.5, 0.5, 11)
    strikes = S0 * np.exp(log_moneyness)
    
    # Initialize plot
    plt.figure(figsize=(12, 8))
    
    # Simulate and plot implied volatility curves for different N values
    for N in N_values:
        # Initialize mSOE simulator
        msoe = ModifiedSOE(H=H, eta=eta, rho=rho, T=T, n=n, N=N)
        
        # Simulate paths
        S, V = msoe.simulate(xi_0, num_paths=num_paths)
        
        # Calculate option prices and implied volatilities
        implied_vols = []
        
        for K in strikes:
            # Calculate option price as discounted expected payoff
            payoffs = np.maximum(S[:, -1] - K, 0)
            option_price = np.mean(payoffs)
            
            # Calculate implied volatility
            implied_vol = calculate_implied_volatility(option_price, S0, K, 0, T)
            implied_vols.append(implied_vol)
        
        # Plot implied volatility curve
        plt.plot(log_moneyness, implied_vols, marker='o', label=f'mSOE-{N}')
    
    # Add labels and legend
    plt.xlabel('Log Moneyness (k = log(K/S0))')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Curves for Different N Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('implied_volatility_curves.png')
    plt.show()
    
    return S, V

def test_neural_sde():
    """Test the neural SDE approach for learning different forward variance curves"""
    # Define parameters
    H = 0.07
    eta = 1.9
    rho = -0.9
    T = 1.0
    n = 2000
    N = 20
    num_paths = 10000
    S0 = 1.0
    
    # Define target forward variance curves
    xi_0_constant = lambda t: 0.2352
    xi_0_brownian = lambda t: 2.0 * abs(np.sin(t * 5))  # Using sin instead of Brownian for reproducibility
    xi_0_fbm = lambda t: 0.1 * (0.5 + abs(np.sin(t * 2) * t**H))  # Approximation of FBM
    
    # Initialize plot
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    
    # Test for each target forward variance curve
    for i, (xi_0, title) in enumerate(zip(
        [xi_0_constant, xi_0_brownian, xi_0_fbm],
        ["Constant", "Brownian", "Fractional Brownian"]
    )):
        # Initialize mSOE simulator
        msoe = ModifiedSOE(H=H, eta=eta, rho=rho, T=T, n=n, N=N)
        
        # Simulate target paths
        S_target, V_target = msoe.simulate(xi_0, num_paths=num_paths)
        
        # Plot target forward variance curve
        t_grid = np.linspace(0, T, 100)
        xi_0_values = [xi_0(t) for t in t_grid]
        axs[i, 0].plot(t_grid, xi_0_values)
        axs[i, 0].set_title(f'{title} Forward Variance Curve')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Forward Variance')
        axs[i, 0].grid(True)
        
        # Initialize neural network model
        forward_variance_model = ForwardVarianceCurveNetwork().to(device)
        
        # Limit to fewer epochs for testing
        num_epochs = 10
        
        # Train neural SDE
        train_losses, test_losses, option_errors = train_neural_sde(
            forward_variance_model, xi_0, S_target, num_epochs=num_epochs
        )
        
        # Plot losses
        axs[i, 1].plot(train_losses, label='Train')
        axs[i, 1].plot(test_losses, label='Test')
        axs[i, 1].set_title(f'{title} Wasserstein-1 Distance')
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].set_ylabel('Loss')
        axs[i, 1].legend()
        axs[i, 1].grid(True)
        
        # Plot option errors
        axs[i, 2].plot(option_errors)
        axs[i, 2].set_title(f'{title} Option Pricing Error')
        axs[i, 2].set_xlabel('Epoch')
        axs[i, 2].set_ylabel('Max Error')
        axs[i, 2].grid(True)
        
        # Generate learned forward variance curve
        learned_xi_0_values = []
        for t in t_grid:
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
            learned_xi_0_values.append(forward_variance_model(t_tensor).item())
        
        # Plot comparison of target and learned forward variance curves
        axs[i, 3].plot(t_grid, xi_0_values, label='Target')
        axs[i, 3].plot(t_grid, learned_xi_0_values, label='Learned')
        axs[i, 3].set_title(f'{title} Forward Variance Comparison')
        axs[i, 3].set_xlabel('Time')
        axs[i, 3].set_ylabel('Forward Variance')
        axs[i, 3].legend()
        axs[i, 3].grid(True)
        
        # Generate option prices using learned model
        def nn_xi_0(t):
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
            return forward_variance_model(t_tensor).item()
        
        S_pred, _ = msoe.simulate(nn_xi_0, num_paths=1000)  # Reduced for quicker testing
        
        # Calculate option prices for different strikes
        strikes = np.linspace(0.8, 1.2, 11)
        target_prices = []
        pred_prices = []
        
        for K in strikes:
            # Target option price
            target_payoffs = np.maximum(S_target[:, -1] - K, 0)
            target_price = np.mean(target_payoffs)
            target_prices.append(target_price)
            
            # Predicted option price
            pred_payoffs = np.maximum(S_pred[:, -1] - K, 0)
            pred_price = np.mean(pred_payoffs)
            pred_prices.append(pred_price)
        
        # Plot option price comparison
        axs[i, 4].plot(strikes, target_prices, label='Target')
        axs[i, 4].plot(strikes, pred_prices, label='Predicted')
        axs[i, 4].set_title(f'{title} Option Prices')
        axs[i, 4].set_xlabel('Strike')
        axs[i, 4].set_ylabel('Option Price')
        axs[i, 4].legend()
        axs[i, 4].grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_sde_results.png')
    plt.show()

if __name__ == "__main__":
    # Test mSOE scheme
    print("Testing mSOE scheme...")
    S, V = test_mSOE_scheme()
    
    # Test neural SDE approach
    print("\nTesting neural SDE approach...")
    test_neural_sde()