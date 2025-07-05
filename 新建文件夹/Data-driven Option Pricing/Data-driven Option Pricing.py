import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import scipy.stats as stats

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class PolicyNetwork(nn.Module):
    """Neural network for the policy function f(S)"""
    
    def __init__(self, input_dim=1, hidden_dims=[128, 128, 128], output_dim=1):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class ResidualNetwork(nn.Module):
    """Residual network for complex function approximation"""
    
    def __init__(self, input_dim=1, hidden_dim=128, num_blocks=5, output_dim=1):
        super(ResidualNetwork, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x)

class OptionPriceNetwork(nn.Module):
    """Neural network for the option price function V0(S)"""
    
    def __init__(self, input_dim=1, hidden_dims=[256, 256, 256, 256, 256], output_dim=1, activation='relu'):
        super(OptionPriceNetwork, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def generate_CIR_paths(S0, a, b, sigma0, T, dt, num_paths, path_length=None):
    """
    Generate stock price paths using Cox-Ingersoll-Ross (CIR) model:
    dS = a(b - S)dt + sigma0*sqrt(S)*dW
    
    Parameters:
    - S0: Initial stock price
    - a: Mean reversion speed
    - b: Long-term mean
    - sigma0: Volatility parameter
    - T: Time horizon
    - dt: Time step size
    - num_paths: Number of paths to generate
    - path_length: Optional parameter for fixed path length (default is T/dt)
    
    Returns:
    - Array of stock price paths
    """
    if path_length is None:
        path_length = int(T / dt)
    
    paths = np.zeros((num_paths, path_length + 1))
    paths[:, 0] = S0
    
    # Generate paths
    for i in range(num_paths):
        S = S0
        for j in range(1, path_length + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = a * (b - S) * dt + sigma0 * np.sqrt(max(S, 0)) * dW
            S = max(S + dS, 0)  # Ensure S stays positive
            paths[i, j] = S
    
    return paths

def generate_GLV_paths(S0, a, b, sigma_loc_func, T, dt, num_paths, path_length=None):
    """
    Generate stock price paths using a Generalized Local Volatility (GLV) model:
    dS = a(b - S)dt + S*sigma_loc(S)*dW
    
    Parameters:
    - S0: Initial stock price
    - a: Mean reversion speed
    - b: Long-term mean
    - sigma_loc_func: Local volatility function sigma_loc(S)
    - T: Time horizon
    - dt: Time step size
    - num_paths: Number of paths to generate
    - path_length: Optional parameter for fixed path length (default is T/dt)
    
    Returns:
    - Array of stock price paths
    """
    if path_length is None:
        path_length = int(T / dt)
    
    paths = np.zeros((num_paths, path_length + 1))
    paths[:, 0] = S0
    
    # Generate paths
    for i in range(num_paths):
        S = S0
        for j in range(1, path_length + 1):
            sigma = sigma_loc_func(S)
            dW = np.random.normal(0, np.sqrt(dt))
            dS = a * (b - S) * dt + S * sigma * dW
            S = max(S + dS, 0)  # Ensure S stays positive
            paths[i, j] = S
    
    return paths

def extract_sample_paths(full_path, path_length, num_samples):
    """
    Extract overlapping sample paths from a single long trajectory
    
    Parameters:
    - full_path: Full stock price trajectory
    - path_length: Length of each sample path
    - num_samples: Number of sample paths to extract
    
    Returns:
    - Array of sample paths
    """
    if len(full_path) < path_length:
        raise ValueError("Full path is shorter than requested sample path length")
    
    max_start_idx = len(full_path) - path_length
    if num_samples > max_start_idx:
        print(f"Warning: Requested {num_samples} samples but only {max_start_idx} are available without replacement.")
        num_samples = min(num_samples, max_start_idx)
    
    # Randomly select starting indices for sample paths
    start_indices = np.random.choice(max_start_idx, num_samples, replace=False)
    
    # Extract sample paths
    sample_paths = np.array([full_path[i:i+path_length] for i in start_indices])
    
    return sample_paths

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using bisection method
    
    Parameters:
    - option_price: Option price
    - S: Stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - option_type: 'call' or 'put'
    
    Returns:
    - Implied volatility
    """
    def bs_price(sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    def f(sigma):
        return bs_price(sigma) - option_price
    
    # Check if valid option price
    if option_type == 'call':
        if option_price >= S or option_price <= max(0, S - K * np.exp(-r * T)):
            return np.nan
    else:  # put
        if option_price >= K * np.exp(-r * T) or option_price <= max(0, K * np.exp(-r * T) - S):
            return np.nan
    
    # Bisection method
    a, b = 0.001, 5.0
    if f(a) * f(b) > 0:
        return np.nan
    
    for _ in range(50):
        c = (a + b) / 2
        if abs(f(c)) < 1e-8:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

def train_policy_network(model, sample_paths, r, dt, num_epochs, batch_size, learning_rate):
    """
    Train the policy network to maximize expected log utility
    
    Parameters:
    - model: Neural network model for policy function f(S)
    - sample_paths: Dataset of stock price paths
    - r: Risk-free rate
    - dt: Time step size
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    
    Returns:
    - Trained model
    - Loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    num_paths, path_length = sample_paths.shape
    paths_tensor = torch.FloatTensor(sample_paths).to(device)
    dataset = TensorDataset(paths_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    
    for epoch in tqdm(range(num_epochs), desc="Training Policy Network"):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch_paths = batch[0]
            batch_size = batch_paths.shape[0]
            
            # Reset optimizer
            optimizer.zero_grad()
            
            # Calculate loss for batch
            batch_loss = 0
            for j in range(batch_size):
                path = batch_paths[j]
                path_loss = 0
                
                for n in range(path_length - 1):
                    S_n = path[n].unsqueeze(0)
                    S_n_plus_1 = path[n+1]
                    
                    # Get policy value f(S_n)
                    f_S_n = model(S_n)
                    
                    # Calculate terms in the loss function
                    term1 = r * (1 - f_S_n) * dt
                    term2 = f_S_n * (S_n_plus_1 - S_n) / S_n
                    term3 = -0.5 * f_S_n**2 * (S_n_plus_1 - S_n)**2 / (S_n**2)
                    
                    # Sum up terms
                    path_loss += term1 + term2 + term3
                
                batch_loss += path_loss
            
            # Average loss over batch
            batch_loss = -batch_loss / batch_size  # Negative because we want to maximize
            
            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        # Average loss over epoch
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}")
    
    return model, loss_history

def construct_pricing_kernel(policy_model, sample_paths, r, dt):
    """
    Construct pricing kernel (rho) using the trained policy model
    
    Parameters:
    - policy_model: Trained policy network
    - sample_paths: Dataset of stock price paths
    - r: Risk-free rate
    - dt: Time step size
    
    Returns:
    - Pricing kernel values for each path (inverse of wealth process)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = policy_model.to(device)
    policy_model.eval()
    
    num_paths, path_length = sample_paths.shape
    pricing_kernels = np.zeros((num_paths, path_length))
    
    # Initial wealth X_0 = 1, so rho_0 = 1/X_0 = 1
    pricing_kernels[:, 0] = 1.0
    
    # Construct wealth process and pricing kernel for each path
    with torch.no_grad():
        for j in range(num_paths):
            path = sample_paths[j]
            X = 1.0  # Initial wealth
            
            for n in range(path_length - 1):
                S_n = path[n]
                S_n_plus_1 = path[n+1]
                
                # Get policy value f(S_n)
                S_n_tensor = torch.FloatTensor([S_n]).to(device)
                f_S_n = policy_model(S_n_tensor).item()
                
                # Update wealth
                dS_S = (S_n_plus_1 - S_n) / S_n
                X_next = X * (1 + r * dt + f_S_n * (dS_S - r * dt))
                X = X_next
                
                # Update pricing kernel (rho = 1/X)
                pricing_kernels[j, n+1] = 1.0 / X
    
    return pricing_kernels

def train_option_price_network(model, sample_paths, pricing_kernels, payoff_func, r, dt, num_epochs, batch_size, learning_rate):
    """
    Train the option price network to estimate option prices
    
    Parameters:
    - model: Neural network model for option price function V_0(S)
    - sample_paths: Dataset of stock price paths
    - pricing_kernels: Pricing kernel values for each path
    - payoff_func: Function to calculate option payoff at maturity
    - r: Risk-free rate
    - dt: Time step size
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    
    Returns:
    - Trained model
    - Loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate terminal payoffs for each path
    num_paths, path_length = sample_paths.shape
    terminal_payoffs = np.array([payoff_func(sample_paths[j, -1]) for j in range(num_paths)])
    
    # Create dataset and dataloader
    inputs = torch.FloatTensor(sample_paths[:, 0]).unsqueeze(1).to(device)  # Initial stock prices
    targets = torch.FloatTensor(terminal_payoffs * pricing_kernels[:, -1]).unsqueeze(1).to(device)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    criterion = nn.MSELoss()
    
    for epoch in tqdm(range(num_epochs), desc="Training Option Price Network"):
        epoch_loss = 0
        num_batches = 0
        
        for inputs, targets in dataloader:
            # Reset optimizer
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss over epoch
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}")
    
    return model, loss_history

def evaluate_option_prices(option_model, stock_prices, r, T, K, option_type='call'):
    """
    Evaluate option prices using the trained option price network
    
    Parameters:
    - option_model: Trained option price network
    - stock_prices: Stock prices to evaluate option prices at
    - r: Risk-free rate
    - T: Time to maturity
    - K: Strike price
    - option_type: 'call' or 'put'
    
    Returns:
    - Option prices
    - Implied volatilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    option_model = option_model.to(device)
    option_model.eval()
    
    stock_prices_tensor = torch.FloatTensor(stock_prices).unsqueeze(1).to(device)
    
    with torch.no_grad():
        option_prices = option_model(stock_prices_tensor).cpu().numpy().flatten()
    
    # Calculate implied volatilities
    implied_vols = np.array([implied_volatility(option_prices[i], stock_prices[i], K, T, r, option_type) 
                             for i in range(len(stock_prices))])
    
    return option_prices, implied_vols

# Main simulation function
def run_data_driven_option_pricing_simulation():
    # Parameters
    r = 0.019  # Risk-free rate
    T = 0.1  # Time to maturity
    dt = 3e-3  # Time step size
    path_length = int(T / dt)
    
    # Simulation parameters
    num_epochs_policy = 2000
    num_epochs_option = 2000
    batch_size = 128
    learning_rate_policy = 0.001
    learning_rate_option = 0.001
    
    # CIR model parameters
    S0_CIR = 1.0
    a_CIR = 0.1
    b_CIR = 1.3
    sigma0_CIR = 0.2
    
    # Option parameters
    K = 1.0  # Strike price
    
    # Define payoff functions
    call_payoff = lambda S: max(S - K, 0)
    put_payoff = lambda S: max(K - S, 0)
    
    # Define local volatility function for CIR model
    sigma_loc_CIR = lambda S: sigma0_CIR / np.sqrt(max(S, 1e-6))
    
    # Generate long trajectory for CIR model
    print("Generating CIR model trajectory...")
    total_length = 100000
    full_trajectory_CIR = generate_CIR_paths(S0_CIR, a_CIR, b_CIR, sigma0_CIR, total_length*dt, dt, 1, total_length)[0]
    
    # Extract sample paths
    num_samples = 5000
    print(f"Extracting {num_samples} sample paths from trajectory...")
    sample_paths_CIR = extract_sample_paths(full_trajectory_CIR, path_length+1, num_samples)
    
    # Define policy network and option price networks
    policy_model_CIR = PolicyNetwork(input_dim=1, hidden_dims=[128, 128, 128], output_dim=1)
    call_option_model_CIR = OptionPriceNetwork(input_dim=1, hidden_dims=[256, 256, 256, 256, 256], output_dim=1, activation='relu')
    put_option_model_CIR = OptionPriceNetwork(input_dim=1, hidden_dims=[128, 128, 128, 128], output_dim=1, activation='leaky_relu')
    
    # Train policy network
    print("\nTraining policy network for CIR model...")
    policy_model_CIR, policy_loss_history_CIR = train_policy_network(
        policy_model_CIR, sample_paths_CIR, r, dt, num_epochs_policy, batch_size, learning_rate_policy
    )
    
    # Construct pricing kernel
    print("\nConstructing pricing kernel for CIR model...")
    pricing_kernels_CIR = construct_pricing_kernel(policy_model_CIR, sample_paths_CIR, r, dt)
    
    # Train option price networks
    print("\nTraining call option price network for CIR model...")
    call_option_model_CIR, call_loss_history_CIR = train_option_price_network(
        call_option_model_CIR, sample_paths_CIR, pricing_kernels_CIR, call_payoff, r, dt, num_epochs_option, batch_size, learning_rate_option
    )
    
    print("\nTraining put option price network for CIR model...")
    put_option_model_CIR, put_loss_history_CIR = train_option_price_network(
        put_option_model_CIR, sample_paths_CIR, pricing_kernels_CIR, put_payoff, r, dt, num_epochs_option, batch_size, learning_rate_option
    )
    
    # Evaluate option prices and implied volatilities
    stock_prices = np.linspace(0.8, 1.2, 41)
    moneyness = stock_prices / K
    
    # For CIR model
    print("\nEvaluating option prices for CIR model...")
    call_prices_CIR, call_ivs_CIR = evaluate_option_prices(call_option_model_CIR, stock_prices, r, T, K, 'call')
    put_prices_CIR, put_ivs_CIR = evaluate_option_prices(put_option_model_CIR, stock_prices, r, T, K, 'put')
    
    # Calculate true option prices for CIR model (this would be done with finite difference method in practice)
    # For demonstration, we'll use a simple Monte Carlo simulation with the true model parameters
    
    def true_cir_option_prices(S0_values, K, T, r, a, b, sigma0, option_type, num_paths=100000):
        option_prices = []
        
        for S0 in S0_values:
            paths = generate_CIR_paths(S0, a, b, sigma0, T, dt, num_paths, path_length+1)
            terminal_prices = paths[:, -1]
            
            if option_type == 'call':
                payoffs = np.maximum(terminal_prices - K, 0)
            else:  # put
                payoffs = np.maximum(K - terminal_prices, 0)
            
            option_price = np.mean(payoffs) * np.exp(-r * T)
            option_prices.append(option_price)
        
        return np.array(option_prices)
    
    # Calculate true option prices and implied volatilities
    print("\nCalculating true option prices for CIR model...")
    true_call_prices_CIR = true_cir_option_prices(stock_prices, K, T, r, a_CIR, b_CIR, sigma0_CIR, 'call')
    true_put_prices_CIR = true_cir_option_prices(stock_prices, K, T, r, a_CIR, b_CIR, sigma0_CIR, 'put')
    
    true_call_ivs_CIR = np.array([implied_volatility(true_call_prices_CIR[i], stock_prices[i], K, T, r, 'call') 
                                  for i in range(len(stock_prices))])
    true_put_ivs_CIR = np.array([implied_volatility(true_put_prices_CIR[i], stock_prices[i], K, T, r, 'put') 
                                for i in range(len(stock_prices))])
    
    # Plot results
    # 1. Policy function comparison
    x_grid = np.linspace(0.7, 1.5, 100)
    x_tensor = torch.FloatTensor(x_grid).unsqueeze(1)
    
    with torch.no_grad():
        learned_policy_CIR = policy_model_CIR(x_tensor).numpy().flatten()
    
    # True optimal policy function for CIR model
    true_policy_CIR = (a_CIR * (b_CIR / x_grid - 1) + r) / (sigma0_CIR**2 / x_grid)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, learned_policy_CIR, 'b-', label='Estimated value')
    plt.plot(x_grid, true_policy_CIR, 'r--', label='True value')
    plt.xlabel('Stock Price')
    plt.ylabel('Policy Function f(S)')
    plt.title('Optimal Policy Function for CIR Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('policy_function_CIR.png')
    plt.close()
    
    # 2. Implied volatility curves
    plt.figure(figsize=(10, 6))
    
    # Plot out-of-money options: calls for moneyness < 1, puts for moneyness > 1
    plt.plot(moneyness[moneyness < 1], call_ivs_CIR[moneyness < 1], 'b-', label='Our algorithm (Call)')
    plt.plot(moneyness[moneyness > 1], put_ivs_CIR[moneyness > 1], 'b-')
    
    plt.plot(moneyness[moneyness < 1], true_call_ivs_CIR[moneyness < 1], 'r--', label='True value (Call)')
    plt.plot(moneyness[moneyness > 1], true_put_ivs_CIR[moneyness > 1], 'r--')
    
    plt.xlabel('Moneyness (S/K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Curve for CIR Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('implied_volatility_CIR.png')
    plt.close()
    
    # 3. Training loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(policy_loss_history_CIR, label='Policy Network Loss')
    plt.axhline(y=-0.0032, color='r', linestyle='--', label='Theoretical Value')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss Curve for Policy Network (CIR Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('policy_loss_CIR.png')
    plt.close()
    
    # Return results for further analysis
    results = {
        'policy_model_CIR': policy_model_CIR,
        'call_option_model_CIR': call_option_model_CIR,
        'put_option_model_CIR': put_option_model_CIR,
        'policy_loss_history_CIR': policy_loss_history_CIR,
        'call_loss_history_CIR': call_loss_history_CIR,
        'put_loss_history_CIR': put_loss_history_CIR,
        'moneyness': moneyness,
        'call_ivs_CIR': call_ivs_CIR,
        'put_ivs_CIR': put_ivs_CIR,
        'true_call_ivs_CIR': true_call_ivs_CIR,
        'true_put_ivs_CIR': true_put_ivs_CIR
    }
    
    return results

# Run the simulation
if __name__ == "__main__":
    start_time = time.time()
    results = run_data_driven_option_pricing_simulation()
    end_time = time.time()
    print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")


def generate_GLV_model():
    # GLV model parameters
    S0_GLV = 1.0
    a_GLV = 3.0
    b_GLV = 0.98
    r_star = 0.019
    T_star = 0.1
    
    # Define implied volatility function
    def sigma_imp(K):
        if K <= 0.60:
            return 0.667
        elif K >= 1.33:
            return 0.454
        else:
            return 2.681 * K**2 - 5.466 * K + 2.981
    
    # Calculate derivatives of sigma_imp
    def sigma_imp_prime(K):
        if K <= 0.60 or K >= 1.33:
            return 0
        else:
            return 2 * 2.681 * K - 5.466
    
    def sigma_imp_prime2(K):
        if K <= 0.60 or K >= 1.33:
            return 0
        else:
            return 2 * 2.681
    
    # Define local volatility function using Dupire equation
    def sigma_loc(K):
        sigma = sigma_imp(K)
        sigma_prime = sigma_imp_prime(K)
        sigma_prime2 = sigma_imp_prime2(K)
        
        d1 = (-np.log(K) + T_star * (r_star + 0.5 * sigma**2)) / (sigma * np.sqrt(T_star))
        
        numerator = sigma**2 + 2 * r_star * sigma * K * T_star * sigma_prime
        denominator = 1 + K * d1 * np.sqrt(T_star) * sigma_prime**2 + sigma * T_star * K**2 * (sigma_prime2 - d1 * sigma_prime**2 * np.sqrt(T_star))
        
        return np.sqrt(numerator / denominator)
    
    return S0_GLV, a_GLV, b_GLV, sigma_loc

def run_GLV_model_simulation():
    # Parameters
    r = 0.019  # Risk-free rate
    T = 0.1  # Time to maturity
    dt = 3e-3  # Time step size
    path_length = int(T / dt)
    
    # Simulation parameters
    num_epochs_policy = 2000
    num_epochs_option = 2000
    batch_size = 256
    learning_rate_policy = 0.001
    learning_rate_option = 0.001
    
    # GLV model parameters
    S0_GLV, a_GLV, b_GLV, sigma_loc = generate_GLV_model()
    
    # Option parameters
    K = 1.0  # Strike price
    
    # Define payoff functions
    call_payoff = lambda S: max(S - K, 0)
    put_payoff = lambda S: max(K - S, 0)
    
    # Generate long trajectory for GLV model
    print("Generating GLV model trajectory...")
    total_length = 100000
    full_trajectory_GLV = generate_GLV_paths(S0_GLV, a_GLV, b_GLV, sigma_loc, total_length*dt, dt, 1, total_length)[0]
    
    # Extract sample paths
    num_samples = 5000
    print(f"Extracting {num_samples} sample paths from trajectory...")
    sample_paths_GLV = extract_sample_paths(full_trajectory_GLV, path_length+1, num_samples)
    
    # Define policy network and option price networks
    policy_model_GLV = ResidualNetwork(input_dim=1, hidden_dim=128, num_blocks=7, output_dim=1)
    call_option_model_GLV = OptionPriceNetwork(input_dim=1, hidden_dims=[128, 128, 128, 128, 128, 128, 128, 128], output_dim=1, activation='leaky_relu')
    put_option_model_GLV = ResidualNetwork(input_dim=1, hidden_dim=128, num_blocks=6, output_dim=1)
    
    # Train policy network
    print("\nTraining policy network for GLV model...")
    policy_model_GLV, policy_loss_history_GLV = train_policy_network(
        policy_model_GLV, sample_paths_GLV, r, dt, num_epochs_policy, batch_size, learning_rate_policy
    )
    
    # Construct pricing kernel
    print("\nConstructing pricing kernel for GLV model...")
    pricing_kernels_GLV = construct_pricing_kernel(policy_model_GLV, sample_paths_GLV, r, dt)
    
    # Train option price networks
    print("\nTraining call option price network for GLV model...")
    call_option_model_GLV, call_loss_history_GLV = train_option_price_network(
        call_option_model_GLV, sample_paths_GLV, pricing_kernels_GLV, call_payoff, r, dt, num_epochs_option, batch_size, learning_rate_option
    )
    
    print("\nTraining put option price network for GLV model...")
    put_option_model_GLV, put_loss_history_GLV = train_option_price_network(
        put_option_model_GLV, sample_paths_GLV, pricing_kernels_GLV, put_payoff, r, dt, num_epochs_option, batch_size, learning_rate_option
    )
    
    # Evaluate option prices and implied volatilities
    stock_prices = np.linspace(0.8, 1.2, 41)
    moneyness = stock_prices / K
    
    # For GLV model
    print("\nEvaluating option prices for GLV model...")
    call_prices_GLV, call_ivs_GLV = evaluate_option_prices(call_option_model_GLV, stock_prices, r, T, K, 'call')
    put_prices_GLV, put_ivs_GLV = evaluate_option_prices(put_option_model_GLV, stock_prices, r, T, K, 'put')
    
    # Calculate true option prices for GLV model
    print("\nCalculating true option prices for GLV model...")
    true_call_prices_GLV = true_glv_option_prices(stock_prices, K, T, r, a_GLV, b_GLV, sigma_loc, 'call')
    true_put_prices_GLV = true_glv_option_prices(stock_prices, K, T, r, a_GLV, b_GLV, sigma_loc, 'put')
    
    true_call_ivs_GLV = np.array([implied_volatility(true_call_prices_GLV[i], stock_prices[i], K, T, r, 'call') 
                                  for i in range(len(stock_prices))])
    true_put_ivs_GLV = np.array([implied_volatility(true_put_prices_GLV[i], stock_prices[i], K, T, r, 'put') 
                                for i in range(len(stock_prices))])
    
    # Plot results
    # 1. Policy function comparison
    x_grid = np.linspace(0.7, 1.5, 100)
    x_tensor = torch.FloatTensor(x_grid).unsqueeze(1)
    
    with torch.no_grad():
        learned_policy_GLV = policy_model_GLV(x_tensor).numpy().flatten()
    
    # True optimal policy function for GLV model (would need to be calculated numerically)
    # Here we use Monte Carlo to estimate it
    true_policy_GLV = estimate_true_glv_policy(x_grid, a_GLV, b_GLV, sigma_loc, r)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, learned_policy_GLV, 'b-', label='Estimated value')
    plt.plot(x_grid, true_policy_GLV, 'r--', label='True value')
    plt.xlabel('Stock Price')
    plt.ylabel('Policy Function f(S)')
    plt.title('Optimal Policy Function for GLV Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('policy_function_GLV.png')
    plt.close()
    
    # 2. Implied volatility curves
    plt.figure(figsize=(10, 6))
    
    # Plot out-of-money options: calls for moneyness < 1, puts for moneyness > 1
    plt.plot(moneyness[moneyness < 1], call_ivs_GLV[moneyness < 1], 'b-', label='Our algorithm (Call)')
    plt.plot(moneyness[moneyness > 1], put_ivs_GLV[moneyness > 1], 'b-')
    
    plt.plot(moneyness[moneyness < 1], true_call_ivs_GLV[moneyness < 1], 'r--', label='True value (Call)')
    plt.plot(moneyness[moneyness > 1], true_put_ivs_GLV[moneyness > 1], 'r--')
    
    plt.xlabel('Moneyness (S/K)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Curve for GLV Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('implied_volatility_GLV.png')
    plt.close()
    
    # 3. Training loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(policy_loss_history_GLV, label='Policy Network Loss')
    plt.axhline(y=-0.02, color='r', linestyle='--', label='Theoretical Value')  # Approximate theoretical value
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Loss Curve for Policy Network (GLV Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('policy_loss_GLV.png')
    plt.close()
    
    # Return results for further analysis
    results = {
        'policy_model_GLV': policy_model_GLV,
        'call_option_model_GLV': call_option_model_GLV,
        'put_option_model_GLV': put_option_model_GLV,
        'policy_loss_history_GLV': policy_loss_history_GLV,
        'call_loss_history_GLV': call_loss_history_GLV,
        'put_loss_history_GLV': put_loss_history_GLV,
        'moneyness': moneyness,
        'call_ivs_GLV': call_ivs_GLV,
        'put_ivs_GLV': put_ivs_GLV,
        'true_call_ivs_GLV': true_call_ivs_GLV,
        'true_put_ivs_GLV': true_put_ivs_GLV
    }
    
    return results

def true_glv_option_prices(S0_values, K, T, r, a, b, sigma_loc_func, option_type, num_paths=100000):
    option_prices = []
    dt = 3e-3
    path_length = int(T / dt)
    
    for S0 in S0_values:
        paths = generate_GLV_paths(S0, a, b, sigma_loc_func, T, dt, num_paths, path_length+1)
        terminal_prices = paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(terminal_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - terminal_prices, 0)
        
        option_price = np.mean(payoffs) * np.exp(-r * T)
        option_prices.append(option_price)
    
    return np.array(option_prices)

def estimate_true_glv_policy(S_values, a, b, sigma_loc_func, r):
    """
    Estimate the true optimal policy function for GLV model using finite differences
    """
    # For GLV model, the optimal policy is (mu(S) - r) / (sigma(S)^2)
    # where mu(S) = a(b - S)
    
    policy_values = []
    
    for S in S_values:
        mu = a * (b - S)  # Drift
        sigma = sigma_loc_func(S) * S  # Volatility
        
        # Optimal policy
        f_star = (mu - r) / (sigma**2)
        policy_values.append(f_star)
    
    return np.array(policy_values)